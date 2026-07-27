"""Frozen-encoder probes, as evaluation metrics the validation loop can run.

A JEPA run has no samples to look at, so representation quality is the only
signal that the training curve cannot give you. Both probes score the pooled
embeddings the objective's validation step produces against the batch labels,
fitting on the first half of the batch and scoring on the second - cheap
enough to run every epoch, and honest, since nothing is scored on data it was
fit on. They measure trend, not absolute transfer accuracy; a full-dataset
probe is a separate offline job.
"""

import jax
import jax.numpy as jnp
import optax

from ..metrics.common import EvaluationMetric


def _split(embeddings, labels):
    fit = len(embeddings) // 2
    if fit == 0:
        raise ValueError("probes need at least two samples per validation batch")
    return embeddings[:fit], labels[:fit], embeddings[fit:], labels[fit:]


def linear_probe(embeddings, labels, num_classes: int, steps: int = 100,
                 learning_rate: float = 1e-2, weight_decay: float = 1e-4):
    """Accuracy of a logistic regression fit on half the batch, scored on the rest."""
    fit_x, fit_y, test_x, test_y = _split(embeddings, labels)
    mean, std = jnp.mean(fit_x, axis=0), jnp.std(fit_x, axis=0) + 1e-6
    fit_x, test_x = (fit_x - mean) / std, (test_x - mean) / std

    params = {"w": jnp.zeros((embeddings.shape[-1], num_classes)),
              "b": jnp.zeros((num_classes,))}
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)

    def objective(p):
        logits = fit_x @ p["w"] + p["b"]
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, fit_y))

    def step(carry, _):
        p, opt_state = carry
        grads = jax.grad(objective)(p)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return (optax.apply_updates(p, updates), opt_state), None

    (params, _), _ = jax.lax.scan(step, (params, optimizer.init(params)), None, length=steps)
    predicted = jnp.argmax(test_x @ params["w"] + params["b"], axis=-1)
    return jnp.mean(predicted == test_y)


def knn_probe(embeddings, labels, num_classes: int, k: int = 20):
    """Cosine k-NN accuracy, fit half against scored half."""
    fit_x, fit_y, test_x, test_y = _split(embeddings, labels)
    fit_x = fit_x / (jnp.linalg.norm(fit_x, axis=-1, keepdims=True) + 1e-8)
    test_x = test_x / (jnp.linalg.norm(test_x, axis=-1, keepdims=True) + 1e-8)

    similarity = test_x @ fit_x.T
    neighbours = jnp.argsort(-similarity, axis=-1)[:, :min(k, fit_x.shape[0])]
    votes = jnp.sum(jax.nn.one_hot(fit_y[neighbours], num_classes), axis=1)
    return jnp.mean(jnp.argmax(votes, axis=-1) == test_y)


def get_linear_probe_metric(num_classes: int, label_key: str = "label", **kwargs):
    return EvaluationMetric(
        function=lambda embeddings, batch: linear_probe(
            embeddings, jnp.asarray(batch[label_key]), num_classes, **kwargs),
        name="linear_probe_accuracy",
        higher_is_better=True,
    )


def get_knn_probe_metric(num_classes: int, label_key: str = "label", k: int = 20):
    return EvaluationMetric(
        function=lambda embeddings, batch: knn_probe(
            embeddings, jnp.asarray(batch[label_key]), num_classes, k=k),
        name="knn_probe_accuracy",
        higher_is_better=True,
    )
