from .common import EvaluationMetric
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared CLIP loader.
# Module-level cache so that when multiple CLIP-based metrics are enabled in
# the same training run (e.g. legacy `clip_similarity` + canonical `clip_score`),
# the underlying CLIP-L/14 weights are loaded into HBM exactly once instead of
# duplicated per metric. Saves ~600 MB on TPU and a few seconds of startup.
# ---------------------------------------------------------------------------
_clip_cache: dict = {}


def _get_clip(modelname: str):
    """Return a cached (model, processor) pair for the given CLIP modelname.

    The pair is loaded once per process per modelname. Subsequent calls return
    the same objects, so JIT-compiled metric functions all reference the same
    weight arrays.
    """
    if modelname not in _clip_cache:
        from transformers import AutoProcessor, FlaxCLIPModel
        print(f"[metrics] Loading CLIP model '{modelname}' (cached for reuse)...")
        model = FlaxCLIPModel.from_pretrained(modelname, dtype=jnp.float16)
        processor = AutoProcessor.from_pretrained(modelname, use_fast=False, dtype=jnp.float16)
        _clip_cache[modelname] = (model, processor)
    return _clip_cache[modelname]


def _clip_image_text_cosine(model, pixel_values, input_ids, attention_mask):
    """Run a CLIP forward pass and return per-sample cosine(image, text).

    Pure helper so JIT'd metric wrappers can share the body. Returns shape [B].
    """
    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    gen_img_emb = out.image_embeds
    txt_emb = out.text_embeds
    gen_img_emb = gen_img_emb / (jnp.linalg.norm(gen_img_emb, axis=-1, keepdims=True) + 1e-6)
    txt_emb = txt_emb / (jnp.linalg.norm(txt_emb, axis=-1, keepdims=True) + 1e-6)
    return jnp.einsum('bd,bd->b', gen_img_emb, txt_emb)


def get_clip_metric(
    modelname: str = "openai/clip-vit-large-patch14",
):
    """Legacy CLIP distance metric: mean(1 - cos(image, text)). LOWER IS BETTER.

    Kept for backward-compatibility with runs from the prior project's HPO
    sweep that ranked by `best_val/clip_similarity`. New experiments should
    use `get_clip_score_metric()` (canonical Hessel et al. CLIPScore) for the
    primary tracker, but enabling both is essentially free since the CLIP
    forward pass and weights are shared via _get_clip().
    """
    model, processor = _get_clip(modelname)

    @jax.jit
    def calc(pixel_values, input_ids, attention_mask):
        cos = _clip_image_text_cosine(model, pixel_values, input_ids, attention_mask)
        return jnp.mean(1.0 - cos)

    def clip_metric(generated: jnp.ndarray, batch):
        original_conditions = batch['text']
        generated = (((generated + 1.0) / 2.0) * 255).astype(jnp.uint8)
        generated_inputs = processor(images=generated, return_tensors="jax", padding=True)
        return calc(
            generated_inputs['pixel_values'],
            original_conditions['input_ids'],
            original_conditions['attention_mask'],
        )

    return EvaluationMetric(function=clip_metric, name='clip_similarity')


def get_clip_score_metric(
    modelname: str = "openai/clip-vit-large-patch14",
):
    """Canonical CLIPScore (Hessel et al. 2021): 100 * max(cos(img, text), 0).

    HIGHER IS BETTER. Typical T2I models score 25-35 on natural prompts. This
    is the primary metric for our validation loop and best-tracker logic. The
    underlying CLIP forward pass and weights are shared with `clip_similarity`
    via _get_clip(), so enabling both metrics costs only one CLIP forward per
    validation step.
    """
    model, processor = _get_clip(modelname)

    @jax.jit
    def calc(pixel_values, input_ids, attention_mask):
        cos = _clip_image_text_cosine(model, pixel_values, input_ids, attention_mask)
        return jnp.mean(100.0 * jnp.maximum(cos, 0.0))

    def clip_score_metric(generated: jnp.ndarray, batch):
        original_conditions = batch['text']
        generated = (((generated + 1.0) / 2.0) * 255).astype(jnp.uint8)
        generated_inputs = processor(images=generated, return_tensors="jax", padding=True)
        return calc(
            generated_inputs['pixel_values'],
            original_conditions['input_ids'],
            original_conditions['attention_mask'],
        )

    return EvaluationMetric(
        function=clip_score_metric,
        name='clip_score',
        higher_is_better=True,
    )