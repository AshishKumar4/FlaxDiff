"""Train a JEPA encoder (I-JEPA over images, V-JEPA over video).

A sibling of training.py rather than a flag on it: the two share the data
pipeline, the registry and the trainer, but nothing else. A JEPA run has no
noise schedule, no sampler, no text conditioning and no VAE, and folding an
--objective switch into training.py would mean threading None through all of
them. The Objective seam is what makes the same trainer serve both.
"""

import argparse
import json
import os
import resource
import time
from datetime import datetime

import jax
import optax

from flaxdiff.data.dataloaders import get_dataset_grain, get_dataset_online
from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.jepa import (
    JepaObjective, multi_block_mask, get_linear_probe_metric, get_knn_probe_metric,
)
from flaxdiff.models.registry import build_model, canonicalize_architecture
from flaxdiff.trainer import GeneralDiffusionTrainer

os.environ['TOKENIZERS_PARALLELISM'] = "false"

OPTIMIZER_MAP = {'adam': optax.adam, 'adamw': optax.adamw, 'lamb': optax.lamb}


def boolean_string(s):
    return s if isinstance(s, bool) else s == 'True'


parser = argparse.ArgumentParser(description='Train a JEPA encoder')
parser.add_argument('--GRAIN_WORKER_COUNT', type=int, default=32)
parser.add_argument('--GRAIN_READ_THREAD_COUNT', type=int, default=140)
parser.add_argument('--GRAIN_READ_BUFFER_SIZE', type=int, default=96)
parser.add_argument('--GRAIN_WORKER_BUFFER_SIZE', type=int, default=100)

parser.add_argument('--dataset', type=str, default='oxford_flowers102')
parser.add_argument('--dataset_path', type=str, default='/home/mrwhite0racle/gcs_mount')
parser.add_argument('--dataset_seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--frames_per_sample', type=int, default=None,
                    help='Set for video (V-JEPA); leave unset for images (I-JEPA)')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps_per_epoch', type=int, default=None)
parser.add_argument('--val_steps_per_epoch', type=int, default=4)

parser.add_argument('--architecture', type=str, default='jepa_encoder',
                    choices=['jepa_encoder', 'jepa_video_encoder'])
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--emb_features', type=int, default=384)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--num_heads', type=int, default=6)
parser.add_argument('--mlp_ratio', type=int, default=4)
parser.add_argument('--ssm_attention_ratio', type=str, default='all-attn',
                    help='Mixer pattern per layer: "all-attn", "all-ssm", "3:1", ...')
parser.add_argument('--ssm_state_dim', type=int, default=64)
parser.add_argument('--use_hilbert', type=boolean_string, default=False)
parser.add_argument('--use_zigzag', type=boolean_string, default=False)
parser.add_argument('--dropout_rate', type=float, default=0.0)
parser.add_argument('--attention_impl', type=str, default=None,
                    choices=[None, 'xla', 'cudnn', 'tpu'])
parser.add_argument('--dtype', type=str, default=None)
parser.add_argument('--precision', type=str, default='default',
                    choices=['high', 'default', 'highest', 'None', None])

parser.add_argument('--predictor_features', type=int, default=192)
parser.add_argument('--predictor_layers', type=int, default=6)
parser.add_argument('--predictor_heads', type=int, default=6)

parser.add_argument('--num_target_blocks', type=int, default=4)
parser.add_argument('--target_scale', type=float, nargs=2, default=[0.15, 0.2])
parser.add_argument('--target_aspect', type=float, nargs=2, default=[0.75, 1.5])
parser.add_argument('--momentum', type=float, nargs=2, default=[0.996, 1.0],
                    help='Target-encoder EMA momentum, ramped over --momentum_steps')
parser.add_argument('--momentum_steps', type=int, default=None,
                    help='Defaults to the full training run')

parser.add_argument('--optimizer', type=str, default='adamw', choices=list(OPTIMIZER_MAP))
parser.add_argument('--optimizer_opts', type=str, default='{}')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--learning_rate_schedule', type=str, default=None, choices=[None, 'cosine'])
parser.add_argument('--learning_rate_peak', type=float, default=1.5e-3)
parser.add_argument('--learning_rate_end', type=float, default=1e-6)
parser.add_argument('--learning_rate_warmup_steps', type=int, default=10000)
parser.add_argument('--learning_rate_decay_epochs', type=int, default=1)
parser.add_argument('--clip_grads', type=float, default=0)

parser.add_argument('--probe_classes', type=int, default=None,
                    help='Number of classes for the frozen-encoder probes')
parser.add_argument('--probe_label_key', type=str, default='label')
parser.add_argument('--knn_k', type=int, default=20)

parser.add_argument('--distributed_training', type=boolean_string, default=True)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--load_from_checkpoint', type=str, default=None)
parser.add_argument('--resume_last_run', type=str, default=None)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--checkpoint_fs', type=str, default='local', choices=['local', 'gcs'])
parser.add_argument('--max_checkpoints_to_keep', type=int, default=1)
parser.add_argument('--wandb_project', type=str, default='mlops-msml605-project')
parser.add_argument('--wandb_entity', type=str, default='umd-projects')


def main(args):
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_OFILE, (65535, 65535))

    print("Initializing JAX")
    try:
        jax.distributed.initialize()
    except Exception:
        pass
    print(f"Number of devices: {jax.device_count()}")

    checkpoint_dir = (f"gs://{args.checkpoint_dir}" if args.checkpoint_fs == 'gcs'
                      else args.checkpoint_dir)

    dataset_generator = (get_dataset_online if 'online' in args.dataset
                         else get_dataset_grain)
    data = dataset_generator(
        args.dataset,
        batch_size=args.batch_size, image_scale=args.image_size,
        worker_count=args.GRAIN_WORKER_COUNT,
        read_thread_count=args.GRAIN_READ_THREAD_COUNT,
        read_buffer_size=args.GRAIN_READ_BUFFER_SIZE,
        worker_buffer_size=args.GRAIN_WORKER_BUFFER_SIZE,
        seed=args.dataset_seed,
        dataset_source=args.dataset_path,
    )
    steps_per_epoch = args.steps_per_epoch or data['train_len'] // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    architecture, suffix_flags = canonicalize_architecture(args.architecture)
    is_video = args.frames_per_sample is not None
    if is_video != (architecture == 'jepa_video_encoder'):
        raise ValueError(
            "--frames_per_sample and --architecture=jepa_video_encoder go together")

    grid = (args.image_size // args.patch_size,) * 2
    mask = multi_block_mask(
        grid,
        num_targets=args.num_target_blocks,
        scale=tuple(args.target_scale),
        aspect=tuple(args.target_aspect),
    )
    print(f"Mask geometry: {mask.block_area} tokens per target block "
          f"({mask.block_shapes}), {mask.num_context} context tokens of {mask.num_patches}")

    shared = {
        "emb_features": args.emb_features,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "ssm_attention_ratio": args.ssm_attention_ratio,
        "ssm_state_dim": args.ssm_state_dim,
        "dropout_rate": args.dropout_rate,
        "attention_impl": args.attention_impl,
        "dtype": args.dtype,
        "precision": args.precision,
    }
    encoder_config = {
        **shared,
        "patch_size": args.patch_size,
        "num_layers": args.num_layers,
        "use_hilbert": args.use_hilbert or suffix_flags.get('use_hilbert', False),
        "use_zigzag": args.use_zigzag or suffix_flags.get('use_zigzag', False),
    }
    predictor_config = {
        **shared,
        "grid": grid,
        "predictor_features": args.predictor_features,
        "num_layers": args.predictor_layers,
        "num_heads": args.predictor_heads,
        "factorized": is_video,
    }
    encoder = build_model(architecture, encoder_config)
    predictor_config["scan_order"] = encoder.scan_order
    predictor = build_model('jepa_predictor', predictor_config)

    sample_data_shape = ((args.frames_per_sample, args.image_size, args.image_size, 3)
                         if is_video else (args.image_size, args.image_size, 3))
    input_config = DiffusionInputConfig(
        sample_data_key='video' if is_video else 'image',
        sample_data_shape=sample_data_shape,
        conditions=[],
    )
    objective = JepaObjective(
        encoder=encoder,
        predictor=predictor,
        mask=mask,
        sample_data_key=input_config.sample_data_key,
        sample_data_shape=sample_data_shape,
        momentum=tuple(args.momentum),
        momentum_steps=args.momentum_steps or total_steps,
    )

    eval_metrics = []
    if args.probe_classes:
        eval_metrics = [
            get_linear_probe_metric(args.probe_classes, label_key=args.probe_label_key),
            get_knn_probe_metric(args.probe_classes, label_key=args.probe_label_key,
                                 k=args.knn_k),
        ]

    learning_rate = args.learning_rate
    if args.learning_rate_schedule == 'cosine':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate, peak_value=args.learning_rate_peak,
            warmup_steps=args.learning_rate_warmup_steps,
            decay_steps=steps_per_epoch * args.learning_rate_decay_epochs,
            end_value=args.learning_rate_end)
    solver = OPTIMIZER_MAP[args.optimizer](learning_rate, **json.loads(args.optimizer_opts))
    if args.clip_grads > 0:
        solver = optax.chain(optax.clip_by_global_norm(args.clip_grads), solver)

    experiment_name = args.experiment_name or (
        f"jepa-{args.dataset}/res-{args.image_size}/patch-{args.patch_size}/"
        f"mixer-{args.ssm_attention_ratio}/emb-{args.emb_features}/"
        f"lr-{args.learning_rate}/date-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    print("Experiment_Name:", experiment_name)

    wandb_config = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": experiment_name,
        "config": {
            "encoder": encoder_config,
            "predictor": predictor_config,
            "architecture": architecture,
            "mask": {"grid": grid, "block_shapes": mask.block_shapes,
                     "block_area": mask.block_area, "num_context": mask.num_context},
            "dataset": {"name": args.dataset, "length": data['train_len']},
            "arguments": vars(args),
        },
    }
    if args.resume_last_run is not None:
        wandb_config['id'] = args.resume_last_run

    trainer = GeneralDiffusionTrainer(
        model=encoder,
        optimizer=solver,
        input_config=input_config,
        rngs=jax.random.PRNGKey(4),
        objective=objective,
        name=experiment_name,
        wandb_config=wandb_config,
        distributed_training=args.distributed_training,
        checkpoint_base_path=checkpoint_dir,
        load_from_checkpoint=args.load_from_checkpoint,
        max_checkpoints_to_keep=args.max_checkpoints_to_keep,
        eval_metrics=eval_metrics,
        best_tracker_metric="val/knn_probe_accuracy" if eval_metrics else "train/best_loss",
        frames_per_sample=args.frames_per_sample,
    )

    start = time.time()
    trainer.fit(data, training_steps_per_epoch=steps_per_epoch, epochs=args.epochs,
                val_steps_per_epoch=args.val_steps_per_epoch)
    print(f"Training finished in {time.time() - start:.0f}s")


if __name__ == '__main__':
    main(parser.parse_args())
