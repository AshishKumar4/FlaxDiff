from typing import Any, Tuple, Mapping, Callable, List, Dict
from functools import partial
import flax.experimental
import flax.jax_utils
import flax.training
import flax.training.dynamic_scale
import jax.experimental.multihost_utils
import orbax
import orbax.checkpoint
import flax.jax_utils
import wandb.util
import wandb.wandb_run
from flaxdiff.models.common import kernel_init
from flaxdiff.models.simple_unet import Unet
from flaxdiff.models.simple_vit import UViT
import jax.experimental.pallas.ops.tpu.flash_attention
from flaxdiff.predictors import VPredictionTransform, EpsilonPredictionTransform, DiffusionPredictionTransform, DirectPredictionTransform, KarrasPredictionTransform
from flaxdiff.schedulers import CosineNoiseScheduler, NoiseScheduler, GeneralizedNoiseScheduler, KarrasVENoiseScheduler, EDMNoiseScheduler

import struct as st
import flax
import tqdm
from flax import linen as nn
import jax
from typing import Dict, Callable, Sequence, Any, Union
from dataclasses import field
import jax.numpy as jnp
import grain.python as pygrain
import numpy as np
import augmax

import matplotlib.pyplot as plt
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
import optax
from flax import struct                # Flax dataclasses
import time
import os
from datetime import datetime
from flax.training import orbax_utils
import functools

import json
# For CLIP
from transformers import AutoTokenizer, FlaxCLIPTextModel, CLIPTextModel
import wandb
import cv2
import argparse
from dataclasses import dataclass
import resource

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from orbax.checkpoint.utils import fully_replicated_host_local_array_to_global_array
from termcolor import colored
from flaxdiff.data.datasets import get_dataset_grain, get_dataset_online

import warnings
import traceback
from flaxdiff.utils import defaultTextEncodeModel

warnings.filterwarnings("ignore")

#####################################################################################################################
################################################# Initialization ####################################################
#####################################################################################################################

os.environ['TOKENIZERS_PARALLELISM'] = "false"

PROCESS_COLOR_MAP = {
    0: "green",
    1: "yellow",
    2: "magenta",
    3: "cyan", 
    4: "white",
    5: "light_blue",
    6: "light_red",
    7: "light_cyan"
}

#####################################################################################################################
################################################## Data Pipeline ####################################################
#####################################################################################################################

    

#####################################################################################################################
############################################### Training Pipeline ###################################################
#####################################################################################################################

from flaxdiff.trainer.diffusion_trainer import DiffusionTrainer

def boolean_string(s):
    if type(s) == bool:
        return s
    return s == 'True'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a diffusion model')
parser.add_argument('--GRAIN_WORKER_COUNT', type=int,
                    default=32, help='Number of grain workers')
# parser.add_argument('--GRAIN_READ_THREAD_COUNT', type=int,
#                     default=512, help='Number of grain read threads')
# parser.add_argument('--GRAIN_READ_BUFFER_SIZE', type=int,
#                     default=80, help='Grain read buffer size')
# parser.add_argument('--GRAIN_WORKER_BUFFER_SIZE', type=int,
#                     default=500, help='Grain worker buffer size')
# parser.add_argument('--GRAIN_WORKER_COUNT', type=int,
#                     default=32, help='Number of grain workers')
parser.add_argument('--GRAIN_READ_THREAD_COUNT', type=int,
                    default=128, help='Number of grain read threads')
parser.add_argument('--GRAIN_READ_BUFFER_SIZE', type=int,
                    default=80, help='Grain read buffer size')
parser.add_argument('--GRAIN_WORKER_BUFFER_SIZE', type=int,
                    default=50, help='Grain worker buffer size')

parser.add_argument('--dtype', type=str, default=None, help='dtype to use')
parser.add_argument('--attn_dtype', type=str, default=None, help='dtype to use for attention')
parser.add_argument('--precision', type=str, default=None, help='precision to use', choices=['high', 'default', 'highest', 'None', None])

parser.add_argument('--wandb_project', type=str, default='flaxdiff', help='Wandb project name')
parser.add_argument('--distributed_training', type=boolean_string, default=True, help='Should use distributed training or not')
parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name, would be generated if not provided')
parser.add_argument('--load_from_checkpoint', type=str,
                    default=None, help='Load from the best previously stored checkpoint. The checkpoint path should be provided')

parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--image_size', type=int, default=128, help='Image size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--steps_per_epoch', type=int,
                    default=None, help='Steps per epoch')
parser.add_argument('--dataset', type=str,
                    default='cc12m', help='Dataset to use')
parser.add_argument('--dataset_path', type=str,
                    default='/home/mrwhite0racle/gcs_mount/arrayrecord/cc12m', help="Dataset location path")

parser.add_argument('--noise_schedule', type=str, default='edm',
                    choices=['cosine', 'karras', 'edm'], help='Noise schedule')

parser.add_argument('--architecture', type=str, choices=["unet", "uvit"], default="unet", help='Architecture to use')
parser.add_argument('--emb_features', type=int, default=256, help='Embedding features')
parser.add_argument('--feature_depths', type=int, nargs='+', default=[64, 128, 256, 512], help='Feature depths')
parser.add_argument('--attention_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--flash_attention', type=boolean_string, default=False, help='Use Flash Attention')
parser.add_argument('--use_projection', type=boolean_string, default=False, help='Use projection')
parser.add_argument('--use_self_and_cross', type=boolean_string, default=False, help='Use self and cross attention')
parser.add_argument('--only_pure_attention', type=boolean_string, default=True, help='Use only pure attention or proper transformer in the attention blocks') 
parser.add_argument('--norm_groups', type=int, default=8, help='Number of normalization groups. 0 for RMSNorm')

parser.add_argument('--named_norms', type=boolean_string, default=False, help='Use named norms')

parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
parser.add_argument('--num_middle_res_blocks', type=int,  default=1, help='Number of middle residual blocks')
parser.add_argument('--activation', type=str, default='swish', help='activation to use')

parser.add_argument('--patch_size', type=int, default=16, help='Patch size for the transformer if using UViT')
parser.add_argument('--num_layers', type=int, default=12, help='Number of layers in the transformer if using UViT')
parser.add_argument('--num_heads', type=int, default=12, help='Number of heads in the transformer if using UViT')
parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio in the transformer if using UViT')

parser.add_argument('--dataset_seed', type=int, default=0, help='Dataset starting seed')

parser.add_argument('--dataset_test', type=boolean_string,
                    default=False, help='Run the dataset iterator for 3000 steps for testintg/benchmarking')

parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--checkpoint_fs', type=str, default='local', choices=['local', 'gcs'], help='Checkpoint filesystem')

parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adam', 'adamw', 'lamb'], help='Optimizer to use')
parser.add_argument('--optimizer_opts', type=str, default='{}', help='Optimizer options as a dictionary')
parser.add_argument('--learning_rate_schedule', type=str, default=None, choices=[None, 'cosine'], help='Learning rate schedule')
parser.add_argument('--learning_rate', type=float,
                    default=2.7e-4, help='Initial Learning rate')
parser.add_argument('--learning_rate_peak', type=float, default=3e-4, help='Learning rate peak')
parser.add_argument('--learning_rate_end', type=float, default=2e-4, help='Learning rate end')
parser.add_argument('--learning_rate_warmup_steps', type=int, default=10000, help='Learning rate warmup steps')
parser.add_argument('--learning_rate_decay_epochs', type=int, default=1, help='Learning rate decay epochs')

parser.add_argument('--autoencoder', type=str, default=None, help='Autoencoder model for Latend Diffusion technique',
                    choices=[None, 'stable_diffusion'])
parser.add_argument('--autoencoder_opts', type=str, 
                    default='{"modelname":"CompVis/stable-diffusion-v1-4"}', help='Autoencoder options as a dictionary')

parser.add_argument('--use_dynamic_scale', type=boolean_string, default=False, help='Use dynamic scale for training')
parser.add_argument('--clip_grads', type=float, default=0, help='Clip gradients to this value')
parser.add_argument('--add_residualblock_output', type=boolean_string, default=False, help='Add a residual block stage to the final output')
parser.add_argument('--kernel_init', type=None, default=1.0, help='Kernel initialization value')


def main(args):
    resource.setrlimit(
        resource.RLIMIT_CORE,
        (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    resource.setrlimit(
        resource.RLIMIT_OFILE,
        (65535, 65535))

    print("Initializing JAX")
    jax.distributed.initialize()

    # jax.config.update('jax_threefry_partitionable', True)
    print(f"Number of devices: {jax.device_count()}")
    print(f"Local devices: {jax.local_devices()}")

    DTYPE_MAP = {
        'bfloat16': jnp.bfloat16,
        'float32': jnp.float32,
        'None': None,
        None: None,
    }

    PRECISION_MAP = {
        'high': jax.lax.Precision.HIGH,
        'default': jax.lax.Precision.DEFAULT,
        'highes': jax.lax.Precision.HIGHEST,
        'None': None,
        None: None,
    }

    ACTIVATION_MAP = {
        'swish': jax.nn.swish,
        'mish': jax.nn.mish,
    }
    
    OPTIMIZER_MAP = {
        'adam' : optax.adam,
        'adamw' : optax.adamw,
        'lamb' : optax.lamb,
    }
    
    CHECKPOINT_DIR = args.checkpoint_dir
    if args.checkpoint_fs == 'gcs':
        CHECKPOINT_DIR = f"gs://{CHECKPOINT_DIR}"

    DTYPE = DTYPE_MAP[args.dtype]
    ATTN_DTYPE = DTYPE_MAP[args.attn_dtype if args.attn_dtype is not None else args.dtype]
    PRECISION = PRECISION_MAP[args.precision]

    GRAIN_WORKER_COUNT = args.GRAIN_WORKER_COUNT
    GRAIN_READ_THREAD_COUNT = args.GRAIN_READ_THREAD_COUNT
    GRAIN_READ_BUFFER_SIZE = args.GRAIN_READ_BUFFER_SIZE
    GRAIN_WORKER_BUFFER_SIZE = args.GRAIN_WORKER_BUFFER_SIZE

    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size

    dataset_name = args.dataset
    
    if 'online' in dataset_name:
        print("Using Online Dataset Generator")
        dataset_generator = get_dataset_online
        GRAIN_WORKER_BUFFER_SIZE *= 5
        GRAIN_READ_THREAD_COUNT *= 4
    else:
        dataset_generator = get_dataset_grain

    data = dataset_generator(
        args.dataset,
        batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE,
        worker_count=GRAIN_WORKER_COUNT, read_thread_count=GRAIN_READ_THREAD_COUNT,
        read_buffer_size=GRAIN_READ_BUFFER_SIZE, worker_buffer_size=GRAIN_WORKER_BUFFER_SIZE,
        seed=args.dataset_seed,
        dataset_source=args.dataset_path,
    )

    if args.dataset_test:
        dataset = iter(data['train']())

        for _ in tqdm.tqdm(range(2000)):
            batch = next(dataset)
            
    datalen = data['train_len']
    batches = datalen // BATCH_SIZE
    # Define the configuration using the command-line arguments
    attention_configs = [
        None,
    ]

    if args.attention_heads > 0:
        attention_configs += [
            {
                "heads": args.attention_heads, "dtype": ATTN_DTYPE, "flash_attention": args.flash_attention,
                "use_projection": args.use_projection, "use_self_and_cross": args.use_self_and_cross,
                "only_pure_attention": args.only_pure_attention,    
            },
        ] * (len(args.feature_depths) - 2)
        attention_configs += [
            {
                "heads": args.attention_heads, "dtype": ATTN_DTYPE, "flash_attention": False,
                "use_projection": False, "use_self_and_cross": args.use_self_and_cross,
                "only_pure_attention": args.only_pure_attention
            },
        ]
    else:
        print("Attention heads not provided, disabling attention")
        attention_configs += [
            None,
        ] * (len(args.feature_depths) - 1)

    INPUT_CHANNELS = 3
    DIFFUSION_INPUT_SIZE = IMAGE_SIZE
    autoencoder = None
    if args.autoencoder is not None:
        autoencoder_opts = json.loads(args.autoencoder_opts)
        if args.autoencoder == 'stable_diffusion':
            print("Using Stable Diffusion Autoencoder for Latent Diffusion Modeling")
            from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
            autoencoder = StableDiffusionVAE(**autoencoder_opts)
            INPUT_CHANNELS = 4
            DIFFUSION_INPUT_SIZE = DIFFUSION_INPUT_SIZE // 8
    
    model_config = {
        "emb_features": args.emb_features,
        "dtype": DTYPE,
        "precision": PRECISION,
        "activation": args.activation,
        "output_channels": INPUT_CHANNELS,
        "norm_groups": args.norm_groups,
    }
    
    MODEL_ARCHITECUTRES = {
        "unet": {
            "class": Unet,
            "kwargs": {
                "feature_depths": args.feature_depths,
                "attention_configs": attention_configs,
                "num_res_blocks": args.num_res_blocks,
                "num_middle_res_blocks": args.num_middle_res_blocks,
                "named_norms": args.named_norms,
            },
        },
        "uvit": {
            "class": UViT,
            "kwargs": {
                "patch_size":  args.patch_size,
                "num_layers":  args.num_layers,
                "num_heads":  args.num_heads,
                "dropout_rate": 0.1,
                "use_projection": False,
                "add_residualblock_output": args.add_residualblock_output,
            },
        }
    }
    
    model_architecture = MODEL_ARCHITECUTRES[args.architecture]['class']
    model_config.update(MODEL_ARCHITECUTRES[args.architecture]['kwargs'])
    
    if args.architecture == 'uvit':
        model_config['emb_features'] = 768
    
    CONFIG = {
        "model": model_config,
        "architecture": args.architecture,
        "dataset": {
            "name": dataset_name,
            "length": datalen,
            "batches": datalen // BATCH_SIZE,
        },
        "learning_rate": args.learning_rate,
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "input_shapes": {
            "x": (DIFFUSION_INPUT_SIZE, DIFFUSION_INPUT_SIZE, INPUT_CHANNELS),
            "temb": (),
            "textcontext": (77, 768)
        },
        "arguments": vars(args),
        "autoencoder": args.autoencoder,
        "autoencoder_opts": args.autoencoder_opts,
    }
    
    if args.kernel_init is not None:
        model_config['kernel_init'] = partial(kernel_init, scale=float(args.kernel_init))
        print("Using custom kernel initialization with scale", args.kernel_init)

    cosine_schedule = CosineNoiseScheduler(1000, beta_end=1)
    karas_ve_schedule = KarrasVENoiseScheduler(
        1, sigma_max=80, rho=7, sigma_data=0.5)
    edm_schedule = EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)

    if args.experiment_name and args.experiment_name != "":
        experiment_name = args.experiment_name
    else:
        experiment_name = "{name}_{date}".format(
            name="Diffusion_SDE_VE_TEXT", date=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
        
    experiment_name = experiment_name.format(**CONFIG['arguments'])
        
    print("Experiment_Name:", experiment_name)

    model_config['activation'] = ACTIVATION_MAP[model_config['activation']]
    unet = model_architecture(**model_config)

    learning_rate = CONFIG['learning_rate']
    optimizer = OPTIMIZER_MAP[args.optimizer]
    optimizer_opts = json.loads(args.optimizer_opts)
    if args.learning_rate_schedule == 'cosine':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate, peak_value=args.learning_rate_peak, warmup_steps=args.learning_rate_warmup_steps,
            decay_steps=batches * args.learning_rate_decay_epochs, end_value=args.learning_rate_end,
        )
    solver = optimizer(learning_rate, **optimizer_opts)
    
    if args.clip_grads > 0:
        solver = optax.chain(
            optax.clip_by_global_norm(args.clip_grads),
            solver,
        )

    wandb_config = {
        "project": args.wandb_project,
        "config": CONFIG,
        "name": experiment_name,
    }
    
    start_time = time.time()
    
    text_encoder = defaultTextEncodeModel()
    
    trainer = DiffusionTrainer(
        unet, optimizer=solver,
        input_shapes=CONFIG['input_shapes'],
        noise_schedule=edm_schedule,
        rngs=jax.random.PRNGKey(4),
        name=experiment_name,
        model_output_transform=KarrasPredictionTransform(
        sigma_data=edm_schedule.sigma_data),
        load_from_checkpoint=args.load_from_checkpoint,
        wandb_config=wandb_config,
        distributed_training=args.distributed_training,  
        checkpoint_base_path=CHECKPOINT_DIR,
        autoencoder=autoencoder,
        use_dynamic_scale=args.use_dynamic_scale,
        encoder=text_encoder,
    )
    
    if trainer.distributed_training:
        print("Distributed Training enabled")
    batches = batches if args.steps_per_epoch is None else args.steps_per_epoch
    print(f"Training on {CONFIG['dataset']['name']} dataset with {batches} samples")
    
    final_state = trainer.fit(data, batches, epochs=CONFIG['epochs'])
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

"""
New -->

for tpu-v4-32

python3 training.py --dataset=combined_online --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=256 --image_size=512 \
            --learning_rate=9e-5 --num_res_blocks=3 --emb_features 512 \
            --use_self_and_cross=False --precision=default --dtype=bfloat16 --attention_heads=16\
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-32_ldm_data-online_big'\
            --optimizer=adamw --feature_depths 128 256 512 512 --autoencoder=stable_diffusion \
            --norm_groups 0 --clip_grads 0.5 --only_pure_attention=True 

python3 training.py --dataset=combined_online --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=256 --image_size=128 \
            --learning_rate=1e-4 --num_res_blocks=3 --emb_features 512 \
            --use_self_and_cross=False --precision=default --dtype=bfloat16 --attention_heads=16\
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-32_data-online'\
            --optimizer=adamw --feature_depths 128 256 512 512 \
            --norm_groups 0 --clip_grads 0.5 --only_pure_attention=True
    
for tpu-v4-16

python3 training.py --dataset=combined_30m --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=128 --image_size=128 \
            --learning_rate=4e-5 --num_res_blocks=3 \
            --use_self_and_cross=False --dtype=bfloat16 --precision=default --attention_heads=8\
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-16_flaxdiff-0-1-9_light_combined_30m_1'\
            --optimizer=adamw --use_dynamic_scale=True --norm_groups 0 --only_pure_attention=False \
            --load_from_checkpoint='gs://flaxdiff-datasets-regional/checkpoints/dataset-combined_30m/image_size-128/batch-128-v4-16_flaxdiff-0-1-9_light_combined_30m_ldm_1'

----------------------------------------------------------------------------------------------------------------------------
Old -->

for tpu-v4-64

python3 training.py --dataset=combined_online --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=512 --image_size=512 --learning_rate=9e-5 \
            --architecture=uvit --num_layers=12 --emb_features=768 --norm_groups 0 --num_heads=12 \
            --dtype=bfloat16 --precision=default \
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-64_uvit_ldm_combined_online'\
            --optimizer=adamw --clip_grads 0.5 --autoencoder=stable_diffusion \
            --learning_rate_schedule=cosine --learning_rate_peak=2.7e-4 --learning_rate_end=4e-5 --learning_rate_warmup_steps=10000 --learning_rate_decay_epochs=2\
                
                
            --load_from_checkpoint='gs://flaxdiff-datasets-regional/checkpoints/dataset-combined_30m/image_size-512/batch-512-v4-64_flaxdiff-0-1-8_ldm_dyn_scale_NEW_ARCH_combined_30'


            --learning_rate_schedule=cosine --learning_rate_peak=4e-5 --learning_rate_end=9e-6 --learning_rate_warmup_steps=5000 --learning_rate_decay_epochs=2\
                

python3 training.py --dataset=combined_online --dataset_path=/home/mrwhite0racle/gcs_mount/ \
    --checkpoint_dir=flaxdiff-datasets-regional/checkpoints/ --checkpoint_fs=gcs \
    --epochs=40 --batch_size=512 --image_size=256 --learning_rate=4e-5 \
    --architecture=uvit --num_layers=12 --emb_features=768 --norm_groups 0 --num_heads=12 \
    --dtype=bfloat16 --precision=default \
    --experiment_name=dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-64_uvit_combined_online-larger_residualout \
    --optimizer=adamw --clip_grads 1 --add_residualblock_output=True

for tpu-v4-32

python3 training.py --dataset=combined_online --dataset_path=/home/mrwhite0racle/gcs_mount/ --checkpoint_dir=flaxdiff-datasets-regional/checkpoints/ \
    --checkpoint_fs=gcs --epochs=40 --batch_size=512 --image_size=256 --learning_rate=8e-5 \
    --num_res_blocks=3 --emb_features 512 --use_self_and_cross=False \
    --precision=default --dtype=bfloat16 --attention_heads=16 \
    --experiment_name=dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-64-_combined_online-finetuned-more-biggerdata \
    --optimizer=adamw --feature_depths 128 256 512 512 --only_pure_attention=True --named_norms=True --norm_groups=0 \
    --clip_grads=1 --load_from_checkpoint=gs://flaxdiff-datasets-regional/checkpoints/dataset-combined_online/image_size-256/batch-512-v4-64-_combined_online-finetuned-more

for tpu-v4-16

python3 training.py --dataset=combined_aesthetic --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=128 --image_size=512 \
            --learning_rate=8e-5 --num_res_blocks=3 \
            --use_self_and_cross=False --precision=default --attention_heads=16\
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-16_flaxdiff-0-1-8_new-combined_ldm_1'\
            --learning_rate_schedule=cosine --learning_rate_peak=1e-4 --learning_rate_end=4e-5 --learning_rate_warmup_steps=5000 --learning_rate_decay_epochs=1\
            --optimizer=adamw  --autoencoder=stable_diffusion --use_dynamic_scale=True\
            --load_from_checkpoint='gs://flaxdiff-datasets-regional/checkpoints/dataset-combined_aesthetic/image_size-512/batch-128-v4-16_flaxdiff-0-1-8__ldm_1'
"""
