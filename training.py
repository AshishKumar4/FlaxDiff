import jax
# jax.config.update("jax_enable_x64", True)

from typing import Any, Tuple, Mapping, Callable, List, Dict
from functools import partial
import flax.training.dynamic_scale
import jax.experimental.multihost_utils
from flaxdiff.models.common import kernel_init
from flaxdiff.models.simple_unet import Unet
from flaxdiff.models.simple_vit import UViT
from flaxdiff.models.simple_dit import SimpleDiT
from flaxdiff.models.simple_mmdit import SimpleMMDiT, HierarchicalMMDiT
import jax.experimental.pallas.ops.tpu.flash_attention
from flaxdiff.predictors import VPredictionTransform, EpsilonPredictionTransform, DiffusionPredictionTransform, DirectPredictionTransform, KarrasPredictionTransform
from flaxdiff.schedulers import CosineNoiseScheduler, NoiseScheduler, GeneralizedNoiseScheduler, KarrasVENoiseScheduler, EDMNoiseScheduler
from diffusers import FlaxUNet2DConditionModel
from flaxdiff.samplers.euler import EulerAncestralSampler
import struct as st
import flax
import tqdm
import jax.numpy as jnp
import re
import optax
import time
import os
from datetime import datetime

import json
# For CLIP
import argparse
from dataclasses import dataclass
import resource

from flaxdiff.data.dataloaders import get_dataset_grain, get_dataset_online

import warnings
import traceback
from flaxdiff.utils import defaultTextEncodeModel
from flaxdiff.inputs import DiffusionInputConfig, ConditionalInputConfig

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

from flaxdiff.trainer.general_diffusion_trainer import GeneralDiffusionTrainer

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
                    default=140, help='Number of grain read threads')
parser.add_argument('--GRAIN_READ_BUFFER_SIZE', type=int,
                    default=96, help='Grain read buffer size')
parser.add_argument('--GRAIN_WORKER_BUFFER_SIZE', type=int,
                    default=128, help='Grain worker buffer size')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--image_size', type=int, default=128, help='Image size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--steps_per_epoch', type=int,
                    default=None, help='Training Steps per epoch')
parser.add_argument('--val_steps_per_epoch', type=int,
                    default=4, help='Validation Steps per epoch')
parser.add_argument('--dataset', type=str,
                    default='laiona_coco', help='Dataset to use')
parser.add_argument('--dataset_path', type=str,
                    default='/home/mrwhite0racle/gcs_mount', help="Dataset location path")

parser.add_argument('--noise_schedule', type=str, default='edm',
                    choices=['cosine', 'karras', 'edm'], help='Noise schedule')

parser.add_argument('--architecture', type=str, 
                    choices=["unet", "uvit", "diffusers_unet_simple", "simple_dit", "simple_mmdit", "hierarchical_mmdit"], 
                    default="unet", help='Architecture to use')
parser.add_argument('--emb_features', type=int, default=256, help='Embedding features')
parser.add_argument('--feature_depths', type=int, nargs='+', default=[64, 128, 256, 512], help='Feature depths')
parser.add_argument('--attention_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--flash_attention', type=boolean_string, default=False, help='Use Flash Attention')
parser.add_argument('--use_projection', type=boolean_string, default=False, help='Use projection')
parser.add_argument('--use_self_and_cross', type=boolean_string, default=True, help='Use self and cross attention')
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
parser.add_argument('--use_hilbert', type=boolean_string, default=False, help='Use Hilbert patch reordering for the transformer if using UViT')

parser.add_argument('--dtype', type=str, default=None, help='dtype to use')
parser.add_argument('--precision', type=str, default=None, help='precision to use', choices=['high', 'default', 'highest', 'None', None])

parser.add_argument('--distributed_training', type=boolean_string, default=True, help='Should use distributed training or not')
parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name, would be generated if not provided')
parser.add_argument('--load_from_checkpoint', type=str,
                    default=None, help='Load from the best previously stored checkpoint. The checkpoint path should be provided')
parser.add_argument('--resume_last_run', type=str,
                    default=None, help='Resume the last run from the experiment name')
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
                    default='{"modelname":"pcuenq/sd-vae-ft-mse-flax"}', help='Autoencoder options as a dictionary')

parser.add_argument('--use_dynamic_scale', type=boolean_string, default=False, help='Use dynamic scale for training')
parser.add_argument('--clip_grads', type=float, default=0, help='Clip gradients to this value')
parser.add_argument('--add_residualblock_output', type=boolean_string, default=False, help='Add a residual block stage to the final output')
# parser.add_argument('--kernel_init', type=None, default=1.0, help='Kernel initialization value')

parser.add_argument('--max_checkpoints_to_keep', type=int, default=1, help='Max checkpoints to keep')

parser.add_argument('--wandb_project', type=str, default='mlops-msml605-project', help='Wandb project name')
parser.add_argument('--wandb_entity', type=str, default='umd-projects', help='Wandb entity name')

parser.add_argument('--val_metrics', type=str, nargs='+', default=['clip'], help='Validation metrics to use')
parser.add_argument('--best_tracker_metric', type=str, default='val/clip_similarity', help='Best tracker metric to use')

# parser.add_argument('--wandb_project', type=str, default='flaxdiff', help='Wandb project name')
# parser.add_argument('--wandb_entity', type=str, default='ashishkumar4', help='Wandb entity name')

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
                "heads": args.attention_heads, "dtype": DTYPE, "flash_attention": args.flash_attention,
                "use_projection": args.use_projection, "use_self_and_cross": args.use_self_and_cross,
                "only_pure_attention": args.only_pure_attention,    
            },
        ] * (len(args.feature_depths) - 2)
        attention_configs += [
            {
                "heads": args.attention_heads, "dtype": DTYPE, "flash_attention": False,
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
    
    if 'diffusers' in args.architecture:
        model_config = {}
    else:
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
                "add_residualblock_output": args.add_residualblock_output,
                "use_flash_attention": args.flash_attention,
                "use_self_and_cross": args.use_self_and_cross,
                "use_hilbert": args.use_hilbert,
            },
        },
        "simple_dit": {
            "class": SimpleDiT,
            "kwargs": {
                "patch_size":  args.patch_size,
                "num_layers":  args.num_layers,
                "num_heads":  args.num_heads,
                "dropout_rate": 0.1,
                "use_flash_attention": args.flash_attention,
                "mlp_ratio": args.mlp_ratio,
                "use_hilbert": args.use_hilbert,
            },
        },
        "simple_mmdit": {
            "class": SimpleMMDiT,
            "kwargs": {
                "patch_size":  args.patch_size,
                "num_layers":  args.num_layers,
                "num_heads":  args.num_heads,
                "dropout_rate": 0.1,
                "use_flash_attention": args.flash_attention,
                "mlp_ratio": args.mlp_ratio,
                "use_hilbert": args.use_hilbert,
            },
        },
        "hierarchical_mmdit": {
            "class": HierarchicalMMDiT,
            "kwargs": {
                "base_patch_size": args.patch_size // 2,  # Use half the patch size for base
                "emb_features": (args.emb_features - 256, args.emb_features, args.emb_features + 256),  # Default dims per stage
                "num_layers": (args.num_layers // 3, args.num_layers // 2, args.num_layers),  # Default layers per stage
                "num_heads": (args.num_heads - 2, args.num_heads, args.num_heads + 2),  # Default heads per stage
                "dropout_rate": 0.1,
                "use_flash_attention": args.flash_attention,
                "mlp_ratio": args.mlp_ratio,
                "use_hilbert": args.use_hilbert,
            },
        },
        "diffusers_unet_simple": {
            "class": FlaxUNet2DConditionModel,
            "kwargs": {
                "sample_size": DIFFUSION_INPUT_SIZE,
                "in_channels": INPUT_CHANNELS,
                "out_channels": INPUT_CHANNELS,
                "layers_per_block": args.num_res_blocks,
                "block_out_channels":args.feature_depths,
                "cross_attention_dim":args.emb_features,
                "dtype": DTYPE,
                "use_memory_efficient_attention": args.flash_attention,
                "attention_head_dim": args.attention_heads,
                "only_cross_attention": not args.use_self_and_cross,
            },
        }
    }
    
    model_architecture = MODEL_ARCHITECUTRES[args.architecture]['class']
    model_config.update(MODEL_ARCHITECUTRES[args.architecture]['kwargs'])
    
    if args.architecture == 'uvit':
        model_config['emb_features'] = 768
        
    sorted_args_json = json.dumps(vars(args), sort_keys=True)
    arguments_hash = hash(sorted_args_json)
    
    text_encoder = defaultTextEncodeModel()
    
    input_config = DiffusionInputConfig(
        sample_data_key='image',
        sample_data_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        conditions=[
            ConditionalInputConfig(
                encoder=text_encoder,
                conditioning_data_key='text',
                pretokenized=True,
                unconditional_input="",
                model_key_override="textcontext",
            )
        ]
    )
    
    eval_metrics = []
    # Validation metrics 
    if args.val_metrics is not None:
        if 'clip' in args.val_metrics:
            from flaxdiff.metrics.images import get_clip_metric
            print("Using CLIP metric for validation")
            eval_metrics.append(get_clip_metric())
    
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
        "input_shapes": input_config.get_input_shapes(
            autoencoder=autoencoder,
        ),
        "input_config": input_config.serialize(),
        "arguments": vars(args),
        "autoencoder": args.autoencoder,
        "autoencoder_opts": args.autoencoder_opts,
        "arguments_hash": arguments_hash,
    }

    cosine_schedule = CosineNoiseScheduler(1000, beta_end=1)
    karas_ve_schedule = KarrasVENoiseScheduler(
        1, sigma_max=80, rho=7, sigma_data=0.5)
    edm_schedule = EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
    
    if args.experiment_name is not None:
        experiment_name = args.experiment_name
    else:
        experiment_name = "manual-dataset-{dataset}/image_size-{image_size}/batch-{batch_size}/schd-{noise_schedule}/dtype-{dtype}/arch-{architecture}/lr-{learning_rate}/resblks-{num_res_blocks}/emb-{emb_features}/pure-attn-{only_pure_attention}"
    
    # Check if format strings are required using regex
    pattern = r"\{.+?\}"
    if re.search(pattern, experiment_name):
        experiment_name = f"{experiment_name}/" + "arguments_hash-{arguments_hash}/date-{date}"
        if autoencoder is not None:
            experiment_name = f"LDM-{experiment_name}"
                
        if args.use_hilbert:
            experiment_name = f"Hilbert-{experiment_name}"
                
        conf_args = CONFIG['arguments']
        conf_args['date'] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        conf_args['arguments_hash'] = arguments_hash
        # Format the string with the arguments
        experiment_name = experiment_name.format(**vars(args))
    else:
        # If no format strings, just use the provided name
        experiment_name = args.experiment_name
        
    print("Experiment_Name:", experiment_name)

    if 'activation' in model_config:
        model_config['activation'] = ACTIVATION_MAP[model_config['activation']]
        
    model = model_architecture(**model_config)
    
    # If using the Diffusers UNet, we need to wrap it 
    if 'diffusers' in args.architecture:
        from flaxdiff.models.general import BCHWModelWrapper
        model = BCHWModelWrapper(model)

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
        "entity": args.wandb_entity,
        "config": CONFIG,
        "name": experiment_name,
    }
    
    if args.resume_last_run is not None:
        wandb_config['id'] = args.resume_last_run
    
    start_time = time.time()
    
    trainer = GeneralDiffusionTrainer(
        model, optimizer=solver,
        input_config=input_config,
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
        native_resolution=IMAGE_SIZE,
        max_checkpoints_to_keep=args.max_checkpoints_to_keep,
        eval_metrics=eval_metrics,
        best_tracker_metric=args.best_tracker_metric,
    )
    
    if trainer.distributed_training:
        print("Distributed Training enabled")
    batches = batches if args.steps_per_epoch is None else args.steps_per_epoch
    print(f"Training on {CONFIG['dataset']['name']} dataset with {batches} samples")
    
    # Construct a validation set by the prompts
    val_prompts = ['water tulip', ' a water lily', ' a water lily', ' a photo of a rose', ' a photo of a rose', ' a water lily', ' a water lily', ' a photo of a marigold', ' a photo of a marigold', ' a photo of a marigold', ' a water lily', ' a photo of a sunflower', ' a photo of a lotus', ' columbine', ' columbine', ' an orchid', ' an orchid', ' an orchid', ' a water lily', ' a water lily', ' a water lily', ' columbine', ' columbine', ' a photo of a sunflower', ' a photo of a sunflower', ' a photo of a sunflower', ' a photo of a lotus', ' a photo of a lotus', ' a photo of a marigold', ' a photo of a marigold', ' a photo of a rose', ' a photo of a rose', ' a photo of a rose', ' orange dahlia', ' orange dahlia', ' a lenten rose', ' a lenten rose', ' a water lily', ' a water lily', ' a water lily', ' a water lily', ' an orchid', ' an orchid', ' an orchid', ' hard-leaved pocket orchid', ' bird of paradise', ' bird of paradise', ' a photo of a lovely rose', ' a photo of a lovely rose', ' a photo of a globe-flower', ' a photo of a globe-flower', ' a photo of a lovely rose', ' a photo of a lovely rose', ' a photo of a ruby-lipped cattleya', ' a photo of a ruby-lipped cattleya', ' a photo of a lovely rose', ' a water lily', ' a osteospermum', ' a osteospermum', ' a water lily', ' a water lily', ' a water lily', ' a red rose', ' a red rose']

    def get_val_dataset(batch_size=8):
        for i in range(0, len(val_prompts), batch_size):
            prompts = val_prompts[i:i + batch_size]
            tokens = text_encoder.tokenize(prompts)
            yield {"text": tokens}

    data['test'] = get_val_dataset
    data['test_len'] = len(val_prompts)

    final_state = trainer.fit(
        data,
        training_steps_per_epoch=batches,
        epochs=CONFIG['epochs'], 
        sampler_class=EulerAncestralSampler, 
        sampling_noise_schedule=karas_ve_schedule,
        val_steps_per_epoch=args.val_steps_per_epoch,
    )
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

"""
New -->

python3 training/training.py --dataset=oxford_flowers102\
            --checkpoint_dir='./checkpoints/' --checkpoint_fs='local'\
            --epochs=2000 --batch_size=32 --image_size=128 \
            --learning_rate=2e-4 --num_res_blocks=2 \
            --use_self_and_cross=True --dtype=bfloat16 --precision=default --attention_heads=8\
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}/schd-{noise_schedule}/dtype-{dtype}/arch-{architecture}/lr-{learning_rate}/resblks-{num_res_blocks}/emb-{emb_features}/pure-attn-{only_pure_attention}'\
            --optimizer=adamw --use_dynamic_scale=True --norm_groups 0 --only_pure_attention=True --use_projection=False
"""