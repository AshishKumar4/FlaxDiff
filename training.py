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
from flaxdiff.models.simple_unet import Unet
from flaxdiff.models.simple_vit import UViT
import jax.experimental.pallas.ops.tpu.flash_attention
from flaxdiff.predictors import VPredictionTransform, EpsilonPredictionTransform, DiffusionPredictionTransform, DirectPredictionTransform, KarrasPredictionTransform
from flaxdiff.schedulers import CosineNoiseSchedule, NoiseScheduler, GeneralizedNoiseScheduler, KarrasVENoiseScheduler, EDMNoiseScheduler
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

import resource

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from orbax.checkpoint.utils import fully_replicated_host_local_array_to_global_array
from termcolor import colored

import warnings
import traceback

warnings.filterwarnings("ignore")


#####################################################################################################################
################################################# Initialization ####################################################
#####################################################################################################################

os.environ['TOKENIZERS_PARALLELISM'] = "false"


class RandomClass():
    def __init__(self, rng: jax.random.PRNGKey):
        self.rng = rng

    def get_random_key(self):
        self.rng, subkey = jax.random.split(self.rng)
        return subkey

    def get_sigmas(self, steps):
        return jnp.tan(self.theta_min + steps * (self.theta_max - self.theta_min)) / self.kappa

    def reset_random_key(self):
        self.rng = jax.random.PRNGKey(42)


class MarkovState(struct.PyTreeNode):
    pass


class RandomMarkovState(MarkovState):
    rng: jax.random.PRNGKey

    def get_random_key(self):
        rng, subkey = jax.random.split(self.rng)
        return RandomMarkovState(rng), subkey

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

def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
  sharding = jax.sharding.NamedSharding(global_mesh, P(global_mesh.axis_names))
  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
  return global_shape, sharding


def form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
  """Put local sharded array into local devices"""
  global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
  try:
    local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
  except ValueError as array_split_error:
    raise ValueError(
        f"Unable to put to devices shape {array.shape} with "
        f"local device count {len(global_mesh.local_devices)} "
    ) from array_split_error
  local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)

def convert_to_global_tree(global_mesh, pytree):
    return jax.tree_util.tree_map_with_path(partial(form_global_array, global_mesh=global_mesh), pytree)

#####################################################################################################################
################################################## Data Pipeline ####################################################
#####################################################################################################################

def defaultTextEncodeModel(backend="jax"):
    modelname = "openai/clip-vit-large-patch14"
    if backend == "jax":
        model = FlaxCLIPTextModel.from_pretrained(
            modelname, dtype=jnp.bfloat16)
    else:
        model = CLIPTextModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname, dtype=jnp.float16)
    return model, tokenizer


def encodePrompts(prompts, model, tokenizer=None):
    if model == None:
        model, tokenizer = defaultTextEncodeModel()
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")

    # inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np")
    inputs = tokenizer(prompts, padding="max_length",
                       max_length=tokenizer.model_max_length, truncation=True, return_tensors="np")
    outputs = model(input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'])
    # outputs = infer(inputs['input_ids'], inputs['attention_mask'])

    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output  # pooled (EOS token) states
    embed_pooled = pooler_output  # .astype(jnp.float16)
    embed_labels_full = last_hidden_state  # .astype(jnp.float16)

    return embed_pooled, embed_labels_full

class CaptionProcessor:
    def __init__(self, tensor_type="pt", modelname="openai/clip-vit-large-patch14"):
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.tensor_type = tensor_type

    def __call__(self, caption):
        # print(caption)
        tokens = self.tokenizer(caption, padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors=self.tensor_type)
        # print(tokens.keys())
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "caption": caption,
        }

    def __repr__(self):
        return self.__class__.__name__ + '()'

# -----------------------------------------------------------------------------------------------#
# Oxford flowers and other TFDS datasources ----------------------------------------------------#
# -----------------------------------------------------------------------------------------------#

def data_source_tfds(name, use_tf=True, split="all"):
    import tensorflow_datasets as tfds
    if use_tf:
        def data_source(path_override):
            return tfds.load(name, split=split, shuffle_files=True)
    else:
        def data_source(path_override):
            return tfds.data_source(name, split=split, try_gcs=False)
    return data_source

def labelizer_oxford_flowers102(path):
    with open(path, "r") as f:
        textlabels = [i.strip() for i in f.readlines()]

    def load_labels(sample):
        return textlabels[int(sample['label'])]
    return load_labels

def tfds_augmenters(image_scale, method):
    labelizer = labelizer_oxford_flowers102("/home/mrwhite0racle/tensorflow_datasets/oxford_flowers102/2.1.1/label.labels.txt")
    if image_scale > 256:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.caption_processor = CaptionProcessor(tensor_type="np")

        def map(self, element) -> Dict[str, jnp.array]:
            image = element['image']
            image = cv2.resize(image, (image_scale, image_scale),
                            interpolation=interpolation)
            # image = (image - 127.5) / 127.5
            caption = labelizer(element)
            results = self.caption_processor(caption)
            return {
                "image": image,
                "input_ids": results['input_ids'][0],
                "attention_mask": results['attention_mask'][0],
            }
    return augmenters

# -----------------------------------------------------------------------------------------------#
# CC12m and other GCS data sources --------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------#

def data_source_gcs(source='arrayrecord/laion-aesthetics-12m+mscoco-2017'):
    def data_source(base="/home/mrwhite0racle/gcs_mount"):
        records_path = os.path.join(base, source)
        records = [os.path.join(records_path, i) for i in os.listdir(
            records_path) if 'array_record' in i]
        ds = pygrain.ArrayRecordDataSource(records)
        return ds
    return data_source

def data_source_combined_gcs(
    sources=[]):
    def data_source(base="/home/mrwhite0racle/gcs_mount"):
        records_paths = [os.path.join(base, source) for source in sources]
        records = []
        for records_path in records_paths:
            records += [os.path.join(records_path, i) for i in os.listdir(
                records_path) if 'array_record' in i]
        ds = pygrain.ArrayRecordDataSource(records)
        return ds
    return data_source

def unpack_dict_of_byte_arrays(packed_data):
    unpacked_dict = {}
    offset = 0
    while offset < len(packed_data):
        # Unpack the key length
        key_length = st.unpack_from('I', packed_data, offset)[0]
        offset += st.calcsize('I')
        # Unpack the key bytes and convert to string
        key = packed_data[offset:offset+key_length].decode('utf-8')
        offset += key_length
        # Unpack the byte array length
        byte_array_length = st.unpack_from('I', packed_data, offset)[0]
        offset += st.calcsize('I')
        # Unpack the byte array
        byte_array = packed_data[offset:offset+byte_array_length]
        offset += byte_array_length
        unpacked_dict[key] = byte_array
    return unpacked_dict

def image_augmenter(image, image_scale, method=cv2.INTER_AREA):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_scale, image_scale),
                            interpolation=cv2.INTER_AREA)
    return image

def gcs_augmenters(image_scale, method):
    labelizer = lambda sample : sample['txt']
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.caption_processor = CaptionProcessor(tensor_type="np")
            self.image_augmenter = partial(image_augmenter, image_scale=image_scale, method=method)

        def map(self, element) -> Dict[str, jnp.array]:
            element = unpack_dict_of_byte_arrays(element)
            image = np.asarray(bytearray(element['jpg']), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            image = self.image_augmenter(image)
            caption = labelizer(element).decode('utf-8')
            results = self.caption_processor(caption)
            return {
                "image": image,
                "input_ids": results['input_ids'][0],
                "attention_mask": results['attention_mask'][0],
            }
    return augmenters

# Configure the following for your datasets
datasetMap = {
    "oxford_flowers102": {
        "source": data_source_tfds("oxford_flowers102", use_tf=False),
        "augmenter": tfds_augmenters,
    },
    "cc12m": {
        "source": data_source_gcs('arrayrecord2/cc12m'),
        "augmenter": gcs_augmenters,
    },
    "laiona_coco": {
        "source": data_source_gcs('arrayrecord2/laion-aesthetics-12m+mscoco-2017'),
        "augmenter": gcs_augmenters,
    },
    "aesthetic_coyo": {
        "source": data_source_gcs('arrayrecords/aestheticCoyo_0.25clip_6aesthetic'),
        "augmenter": gcs_augmenters,
    },
    "combined_aesthetic": {
        "source": data_source_combined_gcs([
                'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
                'arrayrecord2/cc12m',
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
            ]),
        "augmenter": gcs_augmenters,
    },
    "laiona_coco_coyo": {
        "source": data_source_combined_gcs([
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
                'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
            ]),
        "augmenter": gcs_augmenters,
    },
    "combined_30m": {
        "source": data_source_combined_gcs([
                'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
                'arrayrecord2/cc12m',
                'arrayrecord2/aestheticCoyo_0.26_clip_5.5aesthetic_256plus',
                "arrayrecord2/playground+leonardo_x4+cc3m.parquet",
            ]),
        "augmenter": gcs_augmenters,
    }
}

def batch_mesh_map(mesh):
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def map(self, batch) -> Dict[str, jnp.array]:
            return convert_to_global_tree(mesh, batch)
    return augmenters

def get_dataset_grain(
    data_name="cc12m",
    batch_size=64,
    image_scale=256,
    count=None,
    num_epochs=None,
    method=jax.image.ResizeMethod.LANCZOS3,
    worker_count=32,
    read_thread_count=64,
    read_buffer_size=50,
    worker_buffer_size=20,
    seed=0,
    dataset_source="/mnt/gcs_mount/arrayrecord2/cc12m/",
):
    dataset = datasetMap[data_name]
    data_source = dataset["source"](dataset_source)
    augmenter = dataset["augmenter"](image_scale, method)

    local_batch_size = batch_size // jax.process_count()
    model, tokenizer = defaultTextEncodeModel()

    null_labels, null_labels_full = encodePrompts([""], model, tokenizer)
    null_labels = np.array(null_labels[0], dtype=np.float16)
    null_labels_full = np.array(null_labels_full[0], dtype=np.float16)

    sampler = pygrain.IndexSampler(
        num_records=len(data_source) if count is None else count,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    def get_trainset():
        transformations = [
            augmenter(),
            pygrain.Batch(local_batch_size, drop_remainder=True),
        ]
        
        # if mesh != None:
        #     transformations += [batch_mesh_map(mesh)]

        loader = pygrain.DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader
    

    return {
        "train": get_trainset,
        "train_len": len(data_source),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        "null_labels": null_labels,
        "null_labels_full": null_labels_full,
        "model": model,
        "tokenizer": tokenizer,
    }

# -----------------------------------------------------------------------------------------------#
# Dataloader for directly streaming images from urls --------------------------------------------#
# -----------------------------------------------------------------------------------------------#

import albumentations as A
from flaxdiff.data.online_loader import OnlineStreamingDataLoader, dataMapper, \
        default_collate, load_dataset, concatenate_datasets, \
        ImageBatchIterator, default_image_processor, load_from_disk

import threading
import queue

def default_image_processor(
    image, image_shape, 
    min_image_shape=(128, 128),
    upscale_interpolation=cv2.INTER_CUBIC,
    downscale_interpolation=cv2.INTER_AREA,
):
    try:
        image = np.array(image)
        if len(image.shape) != 3 or image.shape[2] != 3:
            return None, 0, 0
        original_height, original_width = image.shape[:2]
        # check if the image is too small
        if min(original_height, original_width) < min(min_image_shape):
            return None, original_height, original_width
        # check if wrong aspect ratio
        if max(original_height, original_width) / min(original_height, original_width) > 2.4:
            return None, original_height, original_width
        # check if the variance is too low
        if np.std(image) < 1e-5:
            return None, original_height, original_width
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        downscale = max(original_width, original_height) > max(image_shape)
        interpolation = downscale_interpolation if downscale else upscale_interpolation

        image = A.longest_max_size(image, max(
            image_shape), interpolation=interpolation)
        image = A.pad(
            image,
            min_height=image_shape[0],
            min_width=image_shape[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        return image, original_height, original_width
    except Exception as e:
        # print("Error processing image", e, image_shape, interpolation)
        # traceback.print_exc()
        return None, 0, 0


class OnlineStreamingDataLoader():
    def __init__(
        self,
        dataset,
        batch_size=64,
        image_shape=(256, 256),
        min_image_shape=(128, 128),
        num_workers=16,
        num_threads=512,
        default_split="all",
        pre_map_maker=dataMapper,
        pre_map_def={
            "url": "URL",
            "caption": "TEXT",
        },
        global_process_count=1,
        global_process_index=0,
        prefetch=1000,
        collate_fn=default_collate,
        timeout=15,
        retries=3,
        image_processor=default_image_processor,
        upscale_interpolation=cv2.INTER_CUBIC,
        downscale_interpolation=cv2.INTER_AREA,
    ):
        if isinstance(dataset, str):
            dataset_path = dataset
            print("Loading dataset from path")
            if "gs://" in dataset:
                dataset = load_from_disk(dataset_path)
            else:
                dataset = load_dataset(dataset_path, split=default_split)
        elif isinstance(dataset, list):
            if isinstance(dataset[0], str):
                print("Loading multiple datasets from paths")
                dataset = [load_from_disk(dataset_path) if "gs://" in dataset_path else load_dataset(
                    dataset_path, split=default_split) for dataset_path in dataset]
            print("Concatenating multiple datasets")
            dataset = concatenate_datasets(dataset)
            dataset = dataset.shuffle(seed=0)
        # dataset = dataset.map(pre_map_maker(pre_map_def), batched=True, batch_size=10000000)
        self.dataset = dataset.shard(
            num_shards=global_process_count, index=global_process_index)
        print(f"Dataset length: {len(dataset)}")
        self.iterator = ImageBatchIterator(self.dataset, image_shape=image_shape,
                                           min_image_shape=min_image_shape,
                                           num_workers=num_workers, batch_size=batch_size, num_threads=num_threads,
                                            timeout=timeout, retries=retries, image_processor=image_processor,
                                             upscale_interpolation=upscale_interpolation,
                                             downscale_interpolation=downscale_interpolation)
        self.batch_size = batch_size

        # Launch a thread to load batches in the background
        self.batch_queue = queue.Queue(prefetch)

        def batch_loader():
            for batch in self.iterator:
                try:
                    self.batch_queue.put(collate_fn(batch))
                except Exception as e:
                    print("Error collating batch", e)

        self.loader_thread = threading.Thread(target=batch_loader)
        self.loader_thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch_queue.get()
        # return self.collate_fn(next(self.iterator))

    def __len__(self):
        return len(self.dataset)

onlineDatasetMap = {
    "combined_online": {
        "source": [
            # "gs://flaxdiff-datasets-regional/datasets/laion-aesthetics-12m+mscoco-2017.parquet"
            # "ChristophSchuhmann/MS_COCO_2017_URL_TEXT",
            # "dclure/laion-aesthetics-12m-umap",
            "gs://flaxdiff-datasets-regional/datasets/laion-aesthetics-12m+mscoco-2017",
            # "gs://flaxdiff-datasets-regional/datasets/coyo700m-aesthetic-5.4_25M",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/cc12m",
            "gs://flaxdiff-datasets-regional/datasets/cc3m",
            "gs://flaxdiff-datasets-regional/datasets/playground-liked",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/cc3m",
            "gs://flaxdiff-datasets-regional/datasets/cc3m",
        ]
    }
}

def generate_collate_fn(tokenizer):
    caption_processor = CaptionProcessor(tensor_type="np")
    def default_collate(batch):
        try:
            # urls = [sample["url"] for sample in batch]
            captions = [sample["caption"] for sample in batch]
            results = caption_processor(captions)
            images = np.stack([sample["image"] for sample in batch], axis=0)
            return {
                "image": images,
                "input_ids": results['input_ids'],
                "attention_mask": results['attention_mask'],
            }
        except Exception as e:
            print("Error in collate function", e, [sample["image"].shape for sample in batch])
            traceback.print_exc()
            
    return default_collate
    
def get_dataset_online(
        data_name="combined_online",
        batch_size=64,
        image_scale=256,
        count=None,
        num_epochs=None,
        method=jax.image.ResizeMethod.LANCZOS3,
        worker_count=32,
        read_thread_count=64,
        read_buffer_size=50,
        worker_buffer_size=20,
        seed=0,
        dataset_source="/mnt/gcs_mount/arrayrecord2/cc12m/",
    ):
    local_batch_size = batch_size // jax.process_count()
    
    model, tokenizer = defaultTextEncodeModel()

    null_labels, null_labels_full = encodePrompts([""], model, tokenizer)
    null_labels = np.array(null_labels[0], dtype=np.float16)
    null_labels_full = np.array(null_labels_full[0], dtype=np.float16)
    
    sources = onlineDatasetMap[data_name]["source"]
    dataloader = OnlineStreamingDataLoader(
            sources, 
            batch_size=local_batch_size,
            num_workers=worker_count,
            num_threads=read_thread_count,
            image_shape=(image_scale, image_scale),
            global_process_count=jax.process_count(),
            global_process_index=jax.process_index(),
            prefetch=worker_buffer_size,
            collate_fn=generate_collate_fn(tokenizer),
            default_split="train",
        )
    
    def get_trainset(mesh: Mesh = None):
        if mesh != None:
            class dataLoaderWithMesh:
                def __init__(self, dataloader, mesh):
                    self.dataloader = dataloader
                    self.mesh = mesh
                    self.tmp_queue = queue.Queue(worker_buffer_size)
                    def batch_loader():
                        for batch in self.dataloader:
                            try:
                                self.tmp_queue.put(convert_to_global_tree(mesh, batch))
                            except Exception as e:
                                print("Error processing batch", e)
                    self.loader_thread = threading.Thread(target=batch_loader)
                    self.loader_thread.start()
                    
                def __iter__(self):
                    return self
                
                def __next__(self):
                    return self.tmp_queue.get()
                
            dataloader_with_mesh = dataLoaderWithMesh(dataloader, mesh)
                    
            return dataloader_with_mesh
        return dataloader
    
    return {
        "train": get_trainset,
        "train_len": len(dataloader) * jax.process_count(),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        "null_labels": null_labels,
        "null_labels_full": null_labels_full,
        "model": model,
        "tokenizer": tokenizer,
    }
    

#####################################################################################################################
############################################### Training Pipeline ###################################################
#####################################################################################################################

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

# Define the TrainState
class SimpleTrainState(train_state.TrainState):
    metrics: Metrics
    dynamic_scale: flax.training.dynamic_scale.DynamicScale

class SimpleTrainer:
    state: SimpleTrainState
    best_state: SimpleTrainState
    best_loss: float
    model: nn.Module
    ema_decay: float = 0.999

    def __init__(self,
                 model: nn.Module,
                 input_shapes: Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 rngs: jax.random.PRNGKey,
                 train_state: SimpleTrainState = None,
                 name: str = "Simple",
                 load_from_checkpoint: str = None,
                 checkpoint_suffix: str = "",
                 loss_fn=optax.l2_loss,
                 param_transforms: Callable = None,
                 wandb_config: Dict[str, Any] = None,
                 distributed_training: bool = None,
                 checkpoint_base_path: str = "./checkpoints",
                 checkpoint_step: int = None,
                 use_dynamic_scale: bool = False,
                 ):
        if distributed_training is None or distributed_training is True:
            # Auto-detect if we are running on multiple devices
            distributed_training = jax.device_count() > 1
            self.mesh = jax.sharding.Mesh(jax.devices(), 'data')
        else:
            self.mesh = None

        self.distributed_training = distributed_training
        self.model = model
        self.name = name
        self.loss_fn = loss_fn
        self.input_shapes = input_shapes
        self.checkpoint_base_path = checkpoint_base_path
        
        
        if wandb_config is not None and jax.process_index() == 0:
            run = wandb.init(**wandb_config)
            self.wandb = run
            
            # define our custom x axis metric
            self.wandb.define_metric("train/step")
            self.wandb.define_metric("train/epoch")
            
            self.wandb.define_metric("train/loss", step_metric="train/step")
            
            self.wandb.define_metric("train/epoch_time", step_metric="train/epoch")
            self.wandb.define_metric("train/avg_time_per_step", step_metric="train/epoch")
            self.wandb.define_metric("train/avg_loss", step_metric="train/epoch")
            self.wandb.define_metric("train/best_loss", step_metric="train/epoch")
            
        # checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        async_checkpointer = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=60)

        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=4, create=True)
        self.checkpointer = orbax.checkpoint.CheckpointManager(
            self.checkpoint_path() + checkpoint_suffix, async_checkpointer, options)

        if load_from_checkpoint is not None:
            latest_epoch, latest_step, old_state, old_best_state, rngstate = self.load(load_from_checkpoint, checkpoint_step)
        else:
            latest_epoch, latest_step, old_state, old_best_state, rngstate = 0, 0, None, None, None

        self.latest_step = latest_step
        
        if rngstate:
            self.rngstate = RandomMarkovState(**rngstate)
        else:
            self.rngstate = RandomMarkovState(rngs)
            
        self.rngstate, subkey = self.rngstate.get_random_key()

        if train_state == None:
            state, best_state = self.generate_states(
                optimizer, subkey, old_state, old_best_state, model, param_transforms, use_dynamic_scale
            )
            self.init_state(state, best_state)
        else:
            self.state = train_state
            self.best_state = train_state
            self.best_loss = 1e9

    def get_input_ones(self):
        return {k: jnp.ones((1, *v)) for k, v in self.input_shapes.items()}

    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None,
        use_dynamic_scale: bool = False
    ) -> Tuple[SimpleTrainState, SimpleTrainState]:
        print("Generating states for SimpleTrainer")
        rngs, subkey = jax.random.split(rngs)

        if existing_state == None:
            input_vars = self.get_input_ones()
            params = model.init(subkey, **input_vars)
        else:
            params = existing_state['params']

        state = SimpleTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            metrics=Metrics.empty(),
            dynamic_scale = flax.training.dynamic_scale.DynamicScale() if use_dynamic_scale else None
        )
        if existing_best_state is not None:
            best_state = state.replace(
                params=existing_best_state['params'])
        else:
            best_state = state
            
        return state, best_state

    def init_state(
        self,
        state: SimpleTrainState,
        best_state: SimpleTrainState,
    ):
        self.best_loss = 1e9

        self.state = state
        self.best_state = best_state

    def get_state(self):
        return self.get_np_tree(self.state)

    def get_best_state(self):
        return self.get_np_tree(self.best_state)
        
    def get_rngstate(self):
        return self.get_np_tree(self.rngstate)
    
    def get_np_tree(self, pytree):
        return jax.tree_util.tree_map(lambda x : np.array(x), pytree)

    def checkpoint_path(self):
        path = os.path.join(self.checkpoint_base_path, self.name.replace(' ', '_').lower())
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def tensorboard_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./tensorboard'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load(self, checkpoint_path=None, checkpoint_step=None):
        if checkpoint_path is None:
            checkpointer = self.checkpointer
        else:
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            options = orbax.checkpoint.CheckpointManagerOptions(
                max_to_keep=4, create=False)
            checkpointer = orbax.checkpoint.CheckpointManager(
                checkpoint_path, checkpointer, options)    
        
        if checkpoint_step is None:
            step = checkpointer.latest_step()
        else:
            step = checkpoint_step
        
        print("Loading model from checkpoint at step ", step)
        ckpt = checkpointer.restore(step)
        state = ckpt['state']
        best_state = ckpt['best_state']
        rngstate = ckpt['rngs']
        # Convert the state to a TrainState
        self.best_loss = ckpt['best_loss']
        if self.best_loss == 0:
            # It cant be zero as that must have been some problem
            self.best_loss = 1e9
        current_epoch = ckpt.get('epoch', step) # Must be a checkpoint from an older version which used epochs instead of steps
        print(
            f"Loaded model from checkpoint at epoch {current_epoch} step {step}", ckpt['best_loss'])
        return current_epoch, step, state, best_state, rngstate

    def save(self, epoch=0, step=0, state=None, rngstate=None):
        print(f"Saving model at epoch {epoch} step {step}")
        try:
            ckpt = {
                # 'model': self.model,
                'rngs': self.get_rngstate() if rngstate is None else self.get_np_tree(rngstate),
                'state': self.get_state() if state is None else self.get_np_tree(state),
                'best_state': self.get_best_state(),
                'best_loss': np.array(self.best_loss),
                'epoch': epoch,
            }
            try:
                save_args = orbax_utils.save_args_from_target(ckpt)
                self.checkpointer.save(step, ckpt, save_kwargs={
                                    'save_args': save_args}, force=True)
                self.checkpointer.wait_until_finished()
                pass
            except Exception as e:
                print("Error saving checkpoint", e)
        except Exception as e:
            print("Error saving checkpoint outer", e)

    def _define_train_step(self, **kwargs):
        model = self.model
        loss_fn = self.loss_fn
        distributed_training = self.distributed_training

        def train_step(train_state: SimpleTrainState, rng_state: RandomMarkovState, batch, local_device_indexes):
            """Train for a single step."""
            images = batch['image']
            labels = batch['label']

            def model_loss(params):
                preds = model.apply(params, images)
                expected_output = labels
                nloss = loss_fn(preds, expected_output)
                loss = jnp.mean(nloss)
                return loss
            loss, grads = jax.value_and_grad(model_loss)(train_state.params)
            if distributed_training:
                grads = jax.lax.pmean(grads, "data")
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss, rng_state
        
        if distributed_training:
            train_step = shard_map(train_step, mesh=self.mesh, in_specs=(P(), P(), P('data'), P('data')), out_specs=(P(), P('data'), P()))
            train_step = jax.pmap(train_step)
        return train_step

    def _define_compute_metrics(self):
        model = self.model
        loss_fn = self.loss_fn

        @jax.jit
        def compute_metrics(state: SimpleTrainState, batch):
            preds = model.apply(state.params, batch['image'])
            expected_output = batch['label']
            loss = jnp.mean(loss_fn(preds, expected_output))
            metric_updates = state.metrics.single_from_model_output(
                loss=loss, logits=preds, labels=expected_output)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

    def summary(self):
        input_vars = self.get_input_ones()
        print(self.model.tabulate(jax.random.key(0), **input_vars,
              console_kwargs={"width": 200, "force_jupyter": True, }))

    def config(self):
        return {
            "model": self.model,
            "state": self.state,
            "name": self.name,
            "input_shapes": self.input_shapes
        }

    def init_tensorboard(self, batch_size, steps_per_epoch, epochs):
        from flax.metrics import tensorboard
        summary_writer = tensorboard.SummaryWriter(self.tensorboard_path())
        summary_writer.hparams({
            **self.config(),
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "batch_size": batch_size
        })
        return summary_writer

    def fit(self, data, steps_per_epoch, epochs, train_step_args={}):
        train_ds = iter(data['train']())
        if 'test' in data:
            test_ds = data['test']
        else:
            test_ds = None
        train_step = self._define_train_step(**train_step_args)
        compute_metrics = self._define_compute_metrics()
        train_state = self.state
        rng_state = self.rngstate
        global_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        process_index = jax.process_index()
        if self.distributed_training:
            global_device_indexes = jnp.arange(global_device_count)
        else:
            global_device_indexes = 0

        def train_loop(current_step, pbar: tqdm.tqdm, train_state, rng_state):
            epoch_loss = 0
            current_epoch = current_step // steps_per_epoch
            last_save_time = time.time()
            for i in range(steps_per_epoch):
                batch = next(train_ds)
                if i == 0:
                    print(f"First batch loaded at step {current_step}")
                    
                if self.distributed_training and global_device_count > 1:
                #     # Convert the local device batches to a unified global jax.Array 
                    batch = convert_to_global_tree(self.mesh, batch)
                train_state, loss, rng_state = train_step(train_state, rng_state, batch, global_device_indexes)

                if i == 0:
                    print(f"Training started for process index {process_index} at step {current_step}")
                
                if self.distributed_training:
                    # loss = jax.experimental.multihost_utils.process_allgather(loss)
                    loss = jnp.mean(loss) # Just to make sure its a scaler value
                    
                if loss <= 1e-6:
                    # If the loss is too low, we can assume the model has diverged
                    print(colored(f"Loss too low at step {current_step} => {loss}", 'red'))
                    # Reset the model to the old state
                    exit(1)
                            
                epoch_loss += loss
                current_step += 1
                if i % 100 == 0:
                    if pbar is not None:
                        pbar.set_postfix(loss=f'{loss:.4f}')
                        pbar.update(100)
                        if self.wandb is not None:
                            self.wandb.log({
                                "train/step" : current_step,
                                "train/loss": loss,
                            }, step=current_step)
                    # Save the model every few steps
                    if i % 10000 == 0 and i > 0:
                        print(f"Saving model after 10000 step {current_step}")
                        print(f"Devices: {len(jax.devices())}") # To sync the devices
                        self.save(current_epoch, current_step, train_state, rng_state)
                        print(f"Saving done by process index {process_index}")
                        last_save_time = time.time()
            print(colored(f"Epoch done on index {process_index} => {current_epoch} Loss: {epoch_loss/steps_per_epoch}", 'green'))
            return epoch_loss, current_step, train_state, rng_state

        while self.latest_step < epochs * steps_per_epoch:
            current_epoch = self.latest_step // steps_per_epoch
            print(f"\nEpoch {current_epoch}/{epochs}")
            start_time = time.time()
            epoch_loss = 0

            if process_index == 0:
                with tqdm.tqdm(total=steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step') as pbar:
                    epoch_loss, current_step, train_state, rng_state = train_loop(self.latest_step, pbar, train_state, rng_state)
            else:
                epoch_loss, current_step, train_state, rng_state = train_loop(self.latest_step, None, train_state, rng_state)
                print(colored(f"Epoch done on process index {process_index}", PROCESS_COLOR_MAP[process_index]))
            
            self.latest_step = current_step
            end_time = time.time()
            self.state = train_state
            self.rngstate = rng_state
            total_time = end_time - start_time
            avg_time_per_step = total_time / steps_per_epoch
            avg_loss = epoch_loss / steps_per_epoch
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = train_state
                self.save(current_epoch, current_step)
                
            if process_index == 0:
                if self.wandb is not None:
                    self.wandb.log({
                        "train/epoch_time": total_time,
                        "train/avg_time_per_step": avg_time_per_step,
                        "train/avg_loss": avg_loss,
                        "train/best_loss": self.best_loss,
                        "train/epoch": current_epoch,
                    }, step=current_step)
                print(colored(f"\n\tEpoch {current_epoch} completed. Avg Loss: {avg_loss}, Time: {total_time:.2f}s, Best Loss: {self.best_loss}", 'green'))
        self.save(epochs)
        return self.state

# Define the TrainState with EMA parameters

class TrainState(SimpleTrainState):
    rngs: jax.random.PRNGKey
    ema_params: dict

    def apply_ema(self, decay: float = 0.999):
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, param: decay * ema + (1 - decay) * param,
            self.ema_params,
            self.params,
        )
        return self.replace(ema_params=new_ema_params)

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder

class DiffusionTrainer(SimpleTrainer):
    noise_schedule: NoiseScheduler
    model_output_transform: DiffusionPredictionTransform
    ema_decay: float = 0.999

    def __init__(self,
                 model: nn.Module,
                 input_shapes: Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 noise_schedule: NoiseScheduler,
                 rngs: jax.random.PRNGKey,
                 unconditional_prob: float = 0.12,
                 name: str = "Diffusion",
                 model_output_transform: DiffusionPredictionTransform = EpsilonPredictionTransform(),
                 autoencoder: AutoEncoder = None,
                 **kwargs
                 ):
        super().__init__(
            model=model,
            input_shapes=input_shapes,
            optimizer=optimizer,
            rngs=rngs,
            name=name,
            **kwargs
        )
        self.noise_schedule = noise_schedule
        self.model_output_transform = model_output_transform
        self.unconditional_prob = unconditional_prob
        
        self.autoencoder = autoencoder

    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None,
        use_dynamic_scale: bool = False
    ) -> Tuple[TrainState, TrainState]:
        print("Generating states for DiffusionTrainer")
        rngs, subkey = jax.random.split(rngs)

        if existing_state == None:
            input_vars = self.get_input_ones()
            params = model.init(subkey, **input_vars)
            new_state = {"params": params, "ema_params": params}
        else:
            new_state = existing_state

        if param_transforms is not None:
            params = param_transforms(params)

        state = TrainState.create(
            apply_fn=model.apply,
            params=new_state['params'],
            ema_params=new_state['ema_params'],
            tx=optimizer,
            rngs=rngs,
            metrics=Metrics.empty(),
            dynamic_scale = flax.training.dynamic_scale.DynamicScale() if use_dynamic_scale else None
        )
            
        if existing_best_state is not None:
            best_state = state.replace(
                params=existing_best_state['params'], ema_params=existing_best_state['ema_params'])
        else:
            best_state = state

        return state, best_state

    def _define_train_step(self, batch_size, null_labels_seq, text_embedder):
        noise_schedule: NoiseScheduler = self.noise_schedule
        model = self.model
        model_output_transform = self.model_output_transform
        loss_fn = self.loss_fn
        unconditional_prob = self.unconditional_prob

        # Determine the number of unconditional samples
        num_unconditional = int(batch_size * unconditional_prob)

        nS, nC = null_labels_seq.shape
        null_labels_seq = jnp.broadcast_to(
            null_labels_seq, (batch_size, nS, nC))

        distributed_training = self.distributed_training
        
        autoencoder = self.autoencoder

        # @jax.jit
        def train_step(train_state: TrainState, rng_state: RandomMarkovState, batch, local_device_index):
            """Train for a single step."""
            rng_state, subkey = rng_state.get_random_key()
            subkey = jax.random.fold_in(subkey, local_device_index.reshape())
            local_rng_state = RandomMarkovState(subkey)
            
            images = batch['image']
            
            # First get the standard deviation of the images
            # std = jnp.std(images, axis=(1, 2, 3))
            # is_non_zero = (std > 0)
            
            images = jnp.array(images, dtype=jnp.float32)
            # normalize image
            images = (images - 127.5) / 127.5
            
            if autoencoder is not None:
                # Convert the images to latent space
                local_rng_state, rngs = local_rng_state.get_random_key()
                images = autoencoder.encode(images, rngs)

            output = text_embedder(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            label_seq = output.last_hidden_state

            # Generate random probabilities to decide how much of this batch will be unconditional

            label_seq = jnp.concat(
                [null_labels_seq[:num_unconditional], label_seq[num_unconditional:]], axis=0)

            noise_level, local_rng_state = noise_schedule.generate_timesteps(images.shape[0], local_rng_state)
            
            local_rng_state, rngs = local_rng_state.get_random_key()
            noise: jax.Array = jax.random.normal(rngs, shape=images.shape)
            
            rates = noise_schedule.get_rates(noise_level)
            noisy_images, c_in, expected_output = model_output_transform.forward_diffusion(
                images, noise, rates)

            def model_loss(params):
                preds = model.apply(params, *noise_schedule.transform_inputs(noisy_images*c_in, noise_level), label_seq)
                preds = model_output_transform.pred_transform(
                    noisy_images, preds, rates)
                nloss = loss_fn(preds, expected_output)
                # Ignore the loss contribution of images with zero standard deviation
                nloss *= noise_schedule.get_weights(noise_level)
                # nloss = jnp.mean(nloss, axis=(1,2,3))
                # nloss = jnp.where(is_non_zero, nloss, 0)
                # nloss = jnp.mean(nloss, where=nloss != 0)
                nloss = jnp.mean(nloss)
                loss = nloss
                return loss
            
            
            if train_state.dynamic_scale is not None:
                # dynamic scale takes care of averaging gradients across replicas
                grad_fn = train_state.dynamic_scale.value_and_grad(
                    model_loss, axis_name="data"
                )
                dynamic_scale, is_fin, loss, grads = grad_fn(train_state.params)
                train_state = train_state.replace(dynamic_scale=dynamic_scale)
            else:
                grad_fn = jax.value_and_grad(model_loss)
                loss, grads = grad_fn(train_state.params)
                if distributed_training:
                    grads = jax.lax.pmean(grads, "data")
            
            new_state = train_state.apply_gradients(grads=grads)
            
            if train_state.dynamic_scale is not None:
                # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
                # params should be restored (= skip this step).
                select_fn = functools.partial(jnp.where, is_fin)
                new_state = new_state.replace(
                    opt_state=jax.tree_util.tree_map(
                        select_fn, new_state.opt_state, train_state.opt_state
                    ),
                    params=jax.tree_util.tree_map(
                        select_fn, new_state.params, train_state.params
                    ),
                )
    
            train_state = new_state.apply_ema(self.ema_decay)
            
            if distributed_training:
                loss = jax.lax.pmean(loss, "data")
            return train_state, loss, rng_state

        if distributed_training:
            train_step = shard_map(train_step, mesh=self.mesh, in_specs=(P(), P(), P('data'), P('data')), 
                                   out_specs=(P(), P(), P()))
            train_step = jax.jit(train_step)
            
        return train_step

    def _define_compute_metrics(self):
        @jax.jit
        def compute_metrics(state: TrainState, expected, pred):
            loss = jnp.mean(jnp.square(pred - expected))
            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

    def fit(self, data, steps_per_epoch, epochs):
        null_labels_full = data['null_labels_full']
        local_batch_size = data['local_batch_size']
        text_embedder = data['model']
        super().fit(data, steps_per_epoch, epochs, {
            "batch_size": local_batch_size, "null_labels_seq": null_labels_full, "text_embedder": text_embedder})

def boolean_string(s):
    if type(s) == bool:
        return s
    return s == 'True'

# %%
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

parser.add_argument('--dtype', type=str, default=None, help='dtype to use')
parser.add_argument('--precision', type=str, default=None, help='precision to use', choices=['high', 'default', 'highest', 'None', None])

parser.add_argument('--distributed_training', type=boolean_string, default=True, help='Should use distributed training or not')
parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name, would be generated if not provided')
parser.add_argument('--load_from_checkpoint', type=str,
                    default=None, help='Load from the best previously stored checkpoint. The checkpoint path should be provided')
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

    cosine_schedule = CosineNoiseSchedule(1000, beta_end=1)
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
        "project": "flaxdiff",
        "config": CONFIG,
        "name": experiment_name,
    }
    
    start_time = time.time()
    
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
    )
    
    if trainer.distributed_training:
        print("Distributed Training enabled")
    # %%
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

python3 training.py --dataset=combined_30m --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=512 --image_size=128 --learning_rate=9e-5 \
            --architecture=uvit --num_layers=12 --emb_features=768 --norm_groups 0 --num_heads=12 \
            --dtype=bfloat16 --precision=default \
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-64_uvit_combined_30m'\
            --optimizer=adamw --clip_grads 0.5 \
            --learning_rate_schedule=cosine --learning_rate_peak=2.7e-4 --learning_rate_end=9e-5 --learning_rate_warmup_steps=10000 --learning_rate_decay_epochs=2\
                
                
            --load_from_checkpoint='gs://flaxdiff-datasets-regional/checkpoints/dataset-combined_30m/image_size-512/batch-512-v4-64_flaxdiff-0-1-8_ldm_dyn_scale_NEW_ARCH_combined_30'


            --learning_rate_schedule=cosine --learning_rate_peak=4e-5 --learning_rate_end=9e-6 --learning_rate_warmup_steps=5000 --learning_rate_decay_epochs=2\
                

python3 training.py --dataset=combined_30m --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=256 --image_size=128 \
            --learning_rate=9e-5 --architecture=uvit --num_layers=12 \
            --use_self_and_cross=False --precision=default --dtype=bfloat16 --attention_heads=16\
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-64_flaxdiff-0-1-10__new-combined_30m'\
            --optimizer=adamw --feature_depths 128 256 512 512 --use_dynamic_scale=True\
            --load_from_checkpoint='gs://flaxdiff-datasets-regional/checkpoints/dataset-combined_aesthetic/image_size-128/batch-256-v4-32_flaxdiff-0-1-8__new-combined_1'

for tpu-v4-32

python3 training.py --dataset=combined_30m --dataset_path='/home/mrwhite0racle/gcs_mount/'\
            --checkpoint_dir='flaxdiff-datasets-regional/checkpoints/' --checkpoint_fs='gcs'\
            --epochs=40 --batch_size=256 --image_size=128 \
            --learning_rate=8e-5 --num_res_blocks=3 \
            --use_self_and_cross=False --precision=default --dtype=bfloat16 --attention_heads=16\
            --experiment_name='dataset-{dataset}/image_size-{image_size}/batch-{batch_size}-v4-32_flaxdiff-0-1-9_combined_30m'\
            --optimizer=adamw --feature_depths 128 256 512 512 --use_dynamic_scale=True --named_norms=True --only_pure_attention=True\
            --load_from_checkpoint='gs://flaxdiff-datasets-regional/checkpoints/dataset-combined_aesthetic/image_size-128/batch-256-v4-32_flaxdiff-0-1-8__3'

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
