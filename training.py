from typing import Any, Tuple, Mapping, Callable, List, Dict
from functools import partial
from flax.metrics import tensorboard
import orbax
import orbax.checkpoint
import flax.jax_utils
from flaxdiff.models.simple_unet import Unet
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

#####################################################################################################################
################################################# Initialization ####################################################
#####################################################################################################################


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
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.caption_processor = CaptionProcessor(tensor_type="np")

        def map(self, element) -> Dict[str, jnp.array]:
            image = element['image']
            image = cv2.resize(image, (image_scale, image_scale),
                            interpolation=cv2.INTER_AREA)
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
# CC12m and other GCS data sources -------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------#

def data_source_gcs():
    def data_source(source="/mnt/gcs_mount/arrayrecord/cc12m/"):
        records_path = source
        records = [os.path.join(records_path, i) for i in os.listdir(
            records_path) if 'array_record' in i]
        ds = pygrain.ArrayRecordDataSource(records[:-1])
        return ds
    return data_source


def labelizer_gcs(sample):
    return sample['txt']

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

def gcs_augmenters(image_scale, method):
    labelizer = labelizer_gcs
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.caption_processor = CaptionProcessor(tensor_type="np")

        def map(self, element) -> Dict[str, jnp.array]:
            element = unpack_dict_of_byte_arrays(element)
            image = np.asarray(bytearray(element['jpg']), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_scale, image_scale),
                            interpolation=cv2.INTER_AREA)
            # image = (image - 127.5) / 127.5
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
        "source": data_source_gcs(),
        "augmenter": gcs_augmenters,
    },
}


def get_dataset_grain(
    data_name="cc12m",
    batch_size=64,
    image_scale=256,
    count=None,
    num_epochs=None,
    text_encoders=None,
    method=jax.image.ResizeMethod.LANCZOS3,
    grain_worker_count=32,
    grain_read_thread_count=64,
    grain_read_buffer_size=50,
    grain_worker_buffer_size=20,
    seed=0,
    dataset_source="/mnt/gcs_mount/arrayrecord/cc12m/",
):
    dataset = datasetMap[data_name]
    data_source = dataset["source"](dataset_source)
    augmenter = dataset["augmenter"](image_scale, method)

    local_batch_size = batch_size // jax.process_count()
    model, tokenizer = text_encoders

    null_labels, null_labels_full = encodePrompts([""], model, tokenizer)
    null_labels = np.array(null_labels[0], dtype=np.float16)
    null_labels_full = np.array(null_labels_full[0], dtype=np.float16)

    sampler = pygrain.IndexSampler(
        num_records=len(data_source) if count is None else count,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.NoSharding(),
    )

    transformations = [
        augmenter(),
        pygrain.Batch(local_batch_size, drop_remainder=True),
    ]

    loader = pygrain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=grain_worker_count,
        read_options=pygrain.ReadOptions(
            grain_read_thread_count, grain_read_buffer_size
        ),
        worker_buffer_size=grain_worker_buffer_size,
    )

    def get_trainset():
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


#####################################################################################################################
############################################### Training Pipeline ###################################################
#####################################################################################################################

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

# Define the TrainState
class SimpleTrainState(train_state.TrainState):
    rngs: jax.random.PRNGKey
    metrics: Metrics

    def get_random_key(self):
        rngs, subkey = jax.random.split(self.rngs)
        return self.replace(rngs=rngs), subkey

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
                 load_from_checkpoint: bool = False,
                 checkpoint_suffix: str = "",
                 loss_fn=optax.l2_loss,
                 param_transforms: Callable = None,
                 wandb_config: Dict[str, Any] = None,
                 distributed_training: bool = None,
                 ):
        if distributed_training is None or distributed_training is True:
            # Auto-detect if we are running on multiple devices
            distributed_training = jax.device_count() > 1

        self.distributed_training = distributed_training
        self.model = model
        self.name = name
        self.loss_fn = loss_fn
        self.input_shapes = input_shapes

        if wandb_config is not None:
            run = wandb.init(**wandb_config)
            self.wandb = run

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=4, create=True)
        self.checkpointer = orbax.checkpoint.CheckpointManager(
            self.checkpoint_path() + checkpoint_suffix, checkpointer, options)

        if load_from_checkpoint:
            latest_epoch, old_state, old_best_state = self.load()
        else:
            latest_epoch, old_state, old_best_state = 0, None, None

        self.latest_epoch = latest_epoch

        if train_state == None:
            self.init_state(optimizer, rngs, existing_state=old_state,
                            existing_best_state=old_best_state, model=model, param_transforms=param_transforms)
        else:
            self.state = train_state
            self.best_state = train_state
            self.best_loss = 1e9

    def get_input_ones(self):
        return {k: jnp.ones((1, *v)) for k, v in self.input_shapes.items()}

    def __init_fn(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None
    ) -> Tuple[SimpleTrainState, SimpleTrainState]:
        rngs, subkey = jax.random.split(rngs)

        if existing_state == None:
            input_vars = self.get_input_ones()
            params = model.init(subkey, **input_vars)

        state = SimpleTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            rngs=rngs,
            metrics=Metrics.empty()
        )
        if existing_best_state is not None:
            best_state = state.replace(
                params=existing_best_state['params'])
        else:
            best_state = state
            
        return state, best_state

    def init_state(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None
    ):
        
        state, best_state = self.__init_fn(
            optimizer, rngs, existing_state, existing_best_state, model, param_transforms
        )
        self.best_loss = 1e9

        if self.distributed_training:
            devices = jax.local_devices()
            if len(devices) > 1:
                print("Replicating state across devices ", devices)
                state = flax.jax_utils.replicate(state, devices)
                best_state = flax.jax_utils.replicate(best_state, devices)
            else:
                print("Not replicating any state, Only single device connected to the process")

        self.state = state
        self.best_state = best_state

    def get_state(self):
        return flax.jax_utils.unreplicate(self.state)

    def get_best_state(self):
        return flax.jax_utils.unreplicate(self.best_state)

    def checkpoint_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./checkpoints'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def tensorboard_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./tensorboard'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load(self):
        epoch = self.checkpointer.latest_step()
        print("Loading model from checkpoint", epoch)
        ckpt = self.checkpointer.restore(epoch)
        state = ckpt['state']
        best_state = ckpt['best_state']
        # Convert the state to a TrainState
        self.best_loss = ckpt['best_loss']
        print(
            f"Loaded model from checkpoint at epoch {epoch}", ckpt['best_loss'])
        return epoch, state, best_state

    def save(self, epoch=0):
        print(f"Saving model at epoch {epoch}")
        ckpt = {
            # 'model': self.model,
            'state': self.get_state(),
            'best_state': self.get_best_state(),
            'best_loss': self.best_loss
        }
        try:
            save_args = orbax_utils.save_args_from_target(ckpt)
            self.checkpointer.save(epoch, ckpt, save_kwargs={
                                   'save_args': save_args}, force=True)
            pass
        except Exception as e:
            print("Error saving checkpoint", e)

    def _define_train_step(self, **kwargs):
        model = self.model
        loss_fn = self.loss_fn
        distributed_training = self.distributed_training

        def train_step(state: SimpleTrainState, batch):
            """Train for a single step."""
            images = batch['image']
            labels = batch['label']

            def model_loss(params):
                preds = model.apply(params, images)
                expected_output = labels
                nloss = loss_fn(preds, expected_output)
                loss = jnp.mean(nloss)
                return loss
            loss, grads = jax.value_and_grad(model_loss)(state.params)
            if distributed_training:
                grads = jax.lax.pmean(grads, "device")
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        if distributed_training:
            train_step = jax.pmap(axis_name="device")(train_step)
        else:
            train_step = jax.jit(train_step)
            
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
        state = self.state
        device_count = jax.local_device_count()
        # train_ds = flax.jax_utils.prefetch_to_device(train_ds, jax.devices())

        summary_writer = self.init_tensorboard(
            data['global_batch_size'], steps_per_epoch, epochs)

        while self.latest_epoch <= epochs:
            self.latest_epoch += 1
            current_epoch = self.latest_epoch
            print(f"\nEpoch {current_epoch}/{epochs}")
            start_time = time.time()
            epoch_loss = 0

            with tqdm.tqdm(total=steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step') as pbar:
                for i in range(steps_per_epoch):
                    batch = next(train_ds)
                    if self.distributed_training and device_count > 1:
                        batch = jax.tree.map(lambda x: x.reshape(
                            (device_count, -1, *x.shape[1:])), batch)
                    
                    state, loss = train_step(state, batch)
                    loss = jnp.mean(loss)
                    
                    epoch_loss += loss
                    if i % 100 == 0:
                        pbar.set_postfix(loss=f'{loss:.4f}')
                        pbar.update(100)
                        current_step = current_epoch*steps_per_epoch + i
                        summary_writer.scalar(
                            'Train Loss', loss, step=current_step)
                        if self.wandb is not None:
                            self.wandb.log({"train/loss": loss})

            print(f"\n\tEpoch done")
            end_time = time.time()
            self.state = state
            total_time = end_time - start_time
            avg_time_per_step = total_time / steps_per_epoch
            avg_loss = epoch_loss / steps_per_epoch
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = state
                self.save(current_epoch)

            # Compute Metrics
            metrics_str = ''

            print(
                f"\n\tEpoch {current_epoch} completed. Avg Loss: {avg_loss}, Time: {total_time:.2f}s, Best Loss: {self.best_loss} {metrics_str}")

        self.save(epochs)
        return self.state

# Define the TrainState with EMA parameters

class TrainState(SimpleTrainState):
    rngs: jax.random.PRNGKey
    ema_params: dict

    def get_random_key(self):
        rngs, subkey = jax.random.split(self.rngs)
        return self.replace(rngs=rngs), subkey

    def apply_ema(self, decay: float = 0.999):
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, param: decay * ema + (1 - decay) * param,
            self.ema_params,
            self.params,
        )
        return self.replace(ema_params=new_ema_params)


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
                 unconditional_prob: float = 0.2,
                 name: str = "Diffusion",
                 model_output_transform: DiffusionPredictionTransform = EpsilonPredictionTransform(),
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

    def __init_fn(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None
    ) -> Tuple[TrainState, TrainState]:
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
            metrics=Metrics.empty()
        )
            
        if existing_best_state is not None:
            best_state = state.replace(
                params=existing_best_state['params'], ema_params=existing_best_state['ema_params'])
        else:
            best_state = state

        return state, best_state

    def _define_train_step(self, batch_size, null_labels_seq, text_embedder):
        noise_schedule = self.noise_schedule
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

        def train_step(state: TrainState, batch):
            """Train for a single step."""
            images = batch['image']
            # normalize image
            images = (images - 127.5) / 127.5

            output = text_embedder(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            # output = infer(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            label_seq = output.last_hidden_state

            # Generate random probabilities to decide how much of this batch will be unconditional

            label_seq = jnp.concat(
                [null_labels_seq[:num_unconditional], label_seq[num_unconditional:]], axis=0)

            noise_level, state = noise_schedule.generate_timesteps(
                images.shape[0], state)
            state, rngs = state.get_random_key()
            noise: jax.Array = jax.random.normal(rngs, shape=images.shape)
            rates = noise_schedule.get_rates(noise_level)
            noisy_images, c_in, expected_output = model_output_transform.forward_diffusion(
                images, noise, rates)

            def model_loss(params):
                preds = model.apply(
                    params, *noise_schedule.transform_inputs(noisy_images*c_in, noise_level), label_seq)
                preds = model_output_transform.pred_transform(
                    noisy_images, preds, rates)
                nloss = loss_fn(preds, expected_output)
                # nloss = jnp.mean(nloss, axis=1)
                nloss *= noise_schedule.get_weights(noise_level)
                nloss = jnp.mean(nloss)
                loss = nloss
                return loss
            
            loss, grads = jax.value_and_grad(model_loss)(state.params)
            if distributed_training:
                grads = jax.lax.pmean(grads, "device")
            state = state.apply_gradients(grads=grads)
            state = state.apply_ema(self.ema_decay)
            return state, loss
        
        if distributed_training:
            train_step = jax.pmap(axis_name="device")(train_step)
        else:
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


# %%
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a diffusion model')
parser.add_argument('--GRAIN_WORKER_COUNT', type=int,
                    default=16, help='Number of grain workers')
parser.add_argument('--GRAIN_READ_THREAD_COUNT', type=int,
                    default=64, help='Number of grain read threads')
parser.add_argument('--GRAIN_READ_BUFFER_SIZE', type=int,
                    default=50, help='Grain read buffer size')
parser.add_argument('--GRAIN_WORKER_BUFFER_SIZE', type=int,
                    default=20, help='Grain worker buffer size')

parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--image_size', type=int, default=128, help='Image size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--steps_per_epoch', type=int,
                    default=None, help='Steps per epoch')
parser.add_argument('--dataset', type=str,
                    default='cc12m', help='Dataset to use')
parser.add_argument('--dataset_path', type=str,
                    default='/home/mrwhite0racle/gcs_mount', help="Dataset location path")

parser.add_argument('--learning_rate', type=float,
                    default=2e-4, help='Learning rate')
parser.add_argument('--noise_schedule', type=str, default='edm',
                    choices=['cosine', 'karras', 'edm'], help='Noise schedule')

parser.add_argument('--emb_features', type=int, default=256, help='Embedding features')
parser.add_argument('--feature_depths', type=int, nargs='+', default=[64, 128, 256, 512], help='Feature depths')
parser.add_argument('--attention_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--flash_attention', type=bool, default=False, help='Use Flash Attention')
parser.add_argument('--use_projection', type=bool, default=False, help='Use projection')
parser.add_argument('--use_self_and_cross', type=bool, default=False, help='Use self and cross attention')
parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
parser.add_argument('--num_middle_res_blocks', type=int,  default=1, help='Number of middle residual blocks')
parser.add_argument('--activation', type=str, default='swish', help='activation to use')

parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype to use')
parser.add_argument('--precision', type=str, default='high', help='precision to use')

parser.add_argument('--distributed_training', type=bool, default=None, help='Should use distributed training or not')
parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name, would be generated if not provided')
parser.add_argument('--load_from_checkpoint', type=bool,
                    default=False, help='Load from the best previously stored checkpoint. Experiment name needs to be provided')
parser.add_argument('--dataset_seed', type=int, default=0, help='Dataset starting seed')

parser.add_argument('--dataset_test', type=bool,
                    default=False, help='Run the dataset iterator for 3000 steps for testintg/benchmarking')

def main(args):
    if args.load_from_checkpoint:
        assert args.experiment_name is not None and args.experiment_name != "", "Experiment name needs to be given if load_from_checkpoint is True"

    jax.distributed.initialize()

    print(f"Number of devices: {jax.device_count()}")
    print(f"Local devices: {jax.local_devices()}")

    DTYPE_MAP = {
        'bfloat16': jnp.bfloat16,
        'float32': jnp.float32
    }

    PRECISION_MAP = {
        'high': jax.lax.Precision.HIGH,
        'default': jax.lax.Precision.DEFAULT,
        'highes': jax.lax.Precision.HIGHEST
    }

    ACTIVATION_MAP = {
        'swish': jax.nn.swish,
        'mish': jax.nn.mish,
    }

    DTYPE = DTYPE_MAP[args.dtype]
    PRECISION = PRECISION_MAP[args.precision]

    GRAIN_WORKER_COUNT = args.GRAIN_WORKER_COUNT
    GRAIN_READ_THREAD_COUNT = args.GRAIN_READ_THREAD_COUNT
    GRAIN_READ_BUFFER_SIZE = args.GRAIN_READ_BUFFER_SIZE
    GRAIN_WORKER_BUFFER_SIZE = args.GRAIN_WORKER_BUFFER_SIZE

    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size

    dataset_name = args.dataset
    datalen = len(datasetMap[dataset_name]['source'](args.dataset_path))
    batches = datalen // BATCH_SIZE
    # Define the configuration using the command-line arguments
    attention_configs = [
        None,
    ]

    attention_configs += [
        {"heads": args.attention_heads, "dtype": DTYPE, "flash_attention": args.flash_attention,
            "use_projection": args.use_projection, "use_self_and_cross": args.use_self_and_cross},
    ] * (len(args.feature_depths) - 2)

    attention_configs += [
        {"heads": args.attention_heads, "dtype": DTYPE, "flash_attention": False,
            "use_projection": False, "use_self_and_cross": False},
    ]

    CONFIG = {
        "model": {
            "emb_features": args.emb_features,
            "feature_depths": args.feature_depths,
            "attention_configs": attention_configs,
            "num_res_blocks": args.num_res_blocks,
            "num_middle_res_blocks": args.num_middle_res_blocks,
            "dtype": DTYPE,
            "precision": PRECISION,
            "activation": args.activation,
        },
        "dataset": {
            "name": dataset_name,
            "length": datalen,
            "batches": datalen // BATCH_SIZE,
        },
        "learning_rate": args.learning_rate,
        "batch_size": BATCH_SIZE,
        "epochs": args.epochs,
        "input_shapes": {
            "x": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "temb": (),
            "textcontext": (77, 768)
        },
        "arguments": vars(args),
    }

    text_encoders = defaultTextEncodeModel()

    data = get_dataset_grain(
        CONFIG['dataset']['name'],
        batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE,
        grain_worker_count=GRAIN_WORKER_COUNT, grain_read_thread_count=GRAIN_READ_THREAD_COUNT,
        grain_read_buffer_size=GRAIN_READ_BUFFER_SIZE, grain_worker_buffer_size=GRAIN_WORKER_BUFFER_SIZE,
        text_encoders=text_encoders,
        seed=args.dataset_seed,
    )

    if args.dataset_test:
        dataset = iter(data['train']())

        for _ in tqdm.tqdm(range(3000)):
            batch = next(dataset)

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
        
    print("Experiment_Name:", experiment_name)

    model_config = CONFIG['model']
    model_config['activation'] = ACTIVATION_MAP[model_config['activation']]
    unet = Unet(**model_config)

    learning_rate = CONFIG['learning_rate']
    solver = optax.adam(learning_rate)

    wandb_config = {
        "project": "flaxdiff",
        "config": CONFIG,
        "name": experiment_name,
    }
    
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
    )

    # %%
    batches = batches if args.steps_per_epoch is None else args.steps_per_epoch
    print(f"Training on {CONFIG['dataset']['name']} dataset with {batches} samples")
    jax.profiler.start_server(6009)
    final_state = trainer.fit(data, batches, epochs=CONFIG['epochs'])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
