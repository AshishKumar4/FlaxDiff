
%load_ext dotenv
%dotenv

import flax
import tqdm
from flax import linen as nn
import jax
from typing import Dict, Callable, Sequence, Any, Union
from dataclasses import field
import jax.numpy as jnp
import tensorflow_datasets as tfds
import grain.python as pygrain
import tensorflow as tf
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
from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True
import json
# For CLIP
from transformers import AutoTokenizer, FlaxCLIPTextModel, CLIPTextModel
import wandb

# %% [markdown]
# # Global Variables
#####################################################################################################################
############################################## Globasl Variables ####################################################
#####################################################################################################################

GRAIN_WORKER_COUNT = 16
GRAIN_READ_THREAD_COUNT = 64
GRAIN_READ_BUFFER_SIZE = 50
GRAIN_WORKER_BUFFER_SIZE = 20

# %% [markdown]
# # Initialization
#####################################################################################################################
################################################# Initialization ####################################################
#####################################################################################################################

# %%
jax.distributed.initialize() 

# %%
print(f"Number of devices: {jax.device_count()}")
print(f"Local devices: {jax.local_devices()}")

# %%
normalizeImage = lambda x: jax.nn.standardize(x, mean=[127.5], std=[127.5])
denormalizeImage = lambda x: (x + 1.0) * 127.5


def plotImages(imgs, fig_size=(8, 8), dpi=100):
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    imglen = imgs.shape[0]
    for i in range(imglen):
        plt.subplot(fig_size[0], fig_size[1], i + 1)
        plt.imshow(jnp.astype(denormalizeImage(imgs[i, :, :, :]), jnp.uint8))
        plt.axis("off")
    plt.show()

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

# %% [markdown]
# # Data Pipeline

# %%
def defaultTextEncodeModel(backend="jax"):
    modelname = "openai/clip-vit-large-patch14"
    if backend == "jax":
        model = FlaxCLIPTextModel.from_pretrained(modelname, dtype=jnp.bfloat16)
    else:
        model = CLIPTextModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname, dtype=jnp.float16)
    return model, tokenizer
    
def encodePrompts(prompts, model, tokenizer=None):
    if model == None:
        model, tokenizer = defaultTextEncodeModel()
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np")
    inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np")
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    # outputs = infer(inputs['input_ids'], inputs['attention_mask'])
    
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output  # pooled (EOS token) states
    embed_pooled = pooler_output#.astype(jnp.float16)
    embed_labels_full = last_hidden_state#.astype(jnp.float16)
    
    return embed_pooled, embed_labels_full

class CaptionProcessor:
    def __init__(self, tensor_type="pt", modelname="openai/clip-vit-large-patch14"):
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.tensor_type = tensor_type
        
    def __call__(self, caption):
        # print(caption)
        tokens = self.tokenizer(caption, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors=self.tensor_type)
        # print(tokens.keys())
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "caption": caption,
        }
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

# %%
def data_source_tfds(name):
    def data_source():
        return tfds.load(name, split="all", shuffle_files=True)
    return data_source

def data_source_cc12m(source="/home/mrwhite0racle/research/FlaxDiff/datasets/gcs_mount/arrayrecord/cc12m/"):
    def data_source():
        cc12m_records_path = source
        cc12m_records = [os.path.join(cc12m_records_path, i) for i in os.listdir(cc12m_records_path) if 'array_record' in i]
        ds = pygrain.ArrayRecordDataSource(cc12m_records[:-1])
        return ds
    return data_source

def labelizer_oxford_flowers102(path):
    with open(path, "r") as f:
        textlabels = [i.strip() for i in f.readlines()]
    textlabels = tf.convert_to_tensor(textlabels)
    def load_labels(sample):
        return textlabels[sample['label']]
    return load_labels

def labelizer_cc12m(sample):
    return sample['txt']

# Configure the following for your datasets
datasetMap = {
    "oxford_flowers102": {
        "source":data_source_tfds("oxford_flowers102"),
        "labelizer":labelizer_oxford_flowers102("/home/mrwhite0racle/tensorflow_datasets/oxford_flowers102/2.1.1/label.labels.txt"),
    },
    "cc12m": {
        "source":data_source_cc12m(),
        "labelizer":labelizer_cc12m,
    }
}

# %%
import struct as st

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

def get_dataset_grain(data_name="oxford_flowers102", 
                      batch_size=64, image_scale=256, 
                      count=None, num_epochs=None,
                      text_encoders=defaultTextEncodeModel(), 
                      method=jax.image.ResizeMethod.LANCZOS3):
    dataset = datasetMap[data_name]
    data_source = dataset["source"]()
    labelizer = dataset["labelizer"]
    
    import cv2
    
    model, tokenizer = text_encoders

    null_labels, null_labels_full = encodePrompts([""], model, tokenizer)
    null_labels = np.array(null_labels[0], dtype=np.float16)
    null_labels_full  = np.array(null_labels_full[0], dtype=np.float16)

    class augmenter(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.caption_processor = CaptionProcessor(tensor_type="np")
            
        def map(self, element) ->  Dict[str, jnp.array]:
            element = unpack_dict_of_byte_arrays(element)
            image = np.asarray(bytearray(element['jpg']), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_scale, image_scale), interpolation=cv2.INTER_AREA)
            # image = (image - 127.5) / 127.5
            caption = labelizer(element).decode('utf-8')
            results = self.caption_processor(caption)
            return {
                    "image": image,
                    "input_ids": results['input_ids'][0],
                    "attention_mask": results['attention_mask'][0],
                } 

    sampler = pygrain.IndexSampler(
        num_records=len(data_source) if count is None else count,
        shuffle=True,
        seed=0,
        num_epochs=num_epochs,
        shard_options=pygrain.NoSharding(),
    )

    transformations = [augmenter(), pygrain.Batch(batch_size, drop_remainder=True)]

    loader = pygrain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=GRAIN_WORKER_COUNT,
        read_options=pygrain.ReadOptions(GRAIN_READ_THREAD_COUNT, GRAIN_READ_BUFFER_SIZE),
        worker_buffer_size=GRAIN_WORKER_BUFFER_SIZE
        )
    
    def get_trainset():
        return loader
    
    return {
        "train": get_trainset,
        "train_len": len(data_source),
        "batch_size": batch_size,
        "null_labels": null_labels,
        "null_labels_full": null_labels_full,
        "model": model,
        "tokenizer": tokenizer,
    }

# %%
from flaxdiff.schedulers import CosineNoiseSchedule, NoiseScheduler, GeneralizedNoiseScheduler, KarrasVENoiseScheduler, EDMNoiseScheduler
from flaxdiff.predictors import VPredictionTransform, EpsilonPredictionTransform, DiffusionPredictionTransform, DirectPredictionTransform, KarrasPredictionTransform

# %% [markdown]
# # Modeling

# %% [markdown]
# ## Metrics

# %% [markdown]
# ## Callbacks

# %% [markdown]
# ## Model Generator

# %%
import jax.experimental.pallas.ops.tpu.flash_attention
from flaxdiff.models.simple_unet import l2norm, ConvLayer, TimeEmbedding, TimeProjection, Upsample, Downsample, ResidualBlock, PixelShuffle
from flaxdiff.models.simple_unet import FourierEmbedding

from flaxdiff.models.attention import kernel_init
# from flash_attn_jax import flash_mha
# from flaxdiff.models.favor_fastattn import make_fast_generalized_attention, make_fast_softmax_attention

# Kernel initializer to use
def kernel_init(scale, dtype=jnp.float32):
    scale = max(scale, 1e-10)
    return nn.initializers.variance_scaling(scale=scale, mode="fan_avg", distribution="truncated_normal", dtype=dtype)

class EfficientAttention(nn.Module):
    """
    Based on the pallas attention implementation.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_bias: bool = True
    kernel_init: Callable = lambda : kernel_init(1.0)

    def setup(self):
        inner_dim = self.dim_head * self.heads
        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        dense = functools.partial(
            nn.Dense,
            self.heads * self.dim_head,
            precision=self.precision, 
            use_bias=self.use_bias, 
            kernel_init=self.kernel_init(), 
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")
        
        self.proj_attn = nn.DenseGeneral(self.query_dim, use_bias=False, precision=self.precision, 
                                     kernel_init=self.kernel_init(), dtype=self.dtype, name="to_out_0")
        # self.attnfn = make_fast_generalized_attention(qkv_dim=inner_dim, lax_scan_unroll=16)
    
    def _reshape_tensor_to_head_dim(self, tensor):
        batch_size, _, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        return tensor
    
    def _reshape_tensor_from_head_dim(self, tensor):
        batch_size, _, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size, 1, seq_len, dim * head_size)
        return tensor

    @nn.compact
    def __call__(self, x:jax.Array, context=None):
        # print(x.shape)
        # x has shape [B, H * W, C]
        context = x if context is None else context
        
        B, H, W, C = x.shape
        x = x.reshape((B, 1, H * W, C))
        
        B, _H, _W, _C = context.shape
        context = context.reshape((B, 1, _H * _W, _C))
        
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        query = self._reshape_tensor_to_head_dim(query)
        key = self._reshape_tensor_to_head_dim(key)
        value = self._reshape_tensor_to_head_dim(value)
        
        hidden_states = jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(
            query, key, value, None
        )
        
        hidden_states = self._reshape_tensor_from_head_dim(hidden_states)
        
        
        # hidden_states = nn.dot_product_attention(
        #     query, key, value, dtype=self.dtype, broadcast_dropout=False, dropout_rng=None, precision=self.precision
        # )
        
        proj = self.proj_attn(hidden_states)
        
        proj = proj.reshape((B, H, W, C))
        
        return proj


class NormalAttention(nn.Module):
    """
    Simple implementation of the normal attention.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_bias: bool = True
    kernel_init: Callable = lambda : kernel_init(1.0)

    def setup(self):
        inner_dim = self.dim_head * self.heads
        dense = functools.partial(
            nn.DenseGeneral,
            features=[self.heads, self.dim_head], 
            axis=-1, 
            precision=self.precision, 
            use_bias=self.use_bias, 
            kernel_init=self.kernel_init(), 
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")

        self.proj_attn = nn.DenseGeneral(
            self.query_dim, 
            axis=(-2, -1), 
            precision=self.precision, 
            use_bias=self.use_bias, 
            dtype=self.dtype, 
            name="to_out_0",
            kernel_init=self.kernel_init()
            # kernel_init=jax.nn.initializers.xavier_uniform()
        )

    @nn.compact
    def __call__(self, x, context=None):
        # x has shape [B, H, W, C]
        context = x if context is None else context
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        hidden_states = nn.dot_product_attention(
            query, key, value, dtype=self.dtype, broadcast_dropout=False, dropout_rng=None, precision=self.precision
        )
        
        proj = self.proj_attn(hidden_states)
        return proj
    
class AttentionBlock(nn.Module):
    # Has self and cross attention
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_bias: bool = True
    kernel_init: Callable = lambda : kernel_init(1.0)
    use_flash_attention:bool = False
    use_cross_only:bool = False
    
    def setup(self):
        if self.use_flash_attention:
            attenBlock = EfficientAttention
        else:
            attenBlock = NormalAttention
            
        self.attention1 = attenBlock(
         query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention1',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init
        )
        self.attention2 = attenBlock(
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention2',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init
        )
        
        self.ff = nn.DenseGeneral(
            features=self.query_dim,
            use_bias=self.use_bias,
            precision=self.precision,
            dtype=self.dtype,
            kernel_init=self.kernel_init(),
            name="ff"
        )
        self.norm1 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm4 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        
    @nn.compact
    def __call__(self, hidden_states, context=None):
        # self attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.use_cross_only:
            hidden_states = self.attention1(hidden_states, context)
        else:
            hidden_states = self.attention1(hidden_states)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.attention2(hidden_states, context)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states

class TransformerBlock(nn.Module):
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Any = jnp.bfloat16
    precision: Any = jax.lax.Precision.HIGH
    use_projection: bool = False
    use_flash_attention:bool = True
    use_self_and_cross:bool = False

    @nn.compact
    def __call__(self, x, context=None):
        inner_dim = self.heads * self.dim_head
        B, H, W, C = x.shape
        normed_x = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)(x)
        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=inner_dim, 
                                       use_bias=False, precision=self.precision, 
                                       kernel_init=kernel_init(1.0),
                                       dtype=self.dtype, name=f'project_in')(normed_x)
            else:
                projected_x = nn.Conv(
                    features=inner_dim, kernel_size=(1, 1),
                    kernel_init=kernel_init(1.0),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_in_conv',
                )(normed_x)
        else:
            projected_x = normed_x
            inner_dim = C
            
        context = projected_x if context is None else context

        if self.use_self_and_cross:
            projected_x = AttentionBlock(
                query_dim=inner_dim,
                heads=self.heads,
                dim_head=self.dim_head,
                name=f'Attention',
                precision=self.precision,
                use_bias=False,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                use_cross_only=False
            )(projected_x, context)
        elif self.use_flash_attention == True:
            projected_x = EfficientAttention(
                query_dim=inner_dim,
                heads=self.heads,
                dim_head=self.dim_head,
                name=f'Attention',
                precision=self.precision,
                use_bias=False,
                dtype=self.dtype,
            )(projected_x, context)
        else:
            projected_x = NormalAttention(
                query_dim=inner_dim,
                heads=self.heads,
                dim_head=self.dim_head,
                name=f'Attention',
                precision=self.precision,
                use_bias=False,
            )(projected_x, context)
        

        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=C, precision=self.precision, 
                                       dtype=self.dtype, use_bias=False, 
                                       kernel_init=kernel_init(1.0),
                                       name=f'project_out')(projected_x)
            else:
                projected_x = nn.Conv(
                    features=C, kernel_size=(1, 1),
                    kernel_init=kernel_init(1.0),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_out_conv',
                )(projected_x)

        out = x + projected_x
        return out


# %% [markdown]
# ## Attention and other prototyping

# %%
x = jnp.ones((16, 1, 16*16, 64))
batch_size, _, seq_len, dim = x.shape
head_size = 4
dim_head = dim // head_size
k = nn.Dense(dim_head * head_size, precision=jax.lax.Precision.HIGHEST, use_bias=True)
param = k.init(jax.random.PRNGKey(42), x)
tensor = k.apply(param, x)
print(tensor.shape)
tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
tensor = jnp.transpose(tensor, (0, 2, 1, 3))
print(tensor.shape)



# %%
x = jnp.ones((16, 64, 64, 128))
context = jnp.ones((16, 64, 64, 128))
attention_block = TransformerBlock(heads=4, dim_head=64//4, dtype=jnp.bfloat16, use_flash_attention=False, use_projection=False, use_self_and_cross=False)
params = attention_block.init(jax.random.PRNGKey(0), x, context)
@jax.jit
def apply(params, x, context):
    return attention_block.apply(params, x, context)

apply(params, x, context)

%timeit -n 1 apply(params, x, context)

# %%
x = jnp.ones((1, 16, 16, 64))
context = jnp.ones((1, 12, 768))
# pad the context
context = jnp.pad(context, ((0, 0), (0, 4), (0, 0)), mode='constant', constant_values=0)
print(context.shape)
context = None#jnp.reshape(context, (1, 1, 16, 768))
attention_block = TransformerBlock(heads=4, dim_head=64//4, dtype=jnp.bfloat16, use_flash_attention=True, use_projection=False, use_self_and_cross=False)
params = attention_block.init(jax.random.PRNGKey(0), x, context)
out = attention_block.apply(params, x, context)
print("Output :", out.shape)
print(attention_block.tabulate(jax.random.key(0), x, context, console_kwargs={"width": 200, "force_jupyter":True, }))
print(jnp.mean(out), jnp.std(out))
# plt.hist(out.flatten(), bins=100)
# %timeit attention_block.apply(params, x)

# %% [markdown]
# ## Main Model

# %%
class ResidualBlock(nn.Module):
    conv_type:str
    features:int
    kernel_size:tuple=(3, 3)
    strides:tuple=(1, 1)
    padding:str="SAME"
    activation:Callable=jax.nn.swish
    direction:str=None
    res:int=2
    norm_groups:int=8
    kernel_init:Callable=kernel_init(1.0)
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST

    @nn.compact
    def __call__(self, x:jax.Array, temb:jax.Array, textemb:jax.Array=None, extra_features:jax.Array=None):
        residual = x
        out = nn.GroupNorm(self.norm_groups)(x)
        out = self.activation(out)

        out = ConvLayer(
            self.conv_type,
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=self.kernel_init,
            name="conv1",
            dtype=self.dtype,
            precision=self.precision
        )(out)

        temb = nn.DenseGeneral(
            features=self.features, 
            name="temb_projection",
            dtype=self.dtype,
            precision=self.precision)(temb)
        temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
        # scale, shift = jnp.split(temb, 2, axis=-1)
        # out = out * (1 + scale) + shift
        out = out + temb

        out = nn.GroupNorm(self.norm_groups)(out)
        out = self.activation(out)

        out = ConvLayer(
            self.conv_type,
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=self.kernel_init,
            name="conv2",
            dtype=self.dtype,
            precision=self.precision
        )(out)

        if residual.shape != out.shape:
            residual = ConvLayer(
                self.conv_type,
                features=self.features,
                kernel_size=(1, 1),
                strides=1,
                kernel_init=self.kernel_init,
                name="residual_conv",
                dtype=self.dtype,
                precision=self.precision
            )(residual)
        out = out + residual

        out = jnp.concatenate([out, extra_features], axis=-1) if extra_features is not None else out

        return out
    
class Unet(nn.Module):
    emb_features:int=64*4,
    feature_depths:list=[64, 128, 256, 512],
    attention_configs:list=[{"heads":8}, {"heads":8}, {"heads":8}, {"heads":8}],
    num_res_blocks:int=2,
    num_middle_res_blocks:int=1,
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Any = jnp.bfloat16
    precision: Any = jax.lax.Precision.HIGH

    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        # print("embedding features", self.emb_features)
        temb = FourierEmbedding(features=self.emb_features)(temb)
        temb = TimeProjection(features=self.emb_features)(temb)
        
        _, TS, TC = textcontext.shape
        
        # print("time embedding", temb.shape)
        feature_depths = self.feature_depths
        attention_configs = self.attention_configs

        conv_type = up_conv_type = down_conv_type = middle_conv_type = "conv"
        # middle_conv_type = "separable"

        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=kernel_init(1.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        downs = [x]

        # Downscaling blocks
        for i, (dim_out, attention_config) in enumerate(zip(feature_depths, attention_configs)):
            dim_in = x.shape[-1]
            # dim_in = dim_out
            for j in range(self.num_res_blocks):
                x = ResidualBlock(
                    down_conv_type,
                    name=f"down_{i}_residual_{j}",
                    features=dim_in,
                    kernel_init=kernel_init(1.0),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    B, H, W, _ = x.shape
                    if H > TS:
                        padded_context = jnp.pad(textcontext, ((0, 0), (0, H - TS), (0, 0)), mode='constant', constant_values=0).reshape((B, 1, H, TC))
                    else:
                        padded_context = None
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32),
                                       dim_head=dim_in // attention_config['heads'],
                                       use_flash_attention=attention_config.get("flash_attention", True),
                                       use_projection=attention_config.get("use_projection", False),
                                       use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                       precision=attention_config.get("precision", self.precision),
                                       name=f"down_{i}_attention_{j}")(x, padded_context)
                # print("down residual for feature level", i, "is of shape", x.shape, "features", dim_in)
                downs.append(x)
            if i != len(feature_depths) - 1:
                # print("Downsample", i, x.shape)
                x = Downsample(
                    features=dim_out,
                    scale=2,
                    activation=self.activation,
                    name=f"down_{i}_downsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # Middle Blocks
        middle_dim_out = self.feature_depths[-1]
        middle_attention = self.attention_configs[-1]
        for j in range(self.num_middle_res_blocks):
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res1_{j}",
                features=middle_dim_out,
                kernel_init=kernel_init(1.0),
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision
            )(x, temb)
            if middle_attention is not None and j == self.num_middle_res_blocks - 1:   # Apply attention only on the last block
                x = TransformerBlock(heads=middle_attention['heads'], dtype=middle_attention.get('dtype', jnp.float32), 
                                    dim_head=middle_dim_out // middle_attention['heads'],
                                    use_flash_attention=middle_attention.get("flash_attention", True),
                                    use_linear_attention=False,
                                    use_projection=middle_attention.get("use_projection", False),
                                    use_self_and_cross=False,
                                    precision=middle_attention.get("precision", self.precision),
                                    name=f"middle_attention_{j}")(x)
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res2_{j}",
                features=middle_dim_out,
                kernel_init=kernel_init(1.0),
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision
            )(x, temb)

        # Upscaling Blocks
        for i, (dim_out, attention_config) in enumerate(zip(reversed(feature_depths), reversed(attention_configs))):
            # print("Upscaling", i, "features", dim_out)
            for j in range(self.num_res_blocks):
                residual = downs.pop()
                x = jnp.concatenate([x, residual], axis=-1)
                # print("concat==> ", i, "concat", x.shape)
                # kernel_size = (1 + 2 * (j + 1), 1 + 2 * (j + 1))
                kernel_size = (3, 3)
                x = ResidualBlock(
                    up_conv_type,# if j == 0 else "separable",
                    name=f"up_{i}_residual_{j}",
                    features=dim_out,
                    kernel_init=kernel_init(1.0),
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    # B, H, W, _ = x.shape
                    # if H > TS:
                    #     padded_context = jnp.pad(textcontext, ((0, 0), (0, H - TS), (0, 0)), mode='constant', constant_values=0).reshape((B, 1, H, TC))
                    # else:
                    #     padded_context = None
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32), 
                                       dim_head=dim_out // attention_config['heads'],
                                       use_flash_attention=attention_config.get("flash_attention", True),
                                       use_projection=attention_config.get("use_projection", False),
                                       use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                        precision=attention_config.get("precision", self.precision),
                                       name=f"up_{i}_attention_{j}")(x, residual)
            # print("Upscaling ", i, x.shape)
            if i != len(feature_depths) - 1:
                x = Upsample(
                    features=feature_depths[-i],
                    scale=2,
                    activation=self.activation,
                    name=f"up_{i}_upsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # x = nn.GroupNorm(8)(x)
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=kernel_init(0.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
    
        x = jnp.concatenate([x, downs.pop()], axis=-1)

        x = ResidualBlock(
            conv_type,
            name="final_residual",
            features=self.feature_depths[0],
            kernel_init=kernel_init(1.0),
            kernel_size=(3,3),
            strides=(1, 1),
            activation=self.activation,
            norm_groups=self.norm_groups,
            dtype=self.dtype,
            precision=self.precision
        )(x, temb)

        x = nn.GroupNorm(self.norm_groups)(x)
        x = self.activation(x)

        noise_out = ConvLayer(
            conv_type,
            features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            # activation=jax.nn.mish
            kernel_init=kernel_init(0.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        return noise_out#, attentions

# %%
unet = Unet(emb_features=512, 
            feature_depths=[128, 256, 512, 1024],
            attention_configs=[
                None,
                # None,
                # {"heads":32, "dtype":jnp.bfloat16, "flash_attention":True, "use_projection":False, "use_self_and_cross":True}, 
                {"heads":32, "dtype":jnp.bfloat16, "flash_attention":True, "use_projection":True, "use_self_and_cross":True}, 
                {"heads":32, "dtype":jnp.bfloat16, "flash_attention":True, "use_projection":True, "use_self_and_cross":True}, 
                {"heads":32, "dtype":jnp.bfloat16, "flash_attention":False, "use_projection":False, "use_self_and_cross":False}
                ],
            num_res_blocks=4,
            num_middle_res_blocks=1
)

inp = jnp.ones((1, 128, 128, 3))
temb = jnp.ones((1,))
textcontext = jnp.ones((1, 77, 768))

params = unet.init(jax.random.PRNGKey(0), inp, temb, textcontext)

# %%
unet.tabulate(jax.random.key(0), inp, temb, textcontext, console_kwargs={"width": 200, "force_jupyter":True, })

# %% [markdown]
# # Training

# %%
import flax.jax_utils
import orbax.checkpoint
import orbax
from typing import Any, Tuple, Mapping,Callable,List,Dict
from flax.metrics import tensorboard
from functools import partial

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
    state : SimpleTrainState
    best_state : SimpleTrainState
    best_loss : float
    model : nn.Module
    ema_decay:float = 0.999
    
    def __init__(self, 
                 model:nn.Module, 
                 input_shapes:Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 rngs:jax.random.PRNGKey,
                 train_state:SimpleTrainState=None,
                 name:str="Simple",
                 load_from_checkpoint:bool=False,
                 checkpoint_suffix:str="",
                 loss_fn=optax.l2_loss,
                 param_transforms:Callable=None,
                 wandb_config:Dict[str, Any]=None
                 ):
        self.model = model
        self.name = name
        self.loss_fn = loss_fn
        self.input_shapes = input_shapes
        
        if wandb_config is not None:
            run = wandb.init(**wandb_config)
            self.wandb = run

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=4, create=True)
        self.checkpointer = orbax.checkpoint.CheckpointManager(self.checkpoint_path() + checkpoint_suffix, checkpointer, options)

        if load_from_checkpoint:
            latest_epoch, old_state, old_best_state = self.load()
        else:
            latest_epoch, old_state, old_best_state = 0, None, None
            
        self.latest_epoch = latest_epoch

        if train_state == None:
            self.init_state(optimizer, rngs, existing_state=old_state, existing_best_state=old_best_state, model=model, param_transforms=param_transforms)
        else:
            self.state = train_state
            self.best_state = train_state
            self.best_loss = 1e9
    
    def get_input_ones(self):
        return {k:jnp.ones((1, *v)) for k,v in self.input_shapes.items()}

    def init_state(self,
                   optimizer: optax.GradientTransformation, 
                   rngs:jax.random.PRNGKey,
                   existing_state:dict=None,
                   existing_best_state:dict=None,
                   model:nn.Module=None,
                   param_transforms:Callable=None
                   ):
        @partial(jax.pmap, axis_name="device")
        def init_fn(rngs):
            rngs, subkey = jax.random.split(rngs)

            if existing_state == None:
                input_vars = self.get_input_ones()
                params = model.init(subkey, **input_vars)

            # if param_transforms is not None:
            #     params = param_transforms(params)
                
            state = SimpleTrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optimizer,
                rngs=rngs,
                metrics=Metrics.empty()
            )
            return state
        self.state = init_fn(jax.device_put_replicated(rngs, jax.devices()))
        self.best_loss = 1e9
        if existing_best_state is not None:
            self.best_state = self.state.replace(params=existing_best_state['params'], ema_params=existing_best_state['ema_params'])
        else:
            self.best_state = self.state
            
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
        print(f"Loaded model from checkpoint at epoch {epoch}", ckpt['best_loss'])
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
            self.checkpointer.save(epoch, ckpt, save_kwargs={'save_args': save_args}, force=True)
            pass
        except Exception as e:
            print("Error saving checkpoint", e)

    def _define_train_step(self, **kwargs):
        model = self.model
        loss_fn = self.loss_fn
        
        @partial(jax.pmap, axis_name="device")
        def train_step(state:SimpleTrainState, batch):
            """Train for a single step."""
            images = batch['image']
            labels= batch['label']
            
            def model_loss(params):
                preds = model.apply(params, images)
                expected_output = labels
                nloss = loss_fn(preds, expected_output)
                loss = jnp.mean(nloss)
                return loss
            loss, grads = jax.value_and_grad(model_loss)(state.params)
            grads = jax.lax.pmean(grads, "device")
            state = state.apply_gradients(grads=grads) 
            return state, loss
        return train_step
    
    def _define_compute_metrics(self):
        model = self.model
        loss_fn = self.loss_fn
        
        @jax.jit
        def compute_metrics(state:SimpleTrainState, batch):
            preds = model.apply(state.params, batch['image'])
            expected_output = batch['label']
            loss = jnp.mean(loss_fn(preds, expected_output))
            metric_updates = state.metrics.single_from_model_output(loss=loss, logits=preds, labels=expected_output)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

    def summary(self):
        input_vars = self.get_input_ones()
        print(self.model.tabulate(jax.random.key(0), **input_vars, console_kwargs={"width": 200, "force_jupyter":True, }))
    
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
        device_count = jax.device_count()
        # train_ds = flax.jax_utils.prefetch_to_device(train_ds, jax.devices())
        
        summary_writer = self.init_tensorboard(data['batch_size'], steps_per_epoch, epochs)
        
        while self.latest_epoch <= epochs:
            self.latest_epoch += 1
            current_epoch = self.latest_epoch
            print(f"\nEpoch {current_epoch}/{epochs}")
            start_time = time.time()
            epoch_loss = 0
            
            with tqdm.tqdm(total=steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step') as pbar:
                for i in range(steps_per_epoch):
                    batch = next(train_ds)
                    batch = jax.tree.map(lambda x: x.reshape((device_count, -1, *x.shape[1:])), batch)
                    # print(batch['image'].shape)
                    state, loss = train_step(state, batch)
                    loss = jnp.mean(loss)
                    # print("==>", loss)
                    epoch_loss += loss
                    if i % 100 == 0:
                        pbar.set_postfix(loss=f'{loss:.4f}')
                        pbar.update(100)
                        current_step = current_epoch*steps_per_epoch + i
                        summary_writer.scalar('Train Loss', loss, step=current_step)
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
            # if test_ds is not None:
            #     for test_batch in iter(test_ds()):
            #         state = compute_metrics(state, test_batch)
            #     metrics = state.metrics.compute()
            #     for metric,value in metrics.items():
            #         summary_writer.scalar(f'Test {metric}', value, step=current_epoch)
            #         metrics_str += f', Test {metric}: {value:.4f}'
            #     state = state.replace(metrics=Metrics.empty())
                    
            print(f"\n\tEpoch {current_epoch} completed. Avg Loss: {avg_loss}, Time: {total_time:.2f}s, Best Loss: {self.best_loss} {metrics_str}")
            
        self.save(epochs)
        return self.state

# Define the TrainState with EMA parameters
class TrainState(SimpleTrainState):
    rngs: jax.random.PRNGKey
    ema_params: dict

    def get_random_key(self):
        rngs, subkey = jax.random.split(self.rngs)
        return self.replace(rngs=rngs), subkey

    def apply_ema(self, decay: float=0.999):
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, param: decay * ema + (1 - decay) * param,
            self.ema_params,
            self.params,
        )
        return self.replace(ema_params=new_ema_params)

class DiffusionTrainer(SimpleTrainer):
    noise_schedule : NoiseScheduler
    model_output_transform:DiffusionPredictionTransform
    ema_decay:float = 0.999
    
    def __init__(self, 
                 model:nn.Module, 
                 input_shapes:Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 noise_schedule:NoiseScheduler,
                 rngs:jax.random.PRNGKey,
                 unconditional_prob:float=0.2,
                 name:str="Diffusion",
                 model_output_transform:DiffusionPredictionTransform=EpsilonPredictionTransform(),
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

    def init_state(self, 
                   optimizer: optax.GradientTransformation, 
                   rngs:jax.random.PRNGKey,
                   existing_state:dict=None,
                   existing_best_state:dict=None,
                   model:nn.Module=None,
                   param_transforms:Callable=None,
                   ):
        # @partial(jax.pmap, axis_name="device")
        def init_fn(rngs):
            rngs, subkey = jax.random.split(rngs)

            if existing_state == None:
                input_vars = self.get_input_ones()
                params = model.init(subkey, **input_vars)
                new_state = {"params":params, "ema_params":params}
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
            return state
            
        self.best_loss = 1e9
        # self.state = init_fn(jax.device_put_replicated(rngs, jax.devices()))
        state = init_fn(rngs)
        if existing_best_state is not None:
            best_state = state.replace(params=existing_best_state['params'], ema_params=existing_best_state['ema_params'])
        else:
            best_state = state
            
        self.state = flax.jax_utils.replicate(state, jax.devices())
        self.best_state = flax.jax_utils.replicate(best_state, jax.devices())

    def _define_train_step(self, batch_size, null_labels_seq, text_embedder):
        noise_schedule = self.noise_schedule
        model = self.model
        model_output_transform = self.model_output_transform
        loss_fn = self.loss_fn
        unconditional_prob = self.unconditional_prob
        
        # Determine the number of unconditional samples
        num_unconditional = int(batch_size * unconditional_prob)
        
        nS, nC = null_labels_seq.shape
        null_labels_seq = jnp.broadcast_to(null_labels_seq, (batch_size, nS, nC))
        
        # @jax.jit
        @partial(jax.pmap, axis_name="device")
        def train_step(state:TrainState, batch):
            """Train for a single step."""
            images = batch['image']
            # normalize image
            images = (images - 127.5) / 127.5
            
            output = text_embedder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            # output = infer(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
            label_seq = output.last_hidden_state
            
            # Generate random probabilities to decide how much of this batch will be unconditional
            
            label_seq = jnp.concat([null_labels_seq[:num_unconditional], label_seq[num_unconditional:]], axis=0)

            noise_level, state = noise_schedule.generate_timesteps(images.shape[0], state)
            state, rngs = state.get_random_key()
            noise:jax.Array = jax.random.normal(rngs, shape=images.shape)
            rates = noise_schedule.get_rates(noise_level)
            noisy_images, c_in, expected_output = model_output_transform.forward_diffusion(images, noise, rates)
            def model_loss(params):
                preds = model.apply(params, *noise_schedule.transform_inputs(noisy_images*c_in, noise_level), label_seq)
                preds = model_output_transform.pred_transform(noisy_images, preds, rates)
                nloss = loss_fn(preds, expected_output)
                # nloss = jnp.mean(nloss, axis=1)
                nloss *= noise_schedule.get_weights(noise_level)
                nloss = jnp.mean(nloss)
                loss = nloss
                return loss
            loss, grads = jax.value_and_grad(model_loss)(state.params)
            grads = jax.lax.pmean(grads, "device")
            state = state.apply_gradients(grads=grads) 
            state = state.apply_ema(self.ema_decay)
            return state, loss
        return train_step
    
    def _define_compute_metrics(self):
        @jax.jit
        def compute_metrics(state:TrainState, expected, pred):
            loss = jnp.mean(jnp.square(pred - expected))
            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

    def fit(self, data, steps_per_epoch, epochs):
        null_labels_full = data['null_labels_full']
        batch_size = data['batch_size']
        text_embedder = data['model']
        super().fit(data, steps_per_epoch, epochs, {"batch_size":batch_size, "null_labels_seq":null_labels_full, "text_embedder":text_embedder})

# %%
BATCH_SIZE = 64
IMAGE_SIZE = 128

cosine_schedule = CosineNoiseSchedule(1000, beta_end=1)
karas_ve_schedule = KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
edm_schedule = EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)

experiment_name = "{name}_{date}".format(
    name="Diffusion_SDE_VE_TEXT", date=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)
# experiment_name = 'Diffusion_SDE_VE_TEXT_2024-07-16_02:16:07'
# experiment_name = 'Diffusion_SDE_VE_TEXT_2024-07-21_02:12:40'
# experiment_name = 'Diffusion_SDE_VE_TEXT_2024-07-30_05:48:22'
# experiment_name = 'Diffusion_SDE_VE_TEXT_2024-08-01_08:59:00'
print("Experiment_Name:", experiment_name)

dataset_name = "cc12m"
datalen = len(datasetMap[dataset_name]['source'])
batches = datalen // BATCH_SIZE

config = {
    "model" : {
        "emb_features":256, 
        "feature_depths":[64, 128, 256, 512],
        "attention_configs":[
            None,
            # None,
            # None,
            # None,
            # None,
            # {"heads":32, "dtype":jnp.bfloat16, "flash_attention":True, "use_projection":False, "use_self_and_cross":True}, 
            {"heads":8, "dtype":jnp.bfloat16, "flash_attention":True, "use_projection":False, "use_self_and_cross":False}, 
            {"heads":8, "dtype":jnp.bfloat16, "flash_attention":True, "use_projection":False, "use_self_and_cross":False}, 
            {"heads":8, "dtype":jnp.bfloat16, "flash_attention":False, "use_projection":False, "use_self_and_cross":False},
        ],
        "num_res_blocks":2,
        "num_middle_res_blocks":1,
    },
    
    "dataset": {
        "name" : dataset_name,
        "length" : datalen,
        "batches": batches
    },
    "learning_rate": 2e-4,
    
    "input_shapes": {
        "x": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "temb": (),
        "textcontext": (77, 768)
    },
}

unet = Unet(**config['model'])

learning_rate = config['learning_rate']
solver = optax.adam(learning_rate)
# solver = optax.adamw(2e-6)

trainer = DiffusionTrainer(unet, optimizer=solver, 
                           input_shapes=config['input_shapes'], 
                           noise_schedule=edm_schedule,
                           rngs=jax.random.PRNGKey(4), 
                           name=experiment_name,
                           model_output_transform=KarrasPredictionTransform(sigma_data=edm_schedule.sigma_data),
                        #    train_state=trainer.best_state,
                        #    loss_fn=lambda x, y: jnp.abs(x - y),
                            # param_transforms=params_transform,
                        #    load_from_checkpoint=True,
                           wandb_config={
                                 "project": "flaxdiff",
                                 "config": config,
                                 "name": experiment_name,
                               },
                           )


# %%
trainer.summary()

# %%
data = get_dataset_grain(config['dataset']['name'], batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE)

# %%
# jax.profiler.start_server(6009)
final_state = trainer.fit(data, 1000, epochs=3)

# %%
# jax.profiler.start_server(6009)
final_state = trainer.fit(data, 1000, epochs=1)

# %%
# jax.profiler.start_server(6009)
final_state = trainer.fit(data, 1000, epochs=1)

# %%
data = get_dataset("oxford_flowers102", batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE)
final_state = trainer.fit(data, batches, epochs=4000)

# %%
from flaxdiff.utils import clip_images

def clip_images(images, clip_min=-1, clip_max=1):
    return jnp.clip(images, clip_min, clip_max)
    
class DiffusionSampler():
    model:nn.Module
    noise_schedule:NoiseScheduler
    params:dict
    model_output_transform:DiffusionPredictionTransform

    def __init__(self, model:nn.Module, params:dict,  
                 noise_schedule:NoiseScheduler, 
                 model_output_transform:DiffusionPredictionTransform=EpsilonPredictionTransform(),
                 guidance_scale:float = 0.0,
                 null_labels_seq:jax.Array=None
                 ):
        self.model = model
        self.noise_schedule = noise_schedule
        self.params = params
        self.model_output_transform = model_output_transform
        self.guidance_scale = guidance_scale
        if self.guidance_scale > 0:
            # Classifier free guidance
            assert null_labels_seq is not None, "Null labels sequence is required for classifier-free guidance"
            print("Using classifier-free guidance")
            @jax.jit
            def sample_model(x_t, t, *additional_inputs):
                # Concatenate unconditional and conditional inputs
                x_t_cat = jnp.concatenate([x_t] * 2, axis=0)
                t_cat = jnp.concatenate([t] * 2, axis=0)
                rates_cat = self.noise_schedule.get_rates(t_cat)
                c_in_cat = self.model_output_transform.get_input_scale(rates_cat)
                
                text_labels_seq, = additional_inputs
                text_labels_seq = jnp.concatenate([text_labels_seq, jnp.broadcast_to(null_labels_seq, text_labels_seq.shape)], axis=0)
                model_output = self.model.apply(self.params, *self.noise_schedule.transform_inputs(x_t_cat * c_in_cat, t_cat), text_labels_seq)
                # Split model output into unconditional and conditional parts
                model_output_cond, model_output_uncond = jnp.split(model_output, 2, axis=0)
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
                
                x_0, eps = self.model_output_transform(x_t, model_output, t, self.noise_schedule)
                return x_0, eps, model_output
            
            self.sample_model = sample_model
        else:
            # Unconditional sampling
            @jax.jit
            def sample_model(x_t, t, *additional_inputs):
                rates = self.noise_schedule.get_rates(t)
                c_in = self.model_output_transform.get_input_scale(rates)
                model_output = self.model.apply(self.params, *self.noise_schedule.transform_inputs(x_t * c_in, t), *additional_inputs)
                x_0, eps = self.model_output_transform(x_t, model_output, t, self.noise_schedule)
                return x_0, eps, model_output
            
            self.sample_model = sample_model

    # Used to sample from the diffusion model
    def sample_step(self, current_samples:jnp.ndarray, current_step, model_conditioning_inputs, next_step=None, state:MarkovState=None) -> tuple[jnp.ndarray, MarkovState]:
        # First clip the noisy images
        step_ones = jnp.ones((current_samples.shape[0], ), dtype=jnp.int32)
        current_step = step_ones * current_step
        next_step = step_ones * next_step
        pred_images, pred_noise, _ = self.sample_model(current_samples, current_step, *model_conditioning_inputs)
        # plotImages(pred_images)
        pred_images = clip_images(pred_images)
        new_samples, state =  self.take_next_step(current_samples=current_samples, reconstructed_samples=pred_images, 
                             pred_noise=pred_noise, current_step=current_step, next_step=next_step, state=state,
                             model_conditioning_inputs=model_conditioning_inputs
                             )
        return new_samples, state

    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        # estimate the q(x_{t-1} | x_t, x_0). 
        # pred_images is x_0, noisy_images is x_t, steps is t
        return NotImplementedError
    
    def scale_steps(self, steps):
        scale_factor = self.noise_schedule.max_timesteps / 1000
        return steps * scale_factor

    def get_steps(self, start_step, end_step, diffusion_steps):
        step_range = start_step - end_step
        if diffusion_steps is None or diffusion_steps == 0:
            diffusion_steps = start_step - end_step
        diffusion_steps = min(diffusion_steps, step_range)
        steps = jnp.linspace(end_step, start_step, diffusion_steps, dtype=jnp.int16)[::-1]
        return steps
    
    def get_initial_samples(self, num_images, rngs:jax.random.PRNGKey, start_step):
        start_step = self.scale_steps(start_step)
        alpha_n, sigma_n = self.noise_schedule.get_rates(start_step)
        variance = jnp.sqrt(alpha_n ** 2 + sigma_n ** 2) 
        return jax.random.normal(rngs, (num_images, IMAGE_SIZE, IMAGE_SIZE, 3)) * variance

    def generate_images(self,
                        num_images=16, 
                        diffusion_steps=1000, 
                        start_step:int = None,
                        end_step:int = 0,
                        steps_override=None,
                        priors=None, 
                        rngstate:RandomMarkovState=RandomMarkovState(jax.random.PRNGKey(42)),
                        model_conditioning_inputs:tuple=()
                        ) -> jnp.ndarray:
        if priors is None:
            rngstate, newrngs = rngstate.get_random_key()
            samples = self.get_initial_samples(num_images, newrngs, start_step)
        else:
            print("Using priors")
            samples = priors

        # @jax.jit
        def sample_step(state:RandomMarkovState, samples, current_step, next_step):
            samples, state = self.sample_step(current_samples=samples,
                                              current_step=current_step, 
                                              model_conditioning_inputs=model_conditioning_inputs,
                                              state=state, next_step=next_step)
            return samples, state

        if start_step is None:
            start_step = self.noise_schedule.max_timesteps

        if steps_override is not None:
            steps = steps_override
        else:
            steps = self.get_steps(start_step, end_step, diffusion_steps)

        # print("Sampling steps", steps)
        for i in tqdm.tqdm(range(0, len(steps))):
            current_step = self.scale_steps(steps[i])
            next_step = self.scale_steps(steps[i+1] if i+1 < len(steps) else 0)
            if i != len(steps) - 1:
                # print("normal step")
                samples, rngstate = sample_step(rngstate, samples, current_step, next_step)
            else:
                # print("last step")
                step_ones = jnp.ones((num_images, ), dtype=jnp.int32)
                samples, _, _ = self.sample_model(samples, current_step * step_ones, *model_conditioning_inputs)
        samples = clip_images(samples)
        return samples

class DDPMSampler(DiffusionSampler):
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        mean = self.noise_schedule.get_posterior_mean(reconstructed_samples, current_samples, current_step)
        variance = self.noise_schedule.get_posterior_variance(steps=current_step)
        
        state, rng = state.get_random_key()
        # Now sample from the posterior
        noise = jax.random.normal(rng, reconstructed_samples.shape, dtype=jnp.float32)

        return mean + noise * variance, state
    
    def generate_images(self, num_images=16, diffusion_steps=1000, start_step: int = None, *args, **kwargs):
        return super().generate_images(num_images=num_images, diffusion_steps=diffusion_steps, start_step=start_step, *args, **kwargs)
    
class SimpleDDPMSampler(DiffusionSampler):
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        noise = jax.random.normal(rng, reconstructed_samples.shape, dtype=jnp.float32)

        # Compute noise rates and signal rates only once
        current_signal_rate, current_noise_rate = self.noise_schedule.get_rates(current_step)
        next_signal_rate, next_noise_rate = self.noise_schedule.get_rates(next_step)
        
        pred_noise_coeff = ((next_noise_rate ** 2) * current_signal_rate) / (current_noise_rate * next_signal_rate)
        
        noise_ratio_squared = (next_noise_rate ** 2) / (current_noise_rate ** 2)
        signal_ratio_squared = (current_signal_rate ** 2) / (next_signal_rate ** 2)
        gamma = jnp.sqrt(noise_ratio_squared * (1 - signal_ratio_squared))
        
        next_samples = next_signal_rate * reconstructed_samples + pred_noise_coeff * pred_noise + noise * gamma
        return next_samples, state

class DDIMSampler(DiffusionSampler):
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        next_signal_rate, next_noise_rate = self.noise_schedule.get_rates(next_step)
        return reconstructed_samples * next_signal_rate + pred_noise * next_noise_rate, state
    
class EulerSampler(DiffusionSampler):
    # Basically a DDIM Sampler but parameterized as an ODE
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        current_alpha, current_sigma = self.noise_schedule.get_rates(current_step)
        next_alpha, next_sigma = self.noise_schedule.get_rates(next_step)

        dt = next_sigma - current_sigma
        
        x_0_coeff = (current_alpha * next_sigma - next_alpha * current_sigma) / (dt)
        dx = (current_samples - x_0_coeff * reconstructed_samples) / current_sigma
        next_samples = current_samples + dx * dt
        return next_samples, state

class SimplifiedEulerSampler(DiffusionSampler):
    """
    This is for networks with forward diffusion of the form x_{t+1} = x_t + sigma_t * epsilon_t
    """
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        _, current_sigma = self.noise_schedule.get_rates(current_step)
        _, next_sigma = self.noise_schedule.get_rates(next_step)

        dt = next_sigma - current_sigma
        
        dx = (current_samples - reconstructed_samples) / current_sigma
        next_samples = current_samples + dx * dt
        return next_samples, state
    
class HeunSampler(DiffusionSampler):
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        # Get the noise and signal rates for the current and next steps
        current_alpha, current_sigma = self.noise_schedule.get_rates(current_step)
        next_alpha, next_sigma = self.noise_schedule.get_rates(next_step)

        dt = next_sigma - current_sigma
        x_0_coeff = (current_alpha * next_sigma - next_alpha * current_sigma) / dt

        dx_0 = (current_samples - x_0_coeff * reconstructed_samples) / current_sigma
        next_samples_0 = current_samples + dx_0 * dt
        
        # Recompute x_0 and eps at the first estimate to refine the derivative
        estimated_x_0, _, _ = self.sample_model(next_samples_0, next_step, *model_conditioning_inputs)
        
        # Estimate the refined derivative using the midpoint (Heun's method)
        dx_1 = (next_samples_0 - x_0_coeff * estimated_x_0) / next_sigma
        # Compute the final next samples by averaging the initial and refined derivatives
        final_next_samples = current_samples + 0.5 * (dx_0 + dx_1) * dt
        
        return final_next_samples, state

class RK4Sampler(DiffusionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert issubclass(type(self.noise_schedule), GeneralizedNoiseScheduler), "Noise schedule must be a GeneralizedNoiseScheduler"
        @jax.jit
        def get_derivative(x_t, sigma, state:RandomMarkovState, model_conditioning_inputs) -> tuple[jnp.ndarray, RandomMarkovState]:
            t = self.noise_schedule.get_timesteps(sigma)
            x_0, eps, _ = self.sample_model(x_t, t, *model_conditioning_inputs)
            return eps, state
        
        self.get_derivative = get_derivative

    def sample_step(self, current_samples:jnp.ndarray, current_step, model_conditioning_inputs, next_step=None, state:MarkovState=None) -> tuple[jnp.ndarray, MarkovState]:
        step_ones = jnp.ones((current_samples.shape[0], ), dtype=jnp.int32)
        current_step = step_ones * current_step
        next_step = step_ones * next_step
        _, current_sigma = self.noise_schedule.get_rates(current_step)
        _, next_sigma = self.noise_schedule.get_rates(next_step)

        dt = next_sigma - current_sigma

        k1, state = self.get_derivative(current_samples, current_sigma, state, model_conditioning_inputs)
        k2, state = self.get_derivative(current_samples + 0.5 * k1 * dt, current_sigma + 0.5 * dt, state, model_conditioning_inputs)
        k3, state = self.get_derivative(current_samples + 0.5 * k2 * dt, current_sigma + 0.5 * dt, state, model_conditioning_inputs)
        k4, state = self.get_derivative(current_samples + k3 * dt, current_sigma + dt, state, model_conditioning_inputs)

        next_samples = current_samples + (((k1 + 2 * k2 + 2 * k3 + k4) * dt) / 6)
        return next_samples, state

class MultiStepDPM(DiffusionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        # Get the noise and signal rates for the current and next steps
        current_alpha, current_sigma = self.noise_schedule.get_rates(current_step)
        next_alpha, next_sigma = self.noise_schedule.get_rates(next_step)

        dt = next_sigma - current_sigma

        def first_order(current_noise, current_sigma):
            dx = current_noise
            return dx
        
        def second_order(current_noise, current_sigma, last_noise, last_sigma):
            dx_2 = (current_noise - last_noise) / (current_sigma - last_sigma)
            return dx_2
        
        def third_order(current_noise, current_sigma, last_noise, last_sigma, second_last_noise, second_last_sigma):
            dx_2 = second_order(current_noise, current_sigma, last_noise, last_sigma)
            dx_2_last = second_order(last_noise, last_sigma, second_last_noise, second_last_sigma)

            dx_3 = (dx_2 - dx_2_last) / (0.5 * ((current_sigma + last_sigma) - (last_sigma + second_last_sigma)))
            
            return dx_3

        if len(self.history) == 0:
            # First order only
            dx_1 = first_order(pred_noise, current_sigma)
            next_samples = current_samples + dx_1 * dt
        elif len(self.history) == 1:
            # First + Second order
            dx_1 = first_order(pred_noise, current_sigma)
            last_step = self.history[-1]
            dx_2 = second_order(pred_noise, current_sigma, last_step['eps'], last_step['sigma'])
            next_samples = current_samples + dx_1 * dt + 0.5 * dx_2 * dt**2
        else:
            # First + Second + Third order
            last_step = self.history[-1]
            second_last_step = self.history[-2]

            dx_1 = first_order(pred_noise, current_sigma)
            dx_2 = second_order(pred_noise, current_sigma, last_step['eps'], last_step['sigma'])
            dx_3 = third_order(pred_noise, current_sigma, last_step['eps'], last_step['sigma'], second_last_step['eps'], second_last_step['sigma'])
            next_samples = current_samples + (dx_1 * dt) + (0.5 * dx_2 * dt**2) + ((1/6) * dx_3 * dt**3)

        self.history.append({
            "eps": pred_noise,
            "sigma" : current_sigma,
        })
        return next_samples, state
    
class EulerAncestralSampler(DiffusionSampler):
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        current_alpha, current_sigma = self.noise_schedule.get_rates(current_step)
        next_alpha, next_sigma = self.noise_schedule.get_rates(next_step)

        sigma_up = (next_sigma**2 * (current_sigma**2 - next_sigma**2) / current_sigma**2) ** 0.5
        sigma_down = (next_sigma**2 - sigma_up**2) ** 0.5
        
        dt = sigma_down - current_sigma
        
        x_0_coeff = (current_alpha * next_sigma - next_alpha * current_sigma) / (next_sigma - current_sigma)
        dx = (current_samples - x_0_coeff * reconstructed_samples) / current_sigma
        
        state, subkey = state.get_random_key()
        dW = jax.random.normal(subkey, current_samples.shape) * sigma_up
        
        next_samples = current_samples + dx * dt + dW
        return next_samples, state

# %%
images = next(iter(data))
plotImages(images, dpi=300)
print(images.shape)
noise_schedule = karas_ve_schedule
predictor = trainer.model_output_transform

rng = jax.random.PRNGKey(4)
noise = jax.random.normal(rng, shape=images.shape, dtype=jnp.float32)
noise_level = 0.9999
noise_levels = jnp.ones((images.shape[0],), dtype=jnp.int32) * noise_level

rates = noise_schedule.get_rates(noise_levels)
noisy_images, c_in, expected_output = predictor.forward_diffusion(images, noise, rates=rates)
plotImages(noisy_images)
print(jnp.mean(noisy_images), jnp.std(noisy_images))
regenerated_images = noise_schedule.remove_all_noise(noisy_images, noise, noise_levels)
plotImages(regenerated_images)

sampler = EulerSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=16, diffusion_steps=20, start_step=int(noise_level*1000), end_step=0, priors=None)
plotImages(samples, dpi=300)

# %%
textEncoderModel, textTokenizer = defaultTextEncodeModel()

# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily', 
    'a photo of a rose'
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.get_state().ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=2, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)


# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily', 
    'a photo of a rose'
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=2, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)


# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily', 
    'a photo of a rose'
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.best_state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=2, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)


# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily', 
    'a water lily', 
    'a photo of a marigold', 
    'a water lily',
    'a water lily',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=2, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)


# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily', 
    'a water lily', 
    'a photo of a marigold', 
    'a water lily',
    'a water lily',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=2, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)


# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily', 
    'a water lily', 
    'a photo of a marigold', 
    'a water lily',
    'a water lily',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=2, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)

# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily', 
    'a water lily', 
    'a photo of a marigold', 
    'a water lily',
    'a water lily',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=4, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=500, fig_size=(4, 5))

# %%
prompts = [
    'water tulip',
    'a green water rose',
    'a green water rose',
    'a green water rose',
    'a water red rose', 
    'a marigold and rose hybrid', 
    'a marigold and rose hybrid', 
    'a marigold and rose hybrid', 
    'a water lily and a marigold',
    'a water lily and a marigold',
    'a water lily and a marigold',
    'a water lily and a marigold',
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=3, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)

# %%
prompts = [
    'water tulip',
    'a water lily',
    'a water lily',
    'a photo of a rose',
    'a photo of a rose',
    'a water lily', 
    'a water lily', 
    'a photo of a marigold', 
    'a photo of a marigold', 
    'a photo of a marigold', 
    'a water lily',
    'a photo of a sunflower',
    'a photo of a lotus',
    "columbine",
    "columbine",
    "an orchid",
    "an orchid",
    "an orchid",
    'a water lily', 
    'a water lily', 
    'a water lily', 
    "columbine",
    "columbine",
    'a photo of a sunflower',
    'a photo of a sunflower',
    'a photo of a sunflower',
    'a photo of a lotus',
    'a photo of a lotus',
    'a photo of a marigold', 
    'a photo of a marigold', 
    'a photo of a rose',
    'a photo of a rose',
    'a photo of a rose',
    "orange dahlia",
    "orange dahlia",
    "a lenten rose",
    "a lenten rose",
    'a water lily', 
    'a water lily', 
    'a water lily', 
    'a water lily', 
    "an orchid",
    "an orchid",
    "an orchid",
    'hard-leaved pocket orchid',
    "bird of paradise",
    "bird of paradise",
    "a photo of a lovely rose",
    "a photo of a lovely rose",
    "a photo of a globe-flower",
    "a photo of a globe-flower",
    "a photo of a lovely rose",
    "a photo of a lovely rose",
    "a photo of a ruby-lipped cattleya",
    "a photo of a ruby-lipped cattleya",
    "a photo of a lovely rose",
    'a water lily', 
    'a osteospermum', 
    'a osteospermum', 
    'a water lily', 
    'a water lily', 
    'a water lily', 
    "a red rose",
    "a red rose",
    ]
pooled_labels, labels_seq = encodePrompts(prompts, textEncoderModel, textTokenizer)

sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform, guidance_scale=4, null_labels_seq=data['null_labels_full'])
samples = sampler.generate_images(num_images=len(prompts), diffusion_steps=200, start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(labels_seq,))
plotImages(samples, dpi=300)

# %%
dataToLabelGenMap["oxford_flowers102"]()

# %%


# %%
sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=64, diffusion_steps=200, start_step=1000, end_step=0, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=64, diffusion_steps=200, start_step=1000, end_step=0, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = RK4Sampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=64, diffusion_steps=6, start_step=1000, end_step=0, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = EulerAncestralSampler(trainer.model, trainer.state.ema_params, karas_ve_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=64, diffusion_steps=300, start_step=1000, end_step=0, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = DDPMSampler(trainer.model, trainer.state.params, trainer.noise_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=16, start_step=1000, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = DDPMSampler(trainer.model, trainer.best_state.params, trainer.noise_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=16, start_step=1000, priors=None)
plotImages(samples, dpi=300)

# %%
sampler = DDPMSampler(trainer.model, trainer.best_state.params, trainer.noise_schedule, model_output_transform=trainer.model_output_transform)
samples = sampler.generate_images(num_images=64, start_step=1000, priors=None)
plotImages(samples)

# %% [markdown]
# 


