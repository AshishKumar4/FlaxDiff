# ![](images/logo.jpeg "FlaxDiff")

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

## A Versatile and simple Diffusion Library

In recent years, diffusion and score-based multi-step models have revolutionized the generative AI domain. However, the latest research in this field has become highly math-intensive, making it challenging to understand how state-of-the-art diffusion models work and generate such impressive images. Replicating this research in code can be daunting.

FlaxDiff is a library of tools (schedulers, samplers, models, etc.) designed and implemented in an easy-to-understand way. The focus is on understandability and readability over performance. I started this project as a hobby to familiarize myself with Flax and Jax and to learn about diffusion and the latest research in generative AI.

I initially started this project in Keras, being familiar with TensorFlow 2.0, but transitioned to Flax, powered by Jax, for its performance and ease of use. The old notebooks and models, including my first Flax models, are also provided.

The library has since grown beyond images: it now also covers video diffusion, flow matching, latent diffusion with a VAE, and I-JEPA/V-JEPA self-supervised training, all running through the same trainer with data-parallel and FSDP sharding. The bigger text-conditional models in the gallery were trained on a TPU-v4-32 pod.

## Example Notebooks from scratch

In the `tutorial notebooks` folder, you will find comprehensive notebooks for various diffusion techniques, written entirely from scratch and are independent of the FlaxDiff library. Each notebook includes detailed explanations of the underlying mathematics and concepts, making them invaluable resources for learning and understanding diffusion models.

### Available Notebooks and Resources

- **[Diffusion explained (nbviewer link)](https://nbviewer.org/github/AshishKumar4/FlaxDiff/blob/main/tutorial%20notebooks/simple%20diffusion%20flax.ipynb) [(local link)](tutorial%20notebooks/simple%20diffusion%20flax.ipynb)**

  - An in-depth exploration of the concept of Diffusion based generative models, DDPM (Denoising Diffusion Probabilistic Models), DDIM (Denoising Diffusion Implicit Models), and the SDE/ODE generalizations of diffusion, with step-by-step explainations and code.

  <a target="_blank" href="https://colab.research.google.com/github/AshishKumar4/FlaxDiff/blob/main/tutorial%20notebooks/simple%20diffusion%20flax.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **[EDM (Elucidating the Design Space of Diffusion-based Generative Models)](tutorial%20notebooks/edm%20tutorial.ipynb)**
  - **TODO** A thorough guide to EDM, discussing the innovative approaches and techniques used in this advanced diffusion model.

  <a target="_blank" href="https://colab.research.google.com/github/AshishKumar4/FlaxDiff/blob/main/tutorial%20notebooks/edm%20tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Other resources

- **[Multi-host Data parallel training script in JAX](./training.py)**
  - Training script for multi-host data parallel training in JAX, to serve as a reference for training large models on multiple GPUs/TPUs across multiple hosts.

- **[TPU utilities for making life easier](./tpu-tools/)**
  - A collection of utilities and scripts to make working with TPUs easier, such as cli to create/start/stop/setup TPUs, script to setup TPU VMs (install everything you need), mounting gcs datasets etc.

## Features

### Schedulers
Implemented in `flaxdiff.schedulers`:
- **LinearNoiseScheduler** (`flaxdiff.schedulers.LinearNoiseScheduler`): A beta-parameterized discrete scheduler.
- **CosineNoiseScheduler** (`flaxdiff.schedulers.CosineNoiseScheduler`): A beta-parameterized discrete scheduler.
- **ExpNoiseScheduler** (`flaxdiff.schedulers.ExpNoiseScheduler`): A beta-parameterized discrete scheduler.
- **CosineContinuousNoiseScheduler** (`flaxdiff.schedulers.CosineContinuousNoiseScheduler`): A continuous scheduler.
- **CosineGeneralNoiseScheduler** (`flaxdiff.schedulers.CosineGeneralNoiseScheduler`): A continuous sigma parameterized cosine scheduler.
- **SqrtContinuousNoiseScheduler** (`flaxdiff.schedulers.SqrtContinuousNoiseScheduler`): A continuous scheduler using the sqrt schedule proposed for diffusion language models.
- **KarrasVENoiseScheduler** (`flaxdiff.schedulers.KarrasVENoiseScheduler`): A sigma-parameterized continuous scheduler proposed by Karras et al. 2022, best suited for inference.
- **EDMNoiseScheduler** (`flaxdiff.schedulers.EDMNoiseScheduler`): A sigma-parameterized continuous scheduler based on the EDM paper, best suited for training with the KarrasVENoiseScheduler.
- **FlowMatchingScheduler** (`flaxdiff.schedulers.FlowMatchingScheduler`): A rectified-flow scheduler with logit-normal timestep sampling and resolution-dependent shifting, as used in Stable Diffusion 3.

### Model Prediction Transforms
Implemented in `flaxdiff.predictors`:
- **EpsilonPredictionTransform** (`flaxdiff.predictors.EpsilonPredictionTransform`): The model predicts the noise in the data.
- **DirectPredictionTransform** (`flaxdiff.predictors.DirectPredictionTransform`): The model predicts the original data from the noisy data.
- **VPredictionTransform** (`flaxdiff.predictors.VPredictionTransform`): The model predicts a linear combination of the data and noise.
- **FlowMatchPredictionTransform** (`flaxdiff.predictors.FlowMatchPredictionTransform`): The model predicts the flow velocity.
- **KarrasPredictionTransform** (`flaxdiff.predictors.KarrasPredictionTransform`): A generalized transform for the EDM, integrating various parameterizations.
- **get_diffusion_preset** (`flaxdiff.predictors.get_diffusion_preset`): One call which pairs a training schedule, a sampling schedule and a transform for the `"edm"`, `"karras"`, `"cosine"` and `"flow"` setups.

### Samplers
Implemented in `flaxdiff.samplers`:
- **DDPMSampler** (`flaxdiff.samplers.DDPMSampler`): Implements the Denoising Diffusion Probabilistic Model (DDPM) sampling process.
- **DDIMSampler** (`flaxdiff.samplers.DDIMSampler`): Implements the Denoising Diffusion Implicit Model (DDIM) sampling process.
- **EulerSampler** (`flaxdiff.samplers.EulerSampler`): An ODE solver sampler using Euler's method.
- **EulerAncestralSampler** (`flaxdiff.samplers.EulerAncestralSampler`): Euler sampling with ancestral noise injection.
- **HeunSampler** (`flaxdiff.samplers.HeunSampler`): An ODE solver sampler using Heun's method.
- **RK4Sampler** (`flaxdiff.samplers.RK4Sampler`): An ODE solver sampler using the Runge-Kutta method.
- **MultiStepDPM** (`flaxdiff.samplers.MultiStepDPM`): Implements a multi-step sampling method inspired by the Multistep DPM solver as presented here: [tonyduan/diffusion](https://github.com/tonyduan/diffusion/blob/fcc0ed829baf29e1493b460b073e735a848c08ea/src/samplers.py#L44)

All samplers support classifier-free guidance, including interval-limited guidance.

### Training
Implemented in `flaxdiff.trainer`:
- **GeneralDiffusionTrainer** (`flaxdiff.trainer.GeneralDiffusionTrainer`): Manages the training loop, loss calculation, EMA, gradient accumulation, checkpointing and wandb logging, for both image and video data. It runs data-parallel and FSDP sharded training through `jax.jit` with `NamedSharding` on a `(data, fsdp)` mesh.
- **Objectives** (`flaxdiff.trainer.objectives`): What to optimize is pluggable. `DiffusionObjective` is the default; `JepaObjective` (`flaxdiff.jepa`) trains I-JEPA/V-JEPA encoders on the same trainer.

### Models
Implemented in `flaxdiff.models` and constructed via `flaxdiff.models.registry.build_model`:
- **Unet**: A classic convolutional UNet.
- **UNet3D**: A video UNet which can inflate 2D Unet checkpoints.
- **UViT / SimpleUDiT**: U-shaped transformers.
- **SimpleDiT / SimpleMMDiT / HierarchicalMMDiT**: DiT and SD3-style multi-modal DiT variants.
- **HybridSSMAttentionDiT**: Interleaves S5 state-space blocks with attention.
- **VideoDiT**: A factorized spatial-temporal DiT for video.
- Hilbert and zigzag patch scan orders are available via the `+hilbert` and `+zigzag` architecture suffixes.
- **Autoencoders** (`flaxdiff.models.autoencoder`): `StableDiffusionVAE` (vendored Flax VAE, loads Hugging Face hub weights) and `SimpleAutoEncoder` for latent diffusion without any external weights.

### Metrics
Implemented in `flaxdiff.metrics`: FID (vendored InceptionV3), CLIP score, PSNR, SSIM, and linear/kNN probes for JEPA.

## Installation

To install FlaxDiff, you need to have Python 3.11 or higher:

```bash
pip install flaxdiff
```

Optional extras pull in the heavier dependencies only when you need them:

- `flaxdiff[av]`: video/audio sources and readers (OpenCV, decord, moviepy, PyAV)
- `flaxdiff[metrics]`: FID (scipy) and Inception weight download
- `flaxdiff[streaming]`: online URL-streaming loader (Hugging Face `datasets`)
- `flaxdiff[tfds]`: TFDS-backed dataset sources

Or for development, clone the repo and install in editable mode with the test dependencies:

```bash
pip install -e .[test]
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

The test suite covers model forward passes for every architecture, scheduler and transform invariants, sampler convergence against an analytic denoiser, trainer smoke runs for images and videos, FSDP and data-parallel parity on a simulated 8-device mesh, sharded checkpoint round-trips with mid-epoch data resume, and the JEPA objectives. Tests marked `network` download pretrained weights and are excluded by default.

## Getting Started

### Training Example

Here is a simplified example to get you started with training a diffusion model using FlaxDiff:

```python
from datetime import datetime
import jax, optax

from flaxdiff.data.dataloaders import get_dataset_grain
from flaxdiff.inputs import DiffusionInputConfig, ConditionalInputConfig
from flaxdiff.inputs.encoders import CLIPTextEncoder
from flaxdiff.models.registry import build_model
from flaxdiff.predictors import get_diffusion_preset
from flaxdiff.trainer import GeneralDiffusionTrainer
from flaxdiff.samplers.euler import EulerAncestralSampler

BATCH_SIZE, IMAGE_SIZE = 16, 128

data = get_dataset_grain("oxford_flowers102", batch_size=BATCH_SIZE, image_scale=IMAGE_SIZE)

text_encoder = CLIPTextEncoder.from_modelname("openai/clip-vit-large-patch14")
input_config = DiffusionInputConfig(
    sample_data_key="image",
    sample_data_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    conditions=[ConditionalInputConfig(encoder=text_encoder)],
)

train_schedule, sample_schedule, transform = get_diffusion_preset("edm")

model = build_model("simple_dit", dict(
    emb_features=512, num_layers=8, num_heads=8, patch_size=8,
))

trainer = GeneralDiffusionTrainer(
    model=model,
    optimizer=optax.adamw(2e-4),
    input_config=input_config,
    noise_schedule=train_schedule,
    model_output_transform=transform,
    rngs=jax.random.PRNGKey(4),
    name=f"flowers-edm-{datetime.now():%Y-%m-%d_%H%M}",
    distributed_training=True,
    checkpoint_base_path="./checkpoints",
)

trainer.fit(
    data,
    training_steps_per_epoch=data["train_len"] // BATCH_SIZE,
    epochs=100,
    sampler_class=EulerAncestralSampler,
    sampling_noise_schedule=sample_schedule,
)
```

The full-featured entry points are [`training.py`](./training.py) for diffusion and [`training_jepa.py`](./training_jepa.py) for I-JEPA/V-JEPA.

### Inference Example

Here is a simplified example for generating images using a trained model:

```python
from flaxdiff.inference.pipeline import DiffusionInferencePipeline

pipeline = DiffusionInferencePipeline.from_wandb_registry(
    modelname="diffusion-model", project="my-project",
)
images = pipeline.generate_samples(
    num_samples=16, resolution=128, diffusion_steps=50,
    guidance_scale=3.0, conditioning_data=["a water lily", "a rose"],
)
```

## Disclaimer (and About Me)

I worked as a Machine Learning Researcher at Hyperverge from 2019-2021, focusing on computer vision, specifically facial anti-spoofing and facial detection & recognition. Since switching to my current job in 2021, I haven't engaged in as much R&D work, leading me to start this pet project to revisit and relearn the fundamentals and get familiar with the state-of-the-art. My current role involves primarily Golang system engineering with some applied ML work just sprinkled in. Therefore, the code may reflect my learning journey. Please forgive any mistakes and do open an issue to let me know.

## References and Acknowledgements

### Research papers and preprints
- The Original Denoising Diffusion Probabilistic Models (DDPM) [paper](https://arxiv.org/abs/2006.11239)
- Denoising Diffusion Implicit Models (DDIM) [paper](https://arxiv.org/abs/2010.02502)
- Improved Denoising Diffusion Probabilistic Models [paper](https://arxiv.org/abs/2102.09672)
- Diffusion Models beat GANs on image synthesis [paper](https://arxiv.org/pdf/2105.05233)
- Score-Based Generative Modeling through Stochastic Differential Equations [paper](https://arxiv.org/pdf/2011.13456)
- Elucidating the design space of Diffusion-based generative models (EDM) [paper](https://arxiv.org/abs/2206.00364)
- Perception Prioritized Training of Diffusion Models (P2 Weighting) [paper](https://arxiv.org/abs/2204.00227)
- Pseudo Numerical Methods for Diffusion Models on Manifolds (PNMDM) [paper](https://arxiv.org/abs/2202.09778)
- The DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps [paper](https://arxiv.org/pdf/2206.00927)
- Scalable Diffusion Models with Transformers (DiT) [paper](https://arxiv.org/abs/2212.09748)
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3) [paper](https://arxiv.org/abs/2403.03206)
- Flow Matching for Generative Modeling [paper](https://arxiv.org/abs/2210.02747)
- Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA) [paper](https://arxiv.org/abs/2301.08243)
- Simplified State Space Layers for Sequence Modeling (S5) [paper](https://arxiv.org/abs/2208.04933)
- Applying Guidance in a Limited Interval Improves Sample and Distribution Quality (interval-limited CFG) [paper](https://arxiv.org/abs/2404.07724)
- Diffusion-LM Improves Controllable Text Generation (sqrt schedule) [paper](https://arxiv.org/abs/2205.14217)

### Useful blogs and codebases

- An incredible series of blogs on various diffusion related topics by [Sander Dieleman](https://sander.ai/posts/). The posts particularly on [diffusion models](https://sander.ai/2022/01/31/diffusion.html), [Typicality](https://sander.ai/2020/09/01/typicality.html), [Geometry of Diffusion Guidance](https://sander.ai/2023/08/28/geometry.html#warning) and [Noise Schedules](https://sander.ai/2024/06/14/noise-schedules.html) are a must read
- An awesome blog series by Tony Duan on [Diffusion models from scratch](https://www.tonyduan.com/diffusion/index.html). Although it trains models for MNIST and the implementations are a bit basic, the maths is explained in a very nice way. The codebase is [here](https://github.com/tonyduan/diffusion)
- The [k-diffusion](https://github.com/crowsonkb/k-diffusion/) codebase by Katherine Crowson, which hosts an exhaustive implementation of the EDM paper (Karras et al) along with the DPM-Solver, DPM-Solver++ (both 2S and 2M) in pytorch. Most other diffusion libraries borrow from this.
- The [Official EDM implementation](https://github.com/NVlabs/edm) by Tero Karras, in pytorch. Really neat code and the reference implementation for all the karras based samplers/schedules.
- The [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers). The vendored Flax VAE and parts of the attention module derive from it (Apache-2.0, attribution headers preserved).
- [jax-fid](https://github.com/matthias-wright/jax-fid), the origin of the vendored InceptionV3 used for FID.
- The [Keras DDPM Tutorial](https://keras.io/examples/generative/ddpm/) by A_K Nain, and the [Keras DDIM implementation](https://keras.io/examples/generative/ddim/) by András Béres, which are great starting points for beginners to understand the basics of diffusion models. I started my journey by trying to implement the concepts introduced in these tutorials from scratch.

## Pending things to do list

- **Multi-host validation of the revamped trainer on an actual TPU pod**
- **A proper precision policy (dtype/param_dtype are still threaded ad-hoc)**
- **Full FID-50k evaluation (the current FID metric is per-validation-batch)**
- **Autoregressive LM and diffusion-LM objectives on the same trainer**

## Gallery

### Images generated by Euler Ancestral Sampler in 200 Steps [text2image with CFG]
Model trained on Laion-Aesthetics 12M + CC12M + MS COCO + 1M aesthetic 6+ subset of COYO-700M on TPU-v4-32:
`a beautiful landscape with a river with mountains, a beautiful landscape with a river with mountains, ...`

**Params**:
`Dataset: Laion-Aesthetics 12M + CC12M + MS COCO + 1M aesthetic 6+ subset of COYO-700M`
`Batch size: 256`
`Image Size: 128`
`Training Epochs: 5`
`Steps per epoch: 74573`
`Model Configurations: feature_depths=[128, 256, 512, 1024]`

`Training Noise Schedule: EDMNoiseScheduler`
`Inference Noise Schedule: KarrasVENoiseScheduler`

![EulerA with CFG](images/medium_epoch5.png)

### Images generated by Euler Ancestral Sampler in 200 Steps [text2image with CFG]
Images generated by the following prompts using classifier free guidance with guidance factor = 2:
`'water tulip, a water lily, a water lily, a water lily, a photo of a marigold, a water lily, a water lily, a photo of a lotus, a photo of a lotus, a photo of a lotus, a photo of a rose, a photo of a rose, a photo of a rose, a photo of a rose, a photo of a rose'`

**Params**:
`Dataset: oxford_flowers102`
`Batch size: 16`
`Image Size: 128`
`Training Epochs: 1000`
`Steps per epoch: 511`

`Training Noise Schedule: EDMNoiseScheduler`
`Inference Noise Schedule: KarrasVENoiseScheduler`

![EulerA with CFG](images/text2img%20euler%20ancestral%201.png)

### Images generated by DDPM Sampler in 1000 steps [Unconditional]

**Params**:
`Dataset: oxford_flowers102`
`Batch size: 16`
`Image Size: 64`
`Training Epochs: 1000`
`Steps per epoch: 511`

`Training Noise Schedule: CosineNoiseScheduler`
`Inference Noise Schedule: CosineNoiseScheduler`

`Model: UNet(emb_features=256,
            feature_depths=[64, 128, 256, 512],
            attention_configs=[{"heads":4}, {"heads":4}, {"heads":4}, {"heads":4}, {"heads":4}],
            num_res_blocks=2,
            num_middle_res_blocks=1)`

![DDPM Sampler results](images/ddpm2.png)

### Images generated by Heun Sampler in 10 steps (20 model inferences as Heun takes 2x inference steps) [Unconditional]

**Params**:
`Dataset: oxford_flowers102`
`Batch size: 16`
`Image Size: 64`
`Training Epochs: 1000`
`Steps per epoch: 511`

`Training Noise Schedule: EDMNoiseScheduler`
`Inference Noise Schedule: KarrasVENoiseScheduler`

![Heun Sampler results](images/heun.png)

## Contribution

Feel free to contribute by opening issues or submitting pull requests. Let's make FlaxDiff better together!

## License

This project is licensed under the MIT License.
