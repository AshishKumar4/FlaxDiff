# ![](images/logo.jpeg "FlaxDiff")

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

## A JAX/Flax library for training diffusion models from scratch

FlaxDiff is a library of schedulers, prediction transforms, samplers, model
architectures, data pipelines and a distributed trainer, written in JAX/Flax
(Linen). The code aims to stay readable: every technique is implemented plainly
and tested against its paper's invariants.

It trains image and video diffusion models (UNet, DiT, MMDiT, UViT, hybrid
SSM-attention, factorized video DiT), flow-matching models, and I-JEPA/V-JEPA
self-supervised encoders — all through one trainer and one `Objective` seam.
Data-parallel and FSDP sharding run through `jax.jit` + `NamedSharding` on a
`(data, fsdp)` mesh and work from a single host up to multi-host TPU pods.

## Installation

Python 3.11+:

```bash
pip install flaxdiff
```

Optional extras:

| Extra | Enables |
|---|---|
| `flaxdiff[av]` | video/audio sources and readers (OpenCV, decord, moviepy, PyAV) |
| `flaxdiff[metrics]` | FID (scipy) and Inception weight download |
| `flaxdiff[streaming]` | online URL-streaming loader (HF `datasets`) |
| `flaxdiff[tfds]` | TFDS-backed dataset sources |
| `flaxdiff[test]` | pytest |

Development:

```bash
pip install -e .[test]
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

## The pieces

- **Schedulers** (`flaxdiff.schedulers`): `LinearNoiseScheduler`,
  `CosineNoiseScheduler`, `ExpNoiseScheduler`, `CosineGeneralNoiseScheduler`,
  `SqrtContinuousNoiseScheduler`, `KarrasVENoiseScheduler`, `EDMNoiseScheduler`
  (EDM2 defaults), `FlowMatchingScheduler` (SD3 logit-normal timesteps,
  resolution shift).
- **Prediction transforms** (`flaxdiff.predictors`): `EpsilonPredictionTransform`,
  `DirectPredictionTransform` (x0), `VPredictionTransform`,
  `FlowMatchPredictionTransform`, `KarrasPredictionTransform`, and
  `get_diffusion_preset(name)` — one call pairing a training schedule, a
  sampling schedule and a transform for `"edm"`, `"karras"`, `"cosine"` and
  `"flow"`.
- **Samplers** (`flaxdiff.samplers`): `DDPMSampler`, `DDIMSampler`,
  `EulerSampler`, `EulerAncestralSampler`, `SimplifiedEulerSampler`,
  `HeunSampler`, `RK4Sampler`, `MultiStepDPM` — all with classifier-free
  guidance, including interval-limited guidance.
- **Models** (`flaxdiff.models`, built via `flaxdiff.models.registry.build_model`):
  `unet`, `unet_3d` (with 2D→3D checkpoint inflation), `uvit`, `simple_dit`,
  `simple_udit`, `simple_mmdit`, `hierarchical_mmdit`, `hybrid_dit` (S5 SSM +
  attention), `video_dit`, plus JEPA encoders/predictor. Hilbert and zigzag
  patch scan orders are supported via `+hilbert` / `+zigzag` suffixes.
- **Autoencoders** (`flaxdiff.models.autoencoder`): `StableDiffusionVAE`
  (vendored Flax VAE, HF hub weights) and `SimpleAutoEncoder` for latent
  diffusion without external weights.
- **Trainer** (`flaxdiff.trainer`): `GeneralDiffusionTrainer` on top of
  `SimpleTrainer` — data-parallel + FSDP sharding, mixed precision, EMA,
  gradient accumulation, async Orbax checkpointing with exact data-iterator
  resume, wandb logging and registry publishing. Objectives:
  `DiffusionObjective` and `JepaObjective` (`flaxdiff.jepa`).
- **Data** (`flaxdiff.data`): Grain pipelines over TFDS/GCS/local sources for
  images and audio-video, plus an online URL-streaming loader.
- **Metrics** (`flaxdiff.metrics`): FID (vendored InceptionV3), CLIP score,
  PSNR, SSIM, and JEPA linear/kNN probes.
- **Inference** (`flaxdiff.inference`): `DiffusionInferencePipeline` — load a
  local checkpoint or a wandb run/registry model and generate.

## Training example

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

The full-featured entry points are [`training.py`](./training.py) (diffusion,
all architectures/datasets/parallelism flags) and
[`training_jepa.py`](./training_jepa.py) (I-JEPA/V-JEPA on the same trainer).

## Inference example

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

## Tutorials

Notebooks written from scratch, independent of the library, with the maths
explained step by step:

- **[Diffusion explained (nbviewer)](https://nbviewer.org/github/AshishKumar4/FlaxDiff/blob/main/tutorial%20notebooks/simple%20diffusion%20flax.ipynb)
  [(local)](tutorial%20notebooks/simple%20diffusion%20flax.ipynb)** — DDPM,
  DDIM, and the SDE/ODE view of diffusion.
  <a target="_blank" href="https://colab.research.google.com/github/AshishKumar4/FlaxDiff/blob/main/tutorial%20notebooks/simple%20diffusion%20flax.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- **[EDM tutorial](tutorial%20notebooks/edm%20tutorial.ipynb)** — work in
  progress.

Other useful pieces:

- **[Multi-host data-parallel training script](./training.py)** — reference for
  multi-host JAX training on TPU pods.
- **[TPU utilities](./tpu-tools/)** — CLI to create/setup/manage TPU VMs and
  pods, mount GCS datasets, and launch multi-host runs.

## Testing

```bash
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

The suite covers model forwards for every architecture, scheduler/transform
invariants, sampler convergence against an analytic denoiser, flow matching,
trainer smoke runs (image + video), FSDP/data-parallel parity and sharding on a
simulated 8-device mesh, sharded checkpoint round-trips with mid-epoch data
resume, JEPA objectives under FSDP, and metrics. Tests marked `network`
download pretrained weights and are excluded by default.

## About

I worked as a Machine Learning Researcher at Hyperverge from 2019-2021,
focusing on computer vision. I started this project to relearn the fundamentals
and keep up with the state of the art. The code reflects that journey — if you
find mistakes, please open an issue.

## References and Acknowledgements

### Research papers and preprints
- Denoising Diffusion Probabilistic Models (DDPM) [paper](https://arxiv.org/abs/2006.11239)
- Denoising Diffusion Implicit Models (DDIM) [paper](https://arxiv.org/abs/2010.02502)
- Improved Denoising Diffusion Probabilistic Models [paper](https://arxiv.org/abs/2102.09672)
- Diffusion Models beat GANs on image synthesis [paper](https://arxiv.org/pdf/2105.05233)
- Score-Based Generative Modeling through Stochastic Differential Equations [paper](https://arxiv.org/pdf/2011.13456)
- Elucidating the design space of Diffusion-based generative models (EDM) [paper](https://arxiv.org/abs/2206.00364)
- Perception Prioritized Training of Diffusion Models (P2 weighting) [paper](https://arxiv.org/abs/2204.00227)
- Pseudo Numerical Methods for Diffusion Models on Manifolds [paper](https://arxiv.org/abs/2202.09778)
- DPM-Solver [paper](https://arxiv.org/pdf/2206.00927)
- Scalable Diffusion Models with Transformers (DiT) [paper](https://arxiv.org/abs/2212.09748)
- Scaling Rectified Flow Transformers (SD3 / MMDiT) [paper](https://arxiv.org/abs/2403.03206)
- Flow Matching for Generative Modeling [paper](https://arxiv.org/abs/2210.02747)
- Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA) [paper](https://arxiv.org/abs/2301.08243)
- Simplified State Space Layers for Sequence Modeling (S5) [paper](https://arxiv.org/abs/2208.04933)
- Applying Guidance in a Limited Interval Improves Sample and Distribution Quality (interval-limited CFG) [paper](https://arxiv.org/abs/2404.07724)
- Diffusion-LM Improves Controllable Text Generation (sqrt schedule) [paper](https://arxiv.org/abs/2205.14217)

### Useful blogs and codebases
- [Sander Dieleman's blog](https://sander.ai/posts/) — particularly the posts on [diffusion models](https://sander.ai/2022/01/31/diffusion.html), [typicality](https://sander.ai/2020/09/01/typicality.html), [the geometry of diffusion guidance](https://sander.ai/2023/08/28/geometry.html) and [noise schedules](https://sander.ai/2024/06/14/noise-schedules.html).
- Tony Duan's [Diffusion models from scratch](https://www.tonyduan.com/diffusion/index.html) and its [codebase](https://github.com/tonyduan/diffusion).
- Katherine Crowson's [k-diffusion](https://github.com/crowsonkb/k-diffusion/) — the reference for EDM-style samplers in PyTorch.
- The [official EDM implementation](https://github.com/NVlabs/edm) by Tero Karras.
- The [Hugging Face Diffusers library](https://github.com/huggingface/diffusers) — the vendored Flax VAE and parts of the attention module derive from it (Apache-2.0, headers preserved).
- [jax-fid](https://github.com/matthias-wright/jax-fid) — origin of the vendored InceptionV3.
- The [Keras DDPM](https://keras.io/examples/generative/ddpm/) and [DDIM](https://keras.io/examples/generative/ddim/) tutorials, where this journey started.

## Gallery

### Text-to-image with CFG, Euler Ancestral, 200 steps
Model trained on LAION-Aesthetics 12M + CC12M + MS COCO + 1M aesthetic 6+ subset of COYO-700M on a TPU-v4-32:
`a beautiful landscape with a river with mountains …`

**Params:** `feature_depths=[128, 256, 512, 1024]`, batch 256, image 128, 5 epochs, 74573 steps/epoch, EDM schedule.

![EulerA with CFG](images/medium_epoch5.png)

### Text-to-image with CFG (guidance 2), oxford_flowers102
`water tulip, a water lily, …`

![EulerA with CFG](images/text2img%20euler%20ancestral%201.png)

### Text-to-image with CFG (guidance 4), oxford_flowers102
`water tulip, a water lily, a photo of a rose, …`

![EulerA with CFG](images/text2img%20euler%20ancestral%202.png)

### Unconditional, DDPM sampler, 1000 steps, oxford_flowers102 (64px)
Cosine schedule, `UNet(emb_features=256, feature_depths=[64, 128, 256, 512])`.

![DDPM Sampler results](images/ddpm2.png)

### Unconditional, Heun sampler, 10 steps (20 model evals), oxford_flowers102 (64px)
EDM schedule.

![Heun Sampler results](images/heun.png)

## Contribution

Feel free to contribute by opening issues or submitting pull requests.

## License

This project is licensed under the MIT License.
