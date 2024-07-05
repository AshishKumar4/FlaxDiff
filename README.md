# FlaxDiff

## A versatile and easy to understand Diffusion library

In recent years, Diffusion and Score based multi step models have taken over the generative AI domain. However, the latest research in this field have been very math intensive. Its not very easy to understand how exactly some of the current state-of-the-art diffusion models work and generate such awesome images. Replicating the research in code can be at times quite daunting. 

FlaxDiff a library of tools (schedulers, samplers, models, etc) designed and implemented in a very easy to understand way. The moto is understandability and readibility over performance. I started this project as a hobby to get familiar with Flax and Jax and learn about diffusion and the latest research in generative AI.

I originally started this project in keras, as I have been intimately familiar with tensorlflow 2.0 in the past and used to love it, but as it turns out its pretty ancient now, and so I reimplmeneted everything in the newer framework Flax, powered by Jax due to its performance and ease of use. But the old notebooks and even old models with one of my first flax models is also provided.

The `Diffusion flax linen.ipynb` notebook is my main workspace where I do all my latest experiements. Few checkpoints I feel good enough are uploaded to the `pretrained` folder along with a copy of my working notebook associated with that checkpoint. *You may need to copy that notebook over to the working root for it to work.*

## Disclaimer (and about me)
I used to work as a Machine Learning Researcher at Hyperverge around 2019-2021 in the domain of computer vision, specifically facial anti spoofind and facial detection & recognition. Since switching to my latest job in 2021, I haven't been able to get the kind of R&D work to keep my skills and expertise sharpened. Therefore, I started this pet project to revisit and relearn the fundamentals and get familiar with the state of the art. Most of my full time work and experience now has been as a golang system engineer with some applied ML work (not R&D/desiging/training) sprinkled in. Therefore, the code may reflect the fact that I am a developer trying to tackle this vast maths heavy domain, and I may not end up doing a good job. Certain implementation may be off or outright wrong. Please forgive me for those mistakes, and do open up an issue to let me know.

## 