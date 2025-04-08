import cv2
import jax.numpy as jnp
import grain.python as pygrain
from flaxdiff.utils import AutoTextTokenizer
from typing import Dict
import random
import augmax
import jax

# -----------------------------------------------------------------------------------------------#
# Oxford flowers and other TFDS datasources -----------------------------------------------------#
# -----------------------------------------------------------------------------------------------#

PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a photo of a {} flower",
    "This is a photo of a {}",
    "This is a photo of a {} flower",
    "A photo of a {} flower",
]

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
        raw = textlabels[int(sample['label'])]
        # randomly select a prompt template
        template = random.choice(PROMPT_TEMPLATES)
        # format the template with the label
        caption = template.format(raw)
        # return the caption
        return caption
    return load_labels

def tfds_augmenters(image_scale, method):
    labelizer = labelizer_oxford_flowers102("/home/mrwhite0racle/tensorflow_datasets/oxford_flowers102/2.1.1/label.labels.txt")
    if image_scale > 256:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
        
    augments = augmax.Chain(
        augmax.HorizontalFlip(0.5),
        augmax.RandomContrast((-0.05, 0.05), 1.),
        augmax.RandomBrightness((-0.2, 0.2), 1.)
    )

    augments = jax.jit(augments, backend="cpu")
        
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenize = AutoTextTokenizer(tensor_type="np")

        def map(self, element) -> Dict[str, jnp.array]:
            image = element['image']
            image = cv2.resize(image, (image_scale, image_scale),
                            interpolation=interpolation)
            # image = augments(image)
            # image = (image - 127.5) / 127.5
            caption = labelizer(element)
            results = self.tokenize(caption)
            return {
                "image": image,
                "input_ids": results['input_ids'][0],
                "attention_mask": results['attention_mask'][0],
            }
    return augmenters