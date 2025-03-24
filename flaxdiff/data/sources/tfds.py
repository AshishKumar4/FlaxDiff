import cv2
import jax.numpy as jnp
import grain.python as pygrain
from flaxdiff.utils import AutoTextTokenizer
from typing import Dict

# -----------------------------------------------------------------------------------------------#
# Oxford flowers and other TFDS datasources -----------------------------------------------------#
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
            self.tokenize = AutoTextTokenizer(tensor_type="np")

        def map(self, element) -> Dict[str, jnp.array]:
            image = element['image']
            image = cv2.resize(image, (image_scale, image_scale),
                            interpolation=interpolation)
            # image = (image - 127.5) / 127.5
            caption = labelizer(element)
            results = self.tokenize(caption)
            return {
                "image": image,
                "input_ids": results['input_ids'][0],
                "attention_mask": results['attention_mask'][0],
            }
    return augmenters