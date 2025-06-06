import cv2
import jax.numpy as jnp
import grain.python as pygrain
from flaxdiff.utils import AutoTextTokenizer
from typing import Dict, Any, Callable, List, Optional
import random
import augmax
import jax
import os
import struct as st
from functools import partial
import numpy as np
from .base import DataSource, DataAugmenter
import traceback

# ----------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------

def unpack_dict_of_byte_arrays(packed_data):
    """Unpacks a dictionary of byte arrays from a packed binary format."""
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


# ----------------------------------------------------------------------------------
# Image augmentation utilities
# ----------------------------------------------------------------------------------

def image_augmenter(image, image_scale, method=cv2.INTER_AREA):
    """Basic image augmentation: convert color and resize."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_scale, image_scale),
                       interpolation=method)
    return image


PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a photo of a {} flower",
    "This is a photo of a {}",
    "This is a photo of a {} flower",
    "A photo of a {} flower",
]


def labelizer_oxford_flowers102(path):
    """Creates a label generator for Oxford Flowers 102 dataset."""
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


# ----------------------------------------------------------------------------------
# TFDS Image Source
# ----------------------------------------------------------------------------------

def get_oxford_valset(text_encoder):
    # Construct a validation set by the prompts for consistency
    val_prompts = ['water tulip', ' a water lily', ' a water lily', ' a photo of a rose', ' a photo of a rose', ' a water lily', ' a water lily', ' a photo of a marigold', ' a photo of a marigold', ' a photo of a marigold', ' a water lily', ' a photo of a sunflower', ' a photo of a lotus', ' columbine', ' columbine', ' an orchid', ' an orchid', ' an orchid', ' a water lily', ' a water lily', ' a water lily', ' columbine', ' columbine', ' a photo of a sunflower', ' a photo of a sunflower', ' a photo of a sunflower', ' a photo of a lotus', ' a photo of a lotus', ' a photo of a marigold', ' a photo of a marigold', ' a photo of a rose', ' a photo of a rose', ' a photo of a rose', ' orange dahlia', ' orange dahlia', ' a lenten rose', ' a lenten rose', ' a water lily', ' a water lily', ' a water lily', ' a water lily', ' an orchid', ' an orchid', ' an orchid', ' hard-leaved pocket orchid', ' bird of paradise', ' bird of paradise', ' a photo of a lovely rose', ' a photo of a lovely rose', ' a photo of a globe-flower', ' a photo of a globe-flower', ' a photo of a lovely rose', ' a photo of a lovely rose', ' a photo of a ruby-lipped cattleya', ' a photo of a ruby-lipped cattleya', ' a photo of a lovely rose', ' a water lily', ' a osteospermum', ' a osteospermum', ' a water lily', ' a water lily', ' a water lily', ' a red rose', ' a red rose']
    val_prompts *= 100

    def get_val_dataset(batch_size=128):
        for i in range(0, len(val_prompts), batch_size):
            try:
                prompts = val_prompts[i:i + batch_size]
                tokens = text_encoder.tokenize(prompts)
                yield {"text": tokens}
            except Exception as e:
                print(f"Error in get_val_dataset: {e}")
                traceback.print_exc()
                continue

    return get_val_dataset, len(val_prompts)
    
class ImageTFDSSource(DataSource):
    """Data source for TensorFlow Datasets (TFDS) image datasets."""
    
    def __init__(self, name: str, use_tf: bool = True, split: str = "all"):
        """Initialize a TFDS image data source.
        
        Args:
            name: Name of the TFDS dataset.
            use_tf: Whether to use TensorFlow for loading.
            split: Dataset split to use.
        """
        self.name = name
        self.use_tf = use_tf
        self.split = split
    
    def get_source(self, path_override: str) -> Any:
        """Get the TFDS data source.
        
        Args:
            path_override: Override path for the dataset.
            
        Returns:
            A TFDS dataset.
        """
        import tensorflow_datasets as tfds
        if self.use_tf:
            return tfds.load(self.name, split=self.split, shuffle_files=True)
        else:
            return tfds.data_source(self.name, split=self.split, try_gcs=False)


class ImageTFDSAugmenter(DataAugmenter):
    """Augmenter for TFDS image datasets."""
    
    def __init__(self, label_path: str = None):
        """Initialize a TFDS image augmenter.
        
        Args:
            label_path: Path to the labels file for datasets like Oxford Flowers.
        """
        if label_path is None:
            label_path = os.path.join(os.path.expanduser("~"), "tensorflow_datasets/oxford_flowers102/2.1.1/label.labels.txt")
        self.label_path = label_path
    
    def create_transform(self, image_scale: int = 256, method: Any = None) -> Callable[[], pygrain.MapTransform]:
        """Create a transform for TFDS image datasets.
        
        Args:
            image_scale: Size to scale images to.
            method: Interpolation method for resizing.
            
        Returns:
            A callable that returns a pygrain.MapTransform.
        """
        labelizer = labelizer_oxford_flowers102(self.label_path)
        
        if image_scale > 256:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        
        from torchvision.transforms import v2
        augments = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.05, saturation=0.2)
        ])
        
        class TFDSTransform(pygrain.MapTransform):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokenize = AutoTextTokenizer(tensor_type="np")
                
            def map(self, element) -> Dict[str, jnp.array]:
                image = element['image']
                image = cv2.resize(image, (image_scale, image_scale),
                                  interpolation=interpolation)
                image = augments(image)
                
                caption = labelizer(element)
                results = self.tokenize(caption)
                return {
                    "image": image,
                    "text": {
                        "input_ids": results['input_ids'][0],
                        "attention_mask": results['attention_mask'][0],
                    }
                }
        
        return TFDSTransform
    
    def create_filter(self, image_scale: int = 256):
        class FilterTransform(pygrain.FilterTransform):
            def map(self, element) -> bool:
                return True
"""
Batch structure:
{
"image": image_batch,
"text": {
    "input_ids": input_ids_batch,
    "attention_mask": attention_mask_batch,
}

"""

# ----------------------------------------------------------------------------------
# GCS Image Source
# ----------------------------------------------------------------------------------

class ImageGCSSource(DataSource):
    """Data source for Google Cloud Storage (GCS) image datasets."""
    
    def __init__(self, source: str = 'arrayrecord/laion-aesthetics-12m+mscoco-2017'):
        """Initialize a GCS image data source.
        
        Args:
            source: Path to the GCS dataset.
        """
        self.source = source
    
    def get_source(self, path_override: str = "/home/mrwhite0racle/gcs_mount") -> Any:
        """Get the GCS data source.
        
        Args:
            path_override: Base path for GCS mounts.
            
        Returns:
            A grain ArrayRecordDataSource.
        """
        records_path = os.path.join(path_override, self.source)
        records = [os.path.join(records_path, i) for i in os.listdir(
            records_path) if 'array_record' in i]
        return pygrain.ArrayRecordDataSource(records)


class CombinedImageGCSSource(DataSource):
    """Data source that combines multiple GCS image datasets."""
    
    def __init__(self, sources: List[str] = []):
        """Initialize a combined GCS image data source.
        
        Args:
            sources: List of paths to GCS datasets.
        """
        self.sources = sources
    
    def get_source(self, path_override: str = "/home/mrwhite0racle/gcs_mount") -> Any:
        """Get the combined GCS data source.
        
        Args:
            path_override: Base path for GCS mounts.
            
        Returns:
            A grain ArrayRecordDataSource.
        """
        records_paths = [os.path.join(path_override, source) for source in self.sources]
        records = []
        for records_path in records_paths:
            records += [os.path.join(records_path, i) for i in os.listdir(
                records_path) if 'array_record' in i]
        return pygrain.ArrayRecordDataSource(records)

class ImageGCSAugmenter(DataAugmenter):
    """Augmenter for GCS image datasets."""
    
    def __init__(self, labelizer: Callable = None):
        """Initialize a GCS image augmenter.
        
        Args:
            labelizer: Function to extract text labels from samples.
        """
        self.labelizer = labelizer or (lambda sample: sample['txt'])
    
    def create_transform(self, image_scale: int = 256, method: Any = None) -> Callable[[], pygrain.MapTransform]:
        """Create a transform for GCS image datasets.
        
        Args:
            image_scale: Size to scale images to.
            method: Interpolation method for resizing.
            
        Returns:
            A callable that returns a pygrain.MapTransform.
        """
        labelizer = self.labelizer
        if method is None:
            if image_scale > 256:
                method = cv2.INTER_CUBIC
            else:
                method = cv2.INTER_AREA
                
        print(f"Using method: {method}")
        
        from torchvision.transforms import v2
        augments = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.05, saturation=0.2)
        ])
        
        class GCSTransform(pygrain.MapTransform):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.auto_tokenize = AutoTextTokenizer(tensor_type="np")
                self.image_augmenter = partial(image_augmenter, image_scale=image_scale, method=method)
                
            def map(self, element) -> Dict[str, jnp.array]:
                element = unpack_dict_of_byte_arrays(element)
                image = np.asarray(bytearray(element['jpg']), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                image = self.image_augmenter(image)
                image = augments(image)
                caption = labelizer(element).decode('utf-8')
                results = self.auto_tokenize(caption)
                return {
                    "image": image,
                    "text": {
                        "input_ids": results['input_ids'][0],
                        "attention_mask": results['attention_mask'][0],
                    }
                }
        
        return GCSTransform
    
    def create_filter(self, image_scale: int = 256):
        import torch.nn.functional as F
        class FilterTransform(pygrain.FilterTransform):
            """
            Filter transform for GCS data source.
            """
            def __init__(self, model=None, processor=None, method=cv2.INTER_AREA):
                super().__init__()
                self.image_scale = image_scale
                if model is None:
                    from transformers import AutoProcessor, CLIPVisionModelWithProjection, FlaxCLIPModel, CLIPModel
                    model_name = "openai/clip-vit-base-patch32"
                    model = CLIPModel.from_pretrained(model_name)
                    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
                self.method = method
                self.model = model
                self.processor = processor
                
                # def _filter_(pixel_values, input_ids):
                #     image_embeds = self.model.get_image_features(pixel_values=pixel_values)
                #     text_embeds = self.model.get_text_features(input_ids=input_ids)
                #     image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
                #     text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)
                #     similarity = jnp.sum(image_embeds * text_embeds, axis=-1)
                #     return jnp.all(similarity >= 0.25)
                
                # self._filter_ = _filter_

            def filter(self, data: Dict[str, Any]) -> bool:
                images = [data['image']]
                texts = [data['caption']]
                inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
                # result = self._filter_(
                #     pixel_values=inputs['pixel_values'],
                #     input_ids=inputs['input_ids']
                # )
                # return result
                
                image_embeds = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                text_embeds = self.model.get_text_features(input_ids=inputs['input_ids'])
                similarity = F.cosine_similarity(image_embeds, text_embeds)
                # Filter out images with similarity less than 0.25
                return similarity[0] >= 0.25
                
        return FilterTransform


# ----------------------------------------------------------------------------------
# Legacy compatibility functions
# ----------------------------------------------------------------------------------

# These functions maintain backward compatibility with existing code

def data_source_tfds(name, use_tf=True, split="all"):
    """Legacy function for TFDS data sources."""
    source = ImageTFDSSource(name=name, use_tf=use_tf, split=split)
    return source.get_source


def tfds_augmenters(image_scale, method):
    """Legacy function for TFDS augmenters."""
    augmenter = ImageTFDSAugmenter()
    return augmenter.create_transform(image_scale=image_scale, method=method)


def data_source_gcs(source='arrayrecord/laion-aesthetics-12m+mscoco-2017'):
    """Legacy function for GCS data sources."""
    source_obj = ImageGCSSource(source=source)
    return source_obj.get_source


def data_source_combined_gcs(sources=[]):
    """Legacy function for combined GCS data sources."""
    source_obj = CombinedImageGCSSource(sources=sources)
    return source_obj.get_source


def gcs_augmenters(image_scale, method):
    """Legacy function for GCS augmenters."""
    augmenter = ImageGCSAugmenter()
    return augmenter.create_transform(image_scale=image_scale, method=method)

def gcs_filters(image_scale):
    """Legacy function for GCS Filters."""
    augmenter = ImageGCSAugmenter()
    return augmenter.create_filter(image_scale=image_scale)