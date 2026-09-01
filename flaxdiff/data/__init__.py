"""Data layer for FlaxDiff: sources, augmenters and dataset loaders.

Importing this package is deliberately cheap. The data layer's heavy
dependencies (HF `datasets`, opencv, torch, decord/pyav, tensorflow_datasets)
are pulled in only when a name is actually used, so `import flaxdiff.data`
works on hosts that have none of them installed - a training run that only
needs the grain pipelines never pays for the streaming stack, and vice versa.

Names below resolve on first access via PEP 562 module `__getattr__`; import a
submodule directly (`from flaxdiff.data.dataloaders import get_dataset_grain`)
when you want the dependency error eagerly.
"""

import importlib

# Public name -> submodule that defines it.
_EXPORTS = {
    # dataloaders: grain (offline) and streaming (online) dataset factories
    "generate_collate_fn": ".dataloaders",
    "CaptionDeletionTransform": ".dataloaders",
    "get_dataset_grain": ".dataloaders",
    "get_dataset_online": ".dataloaders",
    "get_media_dataset_grain": ".dataloaders",
    "get_media_dataset_online": ".dataloaders",
    # online_loader: the streaming stack (needs HF datasets)
    "ResourceManager": ".online_loader",
    "OnlineStreamingDataLoader": ".online_loader",
    "MediaBatchIterator": ".online_loader",
    "dataMapper": ".online_loader",
    "fetch_single_image": ".online_loader",
    "fetch_single_video": ".online_loader",
    "default_image_processor": ".online_loader",
    "default_video_processor": ".online_loader",
    "default_feature_extractor": ".online_loader",
    "default_image_collate": ".online_loader",
    "default_video_collate": ".online_loader",
    "get_default_collate": ".online_loader",
    "map_image_sample": ".online_loader",
    "map_video_sample": ".online_loader",
    "map_batch": ".online_loader",
    "parallel_media_loader": ".online_loader",
    # source/augmenter seam
    "DataSource": ".sources.base",
    "DataAugmenter": ".sources.base",
    "MediaDataset": ".sources.base",
    # image sources
    "ImageTFDSSource": ".sources.images",
    "ImageTFDSAugmenter": ".sources.images",
    "ImageGCSSource": ".sources.images",
    "CombinedImageGCSSource": ".sources.images",
    "ImageGCSAugmenter": ".sources.images",
    "get_oxford_valset": ".sources.images",
    "labelizer_oxford_flowers102": ".sources.images",
    "image_augmenter": ".sources.images",
    "unpack_dict_of_byte_arrays": ".sources.images",
    "PROMPT_TEMPLATES": ".sources.images",
    "data_source_tfds": ".sources.images",
    "data_source_gcs": ".sources.images",
    "data_source_combined_gcs": ".sources.images",
    "tfds_augmenters": ".sources.images",
    "gcs_augmenters": ".sources.images",
    "gcs_filters": ".sources.images",
    # video / audio-video sources
    "VideoTFDSSource": ".sources.videos",
    "VideoLocalSource": ".sources.videos",
    "AudioVideoAugmenter": ".sources.videos",
    "gather_video_paths": ".sources.videos",
    "gather_video_paths_iter": ".sources.videos",
    "VoxCeleb2Source": ".sources.voxceleb2",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value  # first access only; later lookups skip __getattr__
    return value


def __dir__():
    return __all__
