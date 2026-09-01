import grain.python as pygrain
from typing import Dict, Any, Optional, Union, List, Callable
import numpy as np
import jax
import cv2  # Added missing import
from flaxdiff.utils import AutoTextTokenizer
from .dataset_map import datasetMap, onlineDatasetMap, mediaDatasetMap
import traceback
# NOTE: .online_loader is imported lazily inside the two `*_online` factories.
# It needs HF `datasets`, which the grain paths must not require.
from functools import partial


def generate_collate_fn(media_type="image"):
    """Generate a collate function based on media type.
    
    Args:
        media_type: Type of media ("image" or "video").
        
    Returns:
        A collate function for the specified media type.
    """
    auto_tokenize = AutoTextTokenizer(tensor_type="np")
    
    def image_collate(batch):
        try:
            # Check if batch is valid
            if not batch or len(batch) == 0:
                print("Warning: Empty batch received")
                # Return an empty batch with the correct structure
                return {
                    "image": np.zeros((0, 0, 0, 3), dtype=np.float32),
                    "text": {
                        "input_ids": np.zeros((0, 0), dtype=np.int32),
                        "attention_mask": np.zeros((0, 0), dtype=np.int32),
                    }
                }
                
            captions = [sample.get("caption", "") for sample in batch]
            results = auto_tokenize(captions)
            
            # Check if all images have the same shape
            image_shapes = [sample["image"].shape for sample in batch]
            if len(set(str(shape) for shape in image_shapes)) > 1:
                # Different shapes, need to resize all to the same shape
                target_shape = max(shape[0] for shape in image_shapes), max(shape[1] for shape in image_shapes)
                images = np.stack([
                    cv2.resize(sample["image"], target_shape) if sample["image"].shape[:2] != target_shape else sample["image"] 
                    for sample in batch
                ], axis=0)
            else:
                # All same shape, can just stack
                images = np.stack([sample["image"] for sample in batch], axis=0)
                
            return {
                "image": images,
                "text": {
                    "input_ids": results['input_ids'],
                    "attention_mask": results['attention_mask'],
                }
            }
        except Exception as e:
            print("Error in image collate function", e)
            traceback.print_exc()
            # Return a fallback batch
            return fallback_batch(batch, media_type="image")
            
    def video_collate(batch):
        try:
            # Check if batch is valid
            if not batch or len(batch) == 0:
                print("Warning: Empty batch received")
                # Return an empty batch with the correct structure
                return {
                    "video": np.zeros((0, 0, 0, 0, 3), dtype=np.float32),
                    "text": {
                        "input_ids": np.zeros((0, 0), dtype=np.int32),
                        "attention_mask": np.zeros((0, 0), dtype=np.int32),
                    }
                }
                
            captions = [sample.get("caption", "") for sample in batch]
            results = auto_tokenize(captions)
            
            # Check if all videos have the same shape
            video_shapes = [sample["video"].shape for sample in batch]
            if len(set(str(shape) for shape in video_shapes)) > 1:
                # Get max dimensions
                max_frames = max(shape[0] for shape in video_shapes)
                max_height = max(shape[1] for shape in video_shapes)
                max_width = max(shape[2] for shape in video_shapes)
                
                # Resize videos to the same shape
                videos = []
                for sample in batch:
                    video = sample["video"]
                    num_frames, height, width = video.shape[:3]
                    
                    if height != max_height or width != max_width:
                        # Resize each frame
                        resized_frames = np.array([
                            cv2.resize(frame, (max_width, max_height))
                            for frame in video
                        ])
                        video = resized_frames
                    
                    if num_frames < max_frames:
                        # Pad with duplicates of the last frame
                        padding = np.tile(video[-1:], (max_frames - num_frames, 1, 1, 1))
                        video = np.concatenate([video, padding], axis=0)
                    
                    videos.append(video)
                
                videos = np.stack(videos, axis=0)
            else:
                # All videos have the same shape, can just stack
                videos = np.stack([sample["video"] for sample in batch], axis=0)
                
            return {
                "video": videos,
                "text": {
                    "input_ids": results['input_ids'],
                    "attention_mask": results['attention_mask'],
                }
            }
        except Exception as e:
            print("Error in video collate function", e)
            traceback.print_exc()
            # Return a fallback batch
            return fallback_batch(batch, media_type="video")
    
    def fallback_batch(batch, media_type="image"):
        """Create a fallback batch when an error occurs."""
        try:
            batch_size = len(batch) if batch else 1
            if media_type == "video":
                # Create a small valid video batch
                dummy_video = np.zeros((batch_size, 4, 32, 32, 3), dtype=np.uint8)
                dummy_text = auto_tokenize(["Error processing video"] * batch_size)
                return {
                    "video": dummy_video,
                    "text": {
                        "input_ids": dummy_text['input_ids'],
                        "attention_mask": dummy_text['attention_mask'],
                    }
                }
            else:
                # Create a small valid image batch
                dummy_image = np.zeros((batch_size, 32, 32, 3), dtype=np.uint8)
                dummy_text = auto_tokenize(["Error processing image"] * batch_size)
                return {
                    "image": dummy_image,
                    "text": {
                        "input_ids": dummy_text['input_ids'],
                        "attention_mask": dummy_text['attention_mask'],
                    }
                }
        except Exception as e:
            print("Error creating fallback batch", e)
            # Last resort fallback
            if media_type == "video":
                return {
                    "video": np.zeros((1, 4, 32, 32, 3), dtype=np.uint8),
                    "text": {
                        "input_ids": np.zeros((1, 16), dtype=np.int32),
                        "attention_mask": np.zeros((1, 16), dtype=np.int32),
                    }
                }
            else:
                return {
                    "image": np.zeros((1, 32, 32, 3), dtype=np.uint8),
                    "text": {
                        "input_ids": np.zeros((1, 16), dtype=np.int32),
                        "attention_mask": np.zeros((1, 16), dtype=np.int32),
                    }
                }
            
    if media_type == "video":
        return video_collate
    else:  # Default to image
        return image_collate

class CaptionDeletionTransform(pygrain.MapTransform):
    def map(self, element):
        """Delete the caption from the element."""
        if "caption" in element:
            del element["caption"]
        return element

def get_dataset_grain(
    data_name="cc12m",
    batch_size=64,
    image_scale=256,
    count=None,
    num_epochs=None,
    method=None, #jax.image.ResizeMethod.LANCZOS3,
    worker_count=32,
    read_thread_count=64,
    read_buffer_size=50,
    worker_buffer_size=20,
    seed=0,
    dataset_source="/mnt/gcs_mount/arrayrecord2/cc12m/",
    val_batch_size=32,
    val_worker_count=8,
):
    """Legacy function for getting grain dataset loaders for images.
    
    Args:
        data_name: Name of the dataset in datasetMap.
        batch_size: Batch size for the dataset.
        image_scale: Size to scale images to.
        count: Optional count limit for the dataset.
        num_epochs: Number of epochs to iterate.
        method: Interpolation method for resizing.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        read_buffer_size: Size of the read buffer.
        worker_buffer_size: Size of the worker buffer.
        seed: Random seed.
        dataset_source: Source path for the dataset.
        val_batch_size: Batch size for the validation loader.
        val_worker_count: Number of worker processes for the validation loader.
        
    Returns:
        Dictionary with train dataset function and metadata.
    """
    dataset = datasetMap[data_name]
    data_source = dataset["source"](dataset_source)
    augmenter = dataset["augmenter"](image_scale, method)
    local_batch_size = batch_size // jax.process_count()

    train_sampler = pygrain.IndexSampler(
        num_records=len(data_source) if count is None else count,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    # Validation reads the same records in canonical order: sharing the
    # shuffled train sampler made "validation" a random slice of training data.
    val_sampler = pygrain.IndexSampler(
        num_records=len(data_source) if count is None else count,
        shuffle=False,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )
    
    def get_trainset():
        transformations = [
            augmenter(),
        ]
        transformations.append(pygrain.Batch(local_batch_size, drop_remainder=True))

        loader = pygrain.DataLoader(
            data_source=data_source,
            sampler=train_sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader
    
    def get_valset():
        transformations = [
            augmenter(),
            pygrain.Batch(val_batch_size, drop_remainder=True),
        ]

        loader = pygrain.DataLoader(
            data_source=data_source,
            sampler=val_sampler,
            operations=transformations,
            worker_count=val_worker_count,
            read_options=pygrain.ReadOptions(
                32, 128
            ),
            worker_buffer_size=32,
        )
        return loader

    return {
        "train": get_trainset,
        "train_len": len(data_source),
        "val": get_valset,
        "val_len": len(data_source),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
    }


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
    """Legacy function for getting online streaming dataloader for images.
    
    Args:
        data_name: Name of the dataset in onlineDatasetMap.
        batch_size: Batch size for the dataset.
        image_scale: Size to scale images to.
        count: Optional count limit for the dataset.
        num_epochs: Number of epochs to iterate.
        method: Interpolation method for resizing.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        read_buffer_size: Size of the read buffer.
        worker_buffer_size: Size of the worker buffer.
        seed: Random seed.
        dataset_source: Source path for the dataset.
        
    Returns:
        Dictionary with train dataset function and metadata.
    """
    if data_name not in onlineDatasetMap:
        raise ValueError(f"Dataset {data_name} not found in onlineDatasetMap")

    # Imported here, not at module scope: the streaming stack needs HF
    # `datasets`, which the grain paths above deliberately do without.
    from .online_loader import OnlineStreamingDataLoader

    local_batch_size = batch_size // jax.process_count()
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
            collate_fn=generate_collate_fn(),
            default_split="train",
        )
    
    def get_trainset():
        return dataloader
    
    return {
        "train": get_trainset,
        "train_len": len(dataloader) * jax.process_count(),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
    }


# ---------------------------------------------------------------------------------
# New unified dataset loader for both images and videos
# ---------------------------------------------------------------------------------

class _SourceSlice:
    """Random-access view over `source[start:stop]`.

    Gives the train and validation loaders disjoint index ranges while each
    keeps a stock grain IndexSampler, so sharding and epoch handling stay
    grain's. Plain attributes only: grain pickles the source to its workers.
    """

    def __init__(self, source: Any, start: int, stop: int):
        self.source = source
        self.start = start
        self.length = stop - start

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source[self.start + index]


def get_media_dataset_grain(
    data_name: str,
    batch_size: int = 64,
    media_scale: int = 256,
    sequence_length: int = 1,
    count: Optional[int] = None,
    num_epochs: Optional[int] = None,
    method: Any = cv2.INTER_AREA,
    worker_count: int = 32,
    read_thread_count: int = 64,
    read_buffer_size: int = 50,
    worker_buffer_size: int = 20,
    seed: int = 0,
    dataset_source: Optional[str] = None,
    media_type: Optional[str] = None,  # Will be auto-detected if None
    additional_transform_kwargs: Dict[str, Any] = None,
    val_count: Optional[int] = None,
    val_batch_size: Optional[int] = None,
):
    """Get a grain dataset loader for any media type (image or video).
    
    Args:
        data_name: Name of the dataset in mediaDatasetMap.
        batch_size: Batch size for the dataset.
        media_scale: Size to scale media (image or video frames) to.
        sequence_length: Length of the sequence for video data.
        count: Optional count limit for the dataset.
        num_epochs: Number of epochs to iterate.
        method: Interpolation method for resizing.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        read_buffer_size: Size of the read buffer.
        worker_buffer_size: Size of the worker buffer.
        seed: Random seed.
        dataset_source: Root path the dataset's source resolves its files
            against. Required - there is no sane default for a bucket mount
            or a local media tree.
        media_type: Type of media ("image" or "video"). Auto-detected if None.
        additional_transform_kwargs: Additional arguments for the transform.
        val_count: Records held out, in canonical source order, as a
            validation split; the train loader then covers the rest. None
            (default) means no validation loader.
        val_batch_size: Batch size of the validation loader. Defaults to the
            train loader's local batch size.

    Returns:
        Dictionary with train dataset function and metadata; with val_count set
        it also carries "val" and "val_len" for the held-out split.
    """
    if data_name not in mediaDatasetMap:
        raise ValueError(f"Dataset {data_name} not found in mediaDatasetMap")
    if not dataset_source:
        raise ValueError(
            f"get_media_dataset_grain('{data_name}') needs an explicit dataset_source: "
            "media sources resolve their files against it, and an unset one used to "
            "reach os.path.join(None, ...) inside the source itself."
        )

    media_dataset = mediaDatasetMap[data_name]

    # Auto-detect media_type if not provided
    if media_type is None:
        media_type = media_dataset.media_type

    # Get the data source and augmenter
    data_source = media_dataset.get_source(dataset_source)

    # Prepare transform kwargs
    transform_kwargs = {
        "image_scale" if media_type == "image" else "frame_size": media_scale,
        "method": method,
        "sequence_length": sequence_length,
    }
    if additional_transform_kwargs:
        transform_kwargs.update(additional_transform_kwargs)

    augmenter = media_dataset.get_augmenter(**transform_kwargs)

    # Calculate local batch size for distributed training
    local_batch_size = batch_size // jax.process_count()

    # Create a sampler for the dataset
    if hasattr(data_source, "__len__"):
        dataset_length = len(data_source) if count is None else count
    else:
        # Some data sources like video files list don't have __len__
        dataset_length = count if count is not None else 1000000  # Default large number

    train_source, val_source = data_source, None
    if val_count:
        if not hasattr(data_source, "__len__"):
            raise ValueError(
                "A validation split needs a data source of known length; "
                f"'{data_name}' does not report one."
            )
        if not 0 < val_count < dataset_length:
            raise ValueError(
                f"val_count must be within 1..{dataset_length - 1} records for "
                f"'{data_name}', got {val_count}"
            )
        val_source = _SourceSlice(data_source, 0, val_count)
        train_source = _SourceSlice(data_source, val_count, dataset_length)

    train_length = dataset_length if val_source is None else len(train_source)

    train_sampler = pygrain.IndexSampler(
        num_records=train_length,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    def get_trainset():
        """Get a training dataset iterator."""
        transformations = [
            augmenter(),
            pygrain.Batch(local_batch_size, drop_remainder=True),
        ]

        loader = pygrain.DataLoader(
            data_source=train_source,
            sampler=train_sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader

    dataset = {
        "train": get_trainset,
        "train_len": train_length,
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        "media_type": media_type,
    }

    if val_source is None:
        return dataset

    # Its own unshuffled sampler: sharing the train sampler turned validation
    # into a random slice of the training stream.
    val_sampler = pygrain.IndexSampler(
        num_records=len(val_source),
        shuffle=False,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    def get_valset():
        """Get a validation dataset iterator."""
        transformations = [
            augmenter(),
            pygrain.Batch(val_batch_size or local_batch_size, drop_remainder=True),
        ]

        loader = pygrain.DataLoader(
            data_source=val_source,
            sampler=val_sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader

    dataset["val"] = get_valset
    dataset["val_len"] = len(val_source)
    return dataset


def get_media_dataset_online(
    data_name: str = "combined_online",
    batch_size: int = 64,
    media_scale: int = 256,
    worker_count: int = 16,
    read_thread_count: int = 512,
    worker_buffer_size: int = 20,
    dataset_sources: List[str] = None,
    media_type: str = "image",  # Default to image for online datasets
    timeout: int = 15,
    retries: int = 3,
    min_media_scale: int = 128,
):
    """Get an online streaming dataset loader for any media type.
    
    Args:
        data_name: Name of the dataset in onlineDatasetMap, or "custom" for custom sources.
        batch_size: Batch size for the dataset.
        media_scale: Size to scale media (image or video frames) to.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        worker_buffer_size: Size of the worker buffer.
        dataset_sources: Custom dataset sources if data_name is "custom".
        media_type: Type of media ("image" or "video"). 
        timeout: Timeout for dataset operations.
        retries: Number of retries for dataset operations.
        min_media_scale: Minimum scale for media items.
        
    Returns:
        Dictionary with train dataset function and metadata.
    """
    local_batch_size = batch_size // jax.process_count()
    
    # Get dataset sources
    if dataset_sources is None:
        if data_name not in onlineDatasetMap:
            raise ValueError(f"Dataset {data_name} not found in onlineDatasetMap")
        sources = onlineDatasetMap[data_name]["source"]
    else:
        sources = dataset_sources
    
    # Configure shape parameter based on media type
    shape_param = "image_shape" if media_type == "image" else "frame_size"
    shape_value = (media_scale, media_scale) if media_type == "image" else media_scale
    
    # Configure min scale parameter based on media type
    min_scale_param = "min_image_shape" if media_type == "image" else "min_frame_size"
    min_scale_value = (min_media_scale, min_media_scale) if media_type == "image" else min_media_scale
    
    # Prepare dataloader kwargs
    dataloader_kwargs = {
        "batch_size": local_batch_size,
        "num_workers": worker_count,
        "num_threads": read_thread_count,
        shape_param: shape_value,
        min_scale_param: min_scale_value,
        "global_process_count": jax.process_count(),
        "global_process_index": jax.process_index(),
        "prefetch": worker_buffer_size,
        "collate_fn": generate_collate_fn(media_type),
        "default_split": "train",
        "timeout": timeout,
        "retries": retries,
    }

    # Imported here, not at module scope: the streaming stack needs HF
    # `datasets`, which the grain paths above deliberately do without.
    from .online_loader import OnlineStreamingDataLoader

    dataloader = OnlineStreamingDataLoader(sources, **dataloader_kwargs)
    
    def get_trainset():
        """Get a training dataset iterator."""
        return dataloader
    
    return {
        "train": get_trainset,
        "train_len": len(dataloader) * jax.process_count(),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        "media_type": media_type,
    }