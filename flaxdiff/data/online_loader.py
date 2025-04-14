import multiprocessing
import threading
from multiprocessing import Queue
import time
import albumentations as A
import queue
import cv2
from functools import partial
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

import numpy as np
from functools import partial

from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from datasets.utils.file_utils import get_datasets_user_agent
from concurrent.futures import ThreadPoolExecutor
import io
import urllib
import os

import PIL.Image
import traceback

USER_AGENT = get_datasets_user_agent()


class ResourceManager:
    """A manager for shared resources across data loading processes."""
    
    def __init__(self, max_queue_size: int = 32000):
        """Initialize a resource manager.
        
        Args:
            max_queue_size: Maximum size of the data queue.
        """
        self.data_queue = Queue(max_queue_size)
    
    def get_data_queue(self) -> Queue:
        """Get the data queue."""
        return self.data_queue


def fetch_single_image(image_url: str, timeout: Optional[int] = None, retries: int = 0) -> Optional[PIL.Image.Image]:
    """Fetch a single image from a URL.
    
    Args:
        image_url: URL of the image to fetch.
        timeout: Timeout in seconds for the request.
        retries: Number of times to retry the request.
        
    Returns:
        A PIL image or None if the image couldn't be fetched.
    """
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            return image
        except Exception as e:
            if attempt < retries:
                # Wait a bit before retrying
                time.sleep(0.1 * (attempt + 1))
                continue
            # Log the error on the final attempt
            print(f"Error fetching image {image_url}: {e}")
            return None


def fetch_single_video(video_url: str, timeout: Optional[int] = None, retries: int = 0, 
                       max_frames: int = 32) -> Optional[List[np.ndarray]]:
    """Fetch a single video from a URL.
    
    Args:
        video_url: URL of the video to fetch.
        timeout: Timeout in seconds for the request.
        retries: Number of times to retry the request.
        max_frames: Maximum number of frames to extract.
        
    Returns:
        A list of video frames as numpy arrays or None if the video couldn't be fetched.
    """
    # Create a temporary file to download the video
    import tempfile
    
    for attempt in range(retries + 1):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
            request = urllib.request.Request(
                video_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                with open(tmp_path, 'wb') as f:
                    f.write(req.read())
            
            # Load the video frames
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()
            
            # Delete the temporary file
            try:
                os.remove(tmp_path)
            except:
                pass
                
            return frames if frames else None
            
        except Exception as e:
            if attempt < retries:
                # Wait a bit before retrying
                time.sleep(0.1 * (attempt + 1))
                continue
            # Log the error on the final attempt
            print(f"Error fetching video {video_url}: {e}")
            
            # Clean up the temporary file
            try:
                if 'tmp_path' in locals():
                    os.remove(tmp_path)
            except:
                pass
                
            return None


def default_image_processor(
    image: PIL.Image.Image, 
    image_shape: Tuple[int, int], 
    min_image_shape: Tuple[int, int] = (128, 128),
    upscale_interpolation: int = cv2.INTER_CUBIC,
    downscale_interpolation: int = cv2.INTER_AREA,
) -> Tuple[Optional[np.ndarray], int, int]:
    """Process an image for training.
    
    Args:
        image: PIL image to process.
        image_shape: Target shape (height, width).
        min_image_shape: Minimum acceptable shape.
        upscale_interpolation: Interpolation method for upscaling.
        downscale_interpolation: Interpolation method for downscaling.
        
    Returns:
        Tuple of (processed image, original height, original width).
        Processed image may be None if the image couldn't be processed.
    """
    try:
        # Convert to numpy
        image = np.array(image)
        
        # Check if image has 3 channels
        if len(image.shape) != 3 or image.shape[2] != 3:
            return None, 0, 0
            
        original_height, original_width = image.shape[:2]
        
        # Check if the image is too small
        if min(original_height, original_width) < min(min_image_shape):
            return None, original_height, original_width
            
        # Check if wrong aspect ratio
        if max(original_height, original_width) / min(original_height, original_width) > 2.4:
            return None, original_height, original_width
            
        # Check if the variance is too low (likely a blank/solid color image)
        if np.std(image) < 1e-5:
            return None, original_height, original_width
            
        # Choose interpolation method based on whether we're upscaling or downscaling
        downscale = max(original_width, original_height) > max(image_shape)
        interpolation = downscale_interpolation if downscale else upscale_interpolation

        # Resize while keeping aspect ratio
        image = A.longest_max_size(image, max(image_shape), interpolation=interpolation)
        
        # Pad to target shape
        image = A.pad(
            image,
            min_height=image_shape[0],
            min_width=image_shape[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        
        return image, original_height, original_width
        
    except Exception as e:
        # Log the error
        print(f"Error processing image: {e}")
        return None, 0, 0


def default_video_processor(
    frames: List[np.ndarray],
    frame_size: int = 256,
    min_frame_size: int = 128,
    num_frames: int = 16,
    upscale_interpolation: int = cv2.INTER_CUBIC,
    downscale_interpolation: int = cv2.INTER_AREA,
) -> Tuple[Optional[np.ndarray], int, int]:
    """Process video frames for training.
    
    Args:
        frames: List of video frames as numpy arrays.
        frame_size: Target size for each frame.
        min_frame_size: Minimum acceptable frame size.
        num_frames: Target number of frames.
        upscale_interpolation: Interpolation method for upscaling.
        downscale_interpolation: Interpolation method for downscaling.
        
    Returns:
        Tuple of (processed video array, original height, original width).
        Processed video may be None if the video couldn't be processed.
    """
    try:
        if not frames or len(frames) == 0:
            return None, 0, 0
            
        # Get dimensions of the first frame
        first_frame = frames[0]
        original_height, original_width = first_frame.shape[:2]
        
        # Check if frames are too small
        if min(original_height, original_width) < min_frame_size:
            return None, original_height, original_width
            
        # Sample frames evenly
        if len(frames) < num_frames:
            # Not enough frames, duplicate some
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            # Sample frames evenly
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        
        # Process each frame
        processed_frames = []
        for frame in sampled_frames:
            # Choose interpolation method based on whether we're upscaling or downscaling
            downscale = max(frame.shape[1], frame.shape[0]) > frame_size
            interpolation = downscale_interpolation if downscale else upscale_interpolation
            
            # Resize frame
            resized_frame = cv2.resize(frame, (frame_size, frame_size), interpolation=interpolation)
            processed_frames.append(resized_frame)
        
        # Stack frames into a video tensor [num_frames, height, width, channels]
        video_tensor = np.stack(processed_frames, axis=0)
        
        return video_tensor, original_height, original_width
        
    except Exception as e:
        # Log the error
        print(f"Error processing video: {e}")
        return None, 0, 0


def map_image_sample(
    url: str,
    caption: str,
    data_queue: Queue,
    image_shape: Tuple[int, int] = (256, 256),
    min_image_shape: Tuple[int, int] = (128, 128),
    timeout: int = 15,
    retries: int = 3,
    upscale_interpolation: int = cv2.INTER_CUBIC,
    downscale_interpolation: int = cv2.INTER_AREA,
    image_processor: Callable = default_image_processor,
):
    """Process a single image sample and put it in the queue.
    
    Args:
        url: URL of the image.
        caption: Caption for the image.
        data_queue: Queue to put the processed sample in.
        image_shape: Target shape for the image.
        min_image_shape: Minimum acceptable shape.
        timeout: Timeout for image fetching.
        retries: Number of retries for image fetching.
        upscale_interpolation: Interpolation method for upscaling.
        downscale_interpolation: Interpolation method for downscaling.
        image_processor: Function to process the image.
    """
    try:
        # Fetch the image
        image = fetch_single_image(url, timeout=timeout, retries=retries)
        if image is None:
            return

        # Process the image
        image, original_height, original_width = image_processor(
            image, image_shape, min_image_shape=min_image_shape,
            upscale_interpolation=upscale_interpolation,
            downscale_interpolation=downscale_interpolation,
        )

        if image is None:
            return

        # Put the processed sample in the queue
        data_queue.put({
            "url": url,
            "caption": caption,
            "image": image,
            "original_height": original_height,
            "original_width": original_width,
        })
        
    except Exception as e:
        # Log the error
        print(f"Error mapping image sample {url}: {e}")


def map_video_sample(
    url: str,
    caption: str,
    data_queue: Queue,
    frame_size: int = 256,
    min_frame_size: int = 128,
    num_frames: int = 16,
    timeout: int = 30,
    retries: int = 3,
    upscale_interpolation: int = cv2.INTER_CUBIC,
    downscale_interpolation: int = cv2.INTER_AREA,
    video_processor: Callable = default_video_processor,
):
    """Process a single video sample and put it in the queue.
    
    Args:
        url: URL of the video.
        caption: Caption for the video.
        data_queue: Queue to put the processed sample in.
        frame_size: Target size for each frame.
        min_frame_size: Minimum acceptable frame size.
        num_frames: Target number of frames.
        timeout: Timeout for video fetching.
        retries: Number of retries for video fetching.
        upscale_interpolation: Interpolation method for upscaling.
        downscale_interpolation: Interpolation method for downscaling.
        video_processor: Function to process the video.
    """
    try:
        # Fetch the video frames
        frames = fetch_single_video(url, timeout=timeout, retries=retries, max_frames=num_frames*2)
        if frames is None or len(frames) == 0:
            return

        # Process the video
        video, original_height, original_width = video_processor(
            frames, frame_size, min_frame_size=min_frame_size,
            num_frames=num_frames,
            upscale_interpolation=upscale_interpolation,
            downscale_interpolation=downscale_interpolation,
        )

        if video is None:
            return

        # Put the processed sample in the queue
        data_queue.put({
            "url": url,
            "caption": caption,
            "video": video,
            "original_height": original_height,
            "original_width": original_width,
        })
        
    except Exception as e:
        # Log the error
        print(f"Error mapping video sample {url}: {e}")


def default_feature_extractor(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features from a sample.
    
    Args:
        sample: Sample to extract features from.
        
    Returns:
        Dictionary with extracted url and caption.
    """
    # Extract URL
    url = None
    for key in ["url", "URL", "image_url", "video_url"]:
        if key in sample:
            url = sample[key]
            break
    
    if url is None:
        print("No URL found in sample, keys:", sample.keys())
        return {"url": None, "caption": None}
    
    # Extract caption
    caption = None
    for key in ["caption", "CAPTION", "txt", "TEXT", "text"]:
        if key in sample and sample[key] is not None:
            caption = sample[key]
            break
    
    if caption is None:
        caption = "No caption available"
        
    return {
        "url": url,
        "caption": caption,
    }
    

def map_batch(
    batch: Dict[str, Any],
    data_queue: Queue,
    media_type: str = "image",
    num_threads: int = 256,
    image_shape: Tuple[int, int] = (256, 256),
    min_image_shape: Tuple[int, int] = (128, 128),
    frame_size: int = 256,
    min_frame_size: int = 128,
    num_frames: int = 16,
    timeout: int = 15,
    retries: int = 3,
    image_processor: Callable = default_image_processor,
    video_processor: Callable = default_video_processor,
    upscale_interpolation: int = cv2.INTER_CUBIC,
    downscale_interpolation: int = cv2.INTER_AREA,
    feature_extractor: Callable = default_feature_extractor,
):
    """Map a batch of samples and process them in parallel.
    
    Args:
        batch: Batch of samples to process.
        data_queue: Queue to put processed samples in.
        media_type: Type of media ("image" or "video").
        num_threads: Number of threads to use for processing.
        image_shape: Target shape for images.
        min_image_shape: Minimum acceptable shape for images.
        frame_size: Target size for video frames.
        min_frame_size: Minimum acceptable size for video frames.
        num_frames: Target number of frames for videos.
        timeout: Timeout for fetching.
        retries: Number of retries for fetching.
        image_processor: Function to process images.
        video_processor: Function to process videos.
        upscale_interpolation: Interpolation method for upscaling.
        downscale_interpolation: Interpolation method for downscaling.
        feature_extractor: Function to extract features from samples.
    """
    try:
        # Choose mapping function based on media type
        if media_type == "video":
            map_func = partial(
                map_video_sample,
                data_queue=data_queue,
                frame_size=frame_size,
                min_frame_size=min_frame_size,
                num_frames=num_frames,
                timeout=timeout,
                retries=retries,
                video_processor=video_processor,
                upscale_interpolation=upscale_interpolation,
                downscale_interpolation=downscale_interpolation,
            )
        else:  # Default to image
            map_func = partial(
                map_image_sample,
                data_queue=data_queue,
                image_shape=image_shape,
                min_image_shape=min_image_shape,
                timeout=timeout,
                retries=retries,
                image_processor=image_processor,
                upscale_interpolation=upscale_interpolation,
                downscale_interpolation=downscale_interpolation,
            )
        
        # Extract features from batch
        features = feature_extractor(batch)
        urls, captions = features["url"], features["caption"]
        
        if urls is None or captions is None:
            return
        
        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(map_func, urls, captions)
    
    except Exception as e:
        # Log the error
        print(f"Error mapping batch: {e}")
        traceback.print_exc()


def parallel_media_loader(
    dataset: Dataset,
    data_queue: Queue,
    media_type: str = "image",
    num_workers: int = 8,
    image_shape: Tuple[int, int] = (256, 256),
    min_image_shape: Tuple[int, int] = (128, 128),
    frame_size: int = 256,
    min_frame_size: int = 128,
    num_frames: int = 16,
    num_threads: int = 256,
    timeout: int = 15,
    retries: int = 3,
    image_processor: Callable = default_image_processor,
    video_processor: Callable = default_video_processor,
    upscale_interpolation: int = cv2.INTER_CUBIC,
    downscale_interpolation: int = cv2.INTER_AREA,
    feature_extractor: Callable = default_feature_extractor,
):
    """Load and process media from a dataset in parallel.
    
    Args:
        dataset: Dataset to load from.
        data_queue: Queue to put processed samples in.
        media_type: Type of media ("image" or "video").
        num_workers: Number of worker processes.
        image_shape: Target shape for images.
        min_image_shape: Minimum acceptable shape for images.
        frame_size: Target size for video frames.
        min_frame_size: Minimum acceptable size for video frames.
        num_frames: Target number of frames for videos.
        num_threads: Number of threads per worker.
        timeout: Timeout for fetching.
        retries: Number of retries for fetching.
        image_processor: Function to process images.
        video_processor: Function to process videos.
        upscale_interpolation: Interpolation method for upscaling.
        downscale_interpolation: Interpolation method for downscaling.
        feature_extractor: Function to extract features from samples.
    """
    # Create mapping function
    map_batch_fn = partial(
        map_batch,
        data_queue=data_queue,
        media_type=media_type,
        num_threads=num_threads,
        image_shape=image_shape,
        min_image_shape=min_image_shape,
        frame_size=frame_size,
        min_frame_size=min_frame_size,
        num_frames=num_frames,
        timeout=timeout,
        retries=retries,
        image_processor=image_processor,
        video_processor=video_processor,
        upscale_interpolation=upscale_interpolation,
        downscale_interpolation=downscale_interpolation,
        feature_extractor=feature_extractor
    )
    
    # Calculate shard length
    shard_len = len(dataset) // num_workers
    print(f"Local Shard length: {shard_len}")
    
    # Process dataset in parallel
    with multiprocessing.Pool(num_workers) as pool:
        iteration = 0
        while True:
            # Create shards for each worker
            shards = [dataset[i*shard_len:(i+1)*shard_len] for i in range(num_workers)]
            print(f"Mapping {len(shards)} shards")
            
            # Process shards in parallel
            pool.map(map_batch_fn, shards)
            
            # Shuffle dataset for next iteration
            iteration += 1
            print(f"Shuffling dataset with seed {iteration}")
            dataset = dataset.shuffle(seed=iteration)


class MediaBatchIterator:
    """Iterator for batches of media samples."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        media_type: str = "image",
        image_shape: Tuple[int, int] = (256, 256),
        min_image_shape: Tuple[int, int] = (128, 128),
        frame_size: int = 256,
        min_frame_size: int = 128,
        num_frames: int = 16,
        num_workers: int = 8,
        num_threads: int = 256,
        timeout: int = 15,
        retries: int = 3,
        image_processor: Callable = default_image_processor,
        video_processor: Callable = default_video_processor,
        upscale_interpolation: int = cv2.INTER_CUBIC,
        downscale_interpolation: int = cv2.INTER_AREA,
        feature_extractor: Callable = default_feature_extractor,
        resource_manager: Optional[ResourceManager] = None,
    ):
        """Initialize a media batch iterator.
        
        Args:
            dataset: Dataset to iterate over.
            batch_size: Batch size.
            media_type: Type of media ("image" or "video").
            image_shape: Target shape for images.
            min_image_shape: Minimum acceptable shape for images.
            frame_size: Target size for video frames.
            min_frame_size: Minimum acceptable size for video frames.
            num_frames: Target number of frames for videos.
            num_workers: Number of worker processes.
            num_threads: Number of threads per worker.
            timeout: Timeout for fetching.
            retries: Number of retries for fetching.
            image_processor: Function to process images.
            video_processor: Function to process videos.
            upscale_interpolation: Interpolation method for upscaling.
            downscale_interpolation: Interpolation method for downscaling.
            feature_extractor: Function to extract features from samples.
            resource_manager: Resource manager to use. Will create one if None.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.media_type = media_type
        
        # Create or use resource manager
        self.resource_manager = resource_manager or ResourceManager()
        self.data_queue = self.resource_manager.get_data_queue()
        
        # Start loader thread
        loader = partial(
            parallel_media_loader,
            data_queue=self.data_queue,
            media_type=media_type,
            num_workers=num_workers,
            image_shape=image_shape,
            min_image_shape=min_image_shape,
            frame_size=frame_size,
            min_frame_size=min_frame_size,
            num_frames=num_frames,
            num_threads=num_threads,
            timeout=timeout,
            retries=retries,
            image_processor=image_processor,
            video_processor=video_processor,
            upscale_interpolation=upscale_interpolation,
            downscale_interpolation=downscale_interpolation,
            feature_extractor=feature_extractor
        )
        
        # Start loader in background thread
        self.thread = threading.Thread(target=loader, args=(dataset,), daemon=True)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next batch of samples."""
        def fetcher(_):
            try:
                return self.data_queue.get(timeout=60)  # Add timeout to prevent hanging
            except:
                # Return a dummy sample on timeout
                if self.media_type == "video":
                    return {
                        "url": "timeout",
                        "caption": "Timeout occurred while waiting for sample",
                        "video": np.zeros((4, 32, 32, 3), dtype=np.uint8),
                        "original_height": 32,
                        "original_width": 32,
                    }
                else:
                    return {
                        "url": "timeout",
                        "caption": "Timeout occurred while waiting for sample",
                        "image": np.zeros((32, 32, 3), dtype=np.uint8),
                        "original_height": 32,
                        "original_width": 32,
                    }
                
        # Fetch batch in parallel
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            batch = list(executor.map(fetcher, range(self.batch_size)))
            
        return batch

    def __len__(self):
        """Get the number of batches in the dataset."""
        return len(self.dataset) // self.batch_size


def default_image_collate(batch):
    """Default collate function for image batches.
    
    Args:
        batch: Batch of samples to collate.
        
    Returns:
        Collated batch.
    """
    urls = [sample["url"] for sample in batch]
    captions = [sample["caption"] for sample in batch]
    
    # Check if all images have the same shape
    image_shapes = [sample["image"].shape for sample in batch]
    if len(set(str(shape) for shape in image_shapes)) > 1:
        # Get max height and width
        max_height = max(shape[0] for shape in image_shapes)
        max_width = max(shape[1] for shape in image_shapes)
        
        # Resize all images to the same shape
        images = []
        for sample in batch:
            image = sample["image"]
            height, width = image.shape[:2]
            
            if height != max_height or width != max_width:
                # Pad with white
                padded_image = np.ones((max_height, max_width, 3), dtype=image.dtype) * 255
                padded_image[:height, :width] = image
                images.append(padded_image)
            else:
                images.append(image)
                
        images = np.stack(images, axis=0)
    else:
        # All images have the same shape, just stack them
        images = np.stack([sample["image"] for sample in batch], axis=0)
    
    return {
        "url": urls,
        "caption": captions,
        "image": images,
    }


def default_video_collate(batch):
    """Default collate function for video batches.
    
    Args:
        batch: Batch of samples to collate.
        
    Returns:
        Collated batch.
    """
    urls = [sample["url"] for sample in batch]
    captions = [sample["caption"] for sample in batch]
    
    # Check if all videos have the same shape
    video_shapes = [sample["video"].shape for sample in batch]
    if len(set(str(shape) for shape in video_shapes)) > 1:
        # Get max dimensions
        max_frames = max(shape[0] for shape in video_shapes)
        max_height = max(shape[1] for shape in video_shapes)
        max_width = max(shape[2] for shape in video_shapes)
        
        # Resize all videos to the same shape
        videos = []
        for sample in batch:
            video = sample["video"]
            num_frames, height, width = video.shape[:3]
            
            if num_frames != max_frames or height != max_height or width != max_width:
                # Create a new video tensor with the max dimensions
                padded_video = np.zeros((max_frames, max_height, max_width, 3), dtype=video.dtype)
                
                # Copy the original video frames
                padded_video[:num_frames, :height, :width] = video
                
                # If we need more frames, duplicate the last frame
                if num_frames < max_frames:
                    padded_video[num_frames:] = padded_video[num_frames-1:num_frames]
                    
                videos.append(padded_video)
            else:
                videos.append(video)
                
        videos = np.stack(videos, axis=0)
    else:
        # All videos have the same shape, just stack them
        videos = np.stack([sample["video"] for sample in batch], axis=0)
    
    return {
        "url": urls,
        "caption": captions,
        "video": videos,
    }


def get_default_collate(media_type="image"):
    """Get the default collate function for a media type.
    
    Args:
        media_type: Type of media ("image" or "video").
        
    Returns:
        Collate function for the specified media type.
    """
    if media_type == "video":
        return default_video_collate
    else:  # Default to image
        return default_image_collate


def dataMapper(map: Dict[str, Any]):
    """Create a function to map dataset samples to a standard format.
    
    Args:
        map: Dictionary mapping standard keys to dataset-specific keys.
        
    Returns:
        Function that maps a sample to the standard format.
    """
    def _map(sample) -> Dict[str, Any]:
        return {
            "url": sample[map["url"]],
            "caption": sample[map["caption"]],
        }
    return _map


class OnlineStreamingDataLoader:
    """Data loader for streaming media data from online sources."""
    
    def __init__(
        self,
        dataset,
        batch_size=64,
        media_type="image",
        image_shape=(256, 256),
        min_image_shape=(128, 128),
        frame_size=256,
        min_frame_size=128,
        num_frames=16,
        num_workers=16,
        num_threads=512,
        default_split="all",
        pre_map_maker=dataMapper,
        pre_map_def={
            "url": "URL",
            "caption": "TEXT",
        },
        global_process_count=1,
        global_process_index=0,
        prefetch=1000,
        collate_fn=None,
        timeout=15,
        retries=3,
        image_processor=default_image_processor,
        video_processor=default_video_processor,
        upscale_interpolation=cv2.INTER_CUBIC,
        downscale_interpolation=cv2.INTER_AREA,
        feature_extractor=default_feature_extractor,
        resource_manager=None,
    ):
        """Initialize an online streaming data loader.
        
        Args:
            dataset: Dataset to load from, can be a path or a dataset object.
            batch_size: Batch size.
            media_type: Type of media ("image" or "video").
            image_shape: Target shape for images.
            min_image_shape: Minimum acceptable shape for images.
            frame_size: Target size for video frames.
            min_frame_size: Minimum acceptable size for video frames.
            num_frames: Target number of frames for videos.
            num_workers: Number of worker processes.
            num_threads: Number of threads per worker.
            default_split: Default split to use when loading datasets.
            pre_map_maker: Function to create a mapping function.
            pre_map_def: Default mapping definition.
            global_process_count: Total number of processes.
            global_process_index: Index of this process.
            prefetch: Number of batches to prefetch.
            collate_fn: Function to collate samples into batches.
            timeout: Timeout for fetching.
            retries: Number of retries for fetching.
            image_processor: Function to process images.
            video_processor: Function to process videos.
            upscale_interpolation: Interpolation method for upscaling.
            downscale_interpolation: Interpolation method for downscaling.
            feature_extractor: Function to extract features from samples.
            resource_manager: Resource manager to use.
        """
        # Load dataset from path if needed
        if isinstance(dataset, str):
            dataset_path = dataset
            print(f"Loading dataset from path: {dataset_path}")
            if "gs://" in dataset:
                dataset = load_from_disk(dataset_path)
            else:
                dataset = load_dataset(dataset_path, split=default_split)
        elif isinstance(dataset, list):
            if isinstance(dataset[0], str):
                print("Loading multiple datasets from paths")
                dataset = [
                    load_from_disk(dataset_path) if "gs://" in dataset_path 
                    else load_dataset(dataset_path, split=default_split) 
                    for dataset_path in dataset
                ]
            print(f"Concatenating {len(dataset)} datasets")
            dataset = concatenate_datasets(dataset)
            dataset = dataset.shuffle(seed=0)
            
        # Shard dataset for distributed training
        self.dataset = dataset.shard(
            num_shards=global_process_count, index=global_process_index)
        print(f"Dataset length: {len(dataset)}")
        
        # Get or create resource manager
        self.resource_manager = resource_manager or ResourceManager()
        
        # Choose default collate function if not provided
        if collate_fn is None:
            collate_fn = get_default_collate(media_type)
        
        # Create media batch iterator
        self.iterator = MediaBatchIterator(
            self.dataset,
            batch_size=batch_size,
            media_type=media_type,
            image_shape=image_shape,
            min_image_shape=min_image_shape,
            frame_size=frame_size,
            min_frame_size=min_frame_size,
            num_frames=num_frames,
            num_workers=num_workers,
            num_threads=num_threads,
            timeout=timeout,
            retries=retries,
            image_processor=image_processor,
            video_processor=video_processor,
            upscale_interpolation=upscale_interpolation,
            downscale_interpolation=downscale_interpolation,
            feature_extractor=feature_extractor,
            resource_manager=self.resource_manager,
        )
        
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        # Create batch queue for prefetching
        self.batch_queue = queue.Queue(prefetch)
        
        # Start batch loader thread
        def batch_loader():
            try:
                for batch in self.iterator:
                    try:
                        if batch:
                            self.batch_queue.put(collate_fn(batch))
                    except Exception as e:
                        print(f"Error collating batch: {e}")
                        traceback.print_exc()
            except Exception as e:
                print(f"Error in batch loader thread: {e}")
                traceback.print_exc()

        self.loader_thread = threading.Thread(target=batch_loader, daemon=True)
        self.loader_thread.start()

    def __iter__(self):
        """Get an iterator for the data loader."""
        return self

    def __next__(self):
        """Get the next batch."""
        try:
            return self.batch_queue.get(timeout=60)  # Add timeout to prevent hanging
        except queue.Empty:
            if not self.loader_thread.is_alive():
                raise StopIteration("Loader thread died")
            print("Timeout waiting for batch, retrying...")
            return self.__next__()

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.dataset)