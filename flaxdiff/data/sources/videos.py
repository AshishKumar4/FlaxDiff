import cv2
import jax.numpy as jnp
import grain.python as pygrain
from flaxdiff.utils import AutoTextTokenizer
from typing import Dict, Any, Callable, List, Optional, Tuple
import random
import os
import numpy as np
from functools import partial
from .base import DataSource, DataAugmenter
import numpy as np
import subprocess
import shutil
from .av_utils import read_av_random_clip

# ----------------------------------------------------------------------------------
# Video augmentation utilities
# ----------------------------------------------------------------------------------
def gather_video_paths_iter(input_dir, extensions=['.mp4', '.avi', '.mov', '.webm']):
   # Ensure extensions have dots at the beginning and are lowercase
    extensions = {ext.lower() if ext.startswith('.') else f'.{ext}'.lower() for ext in extensions}
        
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            _, ext = os.path.splitext(file)
            if ext.lower() in extensions:
                video_input = os.path.join(root, file)
                yield video_input

def gather_video_paths(input_dir, extensions=['.mp4', '.avi', '.mov', '.webm']):
    """Gather video paths from a directory."""
    video_paths = []
    for video_input in gather_video_paths_iter(input_dir, extensions):
        video_paths.append(video_input)
        
    # Sort the video paths
    video_paths.sort()
    return video_paths

# ----------------------------------------------------------------------------------
# TFDS Video Source
# ----------------------------------------------------------------------------------

class VideoTFDSSource(DataSource):
    """Data source for TensorFlow Datasets (TFDS) video datasets."""
    
    def __init__(self, name: str, use_tf: bool = True, split: str = "train"):
        """Initialize a TFDS video data source.
        
        Args:
            name: Name of the TFDS dataset.
            use_tf: Whether to use TensorFlow for loading.
            split: Dataset split to use.
        """
        self.name = name
        self.use_tf = use_tf
        self.split = split
    
    def get_source(self, path_override: str) -> Any:
        """Get the TFDS video data source.
        
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


# ----------------------------------------------------------------------------------
# Local Video Source
# ----------------------------------------------------------------------------------

class VideoLocalSource(DataSource):
    """Data source for local video files."""
    
    def __init__(
        self, 
        directory: str = "", 
        extensions: List[str] = ['.mp4', '.avi', '.mov', '.webm'],
        clear_cache: bool = False,
        cache_dir: Optional[str] = './cache',
    ):
        """Initialize a local video data source.
        
        Args:
            directory: Directory containing video files.
            extensions: List of valid video file extensions.
            clear_cache: Whether to clear the cache on initialization.
            cache_dir: Directory to cache video paths.
        """
        self.extensions = extensions
        self.cache_dir = cache_dir
        if directory:
            self.load_paths(directory, clear_cache)
    
    def load_paths(self, directory: str, clear_cache: bool = False):
        """Load video paths from a directory."""
        if self.directory == directory and not clear_cache:
            # If the directory hasn't changed and cache is not cleared, return cached paths
            return
        self.directory = directory
        
        # Use gather_video_paths to get all video paths and cache them 
        # in a local dictionary for future use
        
        # Generate a hash for the directory to use as a key
        self.directory_hash = hash(directory)

        # Check if the cache directory exists
        if os.path.exists(self.cache_dir):
            # Load cached video paths if available
            cache_file = os.path.join(self.cache_dir, f"video_paths_{self.directory_hash}.txt")
            import pickle
            if os.path.exists(cache_file) and not clear_cache:
                with open(cache_file, 'rb') as f:
                    video_paths = pickle.load(f)
                print(f"Loaded cached video paths from {cache_file}")
            else:
                # If no cache file, gather video paths and save them
                print(f"Cache file not found or clear_cache is True. Gathering video paths from {directory}")
                video_paths = gather_video_paths(directory, self.extensions)
                with open(cache_file, 'wb') as f:
                    pickle.dump(video_paths, f)
                print(f"Cached video paths to {cache_file}")
        
        self.video_paths = video_paths
    
    def get_source(self, path_override: str = None) -> List[Dict[str, Any]]:
        """Get the local video data source.
        
        Args:
            path_override: Override directory path.
            
        Returns:
            A list of dictionaries with video paths.
        """
        if path_override:
            self.load_paths(path_override)
            
        video_paths = self.video_paths
        dataset = []
        for video_path in video_paths:
            dataset.append({"video_path": video_path})
        return dataset

# ----------------------------------------------------------------------------------
# Video Augmenter
# ----------------------------------------------------------------------------------

class AudioVideoAugmenter(DataAugmenter):
    """Augmenter for audio-video datasets."""
    
    def __init__(self, 
                 preprocess_fn: Callable = None):
        """Initialize a AV augmenter.
        
        Args:
            num_frames: Number of frames to sample from each video.
            preprocess_fn: Optional function to preprocess video frames.
        """
        self.preprocess_fn = preprocess_fn
    
    def create_transform(
        self, 
        frame_size: int = 256, 
        sequence_length: int = 16,
        audio_frame_padding: int = 3,
        method: Any = cv2.INTER_AREA,
    ) -> Callable[[], pygrain.MapTransform]:
        """Create a transform for video datasets.
        
        Args:
            frame_size: Size to scale video frames to.
            sequence_length: Number of frames to sample from each video.
            method: Interpolation method for resizing.
            
        Returns:
            A callable that returns a pygrain.MapTransform.
        """
        num_frames = sequence_length
        
        class AudioVideoTransform(pygrain.RandomMapTransform):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokenize = AutoAudioTokenizer(tensor_type="np")
            
            def random_map(self, element, rng: np.random.Generator) -> Dict[str, jnp.array]:
                video_path = element["video_path"]
                random_seed = rng.integers(0, 2**32 - 1)
                # Read video frames
                framewise_audio, full_audio, video_frames = read_av_random_clip(
                    video_path,
                    num_frames=num_frames,
                    audio_frame_padding=audio_frame_padding,
                    random_seed=random_seed,
                )
                
                # Process caption
                results = self.tokenize(full_audio)
                
                return {
                    "video": video_frames,
                    "audio": {
                        "input_ids": results['input_ids'][0],
                        "attention_mask": results['attention_mask'][0],
                        "full_audio": full_audio,
                        "framewise_audio": framewise_audio,
                    }
                }
        
        return AudioVideoTransform


# ----------------------------------------------------------------------------------
# Helper functions for video datasets
# ----------------------------------------------------------------------------------

# def create_video_dataset_from_directory(
#     directory: str,
#     extensions: List[str] = ['.mp4', '.avi', '.mov', '.webm'],
#     frame_size: int = 256,
# ) -> Tuple[List[Dict[str, Any]], AudioVideoAugmenter]:
#     """Create a video dataset from a directory of video files.
    
#     Args:
#         directory: Directory containing video files.
#         extensions: List of valid video file extensions.
#         frame_size: Size to scale video frames to.
#         num_frames: Number of frames to sample from each video.
        
#     Returns:
#         Tuple of (dataset, augmenter) for the video dataset.
#     """
#     source = VideoLocalSource(
#         directory=directory,
#         extensions=extensions,
#     )
    
#     augmenter = AudioVideoAugmenter(
#         num_frames=num_frames
#     )
    
#     dataset = source.get_source()
#     return dataset, augmenter