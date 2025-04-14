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


# ----------------------------------------------------------------------------------
# Video augmentation utilities
# ----------------------------------------------------------------------------------

def video_frame_augmenter(frame, frame_size, method=cv2.INTER_AREA):
    """Basic frame augmentation: convert color and resize."""
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # Check if it's an RGB frame
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=method)
        return frame
    return None


def video_clip_augmenter(clip, frame_size, num_frames=16, method=cv2.INTER_AREA):
    """Augment a video clip by sampling frames and resizing them."""
    # Sample frames evenly from the clip
    if len(clip) < num_frames:
        # If not enough frames, duplicate some frames
        indices = np.linspace(0, len(clip) - 1, num_frames, dtype=int)
    else:
        # Sample frames evenly
        indices = np.linspace(0, len(clip) - 1, num_frames, dtype=int)
    
    # Get frames at the sampled indices
    sampled_frames = [clip[i] for i in indices]
    
    # Resize frames
    resized_frames = []
    for frame in sampled_frames:
        resized_frame = video_frame_augmenter(frame, frame_size, method)
        if resized_frame is not None:
            resized_frames.append(resized_frame)
    
    if len(resized_frames) < num_frames:
        # If some frames were invalid, duplicate the last valid frame
        while len(resized_frames) < num_frames:
            resized_frames.append(resized_frames[-1])
    
    # Stack frames into a video clip tensor [num_frames, height, width, channels]
    return np.stack(resized_frames, axis=0)


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
    
    def __init__(self, 
                 directory: str, 
                 extensions: List[str] = ['.mp4', '.avi', '.mov', '.webm'],
                 caption_file: str = None):
        """Initialize a local video data source.
        
        Args:
            directory: Directory containing video files.
            extensions: List of valid video file extensions.
            caption_file: Path to a file mapping video filenames to captions.
        """
        self.directory = directory
        self.extensions = extensions
        self.caption_file = caption_file
        self._captions = {}
        
        if caption_file and os.path.exists(caption_file):
            with open(caption_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        video_name = parts[0]
                        caption = parts[1]
                        self._captions[video_name] = caption
    
    def get_source(self, path_override: str = None) -> List[Dict[str, Any]]:
        """Get the local video data source.
        
        Args:
            path_override: Override directory path.
            
        Returns:
            A list of dictionaries with video paths and captions.
        """
        video_dir = path_override or self.directory
        video_files = []
        
        for root, _, files in os.walk(video_dir):
            for file in files:
                if any(file.endswith(ext) for ext in self.extensions):
                    video_path = os.path.join(root, file)
                    video_name = os.path.basename(file)
                    caption = self._captions.get(video_name, f"A video of {os.path.splitext(video_name)[0]}")
                    
                    video_files.append({
                        'video_path': video_path,
                        'caption': caption
                    })
        
        return video_files


# ----------------------------------------------------------------------------------
# Video Augmenter
# ----------------------------------------------------------------------------------

class VideoAugmenter(DataAugmenter):
    """Augmenter for video datasets."""
    
    def __init__(self, 
                 num_frames: int = 16, 
                 preprocess_fn: Callable = None,
                 caption_key: str = 'caption'):
        """Initialize a video augmenter.
        
        Args:
            num_frames: Number of frames to sample from each video.
            preprocess_fn: Optional function to preprocess video frames.
            caption_key: Key for caption field in the dataset.
        """
        self.num_frames = num_frames
        self.preprocess_fn = preprocess_fn
        self.caption_key = caption_key
    
    def create_transform(self, frame_size: int = 256, method: Any = cv2.INTER_AREA) -> Callable[[], pygrain.MapTransform]:
        """Create a transform for video datasets.
        
        Args:
            frame_size: Size to scale video frames to.
            method: Interpolation method for resizing.
            
        Returns:
            A callable that returns a pygrain.MapTransform.
        """
        num_frames = self.num_frames
        caption_key = self.caption_key
        
        class VideoTransform(pygrain.MapTransform):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokenize = AutoTextTokenizer(tensor_type="np")
                self.clip_augmenter = partial(
                    video_clip_augmenter, 
                    frame_size=frame_size, 
                    num_frames=num_frames, 
                    method=method
                )
                
            def _load_video(self, video_path):
                """Load video frames from a file path."""
                cap = cv2.VideoCapture(video_path)
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                cap.release()
                return frames
            
            def map(self, element) -> Dict[str, jnp.array]:
                # Handle different possible input formats
                video_frames = None
                caption = None
                
                if 'video_path' in element:
                    # Local video file path
                    video_frames = self._load_video(element['video_path'])
                    caption = element.get(caption_key, "")
                elif 'video' in element:
                    # Pre-loaded video frames
                    video_frames = element['video']
                    caption = element.get(caption_key, "")
                elif 'sequence' in element:
                    # TFDS video format
                    video_frames = element['sequence']
                    if caption_key in element:
                        caption = element[caption_key]
                    else:
                        caption = str(element.get('label', "A video"))
                
                if video_frames is None or len(video_frames) == 0:
                    # Fallback if video is empty or couldn't be loaded
                    video_tensor = np.zeros((num_frames, frame_size, frame_size, 3), dtype=np.uint8)
                    caption = caption or "Empty video"
                else:
                    video_tensor = self.clip_augmenter(video_frames)
                
                # Process caption
                results = self.tokenize(caption)
                
                return {
                    "video": video_tensor,
                    "text": {
                        "input_ids": results['input_ids'][0],
                        "attention_mask": results['attention_mask'][0],
                    }
                }
        
        return VideoTransform


# ----------------------------------------------------------------------------------
# Helper functions for video datasets
# ----------------------------------------------------------------------------------

def load_video_from_path(video_path: str) -> List[np.ndarray]:
    """Load video frames from a file path.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        List of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames


def create_video_dataset_from_directory(
    directory: str,
    extensions: List[str] = ['.mp4', '.avi', '.mov', '.webm'],
    caption_file: str = None,
    frame_size: int = 256,
    num_frames: int = 16,
) -> Tuple[List[Dict[str, Any]], VideoAugmenter]:
    """Create a video dataset from a directory of video files.
    
    Args:
        directory: Directory containing video files.
        extensions: List of valid video file extensions.
        caption_file: Path to a file mapping video filenames to captions.
        frame_size: Size to scale video frames to.
        num_frames: Number of frames to sample from each video.
        
    Returns:
        Tuple of (dataset, augmenter) for the video dataset.
    """
    source = VideoLocalSource(
        directory=directory,
        extensions=extensions,
        caption_file=caption_file
    )
    
    augmenter = VideoAugmenter(
        num_frames=num_frames
    )
    
    dataset = source.get_source()
    return dataset, augmenter