import cv2
import jax.numpy as jnp
import grain.python as pygrain
from flaxdiff.utils import AutoAudioProcessor
from typing import Dict, Any, Callable, List, Optional
import hashlib
import os
import pickle
import numpy as np
from .base import DataSource, DataAugmenter
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
        self.directory = None
        self.video_paths = []
        if directory:
            self.load_paths(directory, clear_cache)

    def load_paths(self, directory: str, clear_cache: bool = False):
        """Scan `directory` for videos, caching the file list on disk.

        The cache key is a content hash of the directory path, not `hash()`,
        whose string salt changes every interpreter run.
        """
        if self.directory == directory and not clear_cache:
            return
        self.directory = directory

        cache_file = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            digest = hashlib.sha1(directory.encode("utf-8")).hexdigest()[:16]
            cache_file = os.path.join(self.cache_dir, f"video_paths_{digest}.pkl")

        if cache_file and os.path.exists(cache_file) and not clear_cache:
            with open(cache_file, 'rb') as f:
                video_paths = pickle.load(f)
            print(f"Loaded {len(video_paths)} cached video paths from {cache_file}")
        else:
            print(f"Gathering video paths from {directory}")
            video_paths = gather_video_paths(directory, self.extensions)
            if cache_file:
                with open(cache_file, 'wb') as f:
                    pickle.dump(video_paths, f)
                print(f"Cached {len(video_paths)} video paths to {cache_file}")

        self.video_paths = video_paths

    def get_source(self, path_override: str = None) -> List[Dict[str, Any]]:
        """Get the local video data source.

        Args:
            path_override: Directory to scan, overriding the configured one.

        Returns:
            A list of dictionaries with video paths.
        """
        if path_override:
            self.load_paths(path_override)
        if self.directory is None:
            raise ValueError(
                "VideoLocalSource has no directory to read: pass directory=... "
                "when constructing it, or a path override to get_source()."
            )
        return [{"video_path": video_path} for video_path in self.video_paths]

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
        audio_modelname: str = "facebook/wav2vec2-base-960h",
    ) -> Callable[[], pygrain.MapTransform]:
        """Create a transform for video datasets.

        Args:
            frame_size: Size to scale video frames to.
            sequence_length: Number of frames to sample from each video.
            audio_frame_padding: Extra audio frames kept on either side of the
                sampled clip.
            method: Interpolation method for resizing.
            audio_modelname: HF audio model whose feature extractor prepares
                the audio conditioning inputs.

        Returns:
            A callable that returns a pygrain.MapTransform. Records carrying a
            "caption" (e.g. from VoxCeleb2Source) keep it, so the video
            collate_fn can tokenize the prompt.
        """
        num_frames = sequence_length

        def resize_clip(frames: np.ndarray) -> np.ndarray:
            """Clip readers return native resolution; the model wants frame_size."""
            if frames.shape[1] == frame_size and frames.shape[2] == frame_size:
                return frames
            return np.stack([
                cv2.resize(frame, (frame_size, frame_size), interpolation=method)
                for frame in frames
            ])
        
        class AudioVideoTransform(pygrain.RandomMapTransform):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokenize = AutoAudioProcessor(
                    tensor_type="np", modelname=audio_modelname)
            
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
                video_frames = resize_clip(video_frames)

                # Feature-extract the audio; key names differ per model, so
                # pass the processor's output through untouched
                results = self.tokenize(full_audio)

                return {
                    "video": video_frames,
                    "caption": element.get("caption", ""),
                    "audio": {
                        **{key: value[0] for key, value in results.items()},
                        "full_audio": full_audio,
                        "framewise_audio": framewise_audio,
                    }
                }
        
        return AudioVideoTransform
