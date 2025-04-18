from abc import ABC, abstractmethod
import grain.python as pygrain
from typing import Dict, Any, Callable, List, Optional
import jax.numpy as jnp
from functools import partial


class DataSource(ABC):
    """Base class for all data sources in FlaxDiff."""
    
    @abstractmethod
    def get_source(self, path_override: str) -> Any:
        """Return the data source object.
        
        Args:
            path_override: Path to the dataset, overriding the default.
            
        Returns:
            A data source object compatible with grain or other loaders.
        """
        pass
    
    @staticmethod
    def create(source_type: str, **kwargs) -> 'DataSource':
        """Factory method to create a data source of the specified type.
        
        Args:
            source_type: Type of the data source ("image", "video", etc.)
            **kwargs: Additional arguments for the specific data source.
            
        Returns:
            An instance of a DataSource subclass.
        """
        from .images import ImageTFDSSource, ImageGCSSource, CombinedImageGCSSource
        from .videos import VideoTFDSSource, VideoLocalSource
        
        source_map = {
            "image_tfds": ImageTFDSSource,
            "image_gcs": ImageGCSSource,
            "image_combined_gcs": CombinedImageGCSSource,
            "video_tfds": VideoTFDSSource,
            "video_local": VideoLocalSource
        }
        
        if source_type not in source_map:
            raise ValueError(f"Unknown source type: {source_type}")
        return source_map[source_type](**kwargs)


class DataAugmenter(ABC):
    """Base class for all data augmenters in FlaxDiff."""
    
    @abstractmethod
    def create_transform(self, **kwargs) -> Callable[[], pygrain.MapTransform]:
        """Create a transformation function for the data.
        
        Args:
            **kwargs: Additional arguments for the transformation.
            
        Returns:
            A callable that returns a pygrain.MapTransform instance.
        """
        pass
    
    @staticmethod
    def create(augmenter_type: str, **kwargs) -> 'DataAugmenter':
        """Factory method to create a data augmenter of the specified type.
        
        Args:
            augmenter_type: Type of the data augmenter ("image", "video", etc.)
            **kwargs: Additional arguments for the specific augmenter.
            
        Returns:
            An instance of a DataAugmenter subclass.
        """
        from .images import ImageTFDSAugmenter, ImageGCSAugmenter
        from .videos import VideoAugmenter
        
        augmenter_map = {
            "image_tfds": ImageTFDSAugmenter,
            "image_gcs": ImageGCSAugmenter,
            "video": VideoAugmenter
        }
        
        if augmenter_type not in augmenter_map:
            raise ValueError(f"Unknown augmenter type: {augmenter_type}")
        
        return augmenter_map[augmenter_type](**kwargs)


class MediaDataset:
    """A class combining a data source and an augmenter for a complete dataset."""
    
    def __init__(self, 
                 source: DataSource, 
                 augmenter: DataAugmenter,
                 media_type: str = "image"):
        """Initialize a MediaDataset.
        
        Args:
            source: The data source.
            augmenter: The data augmenter.
            media_type: Type of media ("image", "video", etc.)
        """
        self.source = source
        self.augmenter = augmenter
        self.media_type = media_type
    
    def get_source(self, path_override: str) -> Any:
        """Get the data source.
        
        Args:
            path_override: Path to override the default data source path.
            
        Returns:
            A data source object.
        """
        return self.source.get_source(path_override)
    
    def get_augmenter(self, **kwargs) -> Callable[[], pygrain.MapTransform]:
        """Get the augmenter transformation.
        
        Args:
            **kwargs: Additional arguments for the augmenter.
            
        Returns:
            A callable that returns a pygrain.MapTransform instance.
        """
        return self.augmenter.create_transform(**kwargs)