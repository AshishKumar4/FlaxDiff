"""
Voxceleb Data preparation pipeline
1. Resample audio to 16kHz and video to 25 fps
2. Detect scene and segment the video when the scene changes
3. Do face detection and landmark detection and use the landmarks to crop and affine transform the face
4. Do another round of face detection to filter out bad faces
5. Do Audio video synchronization using syncnet
6. Filter out low quality videos using Hyper IQA
"""
from sources import DataSource
from dataclasses import dataclass
from video_reader import PyVideoReader
import os
from typing import List, Tuple
import numpy as np
import queue

def read_video_rsreader(video_path, fast=False):
    vr = PyVideoReader(video_path)
    return vr.decode_fast() if fast else vr.decode()

def gather_video_paths_fast(input_dir, output_dir):
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.endswith(".mp4"):
                rel_path = os.path.relpath(root, input_dir)
                video_input = os.path.join(root, file)
                video_output_dir = os.path.join(output_dir, rel_path)
                video_output = os.path.join(video_output_dir, file)
                if not os.path.isfile(video_output):
                    video_paths.append((video_input, video_output))
    # Sort the paths to ensure consistent order
    video_paths.sort()
    return video_paths

@dataclass
class VideoObjects:
    video_input: str
    video_output: str
    video_frames: List[np.ndarray]

# Define the data source
class AVDataSource(DataSource[VideoObjects]):
    def __init__(self, video_paths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_paths = queue.Queue()
        for video_input, video_output in video_paths:
            self.video_paths.put((video_input, video_output))
        
    def fetch(self) -> VideoObjects:
        if not self.video_paths:
            return None
        video_input, video_output = self.video_paths.get()
        video_frames = read_video_rsreader(video_input, fast=True)
        if video_frames is None:
            return None
        # Emit the video frames and paths
        return VideoObjects(video_input=video_input, video_output=video_output, video_frames=video_frames)
    
    def close(self):
        return super().close()
    
