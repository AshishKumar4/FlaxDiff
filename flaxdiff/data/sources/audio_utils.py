"""
Audio utilities for efficiently loading audio data from video files.
This module provides alternatives to decord's AudioReader/AVReader (which have memory leaks).
"""

import os
import tempfile
import subprocess
import numpy as np
from typing import Tuple, Optional, Union


def read_audio_ffmpeg(
    video_path: str, 
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Extract audio from video file using ffmpeg subprocess calls.
    
    Args:
        video_path: Path to the video file.
        start_time: Start time in seconds (optional).
        duration: Duration to extract in seconds (optional).
        target_sr: Target sample rate for the audio.
        
    Returns:
        Tuple of (audio_data, sample_rate) where audio_data is a numpy array.
    """
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Build the ffmpeg command
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        # Add time parameters if specified
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
            
        if duration is not None:
            cmd.extend(['-t', str(duration)])
            
        # Set output parameters (mono, target sample rate)
        cmd.extend([
            '-ac', '1',  # mono
            '-ar', str(target_sr),  # sample rate
            '-vn',  # no video
            '-f', 'wav',  # wav format
            tmp_path
        ])
        
        # Execute the command
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        # Read the audio file using numpy
        audio_data = np.fromfile(tmp_path, np.int16).astype(np.float32) / 32768.0  # Convert to float in [-1, 1]
        
        return audio_data, target_sr
        
    finally:
        # Always clean up the temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def read_audio_moviepy(
    video_path: str,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Extract audio from video file using moviepy.
    Requires the moviepy package: pip install moviepy
    
    Args:
        video_path: Path to the video file.
        start_time: Start time in seconds (optional).
        duration: Duration to extract in seconds (optional).
        target_sr: Target sample rate for the audio.
        
    Returns:
        Tuple of (audio_data, sample_rate) where audio_data is a numpy array.
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        raise ImportError("moviepy is not installed. Install it with 'pip install moviepy'")
    
    # Load video file
    if start_time is not None or duration is not None:
        start_t = start_time if start_time is not None else 0
        end_t = start_t + duration if duration is not None else None
        video = VideoFileClip(video_path).subclipped(start_t, end_t)
    else:
        video = VideoFileClip(video_path)
    # Extract audio
    audio = video.audio.with_fps(target_sr)
    
    # Get audio data
    audio_data = audio.to_soundarray()
    
    # Convert to mono if stereo
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Clean up
    video.close()
    
    return audio_data, target_sr


# Helper function to choose the best available method
def read_audio(
    video_path: str,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    target_sr: int = 16000,
    method: str = 'ffmpeg'
) -> Tuple[np.ndarray, int]:
    """
    Extract audio from video file using the specified method.
    
    Args:
        video_path: Path to the video file.
        start_time: Start time in seconds (optional).
        duration: Duration to extract in seconds (optional).
        target_sr: Target sample rate for the audio.
        method: Method to use ('ffmpeg' or 'moviepy').
        
    Returns:
        Tuple of (audio_data, sample_rate) where audio_data is a numpy array.
    """
    if method == 'moviepy':
        return read_audio_moviepy(video_path, start_time, duration, target_sr)
    else:  # default to ffmpeg
        return read_audio_ffmpeg(video_path, start_time, duration, target_sr)
