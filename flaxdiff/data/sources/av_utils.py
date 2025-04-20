"""
Functions for reading audio-video data without memory leaks.
"""
import cv2
import os
import shutil
import subprocess
import numpy as np
from typing import Tuple, Optional, Union, List
from .audio_utils import read_audio

def get_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()
    return fps

def read_video(video_path: str, change_fps=False, reader="rsreader"):
    temp_dir = None
    try:
        if change_fps:
            print(f"Changing fps of {video_path} to 25")
            temp_dir = "temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            command = (
                f"ffmpeg -loglevel error -y -nostdin -i {video_path} -r 25 -crf 18 {os.path.join(temp_dir, 'video.mp4')}"
            )
            subprocess.run(command, shell=True)
            target_video_path = os.path.join(temp_dir, "video.mp4")
        else:
            target_video_path = video_path

        if reader == "rsreader":
            return read_video_rsreader(target_video_path)
        elif reader == "rsreader_fast":
            return read_video_rsreader(target_video_path, fast=True)
        elif reader == "decord":
            return read_video_decord(target_video_path)
        elif reader == "opencv":
            return read_video_opencv(target_video_path)
        else:
            raise ValueError(f"Unknown reader: {reader}")
    finally:
        # Clean up temp directory when done
        if change_fps and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def read_video_decord(video_path: str):
    from decord import VideoReader
    vr = VideoReader(video_path)
    video_frames = vr[:].asnumpy()
    vr.seek(0)
    return video_frames

# Fixed OpenCV video reader - properly release resources
def read_video_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return np.array(frames)[:, :, :, ::-1]
    finally:
        cap.release()

def read_video_rsreader(video_path, fast=False):
    from video_reader import PyVideoReader
    vr = PyVideoReader(video_path)
    return vr.decode_fast() if fast else vr.decode()

def read_audio_decord(audio_path:str):
    from decord import AudioReader
    ar = AudioReader(audio_path)
    audio_frames = ar[:].asnumpy()
    ar.seek(0)
    return audio_frames

def read_av_decord(path: str, start: int=0, end: int = None, ctx=None):
    from decord import AVReader, cpu
    if ctx is None:
        ctx = cpu(0)
    vr = AVReader(path, ctx=ctx, sample_rate=16000)
    audio, video = vr[start:end]
    return audio, video.asnumpy()

def read_av_improved(
    path: str,
    start: int = 0,
    end: Optional[int] = None,
    fps: float = 25.0,
    target_sr: int = 16000,
    audio_method: str = 'ffmpeg'
) -> Tuple[Union[List, np.ndarray], np.ndarray]:
    """
    Read audio-video data with explicit cleanup and without memory leaks.
    Uses PyVideoReader for video (which doesn't have memory leaks) and 
    FFmpeg/moviepy for audio extraction.
    
    Args:
        path: Path to the video file.
        start: Start frame index.
        end: End frame index (or None to read until the end).
        fps: Video frames per second (used for audio timing).
        target_sr: Target audio sample rate.
        audio_method: Method to extract audio ('ffmpeg' or 'moviepy').
        
    Returns:
        Tuple of (audio_data, video_frames) where video_frames is a numpy array.
    """
    from video_reader import PyVideoReader
    # Calculate time information for audio extraction
    start_time = start / fps if start > 0 else 0
    duration = None
    if end is not None:
        duration = (end - start) / fps
    
    # Get video frames using PyVideoReader
    vr = PyVideoReader(path)
    video = vr.decode(start_frame=start, end_frame=None)
    
    # Get audio data using our custom audio utilities
    audio, _ = read_audio(
        path, 
        start_time=start_time, 
        duration=duration, 
        target_sr=target_sr,
        method=audio_method
    )
    
    # Convert audio to list for API compatibility with original read_av
    audio_list = list(audio)
    
    return audio_list, video

def read_av_moviepy(
    video_path: str,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    target_fps: float = 25.0,
    target_sr: int = 16000,
):
    """
    Read audio-video data using moviepy.
    
    Args:
        video_path: Path to the video file.
        start_idx: Start frame index (optional).
        end_idx: End frame index (optional).
        target_sr: Target sample rate for the audio.
        
    Returns:
        Tuple of (audio_data, video_frames) where video_frames is a numpy array.
    """
    # Use moviepy to read audio and video
    from moviepy import VideoFileClip
    
    video = VideoFileClip(video_path).with_fps(target_fps)
    
    # Convert frame indexes to time
    start_time = start_idx / target_fps if start_idx is not None else 0
    end_time = end_idx / target_fps if end_idx is not None else None
    video = video.subclipped(start_time, end_time)
    
    # Extract audio
    audio = video.audio.with_fps(target_sr)
    audio_data = audio.to_soundarray()
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    # Extract video frames
    video_frames = []
    for frame in video.iter_frames(fps=target_fps, dtype='uint8'):
        video_frames.append(frame)
    video_frames = np.array(video_frames)
    video.close()
    return audio_data, video_frames
def read_av_random_clip_moviepy(
    video_path: str,
    num_frames: int = 16,
    audio_frames_per_video_frame: int = 1,
    audio_frame_padding: int = 0,
    target_sr: int = 16000,
    target_fps: float = 25.0,
    random_seed: Optional[int] = None,
):
    """
    Read a random clip of audio and video frames.
    Works by first selecting a random appropriate start frame, then reading the specified number of frames (1, N, H, W, C).
    It then selects the audio clip corresponding to the video frames + some extra padding frames on either side. This is 
    of shape (1, P + N + P, K) where P is the padding, N is the number of video frames, and K is the audio data shape per frame.
    if audio_frames_per_video_frame > 1, It then also creates a tensor of shape (1, N, F, K) where F = audio_frames_per_video_frame.
    Otherwise (1, N, 1, K) is returned in the case of audio_frames_per_video_frame = 1.
    
    The final audio and video tensors are returned.
    Args:
        video_path: Path to the video file.
        num_frames: Number of video frames to read.
        audio_frames_per_video_frame: Number of audio frames per video frame.
        audio_frame_padding: Padding for audio frames.
        target_sr: Target sample rate for the audio.
        target_fps: Target frames per second for the video.
        random_seed: Random seed for reproducibility (optional).
        
    Returns:
        Tuple of (frame_wise_audio, full_padded_audio, video_frames) where video_frames is a numpy array.
    """
    from moviepy import VideoFileClip
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    # Load the video
    video = VideoFileClip(video_path).with_fps(target_fps)
    original_duration = video.duration
    total_frames = video.n_frames#int(original_duration * target_fps)
    
    # Calculate effective padding needed based on audio segmentation
    effective_padding = max(audio_frame_padding, (audio_frames_per_video_frame) // 2)

    # Make sure we have enough frames
    if total_frames < num_frames + 2 * effective_padding:
        raise ValueError(f"Video has only {total_frames} frames, but {num_frames + 2 * effective_padding} were requested (including effective padding)")

    # Adjust the range for start_idx to account for effective padding
    min_start_idx = effective_padding
    max_start_idx = total_frames - num_frames - effective_padding

    # Select a random start frame that allows for padding on both sides
    start_idx = np.random.randint(min_start_idx, max_start_idx) if max_start_idx > min_start_idx else min_start_idx
    end_idx = start_idx + num_frames
    
    # Convert to time
    video_start_time = start_idx / target_fps
    video_end_time = end_idx / target_fps
        
    # Extract video frames
    main_clip : VideoFileClip = video.subclipped(video_start_time, video_end_time)
    # Replace the video frame extraction with:
    frame_count = 0
    video_frames = []
    for frame in video.iter_frames(fps=target_fps, dtype='uint8'):
        if frame_count >= start_idx and frame_count < start_idx + num_frames:
            video_frames.append(frame)
        frame_count += 1
        if len(video_frames) == num_frames:
            break
        
    # Convert to numpy array
    video_frames = np.array(video_frames)
    
    audio_start_time = (start_idx - effective_padding) / target_fps
    audio_end_time = (end_idx + effective_padding) / target_fps
    num_audio_frames = num_frames + 2 * effective_padding
    audio_duration = audio_end_time - audio_start_time
    # Ensure we don't go out of bounds
    if audio_start_time < 0 or audio_end_time > original_duration:
        raise ValueError(f"Audio start time {audio_start_time} or end time {audio_end_time} is out of bounds for video duration {original_duration}")
    
    # Extract the subclip
    clip : VideoFileClip = video.subclipped(audio_start_time, audio_end_time)
    # Extract audio
    audio = clip.audio.with_fps(target_sr)
    audio_data = audio.to_soundarray()
    # Make sure len(audio_data) == (num_frames + 2 * effective_padding) * target_sr
    num_audio_samples_required = int(round(audio_duration * target_sr))
    if len(audio_data) < num_audio_samples_required:
        raise ValueError(f"Audio data length {len(audio_data)} is less than required {num_audio_samples_required}")
    audio_data = audio_data[:num_audio_samples_required]
    # Convert to mono if stereo
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    # Close the clips
    clip.close()
    main_clip.close()
    video.close()
    
    # Reshape audio data
    audio_data = np.array(audio_data)   # This is just 1D
    
    # Calculate dimensions for audio
    audio_data_per_frame = int(round(target_sr / target_fps))
    # print(f"Audio {audio_duration * target_sr}->{num_audio_samples_required} data len {audio_data.shape},  shape: {num_audio_frames}, {audio_data_per_frame}")
    audio_data = audio_data.reshape(num_audio_frames, audio_data_per_frame)
    
    # Create frame-wise audio
    if audio_frames_per_video_frame > 1:
        raise NotImplementedError("Frame-wise audio extraction is not implemented yet.")
    else:
        # Extract the central part (for effective frames) and reshape to (1, N, 1, K)
        start_idx = effective_padding
        end_idx = start_idx + num_frames
        central_audio = audio_data[start_idx:end_idx]
        frame_wise_audio = central_audio.reshape(1, num_frames, 1, audio_data_per_frame)
        
    return frame_wise_audio, audio_data, video_frames


def read_av_random_clip_alt(
    video_path: str,
    num_frames: int = 16,
    audio_frames_per_video_frame: int = 1,
    audio_frame_padding: int = 0,
    target_sr: int = 16000,
    target_fps: float = 25.0,
    random_seed: Optional[int] = None,
):
    """
    Read a random clip of audio and video frames.
    Works by first selecting a random appropriate start frame, then reading the specified number of frames (1, N, H, W, C).
    It then selects the audio clip corresponding to the video frames + some extra padding frames on either side. This is 
    of shape (1, P + N + P, K) where P is the padding, N is the number of video frames, and K is the audio data shape per frame.
    if audio_frames_per_video_frame > 1, It then also creates a tensor of shape (1, N, F, K) where F = audio_frames_per_video_frame.
    Otherwise (1, N, 1, K) is returned in the case of audio_frames_per_video_frame = 1.
    
    The final audio and video tensors are returned.
    Args:
        video_path: Path to the video file.
        num_frames: Number of video frames to read.
        audio_frames_per_video_frame: Number of audio frames per video frame.
        audio_frame_padding: Padding for audio frames.
        target_sr: Target sample rate for the audio.
        target_fps: Target frames per second for the video.
        random_seed: Random seed for reproducibility (optional).
        
    Returns:
        Tuple of (frame_wise_audio, full_padded_audio, video_frames) where video_frames is a numpy array.
    """
    from moviepy import VideoFileClip, AudioFileClip
    from video_reader import PyVideoReader
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    # Load the video
    vr = PyVideoReader(video_path)
    info = vr.get_info()
    total_frames = int(info['frame_count'])
    
    # Calculate effective padding needed based on audio segmentation
    effective_padding = max(audio_frame_padding, (audio_frames_per_video_frame) // 2)

    # Make sure we have enough frames
    if total_frames < num_frames + 2 * effective_padding:
        raise ValueError(f"Video has only {total_frames} frames, but {num_frames + 2 * effective_padding} were requested (including effective padding)")

    # Adjust the range for start_idx to account for effective padding
    min_start_idx = effective_padding
    max_start_idx = total_frames - num_frames - effective_padding

    # Select a random start frame that allows for padding on both sides
    start_idx = np.random.randint(min_start_idx, max_start_idx) if max_start_idx > min_start_idx else min_start_idx
    end_idx = start_idx + num_frames
    
    video_frames = vr.decode(start_idx, end_idx)
    
    audio_start_time = (start_idx - effective_padding) / target_fps
    audio_end_time = (end_idx + effective_padding) / target_fps
    num_audio_frames = num_frames + 2 * effective_padding
    audio_duration = audio_end_time - audio_start_time
    
    assert audio_duration > 0, f"Audio duration {audio_duration} is not positive"
    assert audio_start_time >= 0, f"Audio start time {audio_start_time} is negative"
    
    # Extract the subclip
    audio_clip : AudioFileClip = VideoFileClip(video_path).audio.with_fps(target_sr).subclipped(audio_start_time, audio_end_time)
    audio_data = audio_clip.to_soundarray()
    # Make sure len(audio_data) == (num_frames + 2 * effective_padding) * target_sr
    num_audio_samples_required = int(round(audio_duration * target_sr))
    
    if len(audio_data) < num_audio_samples_required:
        raise ValueError(f"Audio data length {len(audio_data)} is less than required {num_audio_samples_required}")
    
    audio_data = audio_data[:num_audio_samples_required]
    # Convert to mono if stereo
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    # Close the clips
    audio_clip.close()
    
    # Reshape audio data
    audio_data = np.array(audio_data)   # This is just 1D
    
    # Calculate dimensions for audio
    audio_data_per_frame = int(round(target_sr / target_fps))
    # print(f"Audio {audio_duration * target_sr}->{num_audio_samples_required} data len {audio_data.shape},  shape: {num_audio_frames}, {audio_data_per_frame}")
    audio_data = audio_data.reshape(num_audio_frames, audio_data_per_frame)
    
    # Create frame-wise audio
    if audio_frames_per_video_frame > 1:
        raise NotImplementedError("Frame-wise audio extraction is not implemented yet.")
    else:
        # Extract the central part (for effective frames) and reshape to (1, N, 1, K)
        start_idx = effective_padding
        end_idx = start_idx + num_frames
        central_audio = audio_data[start_idx:end_idx]
        frame_wise_audio = central_audio.reshape(1, num_frames, 1, audio_data_per_frame)
        
    return frame_wise_audio, audio_data, video_frames

def read_av_random_clip_pyav(
    video_path: str,
    num_frames: int = 16,
    audio_frames_per_video_frame: int = 1,
    audio_frame_padding: int = 0,
    target_sr: int = 16000,
    target_fps: float = 25.0,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes a random video clip and its corresponding audio from `video_path`,
    padding audio by `audio_frame_padding` on each side in terms of video frames.
    Uses PyAV's built-in resampler to produce mono 16-bit audio at `target_sr`.

    Returns:
      (frame_wise_audio, full_padded_audio, video_frames)
        * frame_wise_audio: (1, num_frames, 1, audio_data_per_frame)
        * full_padded_audio: (num_frames + 2*padding, audio_data_per_frame)
        * video_frames: (num_frames, H, W, 3)
    """
    from video_reader import PyVideoReader
    import av

    if random_seed is not None:
        np.random.seed(random_seed)

    # --- 1) Determine which video frames to read ---
    vr = PyVideoReader(video_path)
    total_frames = int(vr.get_info()["frame_count"])
    eff_pad = max(audio_frame_padding, audio_frames_per_video_frame // 2)
    needed_frames = num_frames + 2 * eff_pad
    if total_frames < needed_frames:
        raise ValueError(
            f"Video has only {total_frames} frames but needs {needed_frames} (with padding)."
        )

    min_start = eff_pad
    max_start = total_frames - num_frames - eff_pad
    start_idx = (
        np.random.randint(min_start, max_start)
        if max_start > min_start
        else min_start
    )
    end_idx = start_idx + num_frames

    # --- 2) Decode the chosen video frames ---
    video_frames = vr.decode(start_idx, end_idx)  # shape => (num_frames, H, W, 3)
    del vr

    # --- 3) Define audio time window ---
    audio_start_time = max(0.0, (start_idx - eff_pad) / target_fps)
    audio_end_time = (end_idx + eff_pad) / target_fps
    with av.open(video_path) as container:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio stream found in the file.")

        # --- 4) Decode all audio, resample to s16 mono @ target_sr ---
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        audio_segments = []
        segment_times = []
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                out = resampler.resample(frame)
                out = [out] if not isinstance(out, list) else out
                for oframe in out:
                    # Extract samples from the PyAV audio frame
                    arr = oframe.to_ndarray()   # shape: (1, samples) for mono
                    samples = arr.flatten().astype(np.int16)
                    start_t = float(oframe.pts * audio_stream.time_base)
                    end_t = start_t + oframe.samples / oframe.sample_rate
                    audio_segments.append(samples)
                    segment_times.append((start_t, end_t))
                    
        del resampler
    
    if not audio_segments:
        raise ValueError("No audio frames were decoded.")

    full_audio = np.concatenate(audio_segments, axis=0)
    seg_lens = [len(seg) for seg in audio_segments]
    offsets = np.cumsum([0] + seg_lens)

    # Helper: convert time -> sample index in full_audio
    def time_to_sample(t):
        if t <= segment_times[0][0]:
            return 0
        if t >= segment_times[-1][1]:
            return len(full_audio)
        for i, (st, ed) in enumerate(segment_times):
            if st <= t < ed:
                seg_offset = int(round((t - st) * audio_stream.rate))
                return offsets[i] + min(seg_offset, seg_lens[i] - 1)
        return len(full_audio)

    start_sample = time_to_sample(audio_start_time)
    end_sample = time_to_sample(audio_end_time)
    if end_sample <= start_sample:
        raise ValueError("No audio in the requested range.")

    # Slice out the desired portion
    sliced_audio = full_audio[start_sample:end_sample]

    # --- 5) Convert to float32 in [-1,1], pad or trim to the exact length ---
    # Overall expected sample count for the window
    needed_samples_window = int(round((audio_end_time - audio_start_time) * target_sr))
    if len(sliced_audio) < needed_samples_window:
        pad = needed_samples_window - len(sliced_audio)
        sliced_audio = np.pad(sliced_audio, (0, pad), "constant")
    else:
        sliced_audio = sliced_audio[:needed_samples_window]
    # Convert to float in [-1, 1]
    sliced_audio = sliced_audio.astype(np.float32) / 32768.0

    # We ultimately need (num_frames + 2*pad) * audio_data_per_frame
    num_audio_frames = num_frames + 2 * eff_pad
    audio_data_per_frame = int(round(target_sr / target_fps))
    needed_total_samples = num_audio_frames * audio_data_per_frame

    # Final pad/trim to expected shape
    if len(sliced_audio) < needed_total_samples:
        pad = needed_total_samples - len(sliced_audio)
        sliced_audio = np.pad(sliced_audio, (0, pad), "constant")
    else:
        sliced_audio = sliced_audio[:needed_total_samples]

    full_padded_audio = sliced_audio.reshape(num_audio_frames, audio_data_per_frame)

    # --- 6) Extract the clip's central audio & reshape for per-frame usage ---
    if audio_frames_per_video_frame > 1:
        raise NotImplementedError("Multiple audio frames per video frame not supported.")
    center = full_padded_audio[eff_pad:eff_pad + num_frames]
    frame_wise_audio = center.reshape(1, num_frames, 1, audio_data_per_frame)

    return frame_wise_audio, full_padded_audio, video_frames

# Create a registry of all random clip readers for easier function selection
CLIP_READERS = {
    'moviepy': read_av_random_clip_moviepy,
    'alt': read_av_random_clip_alt,
    'pyav': read_av_random_clip_pyav
}

def read_av_random_clip(
    path: str,
    num_frames: int = 16,
    audio_frames_per_video_frame: int = 1,
    audio_frame_padding: int = 0,
    target_sr: int = 16000,
    target_fps: float = 25.0,
    random_seed: Optional[int] = None,
    method: str = 'alt'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a random clip of audio and video frames using specified method.
    Args:
        path (str): Path to the media file.
        num_frames (int): Number of video frames to read.
        audio_frames_per_video_frame (int): Number of audio frames per video frame.
        audio_frame_padding (int): Padding for audio frames.
        target_sr (int): Target sample rate for audio.
        target_fps (float): Target frames per second for video.
        random_seed (Optional[int]): Seed for random number generator.
        method (str): Method to use for reading the clip.
            Options: 'moviepy', 'alt', 'pyav'.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (frame_wise_audio, full_padded_audio, video_frames).
            - frame_wise_audio: Shape (1, num_frames, 1, audio_data_per_frame)
            - full_padded_audio: Shape (num_frames + 2*padding, audio_data_per_frame)
            - video_frames: Shape (num_frames, H, W, 3)
    """

    if method not in CLIP_READERS:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(CLIP_READERS.keys())}")

    return CLIP_READERS[method](
        path,
        num_frames=num_frames,
        audio_frames_per_video_frame=audio_frames_per_video_frame,
        audio_frame_padding=audio_frame_padding,
        target_sr=target_sr,
        target_fps=target_fps,
        random_seed=random_seed
    )