"""
Voxceleb Data preparation pipeline
1. Resample audio to 16kHz and video to 25 fps
2. Detect scene and segment the video when the scene changes
3. Do face detection and landmark detection and use the landmarks to crop and affine transform the face
4. Do another round of face detection to filter out bad faces
5. Do Audio video synchronization using syncnet
6. Filter out low quality videos using Hyper IQA
"""
from sources import DataSource, NO_DATA
from processors import DataProcessor
from sinks import DataSink
from dataclasses import dataclass
from video_reader import PyVideoReader
import os
from typing import List, Tuple
import numpy as np
import queue
import cv2
import subprocess
import shutil
from decord import VideoReader
from .face_align import FaceAlignmentProcessor
from tqdm import tqdm
import threading
import time
import uuid
import traceback
from concurrent.futures import ProcessPoolExecutor

def read_video(video_path: str, change_fps=True, reader="rsreader"):
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
    elif reader =="rsreader_fast":
        return read_video_rsreader(target_video_path, fast=True)
    if reader == "decord":
        return read_video_decord(target_video_path)
    elif reader == "opencv":
        return read_video_opencv(target_video_path)
    else:
        raise ValueError(f"Unknown reader: {reader}")

def read_video_decord(video_path: str):
    vr = VideoReader(video_path)
    video_frames = vr[:].asnumpy()
    vr.seek(0)
    return video_frames

def read_video_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # Convert BGR to RGB
    return np.array(frames)[:, :, :, ::-1]

def read_video_rsreader(video_path, fast=False):
    vr = PyVideoReader(video_path)
    return vr.decode_fast() if fast else vr.decode()

def gather_video_paths_iter(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.endswith(".mp4"):
                rel_path = os.path.relpath(root, input_dir)
                video_input = os.path.join(root, file)
                video_output_dir = os.path.join(output_dir, rel_path)
                video_output = os.path.join(video_output_dir, file)
                if not os.path.isfile(video_output):
                    yield video_input, video_output, rel_path

def gather_video_paths(input_dir, output_dir):
    video_paths = []
    for paths in gather_video_paths_iter(input_dir, output_dir):
        video_paths.append(paths)
        
    # Sort the video paths
    video_paths.sort()
    return video_paths

def write_video(video_output_path: str, video_frames: np.ndarray, fps: int):
    height, width = video_frames[0].shape[:2]
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"vp09"), fps, (width, height))
    for frame in video_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    
def get_local_partition(objs, world_size, rank):
    """
    Get the local partition of the objects based on the world size and rank.
    """
    partition_size = len(objs) // world_size
    start = rank * partition_size
    end = (rank + 1) * partition_size if rank != world_size - 1 else len(objs)
    return objs[start:end]

@dataclass
class VideoPaths:
    video_input: str
    video_resampled: str
    video_temp: str
    video_output: str

@dataclass
class VideoObjects(VideoPaths):
    video_frames: List[np.ndarray]

class AVPathGenerator(DataSource[VideoPaths]):
    def __init__(self, input_dir, output_dir, world_size=1, rank=0, verbose=False):
        super().__init__(
            buffer_size=None,
            num_workers=1,  # Single-threaded is sufficient for path generation
            verbose=verbose,
        )
        self.input_dir = input_dir
        output_dir = output_dir[:-1] if output_dir.endswith("/") else output_dir
        self.output_dir = output_dir
        # self.video_paths_iter = gather_video_paths_iter(input_dir, output_dir)
        video_paths = gather_video_paths(input_dir, output_dir)
        local_paths = get_local_partition(video_paths, world_size, rank)
        print(f"Total video paths: {len(video_paths)}, Rank {rank} has {len(local_paths)}")
        self.total_paths = len(local_paths)
        self.video_paths_iter = iter(local_paths)
        
    def fetch(self, **kwargs) -> VideoPaths:
        try:
            video_input, video_output, rel_path = next(self.video_paths_iter)
            video_resampled = os.path.join(f"{self.output_dir}_resampled", rel_path, os.path.basename(video_input))
            video_temp = os.path.join(f"{self.output_dir}_temp", rel_path, os.path.basename(video_input).split(".")[0])
            
            if self.verbose:
                print(f"Processing {video_input} -> {video_output}, {video_resampled}, {video_temp}")
            
            return VideoPaths(
                video_input=video_input, 
                video_output=video_output,
                video_resampled=video_resampled,
                video_temp=video_temp,
            )
        except StopIteration:
            return NO_DATA
        except Exception as e:
            print(f"Error in AVPathGenerator: {e}")
            traceback.print_exc()
            return None

    def close(self):
        if self.video_paths_iter is not None:
            self.video_paths_iter = None

def get_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    return fps

# Stage 1: Resample audio and video
class AVResample(DataProcessor[VideoPaths, VideoPaths]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process(self, video_paths: VideoPaths, **kwargs) -> VideoPaths:
        os.makedirs(os.path.dirname(video_paths.video_resampled), exist_ok=True)
        try:
            if get_video_fps(video_paths.video_input) == 25:
                command = f"ffmpeg -loglevel error -y -i {video_paths.video_input} -c:v copy -ar 16000 -q:a 0 {video_paths.video_resampled}"
            else:
                command = f"ffmpeg -loglevel error -y -i {video_paths.video_input} -r 25 -ar 16000 -q:a 0 {video_paths.video_resampled}"
            subprocess.run(command, shell=True)
            if self.verbose:
                print(f"Resampled {video_paths.video_input} to {video_paths.video_resampled}")
        except Exception as e:
            print(f"Error processing {video_paths.video_input}: {e}")
            traceback.print_exc()
        return video_paths
        
# Stage 2: Read video frames for further processing
class AVDataReader(DataProcessor[VideoPaths, VideoObjects]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process(self, video_paths: VideoPaths, **kwargs) -> VideoObjects:
        video_frames = read_video(video_paths.video_resampled, change_fps=False, reader="decord")
        if video_frames is None:
            return None
        if len(video_frames) == 0:
            print(f"Error reading video frames for {video_paths.video_resampled}")
            return None
        # Emit the video frames and paths
        return VideoObjects(
            video_input=video_paths.video_input,
            video_output=video_paths.video_output,
            video_resampled=video_paths.video_resampled,
            video_temp=video_paths.video_temp,
            video_frames=video_frames
        )
    
    def close(self):
        return super().close()
    
    
# Stage 3: Affine Transformation using landmarks
class AVAffineTransform(DataProcessor[VideoObjects, VideoObjects]):
    def __init__(self, resolution=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligners: List[FaceAlignmentProcessor] = []
        for worker in range(self._num_workers):
            self.aligners.append(FaceAlignmentProcessor(resolution=resolution))
        
    def process(self, video_objects: VideoObjects, threadId: int) -> VideoObjects:
        # Perform affine transformation on the frames
        aligner = self.aligners[threadId]
        aligned = aligner.process_frames(video_objects.video_frames, break_on_error=True)
        if aligned is None or len(aligned) != len(video_objects.video_frames):
            print(f"Error in affine transformation for {video_objects.video_input}")
            return None
        if self.verbose:
            print(f"Aligned {video_objects.video_input} with {len(aligned)} frames")
        # Emit the aligned frames and paths
        video_objects.video_frames = aligned
        return video_objects
    
# Stage 4: Face Detection to filter out bad transformed faces
class AVFaceDetection(DataProcessor[VideoObjects, VideoObjects]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligners: List[FaceAlignmentProcessor] = []
        for worker in range(self._num_workers):
            self.aligners.append(FaceAlignmentProcessor(resolution=256))
        
    def process(self, video_objects: VideoObjects, threadId: int) -> VideoObjects:
        # Perform face detection on the frames
        aligner = self.aligners[threadId]
        detected_faces = aligner.detect_faces(video_objects.video_frames, break_on_error=True)
        if detected_faces is None or len(detected_faces) != len(video_objects.video_frames):
            print(f"Error in face detection for {video_objects.video_input}")
            return None
        
        if self.verbose:
            print(f"Detected faces in {video_objects.video_input} with {len(detected_faces)} frames")
        # Emit the detected faces and paths
        return video_objects
        
        
# Stage 5: Write the processed video frames to disk
class AVWrite(DataSink[VideoObjects]):
    def __init__(self, process_temp_dir, total_paths=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_paths = total_paths
        self.lock = threading.Lock()
        if total_paths is not None:
            # Initialize pbar with total paths
            self.pbar = tqdm(total=total_paths, desc="Writing videos")
        else:
            self.pbar = None
        os.makedirs(process_temp_dir, exist_ok=True)
        self.process_temp_dir = process_temp_dir
        
    def write(self, video_objects: VideoObjects, **kwargs) -> None:
        try:
            os.makedirs(os.path.dirname(video_objects.video_output), exist_ok=True)
            
            # Generate a random name for the video to avoid conflicts
            video_name = os.path.splitext(os.path.basename(video_objects.video_resampled))[0]
            video_name = f"{video_name}_{hash(time.time())}_{uuid.uuid4()}"
            
            audio_temp = os.path.join(self.process_temp_dir, f"{video_name}_temp.wav")
            video_temp = os.path.join(self.process_temp_dir, f"{video_name}_temp.mp4")

            write_video(video_temp, video_objects.video_frames, fps=25)

            command = f"ffmpeg -y -loglevel error -i {video_objects.video_resampled} -q:a 0 -map a {audio_temp}"
            subprocess.run(command, shell=True)

            os.makedirs(os.path.dirname(video_objects.video_output), exist_ok=True)
            command = f"ffmpeg -y -loglevel error -i {video_temp} -i {audio_temp} -c:v libx264 -c:a aac -map 0:v -map 1:a -q:v 0 -q:a 0 {video_objects.video_output}"
            subprocess.run(command, shell=True)

            os.remove(audio_temp)
            os.remove(video_temp)
            if self.verbose:
                print(f"Written {video_objects.video_output}")
                
            if self.pbar is not None:
                with self.lock:
                    self.pbar.update(1)
                    # self.pbar.set_postfix_str(f"Processed {video_objects.video_output}")
            return True
        except Exception as e:
            print(f"Error combining video and audio: {str(e)}")
            traceback.print_exc()
            return False
        
def run_pipeline_proc(
    input_dir: str, 
    output_dir: str, 
    process_temp_dir: str, 
    world_size: int = 1, 
    rank: int = 0, 
    num_workers: int = 1,
    use_wandb: bool = False,
):
    if use_wandb:
        import wandb
        wandb.init(project="Voxceleb-prep", name=f"pipeline_{rank}", config={"input_dir": input_dir, "output_dir": output_dir})
    # Create the pipeline
    av_path_generator = AVPathGenerator(input_dir, output_dir, world_size, rank, verbose=False)
    total_paths = av_path_generator.total_paths
    av_resample = AVResample(sources=[av_path_generator], num_workers=num_workers, verbose=False)
    av_data_reader = AVDataReader(sources=[av_resample], num_workers=num_workers, verbose=False)
    av_affine_transform = AVAffineTransform(sources=[av_data_reader], num_workers=num_workers, verbose=False)
    av_face_detection = AVFaceDetection(sources=[av_affine_transform], num_workers=num_workers, verbose=False)
    av_write = AVWrite(process_temp_dir, total_paths, sources=[av_face_detection], num_workers=num_workers, verbose=False)

    # Run the pipeline
    av_path_generator.start()
    av_resample.start()
    av_data_reader.start()
    av_affine_transform.start()
    av_face_detection.start()
    av_write.start()
    
    # Wait for the pipeline to finish
    av_write.join()
    # av_affine_transform.join()
    
def run_pipeline(
    input_dir: str,
    output_dir: str,
    process_temp_dir: str,
    num_processes: int = 1,
    num_workers_per_process: int = 1,
    use_wandb: bool = False,
):
    # Create the process pool
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for rank in range(num_processes):
            futures.append(executor.submit(
                run_pipeline_proc,
                input_dir,
                output_dir,
                process_temp_dir,
                world_size=num_processes,
                rank=rank,
                num_workers=num_workers_per_process,
                use_wandb=use_wandb,
            ))
        # Wait for all processes to finish
        for future in futures:
            future.result()