#!/usr/bin/env python3
"""
Benchmark script to test for memory leaks and performance in decord library.

This script specifically targets the read_av function and provides comprehensive
memory usage tracking and performance metrics.
"""

import os
import sys
import time
import random
import gc
import argparse
import numpy as np
import matplotlib.pyplot as plt
import psutil
from tqdm import tqdm

try:
    from decord import AVReader, VideoReader, cpu, gpu
    HAS_DECORD = True
except ImportError:
    print("Warning: decord library not found. Only OpenCV mode will be available.")
    HAS_DECORD = False

import cv2


def gather_video_paths(directory):
    """Gather all video file paths in a directory (recursively).
    
    Args:
        directory: Directory to search for video files.
    
    Returns:
        List of video file paths.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_paths.append(os.path.join(root, file))
    
    return video_paths


def read_av_standard(path, start=0, end=None, ctx=None):
    """Read audio-video with standard decord approach.
    
    Args:
        path: Path to the video file.
        start: Start frame index.
        end: End frame index.
        ctx: Decord context (CPU or GPU).
    
    Returns:
        Tuple of (audio, video) arrays.
    """
    if not HAS_DECORD:
        raise ImportError("decord library not installed")
    
    ctx = ctx or cpu(0)
    vr = AVReader(path, ctx=ctx)
    audio, video = vr[start:end]
    return audio, video.asnumpy()


def read_av_cleanup(path, start=0, end=None, ctx=None):
    """Read audio-video with explicit cleanup of decord objects.
    
    Args:
        path: Path to the video file.
        start: Start frame index.
        end: End frame index.
        ctx: Decord context (CPU or GPU).
    
    Returns:
        Tuple of (audio, video) arrays.
    """
    if not HAS_DECORD:
        raise ImportError("decord library not installed")
    
    ctx = ctx or cpu(0)
    vr = AVReader(path, ctx=ctx)
    audio, video = vr[start:end]
    audio_list = list(audio)  # Copy audio data
    video_np = video.asnumpy()  # Convert to numpy array
    del vr  # Explicitly delete AVReader object
    return audio_list, video_np


def read_video_opencv(path, max_frames=None):
    """Read video using OpenCV instead of decord.
    
    Args:
        path: Path to the video file.
        max_frames: Maximum number of frames to read.
    
    Returns:
        Video frames as numpy array.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    
    # Stack frames into a video tensor [num_frames, height, width, channels]
    if frames:
        return np.stack(frames, axis=0)
    else:
        return np.array([])  # Empty array if no frames were read


def get_memory_usage():
    """Get current memory usage in MB.
    
    Returns:
        Current memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert bytes to MB


def test_for_memory_leak(video_paths, method='standard', num_iterations=100, sample_size=20):
    """Test for memory leaks by repeatedly loading videos.
    
    Args:
        video_paths: List of video file paths.
        method: Method to use for loading videos ('standard', 'cleanup', or 'opencv').
        num_iterations: Number of iterations to run.
        sample_size: Number of video paths to sample from.
    
    Returns:
        List of memory usage measurements.
    """
    memory_usage = []
    sample_paths = random.sample(video_paths, min(sample_size, len(video_paths)))
    
    # Record baseline memory usage
    gc.collect()
    baseline_memory = get_memory_usage()
    memory_usage.append(baseline_memory)
    
    print(f"Initial memory usage: {baseline_memory:.2f} MB")
    
    # Load videos repeatedly and track memory usage
    for i in tqdm(range(num_iterations), desc=f"Testing {method} method"):
        path = random.choice(sample_paths)
        
        try:
            # Load the video using the specified method
            if method == 'standard' and HAS_DECORD:
                audio, video = read_av_standard(path)
                del audio, video
            elif method == 'cleanup' and HAS_DECORD:
                audio, video = read_av_cleanup(path)
                del audio, video
            elif method == 'opencv':
                video = read_video_opencv(path)
                del video
            else:
                raise ValueError(f"Unknown method: {method}")
                
            # Periodic garbage collection
            if i % 5 == 0:
                gc.collect()
                
            # Record memory
            memory_usage.append(get_memory_usage())
            
        except Exception as e:
            print(f"Error processing video {path}: {e}")
            continue
    
    # Final cleanup
    gc.collect()
    final_memory = get_memory_usage()
    memory_usage.append(final_memory)
    
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory change: {final_memory - baseline_memory:.2f} MB")
    
    return memory_usage


def benchmark_loading_speed(video_paths, method='standard', num_videos=30):
    """Benchmark video loading speed.
    
    Args:
        video_paths: List of video file paths.
        method: Method to use for loading videos ('standard', 'cleanup', or 'opencv').
        num_videos: Number of videos to benchmark.
    
    Returns:
        Tuple of (load times, video sizes).
    """
    # Select random videos to load
    selected_paths = random.sample(video_paths, min(num_videos, len(video_paths)))
    
    load_times = []
    video_sizes = []
    
    print(f"Benchmarking {method} method...")
    
    for path in tqdm(selected_paths, desc=f"Benchmarking {method}"):
        try:
            start_time = time.time()
            
            # Load the video using specified method
            if method == 'standard' and HAS_DECORD:
                audio, video = read_av_standard(path)
            elif method == 'cleanup' and HAS_DECORD:
                audio, video = read_av_cleanup(path)
            elif method == 'opencv':
                video = read_video_opencv(path)
                audio = None
            else:
                raise ValueError(f"Unknown method: {method}")
                
            end_time = time.time()
            
            # Calculate and store metrics
            load_time = end_time - start_time
            load_times.append(load_time)
            
            # Get video size in MB
            video_size = video.nbytes / (1024 * 1024)  # Convert bytes to MB
            video_sizes.append(video_size)
            
            # Cleanup
            del video
            if audio is not None:
                del audio
                
            if len(load_times) % 10 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error benchmarking {path}: {e}")
            continue
    
    if not load_times:
        print("No videos were successfully processed.")
        return [], []
    
    # Calculate statistics
    avg_time = sum(load_times) / len(load_times)
    avg_size = sum(video_sizes) / len(video_sizes) if video_sizes else 0
    avg_speed = sum(video_sizes) / sum(load_times) if sum(load_times) > 0 else 0  # MB/s
    
    print(f"Average load time: {avg_time:.4f} seconds")
    print(f"Average video size: {avg_size:.2f} MB")
    print(f"Average loading speed: {avg_speed:.2f} MB/s")
    
    return load_times, video_sizes


def plot_memory_usage(results, output_dir=None):
    """Plot memory usage over time.
    
    Args:
        results: Dictionary of memory usage results.
        output_dir: Directory to save plots to.
    """
    plt.figure(figsize=(12, 6))
    
    for method, memory_usage in results.items():
        plt.plot(memory_usage, label=method)
    
    plt.title('Memory Usage During Repeated Video Loading')
    plt.xlabel('Iteration')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
    
    plt.show()


def plot_loading_speed(results, output_dir=None):
    """Plot loading speed comparison.
    
    Args:
        results: Dictionary of loading speed results.
        output_dir: Directory to save plots to.
    """
    methods = list(results.keys())
    times = [results[m][0] for m in methods]
    sizes = [results[m][1] for m in methods]
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Load time comparison (box plot)
    plt.subplot(1, 3, 1)
    plt.boxplot(times, labels=methods)
    plt.title('Load Time Comparison')
    plt.ylabel('Time (seconds)')
    
    # Plot 2: Load time vs video size (scatter)
    plt.subplot(1, 3, 2)
    for i, method in enumerate(methods):
        plt.scatter(sizes[i], times[i], alpha=0.7, label=method)
    plt.title('Load Time vs. Video Size')
    plt.xlabel('Video Size (MB)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    
    # Plot 3: Loading speed comparison (box plot)
    plt.subplot(1, 3, 3)
    speeds = []
    for i in range(len(methods)):
        # Calculate MB/s for each video
        speed = [s/t for s, t in zip(sizes[i], times[i]) if t > 0]
        speeds.append(speed)
    
    plt.boxplot(speeds, labels=methods)
    plt.title('Loading Speed Comparison')
    plt.ylabel('Speed (MB/s)')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'loading_speed.png'))
    
    plt.show()


def run_full_benchmark(videos_dir, output_dir=None, iterations=100, num_videos=30, sample_size=20):
    """Run a full benchmark suite.
    
    Args:
        videos_dir: Directory containing video files.
        output_dir: Directory to save results to.
        iterations: Number of iterations for memory leak test.
        num_videos: Number of videos for performance benchmark.
        sample_size: Sample size for memory leak test.
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gather video paths
    print(f"Searching for videos in {videos_dir}...")
    video_paths = gather_video_paths(videos_dir)
    print(f"Found {len(video_paths)} videos.")
    
    if not video_paths:
        print("No videos found. Exiting.")
        return
    
    # Memory leak tests
    print("\n=== Running memory leak tests ===\n")
    memory_results = {}
    
    methods = ['opencv']
    if HAS_DECORD:
        methods = ['standard', 'cleanup', 'opencv']  # Test all methods if decord is available
    
    for method in methods:
        print(f"\nTesting {method} method for memory leaks...")
        memory_usage = test_for_memory_leak(
            video_paths, 
            method=method, 
            num_iterations=iterations, 
            sample_size=sample_size
        )
        memory_results[method] = memory_usage
    
    # Plot memory usage results
    plot_memory_usage(memory_results, output_dir)
    
    # Performance benchmarks
    print("\n=== Running performance benchmarks ===\n")
    performance_results = {}
    
    for method in methods:
        print(f"\nBenchmarking {method} method...")
        times, sizes = benchmark_loading_speed(
            video_paths, 
            method=method, 
            num_videos=num_videos
        )
        performance_results[method] = (times, sizes)
    
    # Plot performance results
    plot_loading_speed(performance_results, output_dir)
    
    # Save results to files if output_dir is specified
    if output_dir:
        # Save memory results
        for method, usage in memory_results.items():
            with open(os.path.join(output_dir, f'memory_{method}.txt'), 'w') as f:
                f.write('\n'.join(str(x) for x in usage))
        
        # Save performance results
        for method, (times, sizes) in performance_results.items():
            with open(os.path.join(output_dir, f'performance_{method}.txt'), 'w') as f:
                f.write('time,size\n')
                for t, s in zip(times, sizes):
                    f.write(f'{t},{s}\n')
    
    print("\nBenchmark complete.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Benchmark decord and OpenCV video loading.')
    parser.add_argument('--videos_dir', '-d', required=True, help='Directory containing video files')
    parser.add_argument('--output_dir', '-o', help='Directory to save results to')
    parser.add_argument('--iterations', '-i', type=int, default=100, help='Number of iterations for memory leak test')
    parser.add_argument('--num_videos', '-n', type=int, default=30, help='Number of videos for performance benchmark')
    parser.add_argument('--sample_size', '-s', type=int, default=20, help='Sample size for memory leak test')
    args = parser.parse_args()
    
    run_full_benchmark(
        args.videos_dir, 
        args.output_dir, 
        args.iterations, 
        args.num_videos, 
        args.sample_size
    )


if __name__ == '__main__':
    main()