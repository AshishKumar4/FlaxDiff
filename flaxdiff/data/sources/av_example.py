#!/usr/bin/env python3
"""
Example script demonstrating how to use the memory-leak-free audio-video reading functions.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from av_utils import read_av_improved, read_av_batch
from audio_utils import read_audio
import argparse


def visualize_av_data(audio_data, video_frames, output_path=None):
    """
    Visualize audio and video data.
    
    Args:
        audio_data: Audio data as numpy array or list.
        video_frames: Video frames as numpy array.
        output_path: Path to save visualization (optional).
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Number of frames to show
    num_frames = min(4, len(video_frames))
    
    # Plot audio waveform
    plt.subplot(2, num_frames, 1)
    plt.plot(audio_data[:10000])
    plt.title('Audio Waveform')
    plt.grid(True)
    
    # Plot audio spectrogram
    plt.subplot(2, num_frames, 2)
    plt.specgram(audio_data, NFFT=1024, Fs=16000)
    plt.title('Audio Spectrogram')
    
    # Plot sample frames
    for i in range(num_frames):
        plt.subplot(2, num_frames, num_frames+i+1)
        plt.imshow(video_frames[i*len(video_frames)//num_frames])
        plt.title(f'Frame {i*len(video_frames)//num_frames}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.show()


def benchmark_av_reading(video_path, num_iterations=10, use_batch=False):
    """
    Benchmark audio-video reading performance.
    
    Args:
        video_path: Path to the video file.
        num_iterations: Number of iterations for benchmarking.
        use_batch: Whether to use batch reading.
    """
    print(f"Benchmarking {'batch' if use_batch else 'single'} reading...")
    
    # Perform warmup
    if use_batch:
        _ = read_av_batch([video_path])
    else:
        _ = read_av_improved(video_path)
    
    # Measure performance
    start_time = time.time()
    
    for i in range(num_iterations):
        if use_batch:
            results = read_av_batch([video_path])
        else:
            audio, video = read_av_improved(video_path)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    print(f"Average time per read: {avg_time:.4f} seconds")
    
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Demo for memory-leak-free audio-video reading")
    parser.add_argument("--video", "-v", required=True, help="Path to the video file")
    parser.add_argument("--output", "-o", help="Path to save visualization")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmarks")
    parser.add_argument("--iterations", "-i", type=int, default=10, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Load audio-video data
    print(f"Reading audio-video data from {args.video}...")
    audio, video = read_av_improved(args.video)
    
    print(f"Video shape: {video.shape}")
    print(f"Audio length: {len(audio)}")
    
    # Visualize data
    visualize_av_data(audio, video, args.output)
    
    # Run benchmarks if requested
    if args.benchmark:
        print("\nRunning benchmarks...")
        single_time = benchmark_av_reading(args.video, args.iterations, use_batch=False)
        batch_time = benchmark_av_reading(args.video, args.iterations, use_batch=True)
        
        print("\nBenchmark results:")
        print(f"Single reading: {single_time:.4f} seconds per video")
        print(f"Batch reading: {batch_time:.4f} seconds per video")
    

if __name__ == "__main__":
    main()