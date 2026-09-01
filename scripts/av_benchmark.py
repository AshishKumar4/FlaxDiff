#!/usr/bin/env python3
"""Benchmark and visualise the leak-free audio-video readers.

Reads a clip with `flaxdiff.data.sources.av_utils.read_av_improved` (PyVideoReader
for frames, ffmpeg for audio - neither leaks the way decord's AVReader did),
plots the waveform/spectrogram/frames, and times repeated reads.

Usage:
    python scripts/av_benchmark.py --video clip.mp4 [--output plot.png] [--benchmark]

Needs the optional AV extras: `video_reader` (PyVideoReader) plus ffmpeg on PATH,
and matplotlib for the plot.
"""

import argparse
import os
import time

import numpy as np

from flaxdiff.data.sources.av_utils import read_av_improved


def visualize_av_data(audio_data, video_frames, output_path=None):
    """
    Visualize audio and video data.

    Args:
        audio_data: Audio data as numpy array or list.
        video_frames: Video frames as numpy array.
        output_path: Path to save visualization (optional).
    """
    import matplotlib.pyplot as plt  # optional dependency, only the plot needs it

    audio = np.asarray(audio_data, dtype=np.float32)
    plt.figure(figsize=(12, 6))

    # Number of frames to show
    num_frames = min(4, len(video_frames))
    columns = max(num_frames, 2)

    # Plot audio waveform
    plt.subplot(2, columns, 1)
    plt.plot(audio[:10000])
    plt.title('Audio Waveform')
    plt.grid(True)

    # Plot audio spectrogram
    plt.subplot(2, columns, 2)
    plt.specgram(audio, NFFT=1024, Fs=16000)
    plt.title('Audio Spectrogram')

    # Plot sample frames
    for i in range(num_frames):
        frame_index = i * len(video_frames) // num_frames
        plt.subplot(2, columns, columns + i + 1)
        plt.imshow(video_frames[frame_index])
        plt.title(f'Frame {frame_index}')
        plt.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")

    plt.show()


def benchmark_av_reading(video_path, num_iterations=10, num_frames=None):
    """
    Benchmark audio-video reading performance.

    Args:
        video_path: Path to the video file.
        num_iterations: Number of iterations for benchmarking.
        num_frames: Frames to read per call, or None for the whole clip.

    Returns:
        Average seconds per read.
    """
    end = None if num_frames is None else num_frames
    label = "whole clip" if end is None else f"first {end} frames"
    print(f"Benchmarking read_av_improved ({label})...")

    # Perform warmup
    read_av_improved(video_path, start=0, end=end)

    # Measure performance
    start_time = time.time()
    for _ in range(num_iterations):
        read_av_improved(video_path, start=0, end=end)
    avg_time = (time.time() - start_time) / num_iterations

    print(f"Average time per read: {avg_time:.4f} seconds")
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Demo for memory-leak-free audio-video reading")
    parser.add_argument("--video", "-v", required=True, help="Path to the video file")
    parser.add_argument("--output", "-o", help="Path to save visualization")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmarks")
    parser.add_argument("--iterations", "-i", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--frames", "-f", type=int, default=16,
                        help="Frames per read for the windowed benchmark")

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
        full_time = benchmark_av_reading(args.video, args.iterations)
        window_time = benchmark_av_reading(args.video, args.iterations, num_frames=args.frames)

        print("\nBenchmark results:")
        print(f"Whole clip:        {full_time:.4f} seconds per read")
        print(f"{args.frames}-frame window: {window_time:.4f} seconds per read")


if __name__ == "__main__":
    main()
