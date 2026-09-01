"""Audio-video reader tests.

`read_av_improved` and the random-clip readers need optional native libraries
(`video_reader`/PyVideoReader, pyav, moviepy) plus ffmpeg. The frame-window
plumbing is checked against a stubbed reader so it runs everywhere; the tests
that decode real media skip when the libraries are missing.
"""

import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from flaxdiff.data.sources import av_utils
from flaxdiff.data.sources.av_utils import read_av_improved, read_av_random_clip

FPS = 25.0
SAMPLE_RATE = 16000
CLIP_SECONDS = 2
CLIP_FRAMES = int(FPS * CLIP_SECONDS)


@pytest.fixture
def stub_reader(monkeypatch):
    """Install a fake `video_reader` module and audio reader, recording calls."""
    calls = {}

    class PyVideoReader:
        def __init__(self, path):
            calls["path"] = path

        def decode(self, start_frame=0, end_frame=None):
            calls["start_frame"] = start_frame
            calls["end_frame"] = end_frame
            stop = CLIP_FRAMES if end_frame is None else end_frame
            return np.zeros((stop - start_frame, 4, 4, 3), dtype=np.uint8)

        def get_info(self):
            return {"frame_count": CLIP_FRAMES}

    module = types.ModuleType("video_reader")
    module.PyVideoReader = PyVideoReader
    monkeypatch.setitem(sys.modules, "video_reader", module)

    def fake_read_audio(path, start_time=None, duration=None, target_sr=SAMPLE_RATE,
                        method='ffmpeg'):
        calls["start_time"] = start_time
        calls["duration"] = duration
        samples = int((duration if duration is not None else CLIP_SECONDS) * target_sr)
        return np.zeros(samples, dtype=np.float32), target_sr

    monkeypatch.setattr(av_utils, "read_audio", fake_read_audio)
    return calls


def test_read_av_improved_stops_at_the_end_frame(stub_reader):
    """`end` was accepted and then dropped: decode() always got end_frame=None,
    so every call decoded the rest of the clip."""
    audio, video = read_av_improved("clip.mp4", start=5, end=10, fps=FPS)

    assert stub_reader["start_frame"] == 5
    assert stub_reader["end_frame"] == 10
    assert len(video) == 5
    # The audio window must line up with the frame window
    assert stub_reader["start_time"] == pytest.approx(5 / FPS)
    assert stub_reader["duration"] == pytest.approx(5 / FPS)
    assert len(audio) == pytest.approx(SAMPLE_RATE * 5 / FPS, abs=1)


def test_read_av_improved_without_an_end_reads_to_the_clip_end(stub_reader):
    _, video = read_av_improved("clip.mp4", start=5, fps=FPS)

    assert stub_reader["end_frame"] is None
    assert stub_reader["duration"] is None
    assert len(video) == CLIP_FRAMES - 5


# ---------------------------------------------------------------------------------
# Real decoding: needs the native readers and ffmpeg
# ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthesized_clip(tmp_path_factory):
    pytest.importorskip("video_reader", reason="PyVideoReader is not installed")
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs the ffmpeg binary")

    path = tmp_path_factory.mktemp("av") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={CLIP_SECONDS}:size=64x64:rate={int(FPS)}",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={CLIP_SECONDS}:sample_rate={SAMPLE_RATE}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def test_read_av_improved_decodes_only_the_requested_window(synthesized_clip):
    audio, video = read_av_improved(str(synthesized_clip), start=5, end=15, fps=FPS)

    assert len(video) == 10
    assert len(audio) == pytest.approx(SAMPLE_RATE * 10 / FPS, rel=0.05)

    _, full_video = read_av_improved(str(synthesized_clip), fps=FPS)
    assert len(full_video) > len(video)


# Each random-clip reader's own decoder; all of them also need PyVideoReader.
_CLIP_READER_DEPS = {"pyav": "av", "alt": "moviepy", "moviepy": "moviepy"}


@pytest.mark.parametrize("method", sorted(_CLIP_READER_DEPS))
def test_random_clip_readers_are_seeded_locally(synthesized_clip, method):
    """Same seed, same clip; and no reader may reseed the process-global RNG."""
    pytest.importorskip(_CLIP_READER_DEPS[method])

    np.random.seed(99)
    global_state = np.random.get_state()[1].copy()

    def read(seed):
        return read_av_random_clip(
            str(synthesized_clip), num_frames=8, audio_frame_padding=1,
            target_sr=SAMPLE_RATE, target_fps=FPS, random_seed=seed, method=method,
        )

    first_framewise, first_audio, first_frames = read(3)
    second_framewise, second_audio, second_frames = read(3)

    assert np.array_equal(first_frames, second_frames)
    assert np.array_equal(first_audio, second_audio)
    assert np.array_equal(first_framewise, second_framewise)
    assert first_frames.shape[0] == 8
    assert first_audio.shape[0] == 8 + 2  # padded on both sides
    assert np.array_equal(global_state, np.random.get_state()[1])
