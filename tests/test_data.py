"""Data layer tests: registries, loader contracts, lazy imports, AV decoding.

The shared environment has no HF `datasets`, decord, pyav, moviepy or
video_reader, which is the point of several of these tests: the grain paths and
`import flaxdiff.data` must not need them. Anything that genuinely requires an
optional dependency skips.
"""

import importlib.util
import inspect
import itertools
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import grain.python as pygrain
import numpy as np
import pytest

import flaxdiff.data
from flaxdiff.data import dataloaders
from flaxdiff.data.dataloaders import get_dataset_online, get_media_dataset_grain
from flaxdiff.data.dataset_map import mediaDatasetMap
from flaxdiff.data.sources.audio_utils import _read_wav_mono, read_audio_ffmpeg
from flaxdiff.data.sources.av_utils import choose_clip_start
from flaxdiff.data.sources.base import DataAugmenter, DataSource, MediaDataset
from flaxdiff.data.sources.images import ImageTFDSAugmenter, gcs_filters
from flaxdiff.data.sources import videos as videos_module
from flaxdiff.data.sources.videos import AudioVideoAugmenter, VideoLocalSource
from flaxdiff.data.sources.voxceleb2 import VoxCeleb2Source

REPO_ROOT = Path(__file__).resolve().parents[1]

# Registry keys the factories advertise; every one must build.
AUGMENTER_KEYS = ("image_tfds", "image_gcs", "video")
SOURCE_KEYS = ("image_tfds", "image_gcs", "image_combined_gcs", "video_tfds", "video_local")


# ---------------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("key", AUGMENTER_KEYS)
def test_every_augmenter_registry_key_constructs(key):
    """"video" used to point at a VideoAugmenter class that never existed."""
    assert isinstance(DataAugmenter.create(key), DataAugmenter)


def test_video_augmenter_key_resolves_to_the_audio_video_augmenter():
    assert isinstance(DataAugmenter.create("video"), AudioVideoAugmenter)


@pytest.mark.parametrize("key", SOURCE_KEYS)
def test_every_source_registry_key_constructs(key):
    kwargs = {"name": "oxford_flowers102"} if key.endswith("tfds") else {}
    assert isinstance(DataSource.create(key, **kwargs), DataSource)


def test_unknown_registry_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown augmenter type"):
        DataAugmenter.create("video_gcs")
    with pytest.raises(ValueError, match="Unknown source type"):
        DataSource.create("audio_gcs")


# ---------------------------------------------------------------------------------
# The filter seam: only the GCS augmenter has one, and it is concrete
# ---------------------------------------------------------------------------------

def test_filtering_is_not_part_of_the_augmenter_contract():
    """No pipeline applies a filter, so the abstract seam (and the two stubs
    that returned None instead of a transform) is gone."""
    assert not hasattr(DataAugmenter, "create_filter")
    assert not hasattr(ImageTFDSAugmenter, "create_filter")
    assert not hasattr(AudioVideoAugmenter, "create_filter")


def test_gcs_filter_returns_a_usable_filter_transform():
    filter_transform = gcs_filters(image_scale=256)
    assert isinstance(filter_transform, type)
    assert issubclass(filter_transform, pygrain.FilterTransform)
    # grain drops elements via `filter`, so that is the method that must exist
    assert callable(filter_transform.filter)


# ---------------------------------------------------------------------------------
# Lazy imports: the data layer must not drag in HF datasets / opencv / decord
# ---------------------------------------------------------------------------------

def test_importing_flaxdiff_data_pulls_in_no_heavy_dependencies():
    """`import flaxdiff.data` used to import online_loader, which imports HF
    `datasets` at module scope and took the whole package down with it."""
    probe = (
        "import sys, flaxdiff.data;"
        "heavy = [m for m in ('datasets', 'cv2', 'decord', 'jax',"
        " 'flaxdiff.data.online_loader', 'flaxdiff.data.dataloaders')"
        " if m in sys.modules];"
        "assert not heavy, heavy;"
        "assert flaxdiff.data.MediaDataset.__name__ == 'MediaDataset'"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lazy_exports_resolve_and_unknown_names_raise_attribute_error():
    assert flaxdiff.data.get_media_dataset_grain is get_media_dataset_grain
    assert flaxdiff.data.VoxCeleb2Source is VoxCeleb2Source
    assert "get_dataset_grain" in dir(flaxdiff.data)
    with pytest.raises(AttributeError, match="has no attribute"):
        flaxdiff.data.get_dataset_from_thin_air


def test_online_dataset_factories_defer_the_streaming_import():
    """The streaming stack needs HF datasets; asking for a bad dataset name must
    fail on the name, not on the missing dependency."""
    already_imported = "flaxdiff.data.online_loader" in sys.modules
    with pytest.raises(ValueError, match="not found in onlineDatasetMap"):
        get_dataset_online("no_such_dataset")
    if not already_imported:
        assert "flaxdiff.data.online_loader" not in sys.modules


def test_online_streaming_loader_no_longer_accepts_the_unapplied_pre_map_args():
    try:
        from flaxdiff.data.online_loader import OnlineStreamingDataLoader
    except ImportError as exc:  # HF datasets missing: nothing to check here
        pytest.skip(f"the streaming loader needs HF datasets ({exc})")

    parameters = inspect.signature(OnlineStreamingDataLoader.__init__).parameters
    assert "pre_map_maker" not in parameters and "pre_map_def" not in parameters


# ---------------------------------------------------------------------------------
# Media grain API: explicit source, and a validation split of its own
# ---------------------------------------------------------------------------------

class _ListSource(DataSource):
    """Minimal random-access source; stands in for arrayrecord/video sources."""

    def __init__(self, length):
        self.records = [{"index": i} for i in range(length)]

    def get_source(self, path_override):
        assert path_override, "the loader must pass the caller's dataset_source through"
        return self.records


class _PassthroughAugmenter(DataAugmenter):
    """Returns records untouched, so tests can read the sampled indices."""

    def create_transform(self, **kwargs):
        self.transform_kwargs = kwargs

        class Passthrough(pygrain.MapTransform):
            def map(self, element):
                return element

        return Passthrough


@pytest.fixture
def fake_media_dataset(monkeypatch):
    """Register a dependency-free media dataset named "fake"."""
    dataset = MediaDataset(source=_ListSource(32), augmenter=_PassthroughAugmenter(),
                           media_type="video")
    monkeypatch.setitem(mediaDatasetMap, "fake", dataset)
    return dataset


def _indices(loader, num_batches):
    return [[int(i) for i in batch["index"]]
            for batch in itertools.islice(iter(loader), num_batches)]


def test_media_dataset_grain_requires_an_explicit_source(fake_media_dataset):
    """dataset_source=None used to reach os.path.join(None, ...) in the source."""
    with pytest.raises(ValueError, match="dataset_source"):
        get_media_dataset_grain("fake")
    with pytest.raises(ValueError, match="not found in mediaDatasetMap"):
        get_media_dataset_grain("not_a_dataset", dataset_source="/tmp")


def test_media_dataset_grain_without_val_count_has_no_validation_loader(fake_media_dataset):
    data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                   worker_count=0, num_epochs=1)
    assert data["train_len"] == 32 and data["media_type"] == "video"
    assert "val" not in data and "val_len" not in data


def test_media_dataset_validation_split_is_ordered_and_disjoint_from_train(fake_media_dataset):
    data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                   val_count=8, val_batch_size=4, worker_count=0,
                                   num_epochs=1, seed=0)

    assert data["val_len"] == 8
    assert data["train_len"] == 24  # the held-out records leave the train stream

    # Validation walks its own records in canonical order, not the shuffled
    # train sampler's, and repeats identically.
    val_batches = _indices(data["val"](), 2)
    assert val_batches == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert _indices(data["val"](), 2) == val_batches

    train_indices = [i for batch in _indices(data["train"](), 3) for i in batch]
    assert set(train_indices).isdisjoint(range(8))
    assert train_indices != sorted(train_indices)  # the train sampler still shuffles


def test_media_dataset_validation_split_rejects_impossible_sizes(fake_media_dataset):
    with pytest.raises(ValueError, match="val_count"):
        get_media_dataset_grain("fake", dataset_source="/tmp", val_count=32)


def test_media_dataset_grain_passes_media_scale_to_the_video_transform(fake_media_dataset):
    get_media_dataset_grain("fake", dataset_source="/tmp", media_scale=64,
                            sequence_length=4, worker_count=0)
    kwargs = fake_media_dataset.augmenter.transform_kwargs
    assert kwargs["frame_size"] == 64 and kwargs["sequence_length"] == 4


def test_legacy_grain_loader_takes_a_validation_batch_size():
    """The legacy val path hardcoded 32 and reused the shuffled train sampler."""
    parameters = inspect.signature(dataloaders.get_dataset_grain).parameters
    assert parameters["val_batch_size"].default == 32
    assert "val_worker_count" in parameters


# ---------------------------------------------------------------------------------
# WAV decoding
# ---------------------------------------------------------------------------------

SAMPLE_RATE = 16000
NUM_SAMPLES = 1000


def _write_wav(path, samples, sample_rate=SAMPLE_RATE, channels=1, sample_width=2):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _tone(num_samples=NUM_SAMPLES):
    return (np.sin(np.linspace(0, 40, num_samples)) * 20000).astype("<i2")


def test_wav_decode_excludes_the_riff_header(tmp_path):
    """np.fromfile(path, np.int16) handed back the 44-byte header as 22 samples."""
    samples = _tone()
    path = tmp_path / "tone.wav"
    _write_wav(path, samples)

    audio = _read_wav_mono(str(path))

    assert len(audio) == NUM_SAMPLES
    assert len(np.fromfile(path, np.int16)) == NUM_SAMPLES + 22  # the old behaviour
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-7)


def test_wav_decode_averages_stereo_to_mono(tmp_path):
    left = _tone()
    right = -left
    interleaved = np.empty(2 * NUM_SAMPLES, dtype="<i2")
    interleaved[0::2], interleaved[1::2] = left, right
    path = tmp_path / "stereo.wav"
    _write_wav(path, interleaved, channels=2)

    audio = _read_wav_mono(str(path))

    assert len(audio) == NUM_SAMPLES
    np.testing.assert_allclose(audio, np.zeros(NUM_SAMPLES), atol=1e-7)


def test_wav_decode_rejects_unsupported_sample_widths(tmp_path):
    path = tmp_path / "24bit.wav"
    _write_wav(path, np.zeros(3 * NUM_SAMPLES, dtype=np.uint8), sample_width=3)
    with pytest.raises(ValueError, match="sample width"):
        _read_wav_mono(str(path))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_read_audio_ffmpeg_returns_the_exact_sample_count(tmp_path):
    samples = _tone()
    path = tmp_path / "tone.wav"
    _write_wav(path, samples)

    audio, sample_rate = read_audio_ffmpeg(str(path), target_sr=SAMPLE_RATE)

    assert sample_rate == SAMPLE_RATE
    assert audio.dtype == np.float32
    assert len(audio) == NUM_SAMPLES
    np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_read_audio_ffmpeg_honours_start_and_duration(tmp_path):
    _write_wav(tmp_path / "tone.wav", _tone())

    audio, _ = read_audio_ffmpeg(str(tmp_path / "tone.wav"), start_time=0.01,
                                 duration=0.02, target_sr=SAMPLE_RATE)

    assert len(audio) == int(0.02 * SAMPLE_RATE)


# ---------------------------------------------------------------------------------
# Clip sampling uses a local RNG
# ---------------------------------------------------------------------------------

def test_clip_start_is_reproducible_without_touching_the_global_rng():
    """The readers used to call np.random.seed() inside data-loading workers."""
    np.random.seed(1234)
    global_state = np.random.get_state()[1].copy()

    starts = [choose_clip_start(100, 16, 3, np.random.default_rng(7)) for _ in range(3)]

    assert len(set(starts)) == 1
    assert np.array_equal(global_state, np.random.get_state()[1])


def test_clip_start_stays_inside_the_padded_window():
    starts = {choose_clip_start(100, 16, 3, np.random.default_rng(seed))
              for seed in range(64)}
    assert min(starts) >= 3
    assert max(starts) <= 100 - 16 - 3
    assert len(starts) > 1  # different seeds really do move the window


def test_clip_start_falls_back_to_the_only_valid_offset():
    assert choose_clip_start(22, 16, 3, np.random.default_rng(0)) == 3


# ---------------------------------------------------------------------------------
# VoxCeleb2 source
# ---------------------------------------------------------------------------------

def _voxceleb_tree(root, split="train", identities=("id00012", "id00015")):
    """Build <root>/<split>/<identity>/<clip>/<utterance>.mp4, plus some noise."""
    clips = []
    for identity in identities:
        for clip in ("_raOc3-IRsw", "21Uxsk56VDQ"):
            path = root / split / identity / clip / "00001.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not really an mp4")
            clips.append(path)
    (root / split / "filelist.txt").write_text("ignored\n")
    return clips


def test_voxceleb2_source_scans_the_tree_recursively(tmp_path):
    clips = _voxceleb_tree(tmp_path)
    records = VoxCeleb2Source().get_source(str(tmp_path))

    assert [record["video_path"] for record in records] == sorted(str(c) for c in clips)


def test_voxceleb2_source_renders_captions_from_the_template(tmp_path):
    _voxceleb_tree(tmp_path)

    templated = VoxCeleb2Source(prompt_template="a video of {identity} speaking")
    captions = {record["caption"] for record in templated.get_source(str(tmp_path))}
    assert captions == {"a video of id00012 speaking", "a video of id00015 speaking"}

    plain = VoxCeleb2Source().get_source(str(tmp_path))
    assert {record["caption"] for record in plain} == {VoxCeleb2Source.DEFAULT_PROMPT_TEMPLATE}

    # A template with a placeholder we do not supply must not explode
    odd = VoxCeleb2Source(prompt_template="a video of {speaker}")
    assert {record["caption"] for record in odd.get_source(str(tmp_path))} == {"a video of {speaker}"}


def test_voxceleb2_source_reads_the_requested_split(tmp_path):
    _voxceleb_tree(tmp_path, split="train")
    _voxceleb_tree(tmp_path, split="test", identities=("id00017",))

    assert len(VoxCeleb2Source(split="train").get_source(str(tmp_path))) == 4
    assert len(VoxCeleb2Source(split="test").get_source(str(tmp_path))) == 2


def test_voxceleb2_source_reports_missing_roots_clearly(tmp_path):
    with pytest.raises(ValueError, match="dataset root directory"):
        VoxCeleb2Source().get_source(None)
    with pytest.raises(ValueError, match="split 'train' not found"):
        VoxCeleb2Source().get_source(str(tmp_path))


def test_voxceleb2_is_registered_on_the_media_pipeline():
    dataset = mediaDatasetMap["voxceleb2"]
    assert isinstance(dataset.source, VoxCeleb2Source)
    assert isinstance(dataset.augmenter, AudioVideoAugmenter)
    assert dataset.media_type == "video"


def test_voxceleb2_records_flow_through_the_audio_video_transform(tmp_path, monkeypatch):
    """End to end with the AV reader and audio model stubbed out: the parts that
    need decord/pyav/wav2vec2 weights are exactly the parts we fake."""
    _voxceleb_tree(tmp_path)
    records = VoxCeleb2Source(prompt_template="a video of {identity}").get_source(str(tmp_path))

    frame_samples = 640
    seen = {}

    def fake_read_av_random_clip(video_path, num_frames, audio_frame_padding, random_seed):
        seen.update(video_path=video_path, num_frames=num_frames,
                    audio_frame_padding=audio_frame_padding, random_seed=random_seed)
        padded = num_frames + 2 * audio_frame_padding
        full_audio = np.zeros((padded, frame_samples), dtype=np.float32)
        framewise = np.zeros((1, num_frames, 1, frame_samples), dtype=np.float32)
        # Native resolution differs from frame_size, so the resize must happen
        frames = np.zeros((num_frames, 96, 48, 3), dtype=np.uint8)
        return framewise, full_audio, frames

    class FakeAudioProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, audio):
            return {"input_values": np.asarray(audio)[None, ...]}

    monkeypatch.setattr(videos_module, "read_av_random_clip", fake_read_av_random_clip)
    monkeypatch.setattr(videos_module, "AutoAudioProcessor", FakeAudioProcessor)

    transform = AudioVideoAugmenter().create_transform(
        frame_size=32, sequence_length=4, audio_frame_padding=1)()
    batch = transform.random_map(records[0], np.random.default_rng(0))

    assert seen["video_path"] == records[0]["video_path"]
    assert seen["num_frames"] == 4 and seen["audio_frame_padding"] == 1
    assert batch["video"].shape == (4, 32, 32, 3)          # resized to frame_size
    assert batch["caption"] == "a video of id00012"        # template survives
    assert batch["audio"]["full_audio"].shape == (6, frame_samples)
    assert batch["audio"]["framewise_audio"].shape == (1, 4, 1, frame_samples)
    assert batch["audio"]["input_values"].shape == (6, frame_samples)


# ---------------------------------------------------------------------------------
# Local video source
# ---------------------------------------------------------------------------------

def test_video_local_source_lists_and_caches_paths(tmp_path):
    """Constructing with a directory raised AttributeError: load_paths read
    self.directory before __init__ ever set it."""
    clips = _voxceleb_tree(tmp_path)
    cache_dir = tmp_path / "cache"

    source = VideoLocalSource(directory=str(tmp_path), cache_dir=str(cache_dir))
    records = source.get_source()

    assert [record["video_path"] for record in records] == sorted(str(c) for c in clips)

    # The cache key must survive a new process: hash() of a str is salted per run
    cached = list(cache_dir.iterdir())
    assert len(cached) == 1
    reloaded = VideoLocalSource(directory=str(tmp_path), cache_dir=str(cache_dir))
    assert reloaded.get_source() == records


def test_video_local_source_without_a_directory_says_so():
    with pytest.raises(ValueError, match="no directory to read"):
        VideoLocalSource().get_source()


# ---------------------------------------------------------------------------------
# Moved scripts
# ---------------------------------------------------------------------------------

def test_av_benchmark_script_imports_against_the_real_av_utils():
    """It used to import a read_av_batch that av_utils never defined."""
    script = REPO_ROOT / "scripts" / "av_benchmark.py"
    spec = importlib.util.spec_from_file_location("av_benchmark_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.benchmark_av_reading)
    assert callable(module.read_av_improved)
    assert not (REPO_ROOT / "flaxdiff" / "data" / "sources" / "av_example.py").exists()
