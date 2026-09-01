"""VoxCeleb2 audio-video source for the unified media pipeline.

This module was rebuilt, not restored. Its previous contents were a torch
`Dataset` lifted from another project: it imported `d2lv2_lightning`,
`.face_mask`, `.prompt_templates` and `.audio`, none of which have ever existed
in this repository, so the module could not be imported at all. The pipeline it
fed (face masks, mel windows, template token maps) is out of scope here.

What remains is the part FlaxDiff needs: enumerate a VoxCeleb2 tree and hand the
grain pipeline one record per utterance. Clip sampling, audio features and
batching are the AudioVideoAugmenter's job (see `sources/videos.py`), which
reads clips with `av_utils.read_av_random_clip` and featurises audio with
`flaxdiff.utils.AutoAudioProcessor`. Every AV dependency is therefore imported
lazily, inside those readers - importing this module costs nothing.
"""

import os
from typing import Any, Dict, List, Optional, Sequence

from .base import DataSource


class VoxCeleb2Source(DataSource):
    """Data source over a local VoxCeleb2 directory tree.

    Follows the dataset's own layout, `<root>/<split>/<identity>/<clip>/<utterance>.mp4`,
    and scans it recursively, so extra nesting is tolerated. Each record is
    ``{"video_path": ..., "caption": ...}``: the video path for the augmenter's
    clip reader, and a caption rendered from a configurable template - `{identity}`
    in the template is replaced with the speaker directory name.
    """

    DEFAULT_EXTENSIONS = ('.mp4', '.avi')
    DEFAULT_PROMPT_TEMPLATE = "a video of a person speaking"

    def __init__(
        self,
        split: str = "train",
        extensions: Optional[Sequence[str]] = None,
        prompt_template: Optional[str] = None,
    ):
        """Initialize a VoxCeleb2 data source.

        Args:
            split: Sub-directory of the dataset root to read ("train" or "test").
            extensions: Video file extensions to accept.
            prompt_template: Caption template; may contain "{identity}".
        """
        self.split = split
        self.extensions = tuple(
            ext.lower() if ext.startswith('.') else f'.{ext}'.lower()
            for ext in (extensions or self.DEFAULT_EXTENSIONS)
        )
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def get_source(self, path_override: str) -> List[Dict[str, Any]]:
        """Scan the dataset tree and return one record per utterance.

        Args:
            path_override: VoxCeleb2 dataset root, holding the split directories.

        Returns:
            A list of {"video_path", "caption"} dicts, ordered deterministically.
        """
        if not path_override:
            raise ValueError(
                "VoxCeleb2Source needs the dataset root directory, e.g. "
                "get_media_dataset_grain(..., dataset_source='/data/voxceleb2')."
            )

        split_root = os.path.join(path_override, self.split)
        if not os.path.isdir(split_root):
            raise ValueError(
                f"VoxCeleb2 split {self.split!r} not found: {split_root!r} is not a "
                "directory. Expected <root>/<split>/<identity>/<clip>/<utterance>.mp4."
            )

        records = []
        for directory, _, files in sorted(os.walk(split_root)):
            identity = os.path.relpath(directory, split_root).split(os.sep)[0]
            caption = self._render_caption(identity)
            for file_name in sorted(files):
                if os.path.splitext(file_name)[1].lower() in self.extensions:
                    records.append({
                        "video_path": os.path.join(directory, file_name),
                        "caption": caption,
                    })
        return records

    def _render_caption(self, identity: str) -> str:
        """Fill the template, tolerating templates without a placeholder."""
        try:
            return self.prompt_template.format(identity=identity)
        except (KeyError, IndexError, ValueError):
            return self.prompt_template
