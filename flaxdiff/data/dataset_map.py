from .sources.base import MediaDataset, DataSource, DataAugmenter
from .sources.images import ImageTFDSSource, ImageGCSSource, CombinedImageGCSSource
from .sources.images import ImageTFDSAugmenter, ImageGCSAugmenter
from .sources.videos import VideoTFDSSource, VideoLocalSource, AudioVideoAugmenter

# ---------------------------------------------------------------------------------
# Legacy compatibility mappings
# ---------------------------------------------------------------------------------

from .sources.images import data_source_tfds, tfds_augmenters, data_source_gcs
from .sources.images import data_source_combined_gcs, gcs_augmenters, gcs_filters

# Configure the following for your datasets.
#
# MSML612 project bucket (gs://msml612-diffusion-data/arrayrecord2/...) — the
# four datasets we actually built and uploaded in this project. These are the
# ones you want to use for real LAION-scale training. Paths are relative to
# the fuse mount at /home/mrwhite0racle/gcs_mount (see ImageGCSSource).
datasetMap = {
    "oxford_flowers102": {
        "source": data_source_tfds("oxford_flowers102", use_tf=False),
        "augmenter": tfds_augmenters,
    },

    # --- MSML612 project datasets (our current work) ---
    "laion12m_coco": {
        # LAION-aesthetics-12M (aesthetic score >=6) unioned with MS-COCO 2017.
        # 228 shards, 236 GiB, ~15M successful samples. Built via img2dataset.
        "source": data_source_gcs('arrayrecord2/laion12m_coco'),
        "augmenter": gcs_augmenters,
    },
    "laion2b_aesthetic": {
        # LAION-2B-en aesthetic >=4.2 subset. 569 shards, 550 GiB. Much larger
        # than laion12m_coco but noisier. Built via multiple img2dataset resumes.
        "source": data_source_gcs('arrayrecord2/laion2B-en-aesthetic'),
        "augmenter": gcs_augmenters,
    },
    "diffusiondb": {
        # DiffusionDB (Stable Diffusion synthetic images + prompts).
        # 31 shards, 60 GiB, 1.97M samples. Built via convert_hf_to_arrayrecord.
        "source": data_source_gcs('arrayrecord2/diffusiondb'),
        "augmenter": gcs_augmenters,
    },
    "cc3m": {
        # Conceptual Captions 3M (natural photos with short captions).
        # 50 shards, 37 GiB, ~3.3M samples. Shard 00039 is missing
        # (~65k samples gap, 2% of total). Built via img2dataset.
        "source": data_source_gcs('arrayrecord2/cc3m'),
        "augmenter": gcs_augmenters,
    },
    "combined_msml612": {
        # Union of all 4 MSML612 project datasets for full-scale training.
        # Total ~883 GiB, ~20M+ usable samples. Use this for big training runs.
        "source": data_source_combined_gcs([
            'arrayrecord2/laion12m_coco',
            'arrayrecord2/laion2B-en-aesthetic',
            'arrayrecord2/diffusiondb',
            'arrayrecord2/cc3m',
        ]),
        "augmenter": gcs_augmenters,
    },

    # --- Legacy entries from prior mlops-msml605-project (paths may not
    # exist on the current msml612-diffusion-data bucket; kept for backward
    # compatibility with scripts that still reference these names) ---
    "cc12m": {
        "source": data_source_gcs('arrayrecord2/cc12m'),
        "augmenter": gcs_augmenters,
    },
    "laiona_coco": {
        "source": data_source_gcs('datasets/laion12m+mscoco'),
        "augmenter": gcs_augmenters,
        "filter": gcs_filters,
    },
    "aesthetic_coyo": {
        "source": data_source_gcs('arrayrecords/aestheticCoyo_0.25clip_6aesthetic'),
        "augmenter": gcs_augmenters,
    },
    "combined_aesthetic": {
        "source": data_source_combined_gcs([
                'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
                'arrayrecord2/cc12m',
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
            ]),
        "augmenter": gcs_augmenters,
    },
    "laiona_coco_coyo": {
        "source": data_source_combined_gcs([
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
                'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
                'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
            ]),
        "augmenter": gcs_augmenters,
    },
    "combined_30m": {
        "source": data_source_combined_gcs([
                'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
                'arrayrecord2/cc12m',
                'arrayrecord2/aestheticCoyo_0.26_clip_5.5aesthetic_256plus',
                "arrayrecord2/playground+leonardo_x4+cc3m.parquet",
            ]),
        "augmenter": gcs_augmenters,
    }
}

onlineDatasetMap = {
    "combined_online": {
        "source": [
            "gs://flaxdiff-datasets-regional/datasets/laion-aesthetics-12m+mscoco-2017",
            "gs://flaxdiff-datasets-regional/datasets/coyo700m-aesthetic-5.4_25M",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/cc12m",
            "gs://flaxdiff-datasets-regional/datasets/playground-liked",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/leonardo-liked-1.8m",
            "gs://flaxdiff-datasets-regional/datasets/cc3m",
            "gs://flaxdiff-datasets-regional/datasets/cc3m",
            "gs://flaxdiff-datasets-regional/datasets/laion2B-en-aesthetic-4.2_37M",
        ]
    }
}

# ---------------------------------------------------------------------------------
# New media datasets configuration with the unified architecture
# ---------------------------------------------------------------------------------

mediaDatasetMap = {
    # Image datasets
    "oxford_flowers102": MediaDataset(
        source=ImageTFDSSource(name="oxford_flowers102", use_tf=False),
        augmenter=ImageTFDSAugmenter(),
        media_type="image"
    ),
    "cc12m": MediaDataset(
        source=ImageGCSSource(source='arrayrecord2/cc12m'),
        augmenter=ImageGCSAugmenter(),
        media_type="image"
    ),
    "laiona_coco": MediaDataset(
        source=ImageGCSSource(source='arrayrecord2/laion-aesthetics-12m+mscoco-2017'),
        augmenter=ImageGCSAugmenter(),
        media_type="image"
    ),
    "combined_aesthetic": MediaDataset(
        source=CombinedImageGCSSource(sources=[
            'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
            'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
            'arrayrecord2/cc12m',
            'arrayrecords/aestheticCoyo_0.25clip_6aesthetic',
        ]),
        augmenter=ImageGCSAugmenter(),
        media_type="image"
    ),
    "combined_30m": MediaDataset(
        source=CombinedImageGCSSource(sources=[
            'arrayrecord2/laion-aesthetics-12m+mscoco-2017',
            'arrayrecord2/cc12m',
            'arrayrecord2/aestheticCoyo_0.26_clip_5.5aesthetic_256plus',
            "arrayrecord2/playground+leonardo_x4+cc3m.parquet",
        ]),
        augmenter=ImageGCSAugmenter(),
        media_type="image"
    ),
    
    # Video dataset
    "voxceleb2": MediaDataset(
        source=VideoLocalSource(),
        augmenter=AudioVideoAugmenter(),
        media_type="video"
    ),
}