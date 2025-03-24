from .sources.tfds import data_source_tfds, tfds_augmenters
from .sources.gcs import data_source_gcs, data_source_combined_gcs, gcs_augmenters

# Configure the following for your datasets
datasetMap = {
    "oxford_flowers102": {
        "source": data_source_tfds("oxford_flowers102", use_tf=False),
        "augmenter": tfds_augmenters,
    },
    "cc12m": {
        "source": data_source_gcs('arrayrecord2/cc12m'),
        "augmenter": gcs_augmenters,
    },
    "laiona_coco": {
        "source": data_source_gcs('arrayrecord2/laion-aesthetics-12m+mscoco-2017'),
        "augmenter": gcs_augmenters,
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
            # "gs://flaxdiff-datasets-regional/datasets/laion-aesthetics-12m+mscoco-2017.parquet"
            # "ChristophSchuhmann/MS_COCO_2017_URL_TEXT",
            # "dclure/laion-aesthetics-12m-umap",
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
            # "gs://flaxdiff-datasets-regional/datasets/laiion400m-185M"
        ]
    }
}