#!/bin/bash

img2dataset --url_list ./datacache/cc12m.csv --input_format "csv"\
         --url_col "image_url" --caption_col "caption" --output_format arrayrecord\
           --output_folder gs://flaxdiff-datasets/arrayrecord/cc12m --processes_count 240 --thread_count 64 --image_size 256\
             --enable_wandb True --disallowed_header_directives '[]' --compute_hash None --max_shard_retry 3 --timeout 60

./gcsfuse.sh DATASET_GCS_BUCKET=flaxdiff-datasets MOUNT_PATH=gcs_mount