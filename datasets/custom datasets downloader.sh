#!/bin/bash

img2dataset --url_list $1 --input_format "parquet"\
         --url_col "url" --caption_col "caption" --output_format arrayrecord\
           --output_folder $2 --processes_count 64 --thread_count 64 \
          --image_size 256 --min_image_size 100 --number_sample_per_shard 80000 --max_aspect_ratio 2.4\
             --enable_wandb True --disallowed_header_directives '[]' --compute_hash None --max_shard_retry 3 --timeout 60

# gs://flaxdiff-datasets-regional/arrayrecord/laion-aesthetics-12m+mscoco-2017
