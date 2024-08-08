#!/bin/bash

img2dataset --url_list ./cc12m.tsv --input_format "tsv"\
         --url_col "image_url" --caption_col "caption" --output_format arrayrecord\
           --output_folder gs://flaxdiff-datasets-regional/arrayrecord2/cc12m --processes_count 64\
          --thread_count 64 --image_size 256 --number_sample_per_shard 50000 --min_image_size 100 \
             --enable_wandb True --disallowed_header_directives '[]' --compute_hash None --max_shard_retry 3 --timeout 60
