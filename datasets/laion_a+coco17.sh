#!/bin/bash

img2dataset --url_list $HOME/research/laion-aesthetics-12m+mscoco-2017.parquet --input_format "parquet"\
         --url_col "url" --caption_col "caption" --output_format arrayrecord\
           --output_folder gs://flaxdiff-datasets-regional/arrayrecord/laion-aesthetics-12m+mscoco-2017 --processes_count 64\
          --thread_count 64 --image_size 256 --min_image_size 100 \
             --enable_wandb True --disallowed_header_directives '[]' --compute_hash None --max_shard_retry 3 --timeout 60
