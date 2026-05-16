#!/usr/bin/bash

# Falcon: all 5 stages ran but Stage-6 is 100% null
rm -rf output/im2gps3k_rgb_images/cmp_shard_falcon_{0,1,2,3}_of_4
rm -rf output/im2gps3k_rgb_images/cmp_shard_falcon_merged

sbatch run_comparison.sh --only falcon

python3 merge_all_shards.py output/im2gps3k_rgb_images
