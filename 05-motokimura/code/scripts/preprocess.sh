#!/bin/bash
# example usage: ./scripts/this_script.sh

#python tools/make_folds_v1.py
#python tools/make_folds_v2.py
python tools/make_folds_v3.py
python tools/make_test_csv.py
python tools/prepare_building_masks.py
python tools/prepare_road_masks.py
python tools/warp_post_images.py
python tools/warp_post_images.py --root_dir /data/test/ --test

# optional
python tools/visualize_dataset.py
python tools/visualize_dataset.py --root_dir /data/test/ --test
