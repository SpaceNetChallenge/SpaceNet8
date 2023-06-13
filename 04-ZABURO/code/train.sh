#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate sn8

# A sample call to your training script follows. Note that folder names are for example only, you should not assume that the exact same folder names will be used in testing.
# ./train.sh /data/SN8_flood/train/
# In this sample case the training data looks like this:
#   data/
#     SN8_flood/
#       train/
#         Germany_Training_Public/
#           POST-event/
#           PRE-event/
#           annotations/
#           Germany_Training_Public_label_image_mapping.csv
#           Germany_Training_Public_reference.csv
#         Louisiana-East_Training_Public/
#           POST-event/
#           ... etc., other folders and files for this AOI
#         resolutions.txt

# TEST
touch /wdata/TEST_WORKING_DIRECTORY_IS_WRITABLE

rsync -av $1/ /wdata/

cd /root/
time bash download_external_data.sh

cd /root/
time bash setup_training_data.sh /wdata

cd /root/
for fold_index in 0 1 2 3; do
    python scripts/roads/train.py train \
        --config_path scripts/roads/configs/efficientnetv2m-junction-01.yaml \
        training.device="cuda:${fold_index}" \
        fold_index=${fold_index} &
done
wait

for fold_index in 0 1 2 3; do
    python scripts/buildings/train.py train \
        --config_path scripts/buildings/configs/efficientnetv2s.yaml \
        training.device="cuda:${fold_index}" \
        fold_index=${fold_index} &
done
wait

for fold_index in 0 1 2 3; do
    python scripts/floods/train.py train \
        --config_path scripts/floods/configs/resnet50-cj-dice-half-flag-xview2.yaml \
        training.device="cuda:${fold_index}" \
        fold_index=${fold_index} &
done
wait
