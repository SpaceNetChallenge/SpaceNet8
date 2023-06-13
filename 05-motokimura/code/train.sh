#!/bin/bash

START_TIME=$SECONDS

TRAIN_DIR=$1

# remove motokimura's home-built models
rm -rf /work/models

# preprocess
python tools/make_folds_v3.py --train_dir $TRAIN_DIR
python tools/prepare_building_masks.py --train_dir $TRAIN_DIR
python tools/prepare_road_masks.py --train_dir $TRAIN_DIR
python tools/warp_post_images.py --root_dir $TRAIN_DIR

echo "mosaicing.. this will take ~20 mins"
echo "you can check progress from /wdata/mosaics/train_*.txt"
mkdir -p /wdata/mosaics/
nohup python tools/mosaic.py --fold_id 0 --train_dir $TRAIN_DIR > /wdata/mosaics/train_0.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 1 --train_dir $TRAIN_DIR > /wdata/mosaics/train_1.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 2 --train_dir $TRAIN_DIR > /wdata/mosaics/train_2.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 3 --train_dir $TRAIN_DIR > /wdata/mosaics/train_3.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 4 --train_dir $TRAIN_DIR > /wdata/mosaics/train_4.txt 2>&1 &
wait

python tools/measure_image_similarities.py --train_dir $TRAIN_DIR --train_only

# training
LOG_DIR=/wdata/logs/train
mkdir -p $LOG_DIR

ARGS=" --override_model_dir /work/models --disable_wandb Data.train_dir=$TRAIN_DIR"
# comment out the line below for dryrun
#ARGS=$ARGS" General.epochs=2"

echo ""
echo "training... (1/9)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation \
    --exp_id 50000 \
    --fold_id 0 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50000.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation \
    --exp_id 50001 \
    --fold_id 1 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50001.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task foundation \
    --exp_id 50002 \
    --fold_id 2 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50002.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task foundation \
    --exp_id 50003 \
    --fold_id 3 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50003.txt 2>&1 &

wait

echo ""
echo "training... (2/9)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation \
    --exp_id 50004 \
    --fold_id 4 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50004.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation \
    --exp_id 60400 \
    --fold_id 0 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60400.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task foundation \
    --exp_id 60401 \
    --fold_id 1 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60401.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task foundation \
    --exp_id 60402 \
    --fold_id 2 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60402.txt 2>&1 &

wait

echo ""
echo "training... (3/9)"
echo "this will take ~3 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation \
    --exp_id 60403 \
    --fold_id 3 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60403.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation \
    --exp_id 60404 \
    --fold_id 4 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60404.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood \
    --exp_id 50010 \
    --pretrained 50000 \
    --fold_id 0 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50010.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task flood \
    --exp_id 50011 \
    --pretrained 50001 \
    --fold_id 1 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50011.txt 2>&1 &

wait

echo ""
echo "training... (4/9)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task flood \
    --exp_id 50012 \
    --pretrained 50002 \
    --fold_id 2 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50012.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task flood \
    --exp_id 50013 \
    --pretrained 50003 \
    --fold_id 3 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50013.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood \
    --exp_id 50014 \
    --pretrained 50004 \
    --fold_id 4 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50014.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task flood \
    --exp_id 60420 \
    --pretrained 60400 \
    --fold_id 0 \
    --config configs/flood/effnet-b6_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_60420.txt 2>&1 &

wait

echo ""
echo "training... (5/9)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task flood \
    --exp_id 60421 \
    --pretrained 60401 \
    --fold_id 1 \
    --config configs/flood/effnet-b6_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_60421.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task flood \
    --exp_id 60422 \
    --pretrained 60402 \
    --fold_id 2 \
    --config configs/flood/effnet-b6_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_60422.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood \
    --exp_id 60423 \
    --pretrained 60403 \
    --fold_id 3 \
    --config configs/flood/effnet-b6_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_60423.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task flood \
    --exp_id 60424 \
    --pretrained 60404 \
    --fold_id 4 \
    --config configs/flood/effnet-b6_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_60424.txt 2>&1 &

wait

echo ""
echo "training... (6/9)"
echo "this will take ~3 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation_xdxd_sn5 \
    --exp_id 80000 \
    --fold_id 0 \
    --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold0/fold0_best.pth \
    $ARGS \
    > $LOG_DIR/exp_80000.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation_xdxd_sn5 \
    --exp_id 80001 \
    --fold_id 1 \
    --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold1/fold1_best.pth \
    $ARGS \
    > $LOG_DIR/exp_80001.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task foundation_xdxd_sn5 \
    --exp_id 80002 \
    --fold_id 2 \
    --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold2/fold2_best.pth \
    $ARGS \
    > $LOG_DIR/exp_80002.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task foundation_xdxd_sn5 \
    --exp_id 80003 \
    --fold_id 3 \
    --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold3/fold3_best.pth \
    $ARGS \
    > $LOG_DIR/exp_80003.txt 2>&1 &

wait

echo ""
echo "training... (7/9)"
echo "this will take ~3 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation_xdxd_sn5 \
    --exp_id 80004 \
    --fold_id 4 \
    --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold0/fold0_best.pth \
    $ARGS \
    > $LOG_DIR/exp_80004.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation_selimsef_xview2 \
    --exp_id 80500 \
    --fold_id 0 \
    --pretrained_path /work/selimsef_xview2_models/localization_densenet_unet_densenet161_3_0_best_dice \
    $ARGS \
    > $LOG_DIR/exp_80500.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task foundation_selimsef_xview2 \
    --exp_id 80501 \
    --fold_id 1 \
    --pretrained_path /work/selimsef_xview2_models/localization_densenet_unet_densenet161_3_1_best_dice \
    $ARGS \
    > $LOG_DIR/exp_80501.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task foundation_selimsef_xview2 \
    --exp_id 80502 \
    --fold_id 2 \
    --pretrained_path /work/selimsef_xview2_models/localization_densenet_unet_densenet161_3_0_best_dice \
    $ARGS \
    > $LOG_DIR/exp_80502.txt 2>&1 &

wait

echo ""
echo "training... (8/9)"
echo "this will take ~3 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation_selimsef_xview2 \
    --exp_id 80503 \
    --fold_id 3 \
    --pretrained_path /work/selimsef_xview2_models/localization_densenet_unet_densenet161_3_1_best_dice \
    $ARGS \
    > $LOG_DIR/exp_80503.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation_selimsef_xview2 \
    --exp_id 80504 \
    --fold_id 4 \
    --pretrained_path /work/selimsef_xview2_models/localization_densenet_unet_densenet161_3_0_best_dice \
    $ARGS \
    > $LOG_DIR/exp_80504.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood_selimsef_xview2 \
    --exp_id 90010 \
    --fold_id 0 \
    --pretrained_path /work/selimsef_xview2_models/pseudo_densenet_seamese_unet_shared_densenet161_0_best_xview \
    --config configs/flood_selimsef_xview2/ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_90010.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task flood_selimsef_xview2 \
    --exp_id 90011 \
    --fold_id 1 \
    --pretrained_path /work/selimsef_xview2_models/pseudo_densenet_seamese_unet_shared_densenet161_2_best_xview \
    --config configs/flood_selimsef_xview2/ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_90011.txt 2>&1 &

wait

echo ""
echo "training... (9/9)"
echo "this will take ~3 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task flood_selimsef_xview2 \
    --exp_id 90012 \
    --fold_id 2 \
    --pretrained_path /work/selimsef_xview2_models/pseudo_densenet_seamese_unet_shared_densenet161_0_best_xview \
    --config configs/flood_selimsef_xview2/ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_90012.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task flood_selimsef_xview2 \
    --exp_id 90013 \
    --fold_id 3 \
    --pretrained_path /work/selimsef_xview2_models/pseudo_densenet_seamese_unet_shared_densenet161_2_best_xview \
    --config configs/flood_selimsef_xview2/ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_90013.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood_selimsef_xview2 \
    --exp_id 90014 \
    --fold_id 4 \
    --pretrained_path /work/selimsef_xview2_models/pseudo_densenet_seamese_unet_shared_densenet161_0_best_xview \
    --config configs/flood_selimsef_xview2/ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_90014.txt 2>&1 &

wait

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Total time for training: " $(($ELAPSED_TIME / 60 + 1)) "[min]"
