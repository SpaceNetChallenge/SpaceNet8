#!/bin/bash

START_TIME=$SECONDS

TEST_DIR=$1
OUT_PATH=$2

rm -rf /wdata/test.csv
rm -rf /wdata/warped_posts_test/
rm -rf /wdata/logs/test/
rm -rf /wdata/preds/
rm -rf /wdata/ensembled_preds/
rm -rf /wdata/refined_preds*/
rm -rf /wdata/building_submissions/
rm -rf /wdata/road_vectors/ 
rm -rf /wdata/road_graphs/
rm -rf /wdata/road_submissions/
rm -rf /wdata/submissions

# preprocess
python tools/make_test_csv.py --test_dir $TEST_DIR
python tools/warp_post_images.py --root_dir $TEST_DIR --test
python tools/measure_image_similarities.py --test_dir $TEST_DIR --test_only

# inference
LOG_DIR=/wdata/logs/test
mkdir -p $LOG_DIR

FOUNDATION_ARGS=" --override_model_dir /work/models Data.test_dir=$TEST_DIR"
FLOOD_ARGS=" --use_ema_weight --tta_hflip_channels 0 --override_model_dir /work/models Data.test_dir=$TEST_DIR"

echo ""
echo "predicting... (1/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 50000 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_50000.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 50001 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_50001.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 50002 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_50002.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 50003 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_50003.txt 2>&1 &

wait

echo ""
echo "predicting... (2/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 50004 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_50004.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 60400 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_60400.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 60401 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_60401.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 60402 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_60402.txt 2>&1 &

wait

echo ""
echo "predicting... (3/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 60403 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_60403.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 60404 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_60404.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 50010 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_50010.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 50011 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_50011.txt 2>&1 &

wait

echo ""
echo "predicting... (4/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 50012 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_50012.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 50013 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_50013.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 50014 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_50014.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 60420 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_60420.txt 2>&1 &

wait

echo ""
echo "predicting... (5/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 60421 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_60421.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 60422 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_60422.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 60423 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_60423.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 60424 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_60424.txt 2>&1 &

wait

echo ""
echo "predicting... (6/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 80000 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80000.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 80001 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80001.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 80002 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80002.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 80003 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80003.txt 2>&1 &

wait

echo ""
echo "predicting... (7/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 80004 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80004.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 80500 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80500.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 80501 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80501.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 80502 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80502.txt 2>&1 &

wait

echo ""
echo "predicting... (8/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 80503 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80503.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 80504 \
    $FOUNDATION_ARGS \
    > $LOG_DIR/exp_80504.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 90010 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_90010.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --exp_id 90011 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_90011.txt 2>&1 &

wait

echo ""
echo "predicting... (9/9)"
echo "this will take ~6 min"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    --exp_id 90012 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_90012.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
    --exp_id 90013 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_90013.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/test_net.py \
    --exp_id 90014 \
    $FLOOD_ARGS \
    > $LOG_DIR/exp_90014.txt 2>&1 &

wait

# ensemble
python tools/ensemble.py --exp_id 50000 50001 50002 50003 50004 60400 60401 60402 60403 60404 --root_dir $TEST_DIR
python tools/ensemble.py --exp_id 50010 50011 50012 50013 50014 60420 60421 60422 60423 60424 --root_dir $TEST_DIR
python tools/ensemble.py --exp_id 80000 80001 80002 80003 80004 --root_dir $TEST_DIR
python tools/ensemble.py --exp_id 80500 80501 80502 80503 80504 --root_dir $TEST_DIR
python tools/ensemble.py --exp_id 90010 90011 90012 90013 90014 --root_dir $TEST_DIR

# postprocess
FOUNDATION_DIR="exp_50000-50001-50002-50003-50004-60400-60401-60402-60403-60404"
FLOOD_DIR="exp_50010-50011-50012-50013-50014-60420-60421-60422-60423-60424"
ROAD_DIR="exp_80000-80001-80002-80003-80004"
BUILDING_DIR="exp_80500-80501-80502-80503-80504"
FLOOD_BUILDING_DIR="exp_90010-90011-90012-90013-90014"

python tools/refine_road_mask.py --foundation /wdata/ensembled_preds/$FOUNDATION_DIR --road /wdata/ensembled_preds/$ROAD_DIR  # foundation: ensembled_preds -> refined_preds
python tools/refine_building_mask.py --foundation /wdata/refined_preds/$FOUNDATION_DIR --building /wdata/ensembled_preds/$BUILDING_DIR  # foundation: refined_preds -> refined_preds_2
python tools/refine_flood_building_mask.py --flood /wdata/ensembled_preds/$FLOOD_DIR --building /wdata/ensembled_preds/$FLOOD_BUILDING_DIR  # flood: ensembled_preds -> refined_preds_3

python tools/building/postproc_building.py --foundation /wdata/refined_preds_2/$FOUNDATION_DIR --flood /wdata/refined_preds_3/$FLOOD_DIR

python tools/road/vectorize.py --foundation /wdata/refined_preds_2/$FOUNDATION_DIR
python tools/road/to_graph.py --vector /wdata/road_vectors/$FOUNDATION_DIR
python tools/road/insert_flood.py --graph /wdata/road_graphs/$FOUNDATION_DIR --flood /wdata/refined_preds_3/$FLOOD_DIR

POSTPROCESSED_DIR="exp_50000-50001-50002-50003-50004-60400-60401-60402-60403-60404_50010-50011-50012-50013-50014-60420-60421-60422-60423-60424"
python tools/submit.py --building /wdata/building_submissions/$POSTPROCESSED_DIR --road /wdata/road_submissions/$POSTPROCESSED_DIR --out $OUT_PATH

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Total time for testing: " $(($ELAPSED_TIME / 60 + 1)) "[min]"
