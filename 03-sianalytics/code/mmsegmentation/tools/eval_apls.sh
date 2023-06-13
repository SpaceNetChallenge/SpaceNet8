
ROAD_PRED_DIR=''
OUT_DIR=''
OUT_SKNW_WKT="${OUT_DIR}/sknw_wkt.csv"
GRAPH_NO_SPEED_DIR="$OUT_DIR/graph_nospeed"
IMG_DIR='/nas/Dataset/SpaceNet8/Training/Louisiana-East_Training_Public'


python3 ../baseline/postprocessing/roads/vectorize_roads.py --im_dir ${ROAD_PRED_DIR} \
   --out_dir $OUT_DIR --band 2 --band-val --suffix '' \
   --write_shps --write_graphs --write_csvs --write_skeletons

python3 ../baseline/postprocessing/roads/wkt_to_G.py --wkt_submission $OUT_SKNW_WKT \
    --graph_dir $GRAPH_NO_SPEED_DIR \
    --min_subgraph_length_pix 300 --min_spur_length_m 10

python3 ../apls/apls_sn8.py --test_method=gt_json_prop_pkl --output_name=road \
    --truth_dir="$IMG_DIR/annotations/" \
    --prop_dir=$GRAPH_NO_SPEED_DIR \
    --im_dir="$IMG_DIR/PRE-event/" \
    --n_plots=300
