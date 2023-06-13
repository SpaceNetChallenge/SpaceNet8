. /opt/conda/etc/profile.d/conda.sh
conda activate sn8

cd /root/
wdata_base=$1

# --- SN5 ---
road_out_image_dir="${wdata_base}/road_train/images_8bit_base/PS-RGB"
road_out_mask_dir="${wdata_base}/road_train/masks_base/train_mask_binned"
road_out_mask_dir_mc="${wdata_base}/road_train/masks_base/train_mask_binned_mc"
road_out_mask_dir_junction="${wdata_base}/road_train/masks_base/train_mask_binned_junction"
output_conversion_csv="${wdata_base}/road_train/masks_base/roads_train_speed_conversion_binned.csv"

mkdir -p ${road_out_image_dir}
mkdir -p ${road_out_mask_dir}
mkdir -p ${road_out_mask_dir_mc}
mkdir -p ${road_out_mask_dir_junction}
mkdir -p ${output_conversion_csv}

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum AOI_7_Moscow AOI_8_Mumbai; do
    python scripts/preprocess/create_8bit_images.py \
        --indir ${wdata_base}/${AOI}/PS-MS \
        --outdir ${road_out_image_dir} \
        --rescale_type=perc \
        --percentiles=2,98 \
        --band_order=5,3,2 &
done
wait

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum AOI_7_Moscow AOI_8_Mumbai; do
    python scripts/preprocess/create_speed_masks_sn5.py \
        --geojson_dir ${wdata_base}/${AOI}/geojson_roads_speed \
        --image_dir ${wdata_base}/${AOI}/PS-MS \
        --output_conversion_csv ${output_conversion_csv} \
        --output_mask_dir ${road_out_mask_dir} \
        --output_mask_multidim_dir ${road_out_mask_dir_mc} \
        --buffer_distance_meters 2 &
done
wait

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum AOI_7_Moscow AOI_8_Mumbai; do
    python scripts/preprocess/create_junction_masks_sn5.py \
        --geojson_dir ${wdata_base}/${AOI}/geojson_roads_speed \
        --image_dir ${wdata_base}/${AOI}/PS-MS \
        --output_mask_dir ${road_out_mask_dir_junction} &
done
wait

# --- SN2 ---
building_out_image_dir="${wdata_base}/building_train/images_8bit_base"
building_out_mask_dir="${wdata_base}/building_train/masks_base"

mkdir -p ${building_out_image_dir}
mkdir -p ${building_out_mask_dir}

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    python scripts/preprocess/create_8bit_images.py \
        --indir ${wdata_base}/${AOI}_Train/MUL-PanSharpen \
        --outdir ${building_out_image_dir} \
        --rescale_type=perc \
        --percentiles=2,98 \
        --band_order=5,3,2 &
done
wait

for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
    python scripts/preprocess/create_building_masks.py sn2 \
        --image_dir ${wdata_base}/${AOI}_Train/MUL-PanSharpen \
        --geojson_dir ${wdata_base}/${AOI}_Train/geojson/buildings \
        --output_dir ${building_out_mask_dir} &
done
wait

# --- SN8 ---
python baseline/baseline/data_prep/geojson_prep.py \
    --root_dir ${wdata_base} \
    --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

# For flood mask
python baseline/baseline/data_prep/create_masks.py \
    --root_dir ${wdata_base} \
    --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

# Road
for AOI in Germany Louisiana-East; do
    find ${wdata_base}/${AOI}_Training_Public/PRE-event -name '*.tif' -type f -print \
        | xargs -I {} basename {} \
        | xargs -I {} ln ${wdata_base}/${AOI}_Training_Public/PRE-event/{} ${road_out_image_dir}/{}
done
for AOI in Germany Louisiana-East; do
    python scripts/preprocess/create_speed_masks_sn8.py \
        --data_dir ${wdata_base}/${AOI}_Training_Public \
        --output_conversion_csv ${output_conversion_csv} \
        --output_mask_dir ${road_out_mask_dir} \
        --output_mask_multidim_dir ${road_out_mask_dir_mc} \
        --buffer_distance_meters 2 &
done
wait
for AOI in Germany Louisiana-East; do
    python scripts/preprocess/create_junction_masks_sn8.py \
        --data_dir ${wdata_base}/${AOI}_Training_Public \
        --output_mask_dir ${road_out_mask_dir_junction} &
done
wait

# Building
for AOI in Germany Louisiana-East; do
    find ${wdata_base}/${AOI}_Training_Public/PRE-event -name '*.tif' -type f -print \
        | xargs -I {} basename {} \
        | xargs -I {} ln ${wdata_base}/${AOI}_Training_Public/PRE-event/{} ${building_out_image_dir}/{}
done
for AOI in Germany Louisiana-East; do
    python scripts/preprocess/create_building_masks.py sn8 \
        --image_dir ${wdata_base}/${AOI}_Training_Public/PRE-event \
        --geojson_dir ${wdata_base}/${AOI}_Training_Public/annotations \
        --output_dir ${building_out_mask_dir} &
done
wait

# --- xview2 ---
python scripts/preprocess/create_flood_mask_xview2.py \
    --geotiffs_dir ${wdata_base}/geotiffs \
    --output_df ${wdata_base}/xview2_image_list.csv \
    --n_jobs 16

# make image_list.csv
python scripts/preprocess/make_image_lists.py ${wdata_base}
