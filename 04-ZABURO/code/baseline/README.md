# Algorithmic Baseline for SpaceNet 8 Flood Detection Challenge 


Each year, natural disasters such as hurricanes, tornadoes, earthquakes and floods significantly damage infrastructure and result in loss of life, property and billions of dollars. As these events become more frequent and severe, there is an increasing need to rapidly develop maps and analyze the scale of destruction to better direct resources and first responders. To help address this need, the SpaceNet 8 Flood Detection Challenge will focus on infrastructure and flood mapping related to hurricanes and heavy rains that cause route obstructions and significant damage. The goal of SpaceNet 8 is to leverage the existing repository of datasets and algorithms from SpaceNet Challenges 1-7 (https://spacenet.ai/datasets/) and apply them to a real-world disaster response scenario, expanding to multiclass feature extraction and characterization for flooded roads and buildings and predicting road speed. 


### Setup
1. Download this repo  

2. Build docker image  
`nvidia-docker build -t sn8/baseline:1.0 /path/to/sn8_baseline/docker`

3. Create and run the container (mount volumes to access your data, etc. see https://docs.docker.com/engine/reference/commandline/run/ for options)  
`nvidia-docker run -it --rm sn8/baseline:1.0 bash`


### Data Preparation
Follow these steps to prepare data for training and validation. 

1. Clean the geojson labels and add speed values to the roads based on road type, number of lanes, and surface type.  
`python baseline/data_prep/geojson_prep.py --root_dir /path/to/spacenet8/aws/data/download --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public`


    The cleaning step here also catches geometry problems, makes a single commom schema, and moves roads and buildings to seperate geojsons/shps. It will output a few additional files that are used in the subsequent step for creating image masks. The following new files will be written to the AOI annotations directory:  
        - `{AOI}/annotations/prepped_cleaned/roads_cleaned_{x}_{y}_{id}.geojson`  
        - `{AOI}/annotations/prepped_cleaned/buildings_cleaned_{x}_{y}_{id}.geojson`  
        - `{AOI}/annotations/prepped_cleaned/roads_speed_{x}_{y}_{id}.geojson`  
        - `{AOI}/annotations/prepped_cleaned/roads_speed_{x}_{y}_{id}.shp`  
        - `{AOI}/annotations/prepped_cleaned/buildings_{x}_{y}_{id}.shp`  

2. Create training and validation masks from the geojsons to use during training and validation.  
`python baseline/data_prep/create_masks.py --root_dir /path/to/spacenet8/aws/data/download --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public`

    Four masks are generated during this process:  
        - binary building mask (0 non-building, 1 building)  
        - binary road mask (0 non-building, 1 building)  
        - 8-channel road speed mask  
        - 4-channel flood mask  

3. Create a random train/val split to train the models on
`python baseline/data_prep/generate_train_val_test_csvs.py --root_dir /path/to/spacenet8/aws/data/download --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public --out_csv_basename sn8_data --val_percent 0.15 --out_dir /path/to/output/folder/for/train/val/csvs`

    This will create a csv file with filepaths to training images and labels and csv file with filepaths to validation images. It will do a random train/val split. These csvs are used by the dataloader during training. 

### Train/Validate Foundation Features Network  
`python baseline/train_foundation_features.py --train_csv /path/to/train.csv --val_csv /path/to/val.csv --save_dir /path/to/save/directory/foundation --model_name resnet34 --lr 0.0001 --batch_size 2 --n_epochs 50 --gpu 0`

### Inference with Foundation Features Network  
1. Write prediction tiffs to be used for postprocessing and generating the submission .csv  
`python baseline/foundation_eval.py --model_path /path/to/saved/foundation/best_model.pth --in_csv /path/to/val/or/test.csv --save_preds_dir /path/to/output/foundation/eval_test --gpu 0 --model_name resnet34`  

2. Write prediction .pngs for visual inspection of predictions  
`python baseline/foundation_eval.py --model_path /path/to/saved/foundation/best_model.pth --in_csv /path/to/val/or/test.csv --save_fig_dir /path/to/output/foundation/eval_test/pngs --gpu 0 --model_name resnet34`  

### Train/Validate Flood Features Network  
`python baseline/train_flood.py --train_csv /path/to/train.csv --val_csv /path/to/val.csv --save_dir /path/to/save/directory/flood --model_name resnet34_siamese --lr 0.0001 --batch_size 2 --n_epochs 50 --gpu 0`  

### Inference With Flood Features Network
1. Write prediction tiffs to be used for postprocessing and generating the submission .csv  
    `python baseline/flood_eval.py --model_path /path/to/saved/flood/best_model.pth --in_csv /path/to/val/or/test.csv --save_preds_dir /path/to/output/flood/eval_test --gpu 0 --model_name resnet34_siamese`    

2. Write prediction .pngs for visual inspection of predictions  
    `python lib/flood_eval.py --model_path /path/to/saved/flood/best_model.pth --in_csv /path/to/val/or/test.csv --save_fig_dir /path/to/output/flood/eval_test/pngs --gpu 0 --model_name resnet34_siamese`    

### Postprocessing
Any of the following postprocessing steps require that you have run inference with the flood features network and foundation features network. As input, the postprocessing requires the geotiff predictions.  

##### Roads
1. Skeletonize road raster predictions and convert to vector data.  
    `python baseline/postprocessing/roads/vectorize_roads.py --im_dir /path/to/road/prediction/geotiffs --out_dir /path/to/road/prediction/geotiffs --write_shps --write_graphs --write_csvs --write_skeletons`
2. Generate road network graph from vector data  
    `python baseline/postprocessing/roads/wkt_to_G.py --wkt_submission /path/to/output/csv/from/last/step --graph_dir /output/path/to/save/simplified/graphs --log_file /path/to/log`
3. Assign road speed predictions to road network graph linestrings  
    `python baseline/postprocessing/roads/infer_speed.py `
4. Generate the road submission .csv  
    `python baseline/postprocessing/roads/create_submission.py`

To run all these steps, see the script `/baseline/postprocessing/roads/road_post.sh`. Change only the variables:  
    EVAL_CSV  
    ROAD_PRED_DIR  
    FLOOD_PRED_DIR  

##### Buildings
1. Merge the building predictions from the foundation features network with the flooded predictions from the flood features network to attribute buildings as flooded or non-flooded. Polygonize the building prediction raster, remove polygons below a certain area threshold and simplify polygon geometries.  
    `python baseline/postprocessing/buildings/building_postprocessing.py --root_dir /path/to/foundation/features/building/prediction/geotiffs --flood_dir /path/to/flood/prediction/geotiffs --out_submission_csv /path/to/output/building_submission.csv --out_shapefile_dir /path/to/output/building/pred_shps --square_size 5 --simplify_tolerance 0.75 --min_area 5 --perc_positive 0.5`

### Submission
Your submission files should be in the following format.

| ImageId | Object | Flooded | Wkt_Pix | Wkt_Geo |
| ------ | ------ | ------ | ------ | ------ |

ImageId is the reference image for the prediction. Object should be either "Building" or "Road". Flooded should be set to "True" or "False". Wkt_Pix is well-known-text format for coordinates as (row, col) of the predictions in reference to the image. Wkt_Geo is well-known-text coordinates of the predictions in WGS84 (x, y). 

Example for buildings: 
| ImageId | Object | Flooded | Wkt_Pix | Wkt_Geo |
| ------ | ------ | ------ | ------ | ------ |
| 104001006504F400_0_11_20 | Building | Null | POLYGON EMPTY | POLYGON EMPTY |
| 104001006504F400_0_10_29 | Building | FALSE | POLYGON ((6 706, 0 708, 0 715, 9 719, 12 706, 6 706)) | POLYGON ((-90.4706194788189 29.8296995794957,-90.4706358281571 29.8296930397604,-90.4706390980247 29.8296668808194,-90.4706063993484 29.8296570712164,-90.4705965897455 29.8296963096281,-90.4706194788189 29.8296995794957)) |  


### References and Resources
1. SpaceNet-8 workshop paper: https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Hansch_SpaceNet_8_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.pdf

2. The preprocessing for road speed labels, road speed training, and road speed post-processing leverages: https://github.com/avanetten/cresi  

3. See the following paper for more information on the road speed segmentation: https://openaccess.thecvf.com/content_WACV_2020/html/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.html  

3. SpaceNet 5 blogs for road speed estimation: https://spacenet.ai/sn5-challenge/
