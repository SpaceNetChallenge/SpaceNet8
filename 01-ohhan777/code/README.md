# SpaceNet 8

Korea Aerospace Research Institute (KARI) AI Lab, 2022.

![Building and Road Segmentation](./SpaceNet8_fig.PNG)

## Docker Build
```
docker build --tag kari-sn8:latest .
```

It takes about 20 minutes. Our Source code will be automatically located on /kari-spacenet8.

## Docker Run
```
docker run --gpus all -it --shm-size 4g --rm -v /kari-spacenet8/data:/data:ro -v /kari-spacenet8/runs:wdata kari-sn8:latest bash
```

## Train (4 GPUs)
```
conda activate kari-ai
cd /kari-spacenet8
./train.sh /data/SN8_flood/train
```
Once the script is executed, it will copy all the datasets to wdata directory (writable directory) because the original dataset directory is read-only.
Training will take about one or two days on four GPUs depending on the batch size (at least 16 GB GPU memory is required for batch size 8).

 
## Inference (4 GPUs)
```
test.sh /data/SN8_flood/test /wdata/submission.csv
```
If '/data/SN8_flood/test' directory contains 'MysteryCity_Test_Private', it will for the dataset. 
Otherwise, it will work for 'Louisiana-West_Test_Public' dataset.

If you need our pre-trained weight files, you can download them from the following links.

```
wget http://ohhan.net/wordpress/wp-content/uploads/2022/08/best_building.pt
wget http://ohhan.net/wordpress/wp-content/uploads/2022/08/best_road.pt 
wget http://ohhan.net/wordpress/wp-content/uploads/2022/08/best_flood.pt
```

These files should be copied to /kari-spacenet8/runs/train/weights/ for further processing.

