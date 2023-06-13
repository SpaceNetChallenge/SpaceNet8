# motokimura's solution for SpaceNet-8 challenge

**Please ignore the description below in the final scoring phase** (it is only for the solution development phase).

See also [docs/ec2_setup](docs/ec2_setup).

## docker build

```
./docker/build.sh
./docker/run.sh
```

## data preprocess

```
./scripts/preprocess.sh
./scripts/mosaic.sh
python tools/measure_image_similarities.py
```

## train networks

```
echo "WANDB_API_KEY = {YOUR_WANDB_API_KEY}" > .env
python tools/train_net.py --task {task} --exp_id {exp_id}
```

## test networks

```
python tools/test_net.py --exp_id {exp_id}
```

## ensemble

```
python tools/ensemble.py --exp_id {exp_id_0, exp_id_1, ...}
```

## post-process (building)

```
python tools/building/postproc_building.py --foundation {foundation_dir} --flood {flood_dir}
```

## post-process (road)

```
python tools/road/vectorize.py --foundation {foundation_dir}
python tools/road/to_graph.py --vector {vector_dir}
python tools/road/insert_flood.py --flood {flood_dir} --graph {graph_dir}
```

## create submission csv

```
python tools/submit.py --building {building_dir} --road {road_dir}
```
