Open Source Resources, Frameworks and Libraries used by this solution.

## Pre-trained models:

-   pytorch-image-models (timm) ImageNet pre-trained efficientnet-b5 and efficientnet-b6,
    [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models),
    Apache 2.0 License

-   SpaceNet-5 pre-trained unet (w/ SEResNeXt-50 backbone) by XD_XD (winner of SpaceNet-5 challenge)
    [https://github.com/SpaceNetChallenge/SpaceNet_Optimized_Routing_Solutions/tree/master/xd_xd](https://github.com/SpaceNetChallenge/SpaceNet_Optimized_Routing_Solutions/tree/master/xd_xd),
    Apache 2.0 License

-   xView2 pre-trained unet (w/ densenet161 backbone) and siamese-unet (w/ densenet161 backbone) by selimsef (2nd place of xView2 challenge)
    [https://github.com/selimsef/xview2_solution](https://github.com/selimsef/xview2_solution),
    Apache 2.0 License

SpaceNet-5 pre-trained models and xView2 pre-trained models are downloaded during docker build (see Dockerfile for the download link).

SpaceNet-5 pre-trained unet is uploaded to my own AWS S3 (`https://motokimura-public-sn8.s3.amazonaws.com/xdxd_sn5_serx50_focal.zip`).
Please see `docs/prepare_sn5_XD_XD_models.md` to know how I prepared this model.

## Environment setup:

-   NVIDIA Docker,
    [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker),
    Apache 2.0 License

-   PyTorch NGC Container,
    [https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), 
    [NVIDIA DEEP LEARNING CONTAINER LICENSE](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf?8-4yoqobo3sABIk5HIXNCgHbxf9BoyjA67FpiMeufj88VoUNCKPK29cHYJYxyB1bMD5IL0KosS5LsOZ1NJHnRXe_KAUUScf0d7BZIyE8PfDUV__-vMLKGWpSkd536BFWbrK6EYsrXhLrrfAOZfvmGhI4&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczpcL1wvd3d3Lmdvb2dsZS5jb21cLyJ9)

## CNN model modeling, training, and inference:

-   PyTorch,
    [https://pytorch.org/](https://pytorch.org/),
    BSD 3-Clause License

-   segmentation_models.pytorch,
    [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch),
    MIT License

-   SpaceNet-5 challenge 1st place solution by XD_XD,
    [https://github.com/SpaceNetChallenge/SpaceNet_Optimized_Routing_Solutions/tree/master/xd_xd](https://github.com/SpaceNetChallenge/SpaceNet_Optimized_Routing_Solutions/tree/master/xd_xd),
    Apache 2.0 License

-   xView2 2nd place solution by selimsef,
    [https://github.com/selimsef/xview2_solution](https://github.com/selimsef/xview2_solution),
    Apache 2.0 License

-   PyTorch Image Models,
    [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models),
    Apache 2.0 License

-   Lightning (PyTorch Lightning),
    [https://github.com/Lightning-AI/lightning](https://github.com/Lightning-AI/lightning),
    Apache 2.0 License

## Pre-processing for SN8 dataset:

-   Albumentations,
    [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations),
    MIT License

-   OpenCV,
    [https://opencv.org/](https://opencv.org/),
    BSD 3-Clause License

-   scikit-image,
    [https://scikit-image.org/](https://scikit-image.org/),
    BSD 3-Clause License

-   imagecodecs,
    [https://github.com/cgohlke/imagecodecs](https://github.com/cgohlke/imagecodecs),
    BSD 3-Clause License

-   pygdal,
    [https://github.com/nextgis/pygdal](https://github.com/nextgis/pygdal)
    GDAL/OGR License

-   Shapely,
    [https://github.com/Toblerity/Shapely](https://github.com/Toblerity/Shapely),
    BSD 3-Clause License

## Post-processing for CNN output:

-   utm,
    [https://github.com/Turbo87/utm](https://github.com/Turbo87/utm),
    MIT License

-   OSMnx,
    [https://github.com/gboeing/osmnx](https://github.com/gboeing/osmnx),
    MIT License

-   Fiona,
    [https://github.com/Toblerity/Fiona](https://github.com/Toblerity/Fiona),
    BSD 3-Clause License

-   GeoPandas,
    [https://github.com/geopandas/geopandas](https://github.com/geopandas/geopandas),
    BSD 3-Clause License

## CNN model development (experiment management):

-   OmegaConf,
    [https://github.com/omry/omegaconf](https://github.com/omry/omegaconf),
    BSD 3-Clause License

-   TensorBoard,
    [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard),
    Apache 2.0 License

-   TensorboardX,
    [https://github.com/lanpa/tensorboardX](https://github.com/lanpa/tensorboardX),
    MIT License

-   Horovod,
    [https://github.com/horovod/horovod](https://github.com/horovod/horovod),
    Apache 2.0 License

-   Statsmodels,
    [https://github.com/statsmodels/statsmodels](https://github.com/statsmodels/statsmodels),
    BSD 3-Clause License

-   Weights and Biases (installed and imported, but not used),
    [https://github.com/wandb/wandb](https://github.com/wandb/wandb),
    MIT License

-   Python-dotenv (installed and imported, but not used),
    [https://github.com/theskumar/python-dotenv](https://github.com/theskumar/python-dotenv),
    BSD 3-Clause License

## Others:

-   NumPy,
    [https://numpy.org/](https://numpy.org/),
    BSD 3-Clause License

-   pandas,
    [https://pandas.pydata.org/](https://pandas.pydata.org/),
    BSD 3-Clause License

-   tqdm,
    [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm),
    MIT License

-   overrides
    [https://github.com/mkorpela/overrides](https://github.com/mkorpela/overrides),
    Apache 2.0 License
