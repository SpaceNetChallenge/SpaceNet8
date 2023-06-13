# Winning Solutions and Baseline for SpaceNet 8 Flood Detection Challenge

Each year, natural disasters such as hurricanes, tornadoes, earthquakes and floods significantly damage infrastructure and result in loss of life, property and billions of dollars. As these events become more frequent and severe, there is an increasing need to rapidly develop maps and analyze the scale of destruction to better direct resources and first responders. To help address this need, the SpaceNet 8 Flood Detection Challenge will focus on infrastructure and flood mapping related to hurricanes and heavy rains that cause route obstructions and significant damage. The goal of SpaceNet 8 is to leverage the existing repository of datasets and algorithms from SpaceNet Challenges 1-7 (https://spacenet.ai/datasets/) and apply them to a real-world disaster response scenario, expanding to multiclass feature extraction and characterization for flooded roads and buildings and predicting road speed.

## Top 5 winning solutions
> highlights. You can find more detailed writeup in each of the subdirectories
1. `./01-ohhan777`
    - siamese HRNet+OCR
    - data augmentation during training (motion blur, gaussian noise, hue and saturation shift, random brighness and contrast, random gamma)
    - binary cross entropy loss for building segmentation, regional mutual information loss for road speed and flood segementation
2. `./02-number13`
    - pretrain U-Net (ResNet-50 encoder, initialized with ImageNet weights) on additional training data from spacenet-2 (buildings), -3 (roads), and -5 (roads) to predict roads and buildings. Use focal and dice loss.
    - finetune the pretrained U-Net on spacenet-8 pre-event imagery and reference data.
    - siamese U-Net (initialized using the pretrained U-Net weights) and train on spacenet-8 for flood and non-flood data.
    - heuristic for flood attribution: if 30% of roads in image are flooded, all instances are assigned flood. if 50% of buildings in image are flooded, all instances of building are flooded.
    - use focal and dice loss for segmentating flood and non-flood. Binary cross entropy loss to classify each whole image as 'flooded' or 'not-flooded'.
    - same postprocessing as baseline
3. `./03-sianalytics`
    - pretrain with additional training data. spacenet-2 buildings, spacenet-3 roads, massachusetts buildings and roads, inria aerial image labeling dataset buildings.
    - three independent models were used (building segmentation, road segmentation, and flood segmentation model)
    - swin-transformer backbone (initialized with ImageNet-22K weights). UPerNet for the building and flood network decoder. Segformer used for road network decoder.
    - focal loss + dice loss + lovasz loss 
    - data augmentation during training
    - same postprocessing as baseline - hyperparameters were adjusted
4. `./04-ZABURO`
    - use additional training data from spacenet-2 (buildings), spacenet-3 (roads), spacenet-5 (roads), and xBD dataset (for flood)
    - For buildings and roads, train a U-Net with EffecientNet-V2 encoder
    - For flood prediction, train a siamese U-Net with ResNet-50 encoder.
    - same postprocessing as baseline. slight modifications for road network extraction
5. `./05-motokimura`
    - reviewed input data and excluded tiles with annotation errors. In cases where there were two post-event tiles, the lower quality image was excluded
    - retiled the input data to gather more samples
    - included auxiliary tasks during training of road junction prediction, building body prediction, building edge prediction, and building contact prediction
    - ensemble for buildings, roads, and flood prediction
        - U-Net with EfficientNet-B5 (predicting roads and buildings)
        - U-Net with EfficientNet-B6 (predicting roads and buildings)
        - Finetune the pretrained solution for SpaceNet-5 (predicting roads)
        - Finetune the pretrained solution for xView-2 (predicting buildings)
        - Siamese U-Net with EfficientNet-B5 (predicting flooded roads and buildings)
        - Siamese U-Net with EfficientNet-B6 (predicting flooded roads and buildings)
        - Finetune the pretrained solution for xView-2 (predicting flooded buildings)
    - average and postprocess the results from the ensemble
    - dice loss + binary cross entropy loss
## Baseline
1. `./baseline`

### References and Resources 
1. [R. Hänsch, J. Arndt, D. Lunga, M. Gibb, T. Pedelose, A. Boedihardjo, D. Petrie and T. M. Bacastow, "SpaceNet 8 - The Detection of Flooded Roads and Buildings", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022, pp. 1472-1480.](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Hansch_SpaceNet_8_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.pdf)

2. [R. Hänsch, J. Arndt, M. Gibb, A. Boedihardjo, T. Pedelose and T. M. Bacastow, "The SpaceNet 8 Challenge - From Foundation Mapping to Flood Detection," IGARSS 2022 - 2022 IEEE International Geoscience and Remote Sensing Symposium, Kuala Lumpur, Malaysia, 2022, pp. 5073-5076, doi: 10.1109/IGARSS46834.2022.9883741.](https://ieeexplore.ieee.org/abstract/document/9883741)