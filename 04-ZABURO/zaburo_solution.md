## 1. Introduction

> Tell us a bit about yourself, and why you have decided to participate in the contest.

 - Name: Kaizaburo Chubachi
 - Handle: ZABURO
 - Placement you achieved in the MM:
 - About you: Software Engineer at Preferred Networks, inc.
 - Why you participated in the MM: I have recently been studying satellite image analysis. I participated in this competition to practically test what I have learned.

## 2. Solution Development

> How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

Since generalization to unknown cities is required, we started by using previous SpaceNet data to increase variations of cities to be trained. We created models with additional data from the SN5 and SN3 datasets for road detection and the SN2 dataset for building detection. After evaluating the predictions of these models, I found that while the building detection was reasonably accurate, the road detection seemed to have a lot of room for improvement, so I decided to focus on improving the road detection.

For road detection, I followed the basic framework and hyperparameters of the 1st place solution of SN5 by XD_XD. Using ResNet50 as the baseline, I tried several larger backbones such as EfficientNet, EfficientNetV2, SwinTransformer, etc. In my experiments, EfficientNetV2 achieved good performance in terms of trade-off between training time and accuracy. Finally, EfficientNetV2-M was selected for road detection. From the visualization results, I also noticed that the segmentation accuracy near intersections was poor. I thought that incorrectly determining the connection relationship in intersections would be a significant loss in APLS. Therefore, I introduced an auxiliary task to detect areas with 8-meter radius from the intersections, so that feature extraction would focus more on the intersections. Interestingly, this has slightly improved the accuracy of road detection in the sense of Dice loss in the validation set. Toward the end of the competition, I worked on tuning the loss (bce, focal, dice, lovasz, label-smoothing, etc.) weights to minimize the APLS for the validation set, but preparing the final test image took more time than expected, and we gave up the tuning. For post-processing, in addition to the post-processing in cresi, the road was extended to create intersections with other roads, and small independent roads were erased.

No special effort was made to detect buildings. As in the past solutions, segmentation of the two channels of buildings and their borders was performed.

Flood detection was a more difficult challenge to generalize because we could not use past SpaceNet data. At first, I thought that by performing all of road, building, and flood detections in a single network using multi-task learning, I could obtain a feature extractor that would be more generalized. However, despite various tuning efforts, I could not achieve good accuracy in flood detection while maintaining the same accuracy in road and building detections. I believe the main reason is that optimization with SGD variants is difficult in heterogeneous multi-task learning, where each task has a different dataset. In such a situation, the use of the xView2 xBD dataset was allowed, so I switched to using it to increase the diversity of the training data.

The training of the flood detection model was a binary semantic segmentation that outputs either flood or non-flood, and was treated as a weakly supervised task where only regions with either label were used for the loss calculation. For images without any flood labels, an image level BCE is added to the loss so that the entire prediction of such images becomes negative. This was effective in reducing false positives. For the xView2 xBD dataset, only events with flooding damage were included in the training data. We also considered performing pseudo labeling to obtain better labels, but this would have required major modifications, such as switching to 4 folds CV to perform this on 4 GPUs, and I could not make it in time.

## 3. Final Approach

> Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

Validation
 - After splitting by AOI and whether the image contains flood or not, a 5 fold split stratified by latitude and longitude was done.

Road Segmentation
 - smp.Unet(“tu-tf_efficientnetv2_m”) x 4 folds
 - 9 channels: Road Speed (7 channels) + Road Skeleton + Junction
 - batch size 8, 296 epoch, Adam(lr=1e-4, weight_decay=1.0e-7) with cosine annealing 
 - 0.75 * focal + 0.25 * dice
 - SN8 dataset + SN5 dataset + SN3 dataset
 - Post-processing:
   - Apply cresi post-processing.
   - For each road, extend the road an appropriate short length to check if it intersects another road, and if so, extend the road to the intersection.
   - Eliminate isolated roads that are less than a certain length.

Building Segmentation
 - smp.Unet(“tu-tf_efficientnetv2_s”) x 4 folds
   - EfficientNetV2-M gave only small improvements despite its slow training speed.
 - 2 channels: Building body and border
 - batch size 14, 296 epoch, Adam(lr=1e-4, weight_decay=1.0e-7) with cosine annealing
 - 1.0 * bce + 1.0 * dice
 - SN8 dataset + SN2 dataset
   - The SN2 dataset was downsampled by 1/4 since the image size is half compared to that of the SN8 dataset
 - Post-processing is the same as the baseline

Flood Segmentation
 - Unet Siamese Network with resnet50 backbone
   - Input pre and post images separately to Unet, concatenate the output of the decoder, apply two Conv2d layers, and pass it to the segmentation head.
 - 1 channel: flooded or not
 - batch size 8, 148 epoch, Adam(lr=1e-4, weight_decay=1.0e-7) with cosine annealing
 - 1.0 * bce_w + 1.0 * dice_w
   - bce_w = (bce(output, target) * mask).sum() / (mask.sum() + 1e-6)
   - dice_w = dice(output.sigmoid() * mask, target)
 - For images that do not contain any flood: Add `bce(avgpool(output), 0)` to the loss
   - This encourages the model to output a negative label in the normal area.
 - SN8 dataset + xView2 xBD dataset
   - Only "hurricane-florence", "hurricane-harvey", "midwest-flooding", "nepal-flooding" are used.
   - Areas with more severe damage than “minor-damage” were treated as flooded.
   - Downsampling was performed so that the number of images that do not contain any flood area is 50% of the number of images that contain flood area.
 - Post-processing is the same as for baseline

## 4. Open Source Resources, Frameworks and Libraries

> Please specify the name of the open source resource along with a URL to where it’s housed and it’s license type:

- numpy, https://github.com/numpy/numpy, BSD-3-Clause license
- scipy, https://github.com/scipy/scipy, BSD-3-Clause license
- pandas, https://github.com/pandas-dev/pandas, BSD-3-Clause license
- pytorch, https://github.com/pytorch/pytorch, BSD-3-Clause license
- albumentations, https://github.com/albumentations-team/albumentations, MIT license
- gdal, https://github.com/OSGeo/gdal, MIT license
- fiona, https://github.com/Toblerity/Fiona, BSD-3-Clause license
- geopandas, https://github.com/geopandas/geopandas, BSD-3-Clause license
- somen, https://github.com/zaburo-ch/somen, MIT license
- scikit-image, https://github.com/scikit-image/scikit-image, Modified BSD
- opencv, https://github.com/opencv/opencv-python, MIT license
- shapely, https://github.com/shapely/shapely, BSD-3-Clause license
- osmnx, https://github.com/gboeing/osmnx, MIT license
- tqdm, https://github.com/tqdm/tqdm, MPLv2, MIT
- python-fire, https://github.com/google/python-fire, Apache-2.0 license
- networkx, https://github.com/networkx/networkx, BSD-3-Clause license
- ACL-Python, https://github.com/Mitarushi/ACL-Python, MIT license
- cresi, https://github.com/avanetten/cresi, Apache-2.0 license
- pytorch-pfn-extras, https://github.com/pfnet/pytorch-pfn-extras, MIT license
- segmentation_models_pytorch, https://github.com/qubvel/segmentation_models.pytorch, MIT license
- timm, https://github.com/rwightman/pytorch-image-models, Apache-2.0 license

## 5. Potential Algorithm Improvements
> Please specify any potential improvements that can be made to the algorithm:

 - Semi-supervised learning methods might improve generalization performance to unknown cities since external data such as xView2 can be used for road and building detection as well
 - The GAN framework might be used to suppress blurry forecasts and choppy roads
 - Post-processing to handle the case of partially flooded straight roads might improve the score a bit.

## 6. Algorithm Limitations
> Please specify any potential limitations with the algorithm:

 - Flood prediction should inherently take into account the elevation of the area, but it is quite difficult to do that from optical imagery alone. I think we should also incorporate such information.
 - There is a risk of not generalizing to unknown cities because flood textures vary widely from region to region.

## 7. Deployment Guide
> Please provide the exact steps required to build and deploy the code:

## 8. Final Verification
> Please provide instructions that explain how to train the algorithm and have it execute against sample data:

## 9. Feedback
> Please provide feedback on the following - what worked, and what could have been done better or differently?

 - Problem Statement
   - The task was well understood.
   - It would be better if more information related to the contest was gathered. For example, it would be good to have the link to the dataset paper or excerpts of important parts of the paper.
 - Data
   - The dataset is well designed with well aligned paired images.
   - Annotation seemed to me a little noisier than past competitions.
 - Contest
   - I felt that the contest duration was too short even though the tasks were very complex. It would be better to simplify the task or extend the duration for such a complex task.
 - Scoring
   - It was tough to optimize both of road and building scores.
   - I think it would increase participation if we had separate leaderboards for roads and buildings, with awards also given to those who achieved the highest accuracy in each. Just an idea.
