# Pixel-Wise Gland Segmentation in histology Images

## Dataset

Gland Segmentation Challenge Contest [Competition dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/)

## Repository

### Cell Segmentation

![Screenshot 2019-12-27 at 14.07.13](https://raw.githubusercontent.com/eikekutz/deeplearning_project/master/images/Prediction_result.png )

The whole implementation for the segmentation can be seen in the [Cell segmentation notebook](https://github.com/eikekutz/deeplearning_project/blob/master/Cell_segmentation.ipynb)

The network model and the training script can be seen individually in [model folder](https://github.com/eikekutz/deeplearning_project/tree/master/model)

The data augmentation and all the helper functions are located in [data_utils](https://github.com/eikekutz/deeplearning_project/tree/master/data_utils)

### GAN based data augmentation approach

Initial approach for GAN stage 1 after: [Towards Adversarial Retinal Image Synthesis](https://arxiv.org/abs/1701.08974)

The GAN approach is only working for one image, yet. Training on multiple images lead to mode collapse.

