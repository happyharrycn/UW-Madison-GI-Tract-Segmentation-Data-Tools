# UW-Madison GI Tract Segmentation Data Tools

## Introduction
This repository contains code for loading and visualizing MRI data used in [UW-Madison GI Tract Image Segmentation challenge](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview).

<div align="center">
  <img src="teaser.gif" width="800px"/>
</div>


## Changelog
* 03/21/2023: Initial commit.

## Installation
Follow INSTALL.md for installing necessary dependencies

## Dataset and Data Preparation

**UW-Madison GI Tract Image Segmentation Dataset**

The MRI data and their annotations are hosted on Kaggle. A detailed description of the dataset and the download link can be found in this [webpage](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data).

**Note**: The training set can be directly downloaded, while the test set is only available when the code is submitted to Kaggle. For submission to Kaggle, please refer to instructions [here](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview/evaluation).

**Download and Unpack the Dataset**

To use code in this repository, please following the following steps to download and prepare the dataset
* Download *uw-madison-gi-tract-image-segmentation.zip* from this [webpage](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data). A Kaggle account is needed.
* Unpack the zip file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───train/
│    │	 └───case2
│    │	 └───case6
│    │	 └───...
│    └───train.csv
│   ...
|
```

## Loading MRI Data and Segmentation Masks
* To see how to use the code, run
```shell
python ./load_data.py --help
```
* To load a single scan (e.g., case136_day25), run
```shell
python ./load_data.py ./data/train.csv ./data/train case136_day25
```
* To visualize a single scan (e.g., case136_day25), run
```shell
python ./load_data.py ./data/train.csv ./data/train --viz
```
An animated GIF (e.g., case136_day25.gif) will be saved under *./outputs*.

## Other Part of the Code
This repository also contains sample code for evaluating 3D Dice score and 3D Hausdorff distance. See *common.py*.

## Contact
Yin Li (yin.li@wisc.edu)
