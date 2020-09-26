# 2. Project description

## 2.1. Stanford Cars Dataset <a name="stanford-cars-dataset"></a>

Stanford Cars Dataset [[12]](5_references.md#KrauseStarkDengFei-Fei_3DRR2013) is a dataset published by Jonathan Krause of Stanford University and is publicly available at https://ai.stanford.edu/~jkrause/cars/car_dataset.html.  

![Example images from Stanford Cars Dataset](img/21_1_stanford_cars_examples.png "Example images from Stanford Cars Dataset")  

The dataset contains 16,185 images of 196 classes of car models (precisely, class contains the information about make, model and production year). Dataset has been splitted with stratification into two parts:  

- 8,144 images as a training set  
- 8,041 images as a test set  

In addition to class labels, both subsets have also bounding boxes (as 4 coordinates in metadata files).

Images are originally of different sizes, mostly in RGB, but there are some grayscale images which has to be taken into account during preprocessing. Another thing to be aware of is that the dataset has been updated at some point - the images and the split did not change, but the file names were reordered and metadata was reorganized for the ease of use.

## 2.2. GhostNet architecture <a name="ghostnet-architecture"></a>
