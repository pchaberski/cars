# 1. Introduction

## 1.1. Problem background <a name="problem-background"></a>

One of the ongoing directions of deep learning research in computer vision and image recognition (but not only) is related to the reduction of neural network size and the number of operations needed for inference while preserving good level of performance in terms of classification accuracy or other metrics. The one important reason to do so is to reduce training times and costs and speed up the iterative process od hyperparameter optimization. Another drawback of large networks in some applications can be extensive overfitting. But the most important reason justifying the search for more efficient neural architectures is that in many practical applications models are needed to be deployed not on a multi-GPU servers or on cloud, but rather as a part of embedded systems on devices with very limited computational power and memory like smartphones, car systems or other devices with so-called intelligent modules.

At the time of completing this work there is already a significant number of different propositions of architectures aiming to reduce the number of parameters and FLOPS needed to efficiently perform image classification tasks [[1]](5_references.md#hollemans2020). Those architectures are most commonly trained on ImageNet (http://www.image-net.org/) and, among others, two metrics are reported on this dataset: accuracy and FLOPS, along with the total number of parameters. Those values give an impression about architecture efficiency in terms of trade-off between prediction quality, inference speed and required memory. Some, but definitely not all, of the successful implementations are:  

- SqueezeNet (2016) [[2]](5_references.md#i2016squeezenet)  
- MobileNet (`V1`: 2017, `V2`: 2018, `V3`: 2019) [[3]](5_references.md#howard2017mobilenets)[[4]](5_references.md#Sandler_2018)[[5]](5_references.md#Howard_2019)    
- SqueezeNext (2018) [[6]](5_references.md#Gholami_2018)  
- ShuffleNet (2018) [[7]](5_references.md#Zhang_2018)[[8]](5_references.md#Ma_2018)   
- EfficientNet (2019) [[9]](5_references.md#tan2019efficientnet)  
- HarDNet (2019) [[10]](5_references.md#Chao_2019)  
- GhostNet (2020) [[11]](5_references.md#Han_2020)  

Most of these architectures come with different customizable variants. For example, EfficientNet has 8 different basic configurations (named `b0` to `b7`) that differ in terms of complexity. Others, like MobileNet, were reworked and upgraded resulting in different versions (there are currently three versions of MobileNets, named simply `V1`, `V2` and `V3`).

The above-mentioned architectures are capable of achieving good accuracy scores with very limited number of parameters and floating point operations required. For example, EfficientNet-b0 has 77.1% accuracy on ImageNet with 5.3 M parameters and 0.39 GFLOPS. GhostNet gets 73.98% accuracy with only 4.1 M parameters and 0.142 GFLOPS. On the contrasts, ResNet-50 to achieve 75.3% accuracy requires 25.6 M parameters and 4.1 GFLOPS to process the image of the same size (224x224 RGB).

## 1.2. Problem statement <a name="problem-statement"></a>

This project is a part of a broader conception to create a mobile application to recognize car models from pictures taken by the users. The initial idea was to:  

1. Pick some of the efficient mobile architectures (the project was intendet to be carried out in a group), train them on a open dataset of car images and compare in terms of accuracy, model size and FLOPS.  
2. Prepare custom dataset of images taken and labelled personally, finetune the best model from step 1 to reflect car models distribution on the streets of Poland.
3. Prepare model for deployment, create a simple Android application that allow to take a picture and recognize a car model.  

This work focuses only on step one with selected architecture. Specifically, it describes the process of training and optimizing hyperparameters of GhostNet [[11]](5_references.md#Han_2020) model using Stanford Cars Dataset (https://ai.stanford.edu/~jkrause/cars/car_dataset.html) to check the performance of this particular novel mobile architecture in a car model recognition task.  
