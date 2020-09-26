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


Most of these architectures come with different customizable variants. For example, EfficientNet has 8 different configurations (named `b0` to `b7`) that differ in terms of complexity. Others, like MobileNet were reworked and upgraded resulting in different versions (in this case there are currently three versions of MobileNets, named simply `V1`, `V2` and `V3`).

## 1.2. Problem statement <a name="problem-statement"></a>

