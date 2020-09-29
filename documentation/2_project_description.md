# 2. Project description

## 2.1. Stanford Cars Dataset <a name="stanford-cars-dataset"></a>

Stanford Cars Dataset [[12]](5_references.md#KrauseStarkDengFei-Fei_3DRR2013) is a dataset published by Jonathan Krause of Stanford University and is publicly available at https://ai.stanford.edu/~jkrause/cars/car_dataset.html.

![Example images from Stanford Cars Dataset](img/21_1_stanford_cars_examples.png "Example images from Stanford Cars Dataset")

The dataset contains 16,185 images of 196 classes of car models (precisely, class label contains information about make, model and production year of a car). Dataset has been splitted with stratification into two parts:

- 8,144 images as a training set
- 8,041 images as a test set

In addition to class labels, both subsets have also bounding boxes (as 4 coordinates in metadata files).

Images are originally of different sizes, mostly in RGB, but there are some grayscale images which has to be taken into account during preprocessing. Another thing to be aware of is that the dataset has been updated at some point - the images and the split did not change, but the file names were reordered and metadata was reorganized for the ease of use.  

## 2.2. GhostNet architecture <a name="ghostnet-architecture"></a>

GhostNet [[11]](5_references.md#Han_2020) is the architecture designed and first implemented by the research team at Huawei Noah's Ark Lab (http://www.noahlab.com.hk/). It is based on the observation, that standard convolutional layers with many filters are large in terms of number of parameters and computationally expensive, while often producing redundant feature maps that are very much alike each other (they might be considered as "ghosts" of the original feature map). The goal of the GhostNet design is not to get rid of those redundant feature maps, because they often help the network to comprehensively understand all the features of the input data. Instead of that, the focus is on obtaining those redundant feature maps in a cost-efficient way.

![Redundant feature maps from ResNet-50 (picture from paper)](img/22_1_redundant_feature_maps.png "Redundant feature maps from ResNet-50 (picture from paper)")

This cost-efficiency in creating feature maps is achieved by introducing GhostModule, namely splitting standard convolutional layer with many filters into two parts. The first part, still being a standard convolutional layer but with less filters, produces a set of base feature maps. Then the second part, by applying cheap linear operations, produces redundant feature maps from the original set (so-called "ghosts"). In the end, the outputs of the first and the second part are concatenated.

![Comparison of standard convolution (a) and GhostModule (b) (picture from paper)](img/22_2_ghost_module.png "Comparison of standard convolution (a) and GhostModule (b) (picture from paper)")

The above mentioned cheap linear operations are implemented using depthwise convolutions [[13]](5_references.md#pandey2018) (although other options like affine or wavelet transforms were also tested by the authors). With this assumption, GhostModule can be implemented in PyTorch as follows:

```python
class GhostModule(nn.Module):
    def __init__(
        self, inp, oup,
        kernel_size=1, ratio=2, dw_size=3, stride=1,
        relu=True
    ):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride,
                kernel_size//2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
            dw_size//2, groups=init_channels, bias=False
        ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, input):
        output_1 = self.primary_conv(input)
        output_2 = self.cheap_operation(output_1)
        output = torch.cat([output_1, output_2], dim=1)
        return output[:, :self.oup, :, :]
```  

Two GhostModules combine for a basic building block of GhostNet - the GhostBottleneck, which is based on the concept taken from MobileNet-V3 design [[5]](5_references.md#Howard_2019) (additionally, in some GhostBottlenecks, similarly to MobileNet-V3, Squeeze-and-Excitation modules are used [[14]](5_references.md#Hu_2018)). The first GhostModule in a GhostBottleneck expands the number of channels, while the second one, after ReLU, reduces them again. There is also a residual connection over the two GhostModules. GhostBottleneck has also strided version (with `stride=2` depthwise convolution between GhostModules) which is applied at the end of each stage of GhostNet.

![GhostBottleneck (picture from paper)](img/22_3_ghost_bottleneck.png "GhostBottleneck (picture from paper)")

To form up the entire GhostNet architecture several GhostBottlenecks are combined in a sequence which is followed by global average pooling and a convolution which transforms feature maps to the feature vector of length 1280. This feature vector, after dropout layer, is then transformed with a fully connected layer to the size of output number of classes.

GhostNet architecture based on paper:  

|   Input               |   Operator          |   #exp   |   #out   |   SE   |   Stride   |
|-----------------------|---------------------|----------|----------|--------|------------|
|224 x 224 x 3          |     Conv2d 3x3      |     -    |   16     |   -    |     2      |
|112 x 112 x 16         |       G-bneck       |     16   |   16     |   -    |     1      |
|112 x 112 x 16         |       G-bneck       |     48   |   24     |   -    |     2      |
|56 x 56 x 24           |       G-bneck       |     72   |   24     |   -    |     1      |
|56 x 56 x 24           |       G-bneck       |     72   |   40     |   1    |     2      |
|28 x 28 x 40           |       G-bneck       |     120  |   40     |   1    |     1      |
|28 x 28 x 40           |       G-bneck       |     240  |   80     |   -    |     2      |
|14 x 14 x 80           |       G-bneck       |     200  |   80     |   -    |     1      |
|14 x 14 x 80           |       G-bneck       |     184  |   80     |   -    |     1      |
|14 x 14 x 80           |       G-bneck       |     184  |   80     |   -    |     1      |
|14 x 14 x 80           |       G-bneck       |     480  |   112    |   1    |     1      |
|14 x 14 x 112          |       G-bneck       |     672  |   112    |   1    |     1      |
|14 x 14 x 112          |       G-bneck       |     672  |   160    |   1    |     2      |
|7 x 7 x 160            |       G-bneck       |     960  |   160    |   -    |     1      |
|7 x 7 x 160            |       G-bneck       |     960  |   160    |   1    |     1      |
|7 x 7 x 160            |       G-bneck       |     960  |   160    |   -    |     1      |
|7 x 7 x 160            |       G-bneck       |     960  |   160    |   1    |     1      |
|7 x 7 x 160            |       Conv2d 1x1    |     -    |   960    |   -    |     1      |
|7 x 7 x 960            |       AvgPool 7x7   |     -    |   -      |   -    |     -      |
|1 x 1 x 960            |       Conv2d 1x1    |     -    |   1280   |   -    |     1      |
|1 x 1 x 1280           |       FC            |     -    |   1000   |   -    |     -      |

GhostNet architecture described above (and in original paper as well) is the basic setup which can be modified by structuring GhostBottlenecks in different sequences. This basic setup, as mentioned before, gets 73.98% accuracy on ImageNet with 4.1 M parameters and requires only 0.142 GFLOPS to process 224x224 RGB image. Other more complex variations, as presented in paper, show superiority over previous designs like MobileNet or ShuffleNet getting better accuracy with less FLOPS and latency.

![GhostNet comparison with some other mobile architectures (pictures from paper)](img/22_4_ghostnet_comparison.png "GhostNet comparison with some other mobile architectures (pictures from paper)")

Full PyTorch implementation of GhostNet that was used in this work is available at [GitHub](https://github.com/pchaberski/cars/blob/documentation/models/architectures/ghostnet.py) repository of the project.  

