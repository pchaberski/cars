# 4. Results

## 4.1. Best model <a name="best-model"></a>

## 4.2. Experiments step-by-step <a name="experiments-step-by-step"></a>

|#  |Experiment description                                                       |Training loss (min)|Validation loss (min)|Training accuracy (max)|Validation accuracy (max)|
|---|-----------------------------------------------------------------------------|-------------------|---------------------|-----------------------|-------------------------|
|1  |Baseline (Cross Entropy Loss)                                                |0.2961             |4.849                |0.9249                 |0.0815                   |
|2  |Loss function change (Label Smoothing Cross Entropy)                         |1.1329             |4.873                |0.9889                 |0.0912                   |
|3  |Augmentations: RandomHorizontalFlip, RandomAffine                            |1.0751             |4.792                |0.9945                 |0.1196                   |
|4  |Augmentations: RandomHorizontalFlip, RandomAffine, RandomErasing             |0.9897             |2.8108               |0.9976                 |0.5192                   |
|5  |Augmentations: RandomHorizontalFlip, RandomErasing, ColorJitter              |1.1223             |3.3386               |0.9812                 |0.3808                   |
|6  |Augmentations: RandomHorizontalFlip, RandomAffine, RandomErasing, ColorJitter|1.3148             |3.4524               |0.9368                 |0.3868                   |
|7  |Augmentations: RandomHorizontalFlip, RandomAffine, ColorJitter               |1.0034             |2.7444               |0.9973                 |0.5428                   |
|8  |Grayscale conversion: no normalization, no augmentations                     |1.0891             |5.0804               |0.9949                 |0.0658                   |
|9  |Grayscale conversion: with normalization, no augmentations                   |1.2071             |4.7456               |0.9713                 |0.0868                   |
|10 |Grayscale conversion: with normalization, best augmentations from RGB tests  |4.5697             |5.142                |0.0758                 |0.0391                   |
|11 |Training set cropping with bounding boxes                                    |4.8223             |5.1957               |0.0436                 |0.0307                   |
|12 |Training set cropping + background erasing                                   |1.0169             |2.8432               |0.9967                 |0.5051                   |
|13 |L2 regularization with AdamW: weight decay = 0.1                             |1.0234             |2.3146               |0.9944                 |0.6339                   |
|14 |L2 regularization with AdamW: weight decay = 0.2                             |1.0706             |2.0888               |0.9884                 |0.685                    |
|15 |L2 regularization with AdamW: weight decay = 0.3                             |1.2132             |2.2845               |0.9583                 |0.6184                   |
|16 |L2 regularization with AdamW: weight decay = 0.4                             |1.1928             |2.1737               |0.9595                 |0.6514                   |
|17 |L2 regularization with AdamW: weight decay = 0.5                             |1.3783             |2.3015               |0.9038                 |0.5995                   |
|18 |Dropout rate tests: dropout = 0.1                                            |1.039              |2.1793               |0.9911                 |0.669                    |
|19 |Dropout rate tests: dropout = 0.3                                            |1.0867             |2.1076               |0.9862                 |0.6781                   |
|20 |Dropout rate tests: dropout = 0.4                                            |1.1732             |2.1635               |0.9652                 |0.6488                   |
|21 |Dropout rate tests: dropout = 0.4                                            |1.1911             |2.0956               |0.9628                 |0.6675                   |
|22 |Last layer size tests: out channels = 320                                    |1.1396             |2.0546               |0.9713                 |0.6893                   |
|23 |Last layer size tests: out channels = 640                                    |1.1958             |2.2503               |0.9613                 |0.6313                   |
|24 |Last layer size tests: out channels = 960                                    |1.1082             |2.2023               |0.9823                 |0.6496                   |
|25 |Last layer size tests: out channels = 1600                                   |1.0709             |2.2937               |0.9899                 |0.6311                   |
|26 |Automatic LR scheduling: take #1                                             |0.9948             |1.9026               |0.9982                 |0.746                    |
|27 |Automatic LR scheduling: take #2                                             |1.014              |1.8357               |0.9978                 |0.762                    |
|28 |Automatic LR scheduling: take #3                                             |0.9878             |1.8881               |0.9983                 |0.7514                   |
|29 |Automatic LR scheduling: take #4                                             |1.0056             |1.8743               |0.9978                 |0.7482                   |
|30 |Controlled LR scheduling: milestones = [28, 48, 68, 88]                      |1.7492             |2.3638               |0.8066                 |0.5782                   |
|31 |Controlled LR scheduling: milestones = [36, 56, 76, 96]                      |1.3213             |2.1303               |0.9503                 |0.6493                   |
|32 |Controlled LR scheduling: milestones = [44, 64, 84, 104]                     |1.1425             |2.0221               |0.9868                 |0.6879                   |
|33 |Controlled LR scheduling: milestones = [52, 72, 92, 112]                     |1.0585             |1.9705               |0.996                  |0.7159                   |
|36 |Weight decay adjustment: weight decay = 0.5                                  |1.1026             |1.6685               |0.9884                 |0.794                    |
|37 |Weight decay adjustment: weight decay = 0.3                                  |1.0431             |1.8583               |0.9957                 |0.7444                   |
|38 |Weight decay adjustment: weight decay = 0.4                                  |1.0483             |1.6952               |0.9937                 |0.7882                   |
|39 |Weight decay adjustment: weight decay = 0.6                                  |1.0898             |1.5635               |0.9867                 |0.8255                   |
|40 |Weight decay adjustment: weight decay = 0.7                                  |1.089              |1.8018               |0.9924                 |0.7512                   |
|41 |Dropout rate verification: dropout = 0.3                                     |1.1107             |1.571                |0.9849                 |0.8208                   |
|42 |Dropout rate verification: dropout = 0.4                                     |1.2701             |1.6464               |0.9534                 |0.7957                   |
|43 |Dropout rate verification: dropout = 0.5                                     |1.2695             |1.6918               |0.9608                 |0.7787                   |
|44 |Dropout rate verification: dropout = 0.25                                    |1.0871             |1.5615               |0.9879                 |0.8245                   |
|45 |Additional augmentations test: RandomResizedCrop                             |1.1619             |1.6991               |0.9756                 |0.7873                   |
|46 |Additional augmentations test: RandomRotation                                |1.2029             |1.7207               |0.9703                 |0.7825                   |
|47 |Additional augmentations test: RandomPerpective                              |1.1691             |1.6295               |0.9742                 |0.8022                   |
|48 |Additional augmentations test: RandomErasing                                 |1.3032             |1.633                |0.9368                 |0.8056                   |
|50 |Learning rate scheduler adjustment: milestones = [67, 82, 95, 107]           |1.0641             |1.5208               |0.9894                 |0.8379                   |
|51 |Learning rate scheduler adjustment: milestones = [63, 78, 91, 103]           |1.0766             |1.5719               |0.9886                 |0.8254                   |
|53 |Learning rate scheduler adjustment: milestones = [66, 81, 94, 106]           |1.0735             |1.5298               |0.9896                 |0.8302                   |
|55 |Learning rate scheduler adjustment: milestones = [68, 83, 96, 108]           |1.0701             |1.5258               |0.9878                 |0.8372                   |
|56 |Learning rate scheduler adjustment: milestones = [64, 79, 92, 104]           |1.0661             |1.5597               |0.9899                 |0.8279                   |
|58 |Last layer size sanity check: out channels = 1280                            |1.0495             |1.7201               |0.9944                 |0.7883                   |
|63 |Learning rate annealing test: LR geometric sequence based on best LR drop    |0.9938             |2.0385               |0.998                  |0.7051                   |
|64 |Learning rate annealing test: exponentiation base = 0.955                    |1.1694             |2.3141               |0.9849                 |0.607                    |
|65 |Learning rate annealing test: exponentiation base = 0.975                    |0.9751             |1.9074               |0.9966                 |0.7307                   |
|66 |Learning rate annealing test: exponentiation base = 0.98                     |1.0571             |1.9423               |0.9872                 |0.7046                   |


### 4.2.1. Loss function <a name="loss-function"></a>

### 4.2.2. Normalization <a name="normalization"></a>

### 4.2.3. Augmentations <a name="augmentations"></a>

### 4.2.4. Grayscale conversion <a name="grayscale-conversion"></a>

### 4.2.5. Bounding boxes utilization <a name="bounding-boxes-utilization"></a>

### 4.2.6. Optimizer change and L2 regularization <a name="optimizer-change-and-l2-regularization"></a>

### 4.2.7. Dropout rate tests <a name="dropout-rate-tests"></a>

### 4.2.8. Last layer size tests <a name="last-layer-size-tests"></a>

### 4.2.9. Automatic learning rate scheduling <a name="automatic-learning-rate-scheduling"></a>

### 4.2.10. Controlled learning rate scheduling <a name="controlled-learning-rate-scheduling"></a>

### 4.2.11. Weight decay adjustment <a name="weight-decay-adjustment"></a>

### 4.2.12. Dropout rate verification <a name="dropout-rate-verification"></a>

### 4.2.13. Additional augmentations tests <a name="additional-augmentations-tests"></a>

### 4.2.14. Learning rate scheduler adjustment <a name="learning-rate-scheduler-adjustment"></a>

### 4.2.15. Last layer size sanity check <a name="last-layer-size-sanity-check"></a>

### 4.2.16. Learning rate annealing tests <a name="learning-rate-annealing-tests"></a>
