# 4. Results

## 4.1. Best model <a name="best-model"></a>

## 4.2. Experiments step-by-step <a name="experiments-step-by-step"></a>

|#  |Experiment description                                                       |Training loss (min)|Validation loss (min)|Training accuracy (max)|Validation accuracy (max)|
|---|-----------------------------------------------------------------------------|-------------------|---------------------|-----------------------|-------------------------|
|1  |Baseline (Cross Entropy Loss)                                                |0.2961             |4.849                |92.49%                 |8.15%                    |
|2  |Loss function change (Label Smoothing Cross Entropy)                         |1.1329             |4.873                |98.89%                 |9.12%                    |
|3  |Augmentations: RandomHorizontalFlip, RandomAffine                            |1.0751             |4.792                |99.45%                 |11.96%                   |
|4  |Augmentations: RandomHorizontalFlip, RandomAffine, RandomErasing             |0.9897             |2.8108               |99.76%                 |51.92%                   |
|5  |Augmentations: RandomHorizontalFlip, RandomErasing, ColorJitter              |1.1223             |3.3386               |98.12%                 |38.08%                   |
|6  |Augmentations: RandomHorizontalFlip, RandomAffine, RandomErasing, ColorJitter|1.3148             |3.4524               |93.68%                 |38.68%                   |
|7  |Augmentations: RandomHorizontalFlip, RandomAffine, ColorJitter               |1.0034             |2.7444               |99.73%                 |54.28%                   |
|8  |Grayscale conversion: no normalization, no augmentations                     |1.0891             |5.0804               |99.49%                 |6.58%                    |
|9  |Grayscale conversion: with normalization, no augmentations                   |1.2071             |4.7456               |97.13%                 |8.68%                    |
|10 |Grayscale conversion: with normalization, best augmentations from RGB tests  |4.5697             |5.142                |7.58%                  |3.91%                    |
|11 |Training set cropping with bounding boxes                                    |4.8223             |5.1957               |4.36%                  |3.07%                    |
|12 |Training set cropping + background erasing                                   |1.0169             |2.8432               |99.67%                 |50.51%                   |
|13 |L2 regularization with AdamW: weight decay = 0.1                             |1.0234             |2.3146               |99.44%                 |63.39%                   |
|14 |L2 regularization with AdamW: weight decay = 0.2                             |1.0706             |2.0888               |98.84%                 |68.50%                   |
|15 |L2 regularization with AdamW: weight decay = 0.3                             |1.2132             |2.2845               |95.83%                 |61.84%                   |
|16 |L2 regularization with AdamW: weight decay = 0.4                             |1.1928             |2.1737               |95.95%                 |65.14%                   |
|17 |L2 regularization with AdamW: weight decay = 0.5                             |1.3783             |2.3015               |90.38%                 |59.95%                   |
|18 |Dropout rate tests: dropout = 0.1                                            |1.039              |2.1793               |99.11%                 |66.90%                   |
|19 |Dropout rate tests: dropout = 0.3                                            |1.0867             |2.1076               |98.62%                 |67.81%                   |
|20 |Dropout rate tests: dropout = 0.4                                            |1.1732             |2.1635               |96.52%                 |64.88%                   |
|21 |Dropout rate tests: dropout = 0.4                                            |1.1911             |2.0956               |96.28%                 |66.75%                   |
|22 |Last layer size tests: out channels = 320                                    |1.1396             |2.0546               |97.13%                 |68.93%                   |
|23 |Last layer size tests: out channels = 640                                    |1.1958             |2.2503               |96.13%                 |63.13%                   |
|24 |Last layer size tests: out channels = 960                                    |1.1082             |2.2023               |98.23%                 |64.96%                   |
|25 |Last layer size tests: out channels = 1600                                   |1.0709             |2.2937               |98.99%                 |63.11%                   |
|26 |Automatic LR scheduling: take #1                                             |0.9948             |1.9026               |99.82%                 |74.60%                   |
|27 |Automatic LR scheduling: take #2                                             |1.014              |1.8357               |99.78%                 |76.20%                   |
|28 |Automatic LR scheduling: take #3                                             |0.9878             |1.8881               |99.83%                 |75.14%                   |
|29 |Automatic LR scheduling: take #4                                             |1.0056             |1.8743               |99.78%                 |74.82%                   |
|30 |Controlled LR scheduling: milestones = [28, 48, 68, 88]                      |1.7492             |2.3638               |80.66%                 |57.82%                   |
|31 |Controlled LR scheduling: milestones = [36, 56, 76, 96]                      |1.3213             |2.1303               |95.03%                 |64.93%                   |
|32 |Controlled LR scheduling: milestones = [44, 64, 84, 104]                     |1.1425             |2.0221               |98.68%                 |68.79%                   |
|33 |Controlled LR scheduling: milestones = [52, 72, 92, 112]                     |1.0585             |1.9705               |99.60%                 |71.59%                   |
|36 |Weight decay adjustment: weight decay = 0.5                                  |1.1026             |1.6685               |98.84%                 |79.40%                   |
|37 |Weight decay adjustment: weight decay = 0.3                                  |1.0431             |1.8583               |99.57%                 |74.44%                   |
|38 |Weight decay adjustment: weight decay = 0.4                                  |1.0483             |1.6952               |99.37%                 |78.82%                   |
|39 |Weight decay adjustment: weight decay = 0.6                                  |1.0898             |1.5635               |98.67%                 |82.55%                   |
|40 |Weight decay adjustment: weight decay = 0.7                                  |1.089              |1.8018               |99.24%                 |75.12%                   |
|41 |Dropout rate verification: dropout = 0.3                                     |1.1107             |1.571                |98.49%                 |82.08%                   |
|42 |Dropout rate verification: dropout = 0.4                                     |1.2701             |1.6464               |95.34%                 |79.57%                   |
|43 |Dropout rate verification: dropout = 0.5                                     |1.2695             |1.6918               |96.08%                 |77.87%                   |
|44 |Dropout rate verification: dropout = 0.25                                    |1.0871             |1.5615               |98.79%                 |82.45%                   |
|45 |Additional augmentations test: RandomResizedCrop                             |1.1619             |1.6991               |97.56%                 |78.73%                   |
|46 |Additional augmentations test: RandomRotation                                |1.2029             |1.7207               |97.03%                 |78.25%                   |
|47 |Additional augmentations test: RandomPerpective                              |1.1691             |1.6295               |97.42%                 |80.22%                   |
|48 |Additional augmentations test: RandomErasing                                 |1.3032             |1.633                |93.68%                 |80.56%                   |
|**50** |**Learning rate scheduler adjustment: milestones = [67, 82, 95, 107]**           |**1.0641**             |**1.5208**               |**98.94%**                 |**83.79%**                   |
|51 |Learning rate scheduler adjustment: milestones = [63, 78, 91, 103]           |1.0766             |1.5719               |98.86%                 |82.54%                   |
|53 |Learning rate scheduler adjustment: milestones = [66, 81, 94, 106]           |1.0735             |1.5298               |98.96%                 |83.02%                   |
|55 |Learning rate scheduler adjustment: milestones = [68, 83, 96, 108]           |1.0701             |1.5258               |98.78%                 |83.72%                   |
|56 |Learning rate scheduler adjustment: milestones = [64, 79, 92, 104]           |1.0661             |1.5597               |98.99%                 |82.79%                   |
|58 |Last layer size sanity check: out channels = 1280                            |1.0495             |1.7201               |99.44%                 |78.83%                   |
|63 |Learning rate annealing test: LR geometric sequence based on best LR drop    |0.9938             |2.0385               |99.80%                 |70.51%                   |
|64 |Learning rate annealing test: exponentiation base = 0.955                    |1.1694             |2.3141               |98.49%                 |60.70%                   |
|65 |Learning rate annealing test: exponentiation base = 0.975                    |0.9751             |1.9074               |99.66%                 |73.07%                   |
|66 |Learning rate annealing test: exponentiation base = 0.98                     |1.0571             |1.9423               |98.72%                 |70.46%                   |


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
