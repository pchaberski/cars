# 4. Results

## 4.1. Best model <a name="best-model"></a>

## 4.2. Experiments step-by-step <a name="experiments-step-by-step"></a>

|      |experiment description                                       |train_acc|valid_acc|
|------|-------------------------------------------------------------|---------|---------|
|C-1 |Baseline (Cross Entropy Loss)                                |92.49%   |8.15%    |
|C-2 |Loss function change (Label Smoothing Cross Entropy)         |98.89%   |9.12%    |
|C-3 |Augmentations: horizontal flip, affine                       |99.45%   |11.96%   |
|C-4 |Augmentations: horizontal flip, affine, erasing              |99.76%   |51.92%   |
|C-5 |Augmentations: horizontal flip, erasing, color jitter        |98.12%   |38.08%   |
|C-6 |Augmentations: horizontal flip, affine, erasing, color jitter|93.68%   |38.68%   |
|C-7 |Augmentations: horizontal flip, affine, color jitter         |99.73%   |54.28%   |
|C-8 |Grayscale: no normalization, no augmentations                |99.49%   |6.58%    |
|C-9 |Grayscale: with normalization, no augmentations              |97.13%   |8.68%    |
|C-10|Grayscale: normalization, best RGB augmentations             |7.58%    |3.91%    |
|C-11|Training set cropping with bounding boxes                    |4.36%    |3.07%    |
|C-12|Training set cropping + background erasing                   |99.67%   |50.51%   |
|C-13|L2 regularization with AdamW: weight decay = 0.1             |99.44%   |63.39%   |
|C-14|L2 regularization with AdamW: weight decay = 0.2             |98.84%   |68.50%   |
|C-15|L2 regularization with AdamW: weight decay = 0.3             |95.83%   |61.84%   |
|C-16|L2 regularization with AdamW: weight decay = 0.4             |95.95%   |65.14%   |
|C-17|L2 regularization with AdamW: weight decay = 0.5             |90.38%   |59.95%   |
|C-18|Dropout rate tests: dropout = 0.1                            |99.11%   |66.90%   |
|C-19|Dropout rate tests: dropout = 0.3                            |98.62%   |67.81%   |
|C-20|Dropout rate tests: dropout = 0.4                            |96.52%   |64.88%   |
|C-21|Dropout rate tests: dropout = 0.5                            |96.28%   |66.75%   |
|C-22|Last layer size tests: out channels = 320                    |97.13%   |68.93%   |
|C-23|Last layer size tests: out channels = 640                    |96.13%   |63.13%   |
|C-24|Last layer size tests: out channels = 960                    |98.23%   |64.96%   |
|C-25|Last layer size tests: out channels = 1600                   |98.99%   |63.11%   |
|C-26|Automatic LR scheduling: take #1                             |99.82%   |74.60%   |
|C-27|Automatic LR scheduling: take #2                             |99.78%   |76.20%   |
|C-28|Automatic LR scheduling: take #3                             |99.83%   |75.14%   |
|C-29|Automatic LR scheduling: take #4                             |99.78%   |74.82%   |
|C-30|Controlled LR scheduling: milestones = [28, 48, 68, 88]      |80.66%   |57.82%   |
|C-31|Controlled LR scheduling: milestones = [36, 56, 76, 96]      |95.03%   |64.93%   |
|C-32|Controlled LR scheduling: milestones = [44, 64, 84, 104]     |98.68%   |68.79%   |
|C-33|Controlled LR scheduling: milestones = [52, 72, 92, 112]     |99.60%   |71.59%   |
|C-36|Weight decay adjustment: weight decay = 0.5                  |98.84%   |79.40%   |
|C-37|Weight decay adjustment: weight decay = 0.3                  |99.57%   |74.44%   |
|C-38|Weight decay adjustment: weight decay = 0.4                  |99.37%   |78.82%   |
|C-39|Weight decay adjustment: weight decay = 0.6                  |98.67%   |82.55%   |
|C-40|Weight decay adjustment: weight decay = 0.7                  |99.24%   |75.12%   |
|C-41|Dropout rate verification: dropout = 0.3                     |98.49%   |82.08%   |
|C-42|Dropout rate verification: dropout = 0.4                     |95.34%   |79.57%   |
|C-43|Dropout rate verification: dropout = 0.5                     |96.08%   |77.87%   |
|C-44|Dropout rate verification: dropout = 0.25                    |98.79%   |82.45%   |
|C-45|Additional augmentations test: resized crop                  |97.56%   |78.73%   |
|C-46|Additional augmentations test: rotation                      |97.03%   |78.25%   |
|C-47|Additional augmentations test: perspective                   |97.42%   |80.22%   |
|C-48|Additional augmentations test: erasing                       |93.68%   |80.56%   |
|**C-50**|**LR scheduler adjustment: milestones = [67, 82, 95, 107]**      |**98.94%**   |**83.79%**   |
|C-51|LR scheduler adjustment: milestones = [63, 78, 91, 103]      |98.86%   |82.54%   |
|C-53|LR scheduler adjustment: milestones = [66, 81, 94, 106]      |98.96%   |83.02%   |
|C-55|LR scheduler adjustment: milestones = [68, 83, 96, 108]      |98.78%   |83.72%   |
|C-56|LR scheduler adjustment: milestones = [64, 79, 92, 104]      |98.99%   |82.79%   |
|C-58|Last layer size sanity check: out channels = 1280            |99.44%   |78.83%   |
|C-63|LR annealing test: LR geometric sequence                     |99.80%   |70.51%   |
|C-64|LR annealing test: exponentiation base = 0.955               |98.49%   |60.70%   |
|C-65|LR annealing test: exponentiation base = 0.975               |99.66%   |73.07%   |
|C-66|LR annealing test: exponentiation base = 0.98                |98.72%   |70.46%   |


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
