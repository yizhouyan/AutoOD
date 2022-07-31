# AutoOD - Automatic Outlier Detection

AutoOD is a tuning-free approach that aims to best use existing outlier detection algorithms
yet without requiring human labeling input. Our key intuition is that
selecting one model from many alternate unsupervised anomaly detection models may not always work well. Instead, AutoOD targets
combining the best of them.

## Table of contents

* [What is AutoOD?](#what-is-autood)
* [AutoOD Methods](#autood-methods)
* [Try AutoOD](#try-autood) 

## What is AutoOD?
Outlier detection is critical in enterprises. Due to the existence of
many outlier detection techniques which often return different results 
for the same data set, the users have to address the problem of determining 
which among these techniques is the best suited for their task and tune its parameters. 
This is particularly challenging in the unsupervised setting, where no labels 
are available for cross-validation needed for such method and parameter optimization. 

In this work, we propose AutoOD which uses the existing unsupervised
detection techniques to automatically produce high quality outliers
without any human tuning. AutoOD’s fundamentally new strategy
unifies the merits of unsupervised outlier detection and supervised
classification within one integrated solution. It automatically tests
a diverse set of unsupervised outlier techniques on a target data set,
extracts useful signals from their combined detection results to reliably capture 
key differences between outliers and inliers. It then uses
these signals to produce a “custom anomaly classifier” to classify
anomalies, with its accuracy comparable to supervised outlier classification 
models trained with ground truth labels – without having
access to the much needed labels. On a diverse set of benchmark
outlier detection datasets, AutoOD consistently outperforms the best
unsupervised outlier detector selected from hundreds of detectors.
It also outperforms other tuning-free approaches we adapted to this
unsupervised outlier setting from 12 to 97 points (out of 100) in the
F-1 score.

## AutoOD Methods

The effectiveness of AutoOD relies on the number and quality of the reliable 
objects discovered and used as labeled training data thereafter. 
In this work, we design two approaches to discover
these reliable objects that complement each other, AutoOD-Augment and AutoOD-Clean. 


* AutoOD Augment

AutoOD-Augment starts by automatically discovering a small but re-
liable set of objects to label and keeps augmenting this set iteratively.
* AutoOD Cleaning

As opposed to AutoOD-Augment, AutoOD-Clean starts
with a large set of noisy labels and keeps cleaning this set into an
increasingly reliable set. 

## Try AutoOD
* [Try AutoOD Augment?](#try-autood-augment)
* [Try AutoOD Cleaning](#try-autood-cleaning)

#### Try AutoOD Augment
Step-1: Load datasets

```python
from autood_utils import load_dataset
filename = '.experiments/datasets/PageBlocks_norm_10.arff'
X, y = load_dataset(filename=filename)
```

Step-2: Preprocessing (gather results from various anomaly detection methods)

```python
import numpy as np
from autood_utils import autood_preprocessing
# set up some parameter ranges
lof_krange = list(range(10,110,10)) * 6
knn_krange = list(range(10,110,10)) * 6
if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
mahalanobis_N_range = [300, 400, 500, 600, 700, 800]
if_N_range = np.sort(mahalanobis_N_range * 5)
N_range = np.sort(mahalanobis_N_range * 10)

L, scores = autood_preprocessing(X,y,lof_krange,knn_krange,if_range, mahalanobis_N_range,if_N_range,N_range)
```

Step-3: Run AutoOD Augment
```python
from autood_utils import autood
autood(L, scores,X, y, max_iteration=5)
```

To reproduce what we have included in the paper, we've also included our experimental results on 11 benchmark datasets under folder `experiments/Autood-Augment`

#### Try AutoOD Cleaning
Run Step-1 and Step-2 above to get the preprocessing results. 

Step-3: Run AutoOD Clean
```python
from autood_cleaning import autood_cleaning
autood_cleaning(X, y, L, ratio_to_remove=0.05, max_iteration=20)
```

To reproduce what we have included in the paper experiment, we've also included our results on 11 benchmark datasets under folder `experiments/AutoOD-Cleaning.ipynb`


