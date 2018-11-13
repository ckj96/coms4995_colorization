
## Introduction

The goal of this project is to get hands-on experience concerning the computer vision task of image similarity. Like most tasks in this field, it's been aided by the ability of deep networks to extract image features.

The task of image similarity is retrieve a set of `N` images closest to the query image. One application of this task could involve visual search engine where we provide a query image and want to find an image closest that image in the database.


## Project Description

### Overview

You will design a simplified version of the deep ranking model as discussed in the paper. Your network architecture will look exactly the same, but the details of the triplet sampling layer will be a lot simpler. The architecture consists of $3$ identical networks $(Q,P,N)$. Each of these networks take a single image denoted by $p_i$ , $p_i^+$ , $p_i^-$ respectively.

- $p_i$: Input to the $Q$ (Query) network. This image is randomly sampled across any class.
- $p_i^+$: Input to the $P$ (Positive) network. This image is randomly sampled from the **SAME** class as the query image.
- $p_i^-$: Input to the $N$ (Negative) network. This image is randomly sample from any class **EXCEPT** the class of $p_i$.


The output of each network, denoted by $f(p_i)$, $f(p_i^+)$, $f(p_i^-)$ is the feature embedding of an image. This gets fed to the ranking layer.


### Ranking Layer

The ranking layer just computes the triplet loss. It teaches the network to produce similar feature embeddings for images from the same class (and different embeddings for images from different classes).


$$ l(p_i, p_i^+, p_i^-) = \max \{ 0, g + D \big(f(p_i), f(p_i^+) \big) - D \big( f(p_i), f(p_i^-) \big)  \} $$


$D$ is the Euclidean Distance between $f(p_i)$ and $f(p_i^{+/-})$.


$$ D(p, q) = \sqrt{(q_1 − p_1)^2 + (q_2 − p_2)^2 + \dots + (q_n − p_n)^2} $$


$g$ is the gap parameter that regularizes the gap between the distance of two image pairs: $(p_i, p_i^+)$ and $(p_i, p_i^-)$. We use the default value of $1.0$, but you can tune it if you’d like (make sure it's positive).


### Testing stage

The testing (inference) stage only has one network and accepts only one image. To retrieve the top $n$ similar results of a query image during inference, the following procedure is followed:

1. Compute the feature embedding of the query image.
2. Compare (euclidean distance) the feature embedding of the query image to all the feature embeddings in the training data (i.e. your database).
3. Rank the results - sort the results based on Euclidean distance of the feature embeddings.

set test directory and run
```
python3 utils.py 
```


### Sampling strategies

Since we have used the simplified version of sampling method as follows:

- $p_i$: Input to the $Q$ (Query) network. This image is randomly sampled across any class.
- $p_i^+$: Input to the $P$ (Positive) network. This image is randomly sampled from the **SAME** class as the query image.
- $p_i^-$: Input to the $N$ (Negative) network. This image is randomly sample from any class **EXCEPT** the class of $p_i$.

I have created a file named [`sampler.py`](https://github.com/Zhenye-Na/image-ranking/blob/master/src/sampler.py) which is aimed to random sampling positive images and negative images for each query image.

```
$ python3 sampler.py
Input Directory: ../tiny-imagenet-200/train
Output Directory: ../
Number of Positive image per Query image:  1
Number of Negative image per Query image:  1
==> Sampling Done ... Now Writing ...
```

[`triplets.txt`](https://github.com/Zhenye-Na/image-ranking/blob/master/triplets.txt) looks like this:

```
../tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG,../tiny-imagenet-200/train/n01443537/images/n01443537_219.JPEG,../tiny-imagenet-200/train/n04376876/images/n04376876_418.JPEG
../tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG,../tiny-imagenet-200/train/n01443537/images/n01443537_219.JPEG,../tiny-imagenet-200/train/n02948072/images/n02948072_159.JPEG
../tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG,../tiny-imagenet-200/train/n01443537/images/n01443537_219.JPEG,../tiny-imagenet-200/train/n04099969/images/n04099969_450.JPEG
```
### Training
```
python3 main.py --dataroot data_dir 
```
### Save embedded text
set  the image directory that is going to be extracted
```
python3 utils_2.py
```



## References

[1] Jiang Wang, Yang song, Thomas Leung, Chuck Rosenberg, Jinbin Wang, James Philbin, Bo Chen, Ying Wu. [*"Learning Fine-grained Image Similarity with Deep Ranking"*](https://arxiv.org/abs/1404.4661). arXiv:1404.4661  
[2] Akarsh Zingade [*"Image Similarity using Deep Ranking"*](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978)  
[3] Pytorch Discussion. [Feedback on PyTorch for Kaggle competitions](https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252)  
