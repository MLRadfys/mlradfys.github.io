---
title: "Post: Semi-supervised learning part 3"
categories:
  - Blog
tags:
  - Semi-supervised learning
  - machine learning
  - deep learning
  - AI
classes: 
  - wide
toc:
  - True
---

In [part 1](https://mlradfys.github.io/blog/Semi-supervised-learning-part-1/) of this series about semi-supervised learning we looked at different ways on how to train machine learning models (supervised, unsupervised, semi-supervised), talked about different machine learning paradigms and went through an introduction to semi-supervised learning.
In [part 2](https://mlradfys.github.io/blog/Semi-supervised-learning-part-2/) the concept of consistency regularization was introduced, as well as its application in semi-supervised learning. 
In this part of the series about semi-supervised learning we will have a look at **pseudo-labeling** methods, which are a so called wrapper algorithms.

# 1. Introduction to pseudo-labeling

The idea behind pseudo-labeling methods is pretty simple: we use a model trained on a labeled dataset to generate label predictions for unlabeled data. This new, labeled data can then be added to the training dataset. This type of pseudo-labeling is also called for self-supervised learning. 

# 2. Self-supervised training
## 2.1 Pseudo-labels
Pseudo-labels **[1]** trains a semi-supervised model in a supervised way by using both labeled and unlabeled data at the same time (Figure 1). If a training batch contains labeled data, the model is trained using the cross-entropy loss. When the batch contains unlabeled data, we can make predictions with our model and use the label with the maximum confidence (or highest probability) as a so called pseudo-label.<br />



<figure>
	<img src="/assets/images/post_3/pseudo-labels.png">
	<figcaption>Figure 1. Pseudo-labeling.</figcaption>
</figure>


The network is optmized using the following loss function:

$$L = \frac{1}{n}\sum_{m=1}^{n}\sum_{i=1}^{K}R(y_{i}^{m},f_{i}^{m}) + \alpha (t)\frac{1}{n'}\sum_{m=1}^{n'}\sum_{i=1}^{K}R(y_{i}^{'m},f_{i}^{'m})$$

where $$n$$ and $$n'$$ are the number of samples in the labeled and unlabeled batch respectively, $$f_{i}^{m}$$ is the network output for the labeled sample m,  $$f_{i}^{'m}$$ the network output unlabeled sample m', $$y_{i}^{m}$$ is the true label of sample m and $$y_{i}^{'m}$$ is the pseudo-label for the unlabeled sample $$m'$$. Another interesting term in the loss function is $$\alpha(t)$$, which is a weight applied to the unlabeled part of the loss. The weight is time-dependent and changes with the number of epochs.
The paper suggests the following scheme for the weight:

$$\alpha(t) = \begin{cases}
0 & \text{ if } t <  T_{1}\\ 
\frac{t-T_{1}}{T_{2}-T_{1}}\alpha_{f} & \text{ if } T_{1} \leq t < T_{2} \\ 
\alpha_{f} & \text{ if } T_{2} \leq t
\end{cases}$$

with $$\alpha_{f}  = 3, T_{1} = 100$$ and $$T_{2} = 600$$.  <br />
Basically we keep the weight equal to zero during the first 100 training epochs and increase it linearly up to epoch 600. For epochs > 600, the weight is kept constant. Starting with an unlabeled loss equal to zero allows the model to focus on the labeled data in the beginning of the training. This hopefully leads to more meaningful predictions for unlabaled examples later on in the training process.

## 2.2 Noisy Student

The Noisy student method **[2]**  makes use of two different models: a **teacher** and a **student** (Figure 2). The teacher model is first trained on the labeled training dataset and can then be used to generate pseudo-labels for the unlabeled data. Usually we choose the class with the highest probability (most confident class) as the pseudo-label. The labeled and unlabeled data together with the true and pseudo-labels are then combined to a single dataset, which is used to train the student model. During training, the input data is augmented with random noise, like e.g. RandAugment, as well as additional noise directly incorporated into the model architecture by using Dropout or stochastic depth.
Once the student model is trained, it becomes the new teacher and the unlabeled dataset is re-labeled, generating new, updated pseudo-labels. The whole process is repeated for several iterations.

<figure>
	<img src="/assets/images/2022-01-13-Semi-supervised-learning-part-2_Noisy_Student.png">
	<figcaption>Figure 2. Blockdigram of the Noisy student algorithm.</figcaption>
</figure>



# 3. References
[1] D.-H. Lee, “Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks,” in Work-shop on challenges in representation learning, ICML, vol. 3, no. 2,2013. <br />
[2] Q. Xie, M. -T. Luong, E. Hovy and Q. V. Le, "Self-Training With Noisy Student Improves ImageNet Classification," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 10684-10695, doi: 10.1109/CVPR42600.2020.01070. <br />







