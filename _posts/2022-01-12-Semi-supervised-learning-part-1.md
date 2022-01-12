---
title: "Post: Semi-supervised learning part 1"
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

# An introduction to semi-supervised learning

<div align = 'justify'>
In this series of blog posts we will dive into a machine learning task called for semi-supervised training. We will start by taking a general look at the different machine learning tasks, namely supervised, unsupervised and semi-supervised learning, as well the paradigms of statistical inference. After that we will discuss why we need semi-supervised machine learning, which assumptions we make, and finally we will discuss some semi-supervised algorithms. The intention of this post is not to give a detailed explanation of the different semi-supervised algorithms, but rather to give you an overview of this topic. 
</div> <br />

## 1. Supervised, Unsupervised and Semi-supervised Learning


Machine learning tasks can be split into:
- Supervised learning
- Unsupervised learning and
- Semi-supervised learning
- (Reinforcement learning)


In *supervised* machine learning problems, we have a dataset of labeled data pairs $$D_{l} =(({x_{i}, y_{i}))}_{i = 1}^{N}$$, where $${N}$$ is the number of datapoints and $$x_{i}$$ is an example from a given input space $$X$$ with a matching label  $$y_{i}$$. The goal is now to find a mapping between $$x$$ and $$y$$, generating a function that can be used to predict label $$\hat{y}_{j}$$ from an unseen datapoint $${x}_{j}$$, coming from an unlabeled dataset $$D_{ul} =(({x_{j}))}_{j = 1}^{N}$$. <br />
In *unsupervised* machine learning, we are working with an unlabeled dataset $$D_{ul} =(({x_{i}))}_{i = 1}^{N}$$. The goal in unsupervised machine learning is to find an interesting structure in the data.  <br />
*Semi-supervised* machine learning is somewhere in the middle. Usually we have a small labeled dataset $$D_{l} =(({x_{i}, y_{i}))}_{i = 1}^{N}$$ with examples $$x$$ and labels $$y$$, and a large unlabeled dataset $$D_{ul} =(({x_{i}))}_{i = 1}^{N}$$, with examples $$x$$ only. In a semi-supervised setting, the small labeled dataset is usually used to supervise the learning task. <br />
As you can see, I also added *reinforcement learning* to the list of machine learning tasks, but I will not cover this.

## 2. Inductive, Deductive and Transductive Learning


Before we dive into semi-supervised machine learning, I want to mention some machine learning paradigms used in the context of inference ('reaching an outcome or a decision') called for inductive, deductive and transductive learning (Figure 1).

**Inductive Learning**: *Learning general rules from specific examples* <br />
In inductive learning, or even called inductive reasoning, we use evidence to determine an outcome. In general we can see this as fitting a model using our training dataset, in order to learn general rules and to generalize to new, unseen data.<br /> 
 <br />
**Deductive Learning**: *Learning specific examples from general rules* <br />
In deductive inference, we use our general rules to determine outcomes. In a machine learning context, this can be thought of as making predictions with our model. <br />
 <br />
**Transductive Learning**: *Predicting specific examples given specific examples* <br />
In transductive learning, we don't model our data, but we rather use both training and test data at the same time to make predictions. A classical example of transductive learning is the <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbor algorithm</a>. Every time a prediction is made, a full re-calculation of the algorithm using all data (training + test data) is needed. <br />

<figure>
	<img src="/assets/images/2022-01-12-Semi-supervised-learning-part-1_Relationship-between-Induction-Deduction-and-Transduction.png">
	<figcaption>Figure 1. Paradigms of Induction, deduction and tansduction (from The Nature of Statistical Learning Theory).</figcaption>
</figure>


## 3. Semi-supervised learning

### 3.1 Why do we need semi-supervised learning?

Annotating (labeling) data is a time-consuming and costly process. On the other hand, for many problems, a lot of unlabeled data excists. So what if we could reduce time and costs by making use of this unlabeled data? <br />
This is exactly what semi-supervised learning is about! In semi-supervised learning, we use a small labeled dataset and combine it with an often larger unlabeled dataset. As we use both labeled and unlabeled data, semi-supervised training is sometimes called for a hybrid method.

### 3.2 General assumptions 

As we said before, the goal of semi-supervised learning is to reduce cost and time of data annotation. We also said that this can be achieved by training a model using a small labeled dataset and a larger, unlabeled dataset. Nevertheless, the question is:<br /> <br />
*Can we build a model that performs better, when taking both labeled and unlabled data into consideration, when compared to supervised training only?* <br /><br />
The answere is "Yes", but the distribution of the examples we intend to use must be relevant for our problem. If this is not the case, then our semi-supervised model, will probably not perform any better. <br />

In general, we can make the following assumptions for semi-supervised training [1] :

- **Smoothness**: If two datapoints in the sample space are close to each other, then their labels in the output space should be close to each other as well. So basically, datapoints very close to each other will probably get the same label.
- **Manifold**: Datapoints lying on the same, low-dimensional manifold should have the same label and probably belong to the same class. By low-dimensional manifold, we usually mean the projection of our data from a high dimensional space into a low dimensional manifold, which can be achieved by e.g. dimensional reduction methods like Principal Component Analysis (PCA).
- **Low density separation**: This assumes that the decision boundary between classes does not pass through high-density areas. 
- **Cluster**: Points in the same cluster are likely to have the same class. Basically, if our datapoints (labeled and unlabeled) cannot be clustered in a meaningful way, than our semi-supervised algorithm will not perform better then a supervised method.

### 3.3 Semi-supervised learning methods

Figure 2 shows a summary of different semi-supervised learning algorithms sorted based on their inference paradigms. In general, we can split semi-supervised methods into inductive methods and transductive methods, each consisting of several different algorithms. Here, we focus on the machine learning task of classification.
<br />
The main methods for semi-supervised learning are wrapper- or graph-based, optimization- or cost-function-based or they combine supervised algorithms with an unsupervised setting.

<figure>
	<img src="/assets/images/2022-01-12-Semi-supervised-learning-part-1_methods.PNG">
	<figcaption>Figure 2. Semi-supervised deep learning models based on inference paradigms [2].</figcaption>
</figure>

In this post I will just give you a very brief description of the different methods, without going into much detail. I will cover some popular algorithms in upcoming blog posts. 

#### 3.3.1 Wrapper methods
Wrapper methods train a classifier on labeled data and use this classifier to make predictions on unlabeled and unseen data. These predicions are then used as labels. Labels generated by these wrapper methods are often referred to as *pseudo labels*, where the *wrapper* refers to the methods generating the pseudo-labeled data. The advantage of wrapper-based methods is that they can easily be incorporated into any supervised learning workflow, and extend a supervised method to a semi-supervised setting.

#### 3.3.2 Unsupervised preprocessing 
Unsupervised preprocesssing methods either extract useful features directly from the unlabeled data (*feature extraction*) with e.g. autoencoders, or they make use of pre-clustering techniques (*cluster-then-label*). In addition, initial parameters of a supervised method can be determined using unsupervised methods (*pre-training*), like stacked autoencoders or deep belief networks. In these methods, a model is pre-trained in an unsupervised manner using the unlabeled data, and fine-tuned using the labeled data. Like the wrapper methods, unsupervised preprocessing can be used in combination with any supervised learning algorithm. Nevertheless, compared to the wrapper methods were both labeled and unlabeled data are used for the supervision task, in unsupervised preprocessing, we only use the labeled data for supervision. 

#### 3.3.3 Instrinsically semi-supervised methods:
Intrinsical methods incorporate unlabeled data directly into the optimization process or the cost function of the optimization problem. Often, supervised learning methods are extended to a semi-supervised setting by changing the objective function of the supervised method to include unlabeled data. This method is extremely popular in recent, deep learning-based methods, like the $${\pi}$$-Model, Mean Teacher or Temporal Ensembling. These methods use both labeled and unlabeled data, and rely on data pertubations, which can be expressed and used in the loss function. Other popular methods are adverserial approaches, including generative adverserial networks (GAN) and variational autoencoders (VAE). The advantage with intrinsically semi-supervised methods is, that they can be trained directly on both labeled and unlabeled data, without any intermediate steps like the previous mentioned methods.

#### 3.3.4 Transductive methods
Transductive methods do not train a model to come up with general rules that can be used to make predictions on new data. Instead, we use the whole dataset (training + testing) to make predictions, and need to re-run the algorithm for every new prediction using all data. That means that transductive methods are supplied with both labeled and unlabeled data at the same time, and generate output predictions for the unlabeled data. In the semi-supervised transductive setting, graph-based models can be used, were a graph is computed that connects similar datapoints with each other. For labeled data, the graph is optimized to produce the true label, while similar data should result in the same predictions (for unlabeled data).

### 3.4 Semi-supervised deep learning methods

Semi-supervised learning has become extremely popular in deep learning. As you can imagine, labeling certain kinds of data, e.g. images used for computer vision and medical imaging applications like object recognition, detection or semantic segmentation, is often not feasible due to the huge amount of work. <br />
There are tons of semi-supervised deep learning methods out there and some (e.g. pseudo labeling, pertubation-based) overlap with the already mentioned ones, and are extendet to the deep learning setting. A good overview of deep learning-based semi-supervised methods (Figure 3) can be found in the recently published paper by <a href="https://arxiv.org/abs/2103.00550"> Yang et. al</a>. <br />

<figure>
	<img src="/assets/images/2022-01-12-Semi-supervised-deep-learning-Yang.png">
	<figcaption>Figure 3. Different semi-supervised deep learning models sorted based on model design and loss function [3].</figcaption>
</figure>

## 4. What's next... ?

You should now have a basic understanding about the different types of machine learning methods and paradigms, and about semi-supervised learning. In upcoming posts I will pick some of the most recent and popular semi-supervised deep learning methods, like in example Pseudo-labeling, the $${\pi}$$-Model, Temporal Ensembling and Mean Teacher, and will try to explain them in more detail. 


## 5. References
[1] Chapelle, O., Schölkopf, B. & Zien, A. (eds.) (2006). Semi-Supervised Learning. The MIT Press. ISBN: 9780262033589. <br />
[2] Yang, X., Song, Z., King, I., & Xu, Z. (2021). A Survey on Deep Semi-supervised Learning. ArXiv, abs/2103.00550. <br />
[3] van Engelen, J.E., Hoos, H.H. A survey on semi-supervised learning. Mach Learn 109, 373–440 (2020). <br />







