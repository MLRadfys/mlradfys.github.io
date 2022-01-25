---
title: "Post: Semi-supervised learning part 2"
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
link: 
---

In [part 1](https://mlradfys.github.io/blog/Semi-supervised-learning-part-1/) of this series about semi-supervised learning we looked at different ways on how to train machine learning models (supervised, unsupervised, semi-supervised), talked about different machine learning paradigms and went through an introduction to semi-supervised learning.
In this post I want to introduce the concept of **consistency regularization**, which is also called for **consistency training**. We will first discuss what consistency regularization is and then see how we can use it for semi-supervised training by looking at some of the popular models.

# 1. Consistency Regularization

## 1.1 Introduction

The main idea of consistency regularization is that a prediction on unlabeled data should stay the same, even if we add pertubations to the input data. This is connected to the *cluster assumption* of semi-supervised learning. Pertubations or noise can be added in form of random noise, data augmentation or in the model architecture itself, by e.g. adding Dropout layers. The magnitude of the noise should be chosen, so that model predictions are consistent but they should not be messed up!

<figure>
	<img src="/assets/images/post 2/consistency.png">
	<figcaption>Figure 1. Consistency regularization. a) A dataset with labeled and unlabeled examples. b) predictions inside a circle should be constant, while overlapping regions should result in the same prediction. c) Decision boundary. (from Standford CS224)</figcaption>
</figure>

**Figure 1** visualizes the concept of consistency regularization. We have a dataset with two classes (red, blue), and a few labeled examples. In addition, we have many unlabeled datapoints. By adding noise during consistency training, we try to smooth out the distribution. The model should give constant predictions inside a circle, while predictions in overlapping regions should be the same.

*How can we use consistency regularization for semi-supervised learning?* <br />

In a typical supervised training setting we compare the model output with the true label and optimize the model using a cost function. For unlabeled data, the true label is not available though. Nevertheless, consistency regularization does not need the true label. Instead, we can introduce a consistency loss term that compares input data with slightly different pertubations. What kind of consistency loss we use depends on the application, but some popular ones are the [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) , the [mean squared error (MSE) ](https://en.wikipedia.org/wiki/Mean_squared_error) or the [binary cross-entropy (BCE)](https://en.wikipedia.org/wiki/Cross_entropy) loss. The introduction of the conistency loss term allows us to train our model in a semi-supervised form, by using both labeled and unlabeled data.

# 2. Model architectures using consistency regularization

In this section we will take a look at some of the most famous architectures that use consistency regularization for semi-supervised training. 

## 2.1 Ladder Network

The ladder network was introduced by Rasmus et. al [1] and consists of 2 encoders and a single decoder (**Figure 2**). One of the encoders is referred to as the "clean encoder", mapping and input $$x$$ to output $$y$$, while Gaussian noise is added to all layers of the other encoder. Notice that the clean and the noise-corrupted encoders share the same mappings $$f(\cdot)$$ The decoder takes the noise-corrupted mappings from each layer of the noisy encoder and tries to denoise them. With the help of the decoder, we can add an unsupervised learning component to our setting.

<figure>
	<img src="/assets/images/post 2/ladder_network.PNG">
	<figcaption>Figure 2. Architecture of the ladder network with L = 2.</figcaption>
</figure>


The total cost function is a combination of a supervised and an unsupervised part. The supervised cost is given by the negative log-likelihood, which can be computed for all labeled samples:

$$C_{c} = -\frac{1}{N}\sum_{n=1}^{N}logP(\tilde{y}=t(n)|x(n))$$

where $$\tilde{y}$$ is the with noise corrupted output, $$t(n)$$ the true label and $$x(n)$$ the input to the encoder. The Gaussian noise added to the layers can be seen as a regularizer.

The unsupervised loss is given by the denoising costs of all layers multiplied by a weighting factor, determining the importance of each layer. The unsupervised loss is given by:

$$C_{d} = \sum_{l=0}^{L}\lambda_{l}C_{d}^{(l)} = \sum_{l=0}^{L}\frac{\lambda_{l}}{Nm_{l}}\sum_{n=1}^{N}\left \| z^{(l)}(n)-\hat{z}_{BN}^{(l)}(n) \right \|$$

where $$m_{l}$$ is the layer width, $$N$$ the number of training examples and $$\lambda_{l}$$ is a hyperparameter, defining the weight for each layer. 

The final cost is then given by:

$$C=C_{c}+C_{d}= -\frac{1}{N}\sum_{n=1}^{N}logP(\tilde{y}=t(n)|x(n)) + \sum_{l=0}^{L}\frac{\lambda_{l}}{Nm_{l}}\sum_{n=1}^{N}\left \| z^{(l)}(n)-\hat{z}_{BN}^{(l)}(n) \right \|$$

Network parameters are learned by backpropagation, and the clean encoder can be used to make predictions for new, unseen data. The ladder network is the first step towards *Student-Teacher models* which are further discussed below.

## 2.2 Pi-Model

The $$\pi$$-Model, introduced by Leine et. al [2] can be trained with both labeled and unlabeled data simultaneously. For both we generate two random augmentations, and use the augmented images as the input to our model, resulting in predictions $$z_{i}$$ and $$\tilde{z_{i}}$$ (**Figure 3**).

<figure>
	<img src="/assets/images/post 2/pi-model.PNG">
	<figcaption>Figure 3. Pi-model.</figcaption>
</figure>

The loss function that is minimized during training consists of two different components. The first one ist the supervised component used for labeled images, which is the standard Cross-Entropy (CE) loss. The second component is the mean squared error between $$z_{i}$$ and $$\tilde{z_{i}}$$, which is the consistency loss. The consistency component is computed for all input images (labeled and unlabeled). The total loss is then given by:

$$L = -\frac{1}{\left | B \right |} \sum_{i\epsilon (B\cap L)}^{}log(z_{i})y_{i} + w(t)\frac{1}{C\left | B \right |} \sum_{i \epsilon B}^{} \left \| z_{i} - \tilde{z_{i}}\right \|^2$$

where $$w(t)$$ is a time dependent weight factor. In the original paper, the weight $$w(t)$$ is ramped up during training, starting at $$w = 0$$. That enables the supervised loss to dominate in the beginning, hopefully leading to more robust predictions of the unlabeled samples later on in the training process. 

## 2.3 Temporal Ensembling

In the same paper that introduced the above mentioned $$\pi$$-Model, Leine et al. suggests a modified version, called for temporal ensembling [2]. Temporal ensembling builds upon the $$\pi$$-Model and extend it with an Exponential moving average (EMA) based on previous predictions. 
Noise in form of stochastic augmentations is added to an input image. This image serves as the input to a model with Dropout, to generate a prediction $$z_{i}$$ (**Figure 4**).


<figure>
	<img src="/assets/images/post 2/temporal_ensembling.PNG">
	<figcaption>Figure 4. Temporal Ensembling.</figcaption>
</figure>

The predictions are then compared with the EMA predictions $$\tilde{z_{i}}$$, using the squared difference. This is used as the consistency regularization loss, and computed for both labeled and unlabeled images. In addition, the Cross Entropy loss is computed for labeled images. The total loss is then the weighted sum of the two loss terms. 

$$L = -\frac{1}{\left | B \right |} \sum_{i\epsilon (B\cap L)}^{}log(z_{i})y_{i} + w(t)\frac{1}{C\left | B \right |} \sum_{i \epsilon B}^{} \left \| z_{i} - \tilde{z_{i}}\right \|^2$$

Ensemble predictions are stored as the EMA, given by:

$$Z = \alpha Z + (1-\alpha)z$$

wheras the target vectors are constructed using a start-up bias correction:

$$\tilde{z} = \frac{Z}{1-\alpha^t}$$

The temporal ensembling method is faster when compared to the $$\pi$$-Model, and the training target $$\tilde{z_{i}}$$ is less noisy. The disadvantage is that predictions for all samples have to be stored during training. For huge datasets, this might be a problem.


## 2.4 Mean Teacher

The Mean Teacher algorithm uses two models:

- Teacher model and
- Student model

, where both have the same architecture (**Figure 5**). For labeled and unlabeled data we create two perdubated images. In the paper, noise is added in form of image augmentation (random translations, horizontal flips), Gaussian noise ($$\sigma = 0.15$$) added to the input layer of the  network, as well as dropout with a probability of $$p = 0.5$$ added to the network architecture.  <br />

<figure>
	<img src="/assets/images/post 2/mean_teacher.PNG">
	<figcaption>Figure 5. Mean Teacher.</figcaption>
</figure>

One image is fed to the Student model, which predicts the distributions of the labels. The other image is fed into the Teacher network. We can then compute the squared difference between the two predictions, which serves as our consistency loss. This loss component can be computed for both labeled and unlabeled images. In addition, we can compute the Cross-Entropy loss for labeled images. The final loss is the time-dependent weighted sum of the Cross-Entropy loss and the consistency term.

$$ L = L_{CE} + w(t)L_{consistency}$$

As we have seen in the description of the Temporal Ensembling method, saving the predictions for all samples is not feasible for large datasets. The Mean Teacher approach improves that. Instead of computing the EMA of the predictions, we compute the EMA of the Student models parameters. During training, the Teacher weights are updated to the EMA of the Student parameters. 

The parameters of the Teacher model for the actual training epoch $$t$$ are given by:

$$\theta_{t}^{'} = \alpha \theta_{t-1}^{'} + (1-\alpha)\theta_{t}$$

where $$\theta_{t}^{'}$$ and $$\theta_{t-1}^{'}$$ are the model parameters of the student network in epoch $$t$$ and $$t-1$$, respectively. The smoothing coefficient of the EMA is given by $$\alpha$$.
With that, the Teacher model becomes the average of the Student model, though with a higher weight on most recent Student model parameters. The Teacher becomes a more stable and less fluctuat version when compared to the Student.

## 2.5 Virtual Adverserial Training (VAT)

Virtual Adverserial Training (VAT) was proposed by Miyato et. al [4] and is a regularization method based on an adverserial loss term. VAT is based on the idea of adverserial attack, where adverserial noise is added to the input [5].  <br />
The idea is to make the model robust against this type of noise, leading to a smooth manifold. The model output becomes robust against small pertubations applied to the input.
Similar to the other methods, we take an input image and apply noise to it. An adverserial image of the orignal one is generated by **maximizing** a distance norm between the two (e.g. the KL divergence). 
The adverserial image can the be used to compute the distance between the original image and the adverserial one, which serves as the consistency term:

$$L_{u}^{adv}(x,\theta)= D\left[p_{\theta}(y|x), p_{\theta}(y|x + r_{adv})\right]$$

Here $$p_{\theta}(y|x)$$ is the label distribution predicted given input image $$x$$, while $$p_{\theta}(y|x + r_{adv})$$ is the label distribution predicted given input image $$x$$ + *adverserial pertubation*.
To compute the difference we can use the KL divergence, but now we want to **minimize** it. The consistency term can be computed for both labeled and unlabeled images. In addition we compute the cross entropy loss for the labeled data. <br />
The total loss is then given by:

$$ L = L_{CE} + L_{u}^{adv}$$


## 2.6 Dual Students
As we have seen in section 2.4, the Mean Teacher model uses an EMA of th Student model to update the weights of the Teacher. This method has one major drawback. If trained long enough, the weights of the Teacher will converge to the weights of the Student, resulting in two very similar models. This doesn't necessarily mean that the Teacher model is bad, but any biases or errors contained in Student model will evantually be transfered to the Teacher.<br />
To avoid this situation, Ke et. al [6] proposed a so called Dual Student model. The architecture consists of two models, Student $$M1$$ and Student $$M2$$, both having the same architecture, but with two different initializations (**Figure 6**). 

<figure>
	<img src="/assets/images/post 2/dual_student.PNG">
	<figcaption>Figure 6. Dual student architecture consisting of two studend models with the same architecture but different initializations of the model weights.</figcaption>
</figure>

The authors also define the so called **stable sample**. A sample is stable if it satisfies the following stability conditions:

- using a model, the input $$x$$ and its pertubated version $$\tilde{x}$$ should lead to the same prediction
- the predictions for $$x$$ and $$\tilde{x}$$ should be confidend, meaning they should be larger then a pre-defined threshold $$\epsilon$$.

For both Students, a total of four outputs are predicted, $$M1(x_{1})$$, $$M1(\tilde{x_{1}})$$, $$M2(x_{2})$$ and $$M2(\tilde{x_{2}})$$. For unlabeled and labeled samples we can compute a Consistency loss between the models prediction. In addition, the Cross-Entropy loss is computed for labeled images. <br />
In addition to the two losses, one of the students, will also determine the target values for the other one. The stability of a models output is determined for each sample. If the sample of one Student model is unstable, the target is determined by the other Student and vice versa. 
If both models satisfy the stability criterion for x, the stability is calculated between the two models using the mean squared error:

$$L_{MSE} = \left \| f(\theta^i,x) - f(\theta^j,x) \right \|^2$$
	 
The stability constraint is then applied from the more stable model. 

The total loss for Student i is given by:

$$L_{i} = L_{cls}^{i} +\lambda_{1}L_{con}^{i} + \lambda_{2}L_{sta}^{i}$$

The total loss is determined for both models independently and the model weights for both students are updated using backpropagation. 

# 3. References
[1]Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, and Tapani Raiko. 2015. Semi-supervised learning with Ladder networks. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'15). MIT Press, Cambridge, MA, USA, 3546–3554.
[2] Laine, S. & Aila, T. (2017). Temporal Ensembling for Semi-Supervised Learning.. ICLR (Poster), : OpenReview.net.  <br />
[3] Tarvainen, A. & Valpola, H. (2017). Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan & R. Garnett (eds.), NIPS (p./pp. 1195-1204).  <br />
[4] Miyato, Takeru & Maeda, Shin-ichi & Koyama, Masanori & Ishii, Shin. (2017). Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence. <br />
[5] Goodfellow, Ian & Shlens, Jonathon & Szegedy, Christian. (2014). Explaining and Harnessing Adversarial Examples. arXiv 1412.6572. 
[6] Ke, Zhanghan, Daoye Wang, Qiong Yan, Jimmy S. J. Ren and Rynson W. H. Lau. “Dual Student: Breaking the Limits of the Teacher in Semi-Supervised Learning.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019): 6727-6735.
<br />








