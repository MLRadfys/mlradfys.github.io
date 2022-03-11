---
title: "Post: Machine learning: KNN for classification"
categories:
  - Blog
tags:
  - machine learning
  - algorithms
  - AI
classes: 
  - wide
toc:
  - True
---

# 1. Introduction

The K-nearest neighbor (KNN) algorithm is a supervised learning algorithm, which can be used for both classification and regression problems. Supervised algorithms consist of a dataset having N training examples, where each training example has a (class) label assigned to it. In the case of classification, KNN predicts the class label for a new, unseen data point, by looking at its k nearest neighbors and assigning a class label by majority voting. 

The KNN algorithm is a so called "lazy learner". While other machine learing algorithms, like in example logistic regression, use the training data to learn the parameters of a function, the KNN algorithm has no real learning phase. Instead it *"memorizes"* the training set. </ br>

**Note:** The prediction step in the KNN algorithm can be computationally expensive, depending on the number of training examples. Each time we make a new prediction, the algorithm searches the entire dataset for the k nearest neighbors.


# 2 How it works...

1. Choose a value for K.
2. Compute the distance between a new sample and all training samples using a distance measure (such as the Euclidean distance).
3. Sort the distances and determine the k nearest neighbors, meaning the k neighbors with the shortest distance to the new sample.
4. The majority category / class of the nearest neighbors is the lable for the new sample.

## 2.1 KNN for regression
For regression, predictions are based on the mean or the median value of the K most similar instances.

## 2.2 KNN for classification

For a classificatation task, the class for a new sample is predicted using the class with the highest occurence (frequency) from the K most similar instances.

## 2.3 Prepare your data

Basically there are three important preprocessing steps that you might want to perform before running your KNN algorithm:

- **Rescaling:** KNN works best if your data is on the same scale. A good idea is to normalize your data between 0-1. You can also try to standarize your dataset by subtracting the mean and dividing by the standard deviation.
- **Missing data:** You should adress missing data. If your dataset contains missing values, the distance between a new datapoint and these values cannot be computed. You can choose to either exclude missing values or you could use imputation techniques.
- **Low dimensionality:** KNN works good for low dimensional data, but struggles in high dimensions (*curse of dimensionality*). Try to run the KNN algorithm on your data and evaluate it. If it doesn't perform well, try to decrease the dimenions by using feature selection or dimensional reduction methods.

## 2.4 Choosing the value of K

Choosing the correct value of K isn't a simple task. A small value for K might lead to model with low bias and high variance, whereas a large value for K might lead to a model with low variance and high bias.
One possible solution is to run the KNN algorithm for several different values of K and choose the model with either the highest accuracy, or the model with the lowest error rate on the validation set. 

Another common technique to choose K is by taking the square root of the number of samples N in the dataset:

$$K = \sqrt{N}$$


# 3. KNN in Python 

Now that we know how the KNN algorithm works, lets try to implement it in Python. <br /> 
<br />
We will use the Iris Flower dataset, which includes measurements (features) of iris flowers. These features can be used to predict a flowers species using new, unseen data.

The dataset contains multiple classes and features like sepal lenght/width and petal length/width.
We can use the scikit learn library to load the dataset:

```python
from sklearn import datasets
# Load Iris Dataset
iris = datasets.load_iris()
X = iris.data  
y = iris.target
```

If you want to read the dataset on you own, you can find it as a .csv file here : [Iris dataset](/assets/post_4/iris.csv)

Like mentioned before, the KNN algorithm works best if all features are on the same scale. We should therefore normalize our dataset. One way of doing this is by transforming the features into a range between 0 - 1. 

```python
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)
```

We also need a distance measure that computes the distance between all samples in the training data and a new sample.
In this example we will use the Euclidean distance, which is given as the square root of the sum of the squared distances between point a and b across all features i:

$$E(a,b) = \sqrt{\sum_{i = 1}^{n}(a_{i} - b_{i})^2}$$

Lets implement the Euclidean distance in Python:

```python
from math import sqrt

def euclidean_distance(self, a,b):
  '''
  Args: 
      a (vector): vector containing the features of sample a
      b (vector): vector contraining the features of sample b
  Returns:
      euclidean distance between a and b
  '''
  d = 0.0
  for i in range(len(a)):
      d += (a[i] - b[i])**2
  return sqrt(d)
```

Next we need to use our Euclidean function and compute the distances between all samples in the dataset and a new sample, keep track of the distances, and return the K nearest neighbors for the sample.

```python
def get_NN(self, X_train, y_train, test_sample):
  '''
  Args: 
      X (array): training dataset
      test_sample (array): new test sample
      n_neighbors (int): the number of K neighbors that should be returned
  Returns:
      neighbors (array): the K nearest neighbors
  '''
  # generate a list that stores all distance
  all_distances = [] 
  
  #Loop over all training examples
  for sample, label in zip(X_train, y_train):
      #compute the distance between a training sample an the new test sample
      distance = self.euclidean_distance(test_sample, sample)
      all_distances.append((sample, distance, label))
  
  #Sort the list using the second item as a key
  all_distances.sort(key = lambda a: a[1])

  
  # generate a list to store the neighbors
  neighbors, distances, labels = zip(*([(tp[0], tp[1], tp[2]) for tp in all_distances]))

  #return the k nearest neighbors
  return neighbors[: self.k_neighbors], distances[: self.k_neighbors], labels[: self.k_neighbors]
```
Now that we have a function that computes the euclidean distances between the training examples and a new, unseen test sample, we need to make a class prediction, based on the K nearest neighbors.

```python
def predict(self, X_train, y_train, test_sample):
  '''
  Args: 
      X (array): training dataset
      test_sample (array): new test sample
      n_neighbors (int): the number of K neighbors that should be returned
  Returns:
      prediction (int): predicted class label
  '''
  nearest_neighbors, distances, y = self.get_NN(X_train, y_train, test_sample)

  most_common = Counter(y).most_common(1)[0]
  
  y_pred_i = most_common[0]
          
  predicted_class = self.get_class_prediction(distances, y)
  
  return predicted_class
```

We also need to define a function that loops over several new samples and not only a single instance.

```python
 def k_nearest_neighbors(self, X_train, y_train, X_test):
    """
    Args:
        X_train (array) : training dataset
        y_train (array): training class labels
        X_test (array): test dataset           
    Returns:
        predictions (array): predicted class labels for the test dataset
    """
    predictions = list()
    for test_sample in X_test:
        output = self.predict(X_train, y_train, test_sample)
        predictions.append(output)
    return predictions
```

The last thing we need to implement is some kind of measure, that tells us how good our algorithm performs. For a classification problem like this, we can use accuray as a metric, which is given as the number of correct predictions divided by the number of total predictions:

$$Accuray = \frac{(TP + TN)}{(TP + FP + TN + FN)}$$

```python
# Calculate accuracy percentage
def accuracy_metric(self, gt_label, predicted):
    correct = 0
    for i in range(len(gt_label)):
        if gt_label[i] == predicted[i]:
            correct += 1
    return correct / float(len(gt_label)) * 100.0
```


**Note:** Usually you would like to train the model using k-fold cross-validation, resulting in K different models. Once this is done, you could make a prediction with all K models and average the model outputs to receive a final prediction. The final model can then be used to make predictions on an independent test dataset.

Lets put everything together in a Python class. <br />

```python
class KNN_classifier:

1
    def __init__(self, k_neighbors, folds = 5, normalize = False):
        '''
        Args:
            k_neighbors (int): number of nearest neighbors
            folds (int): number of folds for cross-validation training
        Returns:
            None
        '''

        self.k_neighbors = k_neighbors
        self.folds = folds
        self.normalize = normalize
        self.dataset = None
        self.classes = None

        #get iris dataset and normalize data if normalize = True
        self.initialize()

        
    def initialize(self):

        self.X, self.y = self.get_dataset()
        if self.normalize:
            X = self.normalize(X)
        self.classes = np.unique(self.y)
     

    def get_dataset(self):
        
        '''
        Args:
            path (str): path to dataset file
        Returns:
            dataset (list): dataset
        '''

        # Load Iris Dataset
        iris = datasets.load_iris()
        X = iris.data      

        y = iris.target

        return X, y
    
    def normalize_data(self, X):   
        '''
        Args:
            X (array): numpy array containing the dataset
        Returns:
            X (array): Normalized array
        '''      

        X = MinMaxScaler().fit_transform(X)

        return X
    
    # Calculate accuracy percentage
    def accuracy_metric(self, gt_label, predicted):
        correct = 0
        for i in range(len(gt_label)):
            if gt_label[i] == predicted[i]:
                correct += 1
        return correct / float(len(gt_label)) * 100.0
        
    

    #Split a dataset into k folds
    def cross_validation_split(self, X, y, n_folds):

        '''
        Args:
            X (array): training dataset
            y (array): class labels
            n_folds (int): number of K folds for cross validation

        Returns:        
            X_split (array): returns K valdiation folds
        '''

        X_split = list()
        X_copy = list(X)
        
        # Determine the number of samples in each fold
        fold_size = int(len(X) / n_folds)
        
        for _ in range(n_folds):
            
            fold = list()
            
            while len(fold) < fold_size:

                index = randrange(len(X_copy))
                #append item to fold while deleting it fromX_copy
                fold.append(X_copy.pop(index))
            
            X_split.append(fold)
        
        return X_split
    
    def euclidean_distance(self, a,b):
        '''
        Args: 
            a (vector): vector containing the features of sample a
            b (vector): vector contraining the features of sample b
        Returns:
            euclidean distance between a and b
        '''
        d = 0.0
        for i in range(len(a)):
            d += (a[i] - b[i])**2
        return sqrt(d)


    
    def get_class_prediction(self, distances, y):
        """
        Args:
            distances (array): distances of the k nearest neighbors
            y (array): labels of the k nearest neighbors
        Returns:
            y_pred_i (int): predicted class label obtained by majority voting
        """
        
        most_common = Counter(y).most_common(1)[0]
        y_pred_i = most_common[0]
        
        #P(A) = #outcomes in A / #outcomes in sample Space
        #y_pred_proba_i = most_common[1] / len(y)

        return y_pred_i

    
    def get_NN(self, X_train, y_train, test_sample):
        '''
        Args: 
            X (array): training dataset
            test_sample (array): new test sample
            n_neighbors (int): the number of K neighbors that should be returned
        Returns:
            neighbors (array): the K nearest neighbors
        '''
        # generate a list that stores all distance
        all_distances = [] 
        
        #Loop over all training examples
        for sample, label in zip(X_train, y_train):
            #compute the distance between a training sample an the new test sample
            distance = self.euclidean_distance(test_sample, sample)
            all_distances.append((sample, distance, label))
        
        #Sort the list using the second item as a key
        all_distances.sort(key = lambda a: a[1])

       
        # generate a list to store the neighbors
        neighbors, distances, labels = zip(*([(tp[0], tp[1], tp[2]) for tp in all_distances]))

        #return the k nearest neighbors
        return neighbors[: self.k_neighbors], distances[: self.k_neighbors], labels[: self.k_neighbors]

    def predict(self, X_train, y_train, test_sample):
        '''
        Args: 
            X (array): training dataset
            test_sample (array): new test sample
            n_neighbors (int): the number of K neighbors that should be returned
        Returns:
            prediction (int): predicted class label
        '''
        nearest_neighbors, distances, y = self.get_NN(X_train, y_train, test_sample)

        most_common = Counter(y).most_common(1)[0]
        
        y_pred_i = most_common[0]
               
        predicted_class = self.get_class_prediction(distances, y)
        
        return predicted_class
    
    # kNN Algorithm
    def k_nearest_neighbors(self, X_train, y_train, X_test):
        """
        Args:
            X_train (array) : training dataset
            y_train (array): training class labels
            X_test (array): test dataset           
        Returns:
            predictions (array): predicted class labels for the test dataset
        """
        predictions = list()
        for test_sample in X_test:
            output = self.predict(X_train, y_train, test_sample)
            predictions.append(output)
        return predictions
    

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self):

       
        scores = list()
        
        kf = KFold(n_splits=self.folds, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            
            X_train, X_test = self.X[train_index], self.X[test_index]
            
            y_train, y_test = self.y[train_index], self.y[test_index]

            predicted = self.k_nearest_neighbors(X_train, y_train, X_test)
            
            accuracy = self.accuracy_metric(y_test, predicted)
            
            scores.append(accuracy)
        
        return scores 

```

``` python
# Test the kNN on the Iris Flowers dataset
num_neighbors = 5

knn = KNN_classifier(filename, num_neighbors, folds = 5)

scores = knn.evaluate_algorithm()
#scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

```python
Scores: [100.0, 93.33333333333333, 93.33333333333333, 96.66666666666667, 100.0]
Mean Accuracy: 96.667%
```

Using K-fold cross validation and K = 5 nearest neighbors we obtain a mean accuray of 96.7%.

## 3.1 KNN decision boundary

Another interesting thing is to run the KNN algorithm and to visualize the decision boundary of our classifier. For that, we need to define the upper and lower bounds of our dataset, create a meshgrid with testdata, and run them through our model. Once we have obtained the predictions we can plot the KNNs decision boundary together with the training and test data.

For visualization purposes, we can try to plot the decision boundary for just two of the iris datasets features:

```python
from matplotlib.colors import ListedColormap
def scatter_plot(self, X, y, X_train, y_train, X_test, y_test, k, feature_idxs =[1,3] , h=0.05):
    
    legend  = ['Setosa', 'Versicolour', 'Virginica']

    feature_names = datasets.load_iris().feature_names

    xlbl, ylbl = feature_names[feature_idxs[0]], feature_names[feature_idxs[1]] 


    self.debug = False

    classes = list(set(y))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    colours = ['red', 'green', 'blue']
    pad = 0.5
    x_min, x_max = X[:, feature_idxs[0]].min() - pad, X[:, feature_idxs[0]].max() + pad
    y_min, y_max = X[:, feature_idxs[1]].min() - pad, X[:, feature_idxs[1]].max() + pad

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    X_train = X_train[:,[feature_idxs[0], feature_idxs[1]]]
    X_test = X_test[:,[feature_idxs[0], feature_idxs[1]]]

    mesh_testData = np.c_[xx.ravel(), yy.ravel()]
    
    Z =self.k_nearest_neighbors(X_train, y_train, mesh_testData)
    
    Z = np.array(Z).reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
    for i in classes:
        idx = np.where(y_train == classes[i])
        plt.scatter(X_train[idx, 0],
                    X_train[idx, 1],
                    c=colours[i],
                    label=legend[i],
                    marker='o', s=20)
    for i in classes:
        idx = np.where(y_test == classes[i])
        plt.scatter(X_test[idx, 0],
                    X_test[idx, 1],
                    c=colours[i],  # label=legend[i],
                    marker='x', s=20)
    plt.legend()
    plt.xlabel(xlbl, fontsize=16)
    plt.ylabel(ylbl, fontsize=16)
    plt.title("kNN classification (k = {}) - train (o), test (x)"
            .format(k), fontsize=16)
    plt.show()       
```

And here are the resulting decision boundary plots for K = 5 different training and validation folds:

<figure>
	<img src="/assets/images/post_4/decision_boundary_1.png">
	<figcaption>Figure 1. Decision boundary fold 1.</figcaption>
</figure>

<figure>
	<img src="/assets/images/post_4/decision_boundary_2.png">
	<figcaption>Figure 2. Decision boundary fold 2.</figcaption>
</figure>

<figure>
	<img src="/assets/images/post_4/decision_boundary_3.png">
	<figcaption>Figure 3. Decision boundary fold 3.</figcaption>
</figure>

<figure>
	<img src="/assets/images/post_4/decision_boundary_4.png">
	<figcaption>Figure 4. Decision boundary fold 4.</figcaption>
</figure>

<figure>
	<img src="/assets/images/post_4/decision_boundary_5.png">
	<figcaption>Figure 5. Decision boundary fold 5.</figcaption>
</figure>


