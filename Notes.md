Notes include lessons from:
1. [Machine Learning Engineering](http://www.mlebook.com/wiki/doku.php) by [Andriy Burkov](https://www.linkedin.com/in/andriyburkov/ "https://www.linkedin.com/in/andriyburkov/")



#### Feature Engineering

to obtain tidy data from raw data


#### Tensors


"Most ML models work with numerical and not categorical data."


#### Hyperparameters

The best configuration values for that learning algorithm. For this and choosing the algorithm, validation set is used.

Hyperparameters are inputs of machine learning algorithms or pipelines that influence the performance of the model. They don’t belong to the training data and cannot be learned from it. For example, the maximum depth of the tree in the decision tree learning algorithm, the misclassification penalty in support vector machines, k in the k-nearest neighbors algorithm, the target dimensionality in dimensionality reduction, and the choice of the missing data imputation technique are all examples of hyperparameters.


#### Baseline

A baseline is a simple algorithm for solving a problem, usually based on a heuristic, simple summary statistics, randomization, or very basic machine learning algorithm


#### Parameters

Variables that define the model trained by the learning algorithm. Parameters are directly modified by the learning algorithm based on the training data. The goal of learning is to find such values of parameters that make the model optimal in a certain sense.

#### Classification

Binary - one of two
Multiclass - one-versus-rest


#### Model-Based Learning

Most supervised learning algorithms are model-based. A typical model is a support vector machine (SVM). Model-based learning algorithms use the training data to create a model with parameters learned from the training data. In SVM, the two parameters are w (a vector) and b (a real number). After the model is trained, it can be saved on disk while the training data can be discarded.


#### Instance Based Learning

Instance-based learning algorithms use the whole dataset as the model. One instance based algorithm frequently used in practice is k-Nearest Neighbors (kNN). In classification, to predict a label for an input example, the kNN algorithm looks at the close neighborhood of the input example in the space of feature vectors and outputs the label that it saw most often in this close neighborhood.

An instance-based machine learning algorithm uses the entire training dataset as a model. The training data is exposed to the machine learning algorithm, while holdout data isn’t.


#### Shallow vs Deep Learning

A shallow learning algorithm learns the parameters of the model directly from the features of the training examples. Most machine learning algorithms are shallow.

Neural network learning algorithms, specifically those that build neural networks with more than one layer between input and output. Such neural networks are called deep neural networks. In deep neural network learning. Most model parameters are learned not directly from the features of the training examples, but from the outputs of the preceding layers


#### Machine Learning Project lifecycle
![[Pasted image 20220728092627.png]]

### Data Collection

#### How much data
Sample rule of thumb says, that data for machine learning should be - 
1. 10 times the amount of features
2. 100/1000 times number of classes
3. 10 times number of trainable parameters


#### Data/Target Leakage
"A variable that the analyst is trying to predict is among the features in feature vector"

In supervised learning, It is the unintentional introduction of information about the target that should not be made available.

Causes of data leakage
 1. Target being function of a feature
	 Ex - predicting yearly salary from data where monthly is part of the input itself
 2. Feature hiding a target
	 Ex - A column may indirectly/factually be related to the target. For a M/F classifier, if one column already has information like M18-25
 3. Feature coming from the future
	 Ex - Some features that might be only present in training data, but not when used in production, which can influence the predicition of the system. Like, to decide whether a person will pay loan back, and a feature "sent late payment reminder" will always be absent in production


#### Feedback Loop
Property in the system design when the data used to train the model is obtained from the model itself.


#### Data Labelling

Required before using data as input for model
A technique: noisy pre-labelling is used where a trained model tries to predict label of data, and then validated by human for correction if needed.

"Noisy data can lead to overfitting of the model in cases of small data sizes, while may average out for individual cases in big data"


#### Types of Biases

"Biased data can lead to incorrect results due to the input for the model, which was skewed to a certain type of values, eventutally leading to a model which is biased towards a certain type  to input"
1. Selection Bias - skewing choice to data sources which are easily available
2. Self selection Bias - when data is from sources which volunteered. Ex - Amazon rating givers
3. Ommitted variable Bias - when you train a model with data which didn't have a feature from outside which came later into existence and started affecting the predictions of the model itself, because the model wasn't trained on it. Ex - A new company giving discount for a product
4. Sponsorship Bias - only taking data into consideration which is positive/expected
5. Sampling Bias / Distribution Shift - when the distribution of example for a target value is very different in development, than real world.
6. Stereotype Bias
7. System value distortion
8. Experimenter Bias
9. Labeling Bias


#### Concept Drift
Once a certain model quality monitoring procedure is deployed in the production environment, New training data is added to adjust the model; the model is then retrained and redeployed.

If the cause of an error is explained by the finiteness of the training set, then new data will improve the model. In other cases, the model starts to make errors because of concept drift.

"Concept drift is a fundamental change in the statistical relationship between the features and the label"

Example - Music taste of users might change due to their age, exposure to newer cultures etc, all of which are not something initially captured as part of the training data.

Such cases require defining an additional hyperparameter, and performing hyperparameter turning (Ex - grid search).


#### Outliers

 1. Dissimilar to other examples in dataset.
 2. Usually measured by some property like Euclidean distance.
 3. Sometimes, an outlier in original feature can be a typical example in a feature vector space transformed using tools such as a kernel function, done by models like SVM or Deep Neural Networks.
 4. Algorithms like Linear and Logistic Regression, are particularly sensitive to outliers.
 5. Usage of concepts like autoencoder to identify which example to exclude and consider them outliers.


#### Properties of good data

For the convenience of future reference, let me once again repeat the properties of good data:  
1. It contains enough information that can be used for modeling,  
2. It has good coverage of what you want to do with the model,  
3. It reflects real inputs that the model will see in production,  
4. It is as unbiased as possible,  
5. It is not a result of the model itself,  
6. It has consistent labels, and  
7. It is big enough to allow generalization.


#### Data Partitioning

"In practice, data analysts work with three distinct sets of examples: 
1. training set,
2. validation set
3. test set."

The training set is used by the machine learning algorithm to train the model.
The validation set is needed to find the best values for the hyperparameters of the machine  
learning pipeline
The test set is used for reporting: once you have your best model, you test its performance  
on the test set and report the results

**Splitting data into 3 sets**
The percentage of the split can also be dependent on the chosen machine learning algorithm  
or model. Deep learning models tend to significantly improve when exposed to more training  
data. This is less true for shallow algorithms and models.


#### Dealing with missing data

1. Remove data with missing attributes
2. Using algorithm to deal with missing data
3. Data Imputation


#### Data Imputation

Several techniques like:
a. Replace the missing value with average of the values present.
	Ex - missing height of students can be filled by avg of other student's height
b. Replace missing value with value outside of normal range
	Ex - replace missing values with -2 or 0 where normal range is [-1,1]. Here the model learns what to do with inputs with missing values
c. Using missing value as a target for a regression problem, and filling values

Same concept works while doing prediction


#### Data Augmentation

"To generate more labeled examples without additional labeling"

**For Images**
DA can be done by applying several techniques on input data like:
1. Flip (to ensure that flipping doesn't lose meaning of the object in question. Ex - a ball's image can be flipped vertically and horizontally, but not a car's)
2. Rotation
3. Crop
4. Color shift (changing RGB percentages slightly)
5. Noise addition (Gaussian noise to be introduced on same image in multiple amounts)
6. Perspective change
7. Information loss (Removing parts of image to help model learn about partial images or images with some obstacle in it)
8. Contrast change
9. Mix-Up (Instead of training on raw images, 2 images are taken in linear combination)

**For Text**
1. Replacing words in statements with their synonyms (Replace car with auto)
2. Replacing words in statements with their hypernyms (Replace car with vehicle)
3. Using word or document embeddings, via applying Gaussian Noise for random selection of embedding or via replacing with nearest neighbours.
4. Back translation


#### Imbalanced Data

"When the class imbalance is high, for example when 90% examples are of one class, and 10%  
are of the other, using the standard formulation of the learning algorithm that usually equally  
weights errors made in both classes may not be as effective and would need modification."

Ways to solve
1. Oversampling
	Algorithms that oversample the minority class by creating synthetic examples: SMOTE and ADASYN
2. Undersampling
	"remove from the training set some examples of the majority class, maybe based on some property (Ex - Tomek links)". 
	
	Also read: Cluster Based Undersampling
3. Hybrid of over and under sampling


#### Data Sampling

Generally can be distinguished between  - probability and non-probability sampling

Major probability sampling techniques:
1. Simple Random Sampling
	"each example from the entire dataset is chosen purely by chance"
	
	For example, if your entire dataset contains 1000 examples, tagged from 0 to 999, use groups of three digits from the random number generator to select an example. So, if the first three numbers from the random number generator were 0, 5, and 7, choose the example numbered 57, and so on.
2. Systematic/Interval Sampling
	For example, create a list containing all examples. From that list, you randomly select the first example xstart from the first k elements on the list. Then, you select every kth item on the list starting from x[start]. You choose such a value of k that will give you a sample of the desired size.

	Is inappropriate if the list of examples has periodicity or repetitive patterns.
3. Stratified Sampling
	First divide your dataset into groups (called strata) and then randomly select examples from each stratum, like in simple random sampling.

	If there's existence of several groups in data, then in sample, there should be examples from each of these groups.


#### Data Versioning
"Sometimes, after an update of the data, the new model performs worse, and you would like to investigate why by switching from one version of the data to another."

Also critical in supervised learning when the labeling is done by multiple labelers. 

**Levels In Data Versioning**

0. unversioned data, simple to read/write/store
1. versioned data, by storing as a snapshot at training time.
2. both data and code are versioned as 1 asset.


"Data First Algorithm Second" or in other words, spend most of your effort and time on getting more data of wide variety and high quality, instead of trying to squeeze the maximum out of a learning algorithm.


#### Feature Engineering (2)

"Machine learning algorithms can only apply to feature vectors"
"Feature engineering is a creative process where the analyst applies their imagination, intuition,  
and domain expertise"

FE for text data
1. One Hot Encoding
	Ex - Instead of using color's name in a column, use 1/0 model with 3 columns, assuming there are 3 different type of values for that field.
	red      = [1, 0, 0]  
	yellow = [0, 1, 0]  
	green  = [0, 0, 1]
	
2. Bag of Words or bag-of-words
	"Instead of representing one attribute as a binary vector, you use this technique to represent an entire text document as a binary vector"
	STEPS OF GENERATION
	1. For having number of documents, where each document is a sequence of words, we first tokenize each document. Such that:
		Document1: [Word1, Word2, Word3]
		Document2: [Word2, Word4, Word5] and so on
	2. Then we build a vocabulary from the words.
		Word1 Word2 Word3 Word4 Word5
	3. Ordering every word in the vocabulary and assiging an index
		
		| Word1 | Word2 | Word3 | Word4 | Word5 |
		| --- | --- | --- | --- | --- | 
		| 1 | 2 | 3 | 4 | 5 |
	4. Now every document can be referred via this table of indexes of tokens
		Document1: [1,1,1,0,0]
		Document2: [0,1,0,1,1]
		And so on..
		This is called a "binary model" in Bag of words.
	Other flavors include
		a. Count of tokens
		b. Frequency of tokens
		c. Term Frequency-Inverse Document Frequency
		d. Bag of n-grams
3. Other alternatives may include
	 - **Mean Encoding**
	 - Using odds ratio or log-odds ratio to have an statistic about the assocation between 2 variables, or between a categorial feature and a positive label.
		Example - existence of word "infected" in emails, which were either spam or not-spam.
	 - Working with *ordered* categorical features, like first, second and third. This can be replaced as : 1/3, 2/3 and 3/3 to maintain the relationship between the 3 values.
	 - Working with *cyclical* categorical features, like days of week, where Sunday comes after Saturday again. Instead of using one-hot encoding, using something like **sine-cosine transformation** makes more sense. 


**Feature Hashing**

A drawback while working with one-hot encoding and bag-of-words: having many unique values will create high-dimensional feature vectors.

Instead, a better approach can be to hash several features into a individual number/value to keep data manageable.
Ex - For converting statement "Love is a doing word" into a feature vector of dimension 5, a hash function can be used like MurmurHash3, Jenkins, CityHash, and MD5, generating outputs like similar to the mod function as below:

h(love) mod 5     = 0  
h(is) mod 5         = 3  
h(a) mod 5          = 1  
h(doing) mod 5  = 3  
h(word) mod 5   = 4

And having the final vector like: [1, 1, 0, 2, 1] (Why?)


**Topic Modelling**

"A family of techniques that uses unlabeled data, typically in the form of natural language text documents, where the model learns to represent a document as a vector of topics".

Example - A newspaper, which can be represented by: “sports”, “politics,” “entertainment”,  “finance” and “technology” can have a feature vector as:
[0.04, 0.5, 0.1, 0.3, 0.06] where politics (0.5) and finance (0.3) can have most weight.

Major algorithms: LSA and LDA




