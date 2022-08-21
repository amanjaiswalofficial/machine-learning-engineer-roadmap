Notes include lessons from:
1. [Machine Learning Engineering](http://www.mlebook.com/wiki/doku.php) by [Andriy Burkov](https://www.linkedin.com/in/andriyburkov/ "https://www.linkedin.com/in/andriyburkov/")


### Common terminologies

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


### Machine Learning Project lifecycle
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


### Feature Engineering (2)

"Machine learning algorithms can only apply to feature vectors"
"Feature engineering is a creative process where the analyst applies their imagination, intuition,  
and domain expertise"

#### Feature Engineering for text data
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


#### Feature Hashing

A drawback while working with one-hot encoding and bag-of-words: having many unique values will create high-dimensional feature vectors.

Instead, a better approach can be to hash several features into a individual number/value to keep data manageable.
Ex - For converting statement "Love is a doing word" into a feature vector of dimension 5, a hash function can be used like MurmurHash3, Jenkins, CityHash, and MD5, generating outputs like similar to the mod function as below:

h(love) mod 5     = 0  
h(is) mod 5         = 3  
h(a) mod 5          = 1  
h(doing) mod 5  = 3  
h(word) mod 5   = 4

And having the final vector like: [1, 1, 0, 2, 1] (Why?)


#### Topic Modelling

"A family of techniques that uses unlabeled data, typically in the form of natural language text documents, where the model learns to represent a document as a vector of topics".

Example - A newspaper, which can be represented by: “sports”, “politics,” “entertainment”,  “finance” and “technology” can have a feature vector as: [0.04, 0.5, 0.1, 0.3, 0.06] where politics (0.5) and finance (0.3) can have most weight.

Major algorithms: LSA and LDA


#### Features for time-series

Compared to other datasets, time-series data is usually orderedd by time.

Instead of having multiple records for same time grain, its recommended to use something like count/average etc of those values to have only 1 record represent that time grain.

Such data is used for purposes like:
1. Predicting next observation
2. Identifying the patterns in existing observations.


Identifying Feature Engineering possibilites based on data analysis

Examples:
1. In receiving spam mails, if many of them are sent on Monday, then having a feature like: "sent-on-monday" in the data.
2. In receiving spam mails, if many of them contain multiple emojis, then having a feature like: "no-of-smileys"
And so on.


#### Stacking features

Example - for a problem "movie title classification in tweets", considering:
a. Five words before title are the left context.
b. Then there's the title.
c. Five words after title are the right context.

Stacking can be done by dealing with all 3 of above mentioned parts of the tweet and then combining them together in some order (can be a. b. and c., can be b. c. and a. and so on, only needs to ensure that all examples are combined in similar fashion).

The approach:
1. Take all left contexts from tweets
	1. Apply bag-of-words to convert each left context into a feature vector.
2. Take title/extractions:
	1. Apply bag-of-words to convert each title into a feature vector.
3. Take all right context from tweets
	1. Apply bag-of-words to convert each right context into a feature vector.
4. Combine all the feature vectors, similar to something like:
	[0,1,0,0....]  [0,1,1,0....]  [1,1,0,0....]
5. Ensure all follow the same order in combining


Along with this, stacking individual features can also be done. In cases where high enough predictive power is required rather having it more time-efficient.

Example - after having the feature-vector for a tweet with cinema, an additional feature can be to check whether: "the tweet has the topic as cinema or not". The feature can be from a classifier. If the topic predicted is cinema, then 1 else 0. Similarily, more features like:

1. IMDB score
2. Rotten tomatoes score etc

Can be added along with original feature vectors


#### Properties of good features
1. High predictive power: Features that contribute directly to the lets say, prediction of something should be more focused than features which don't. Example - person's weight and health in predicting if he/she has cancer, and not his car type or car color.
2. Fast computability: It should be possible to compute a feature fast. 
3. Reliable
4. Uncorelatedness
5. Distribution of feature in training and production data should be as closely matched as possible.
6. Should be unitary, as much as possible. Ex - length of car, weight of car, and not length divided by weight of car.


#### Feature Selection
"If we could estimate the importance of features, we would keep only the most important ones. That would allow us to save time, fit more examples in memory, and improve the model’s  
quality"
"Identifying which features directly impact the predictions as well as have enough values to impact the model training."

Ways to identify such features:
1. Cutting long tail: A feature which contains information, only for a handful of examples. A long tail of a distribution is such a part of that distribution that contains elements with substantially lower counts compared to a smaller group of elements with the highest counts.
	1. For such a case, its important to identify the threshold for defining a feature as a long tail.
	2. Can also use a hyperparameter for the problem to discover optimal value for this threshold.
	3. Can consider distribution of counts as well, to identify the threshold.
2. Boruta: Boruta iteratively trains random forest models and runs statistical tests to identify features as important and unimportant.
3. L1 Regularization: L1 regularization produces a sparse model, which is a model that has most of its parameters equal to zero. Therefore, L1 implicitly performs feature selection by deciding  which features are essential for prediction, and which ones are not.
4. Task specific feature selection: Example - To Remove some features from bag-of-words vectors representing natural language texts by excluding the dimensions corresponding to stop words, where stop words refer to words that are too generic for the problem we're trying to solve.


#### Regularization
"Regularization is an umbrella term for a range of techniques that improve the **generalization** of the model. Generalization, in turn, is the model’s ability to correctly predict the  
label for unseen examples"


#### Feature Synthesizing
"Sometimes it may be useful to convert numerical features into categorical ones"


#### Feature Discretization
"If some feature selection technique applies only to categorical features."
"discretization can add useful information to the learning algorithm when the training dataset is relatively small"

#### Binning / Bucketing
"Techniques to convert numerical feature into a categorical one by replacing numerical values in a specific range by a constant categorical value."

**Types of Binning**

1. Uniform Binning
	All bins for a feature will usually have similar lengths. Also, once the model is deployed in production, if the value of the feature in the input feature vector is below or above the range of any bin, then the closest bin is assigned, which is either the leftmost or the rightmost bin.
2. K-Means Binning
	Values in each bin belong to the nearest one-dimensional k-means cluster.
3. Quantile Based Binning
	All bins have the same number of examples

#### Synthesizing features from Relational Data

"A typical approach is to compute various statistics from the data coming from multiple rows and use the value of each statistic as a feature."

For example - calculating values like mean and standard deviation. on multiple row values for a certain property. 

Why? If you want to increase the predictive power of your feature vectors, or when your training set is rather small, you can synthesize additional features that would help in predictions.

#### Synthesizing features from data

"Using algorithms like k-means clustering"


#### Synthesizing features from other features

"Using neural networks"

In practice, the most common way to obtain new features from the existing features is to apply a simple transformation to one or a pair of existing features. Different type of transformations include:

1. Discretization of the feature
2. Squaring the feature
3. Calculating mean and standard deviation of feature j from k-nearest neighbors.
4. For pair of numerical features, simple arithmetic operators can be used as transformation.


#### Learning features from data

Technique include:

1. Word Embeddings

	"Word embeddings are feature vectors that represent words"
	
	"Word embeddings are learned from large corpora of text documents"
	
	Once you have a collection of word embeddings for some language, you can use them to represent individual words in sentences or documents written in that language, instead of using one-hot encoding.
	
	Some algorithms that can be used for this: word2vec, skip-gram, fastText

2. Document Embeddings

	Similar to a word, document embeddings can be generated and used as a feature.
	
	Some algorithm than can be used for this: doc2vec

3. Embeddings of Anything

	A general approach to train embeddings of any type:
	
	• What supervised learning problem to solve (for images, usually object classification),  
	• How to represent the input for the neural network (for images, matrices of pixels, one  
	per channel)
	• What will be the architecture of the neural network before the fully connected layers  
	(for images, usually a deep CNN)


#### Dimensionality Reduction

"Sometimes, it might be necessary to reduce the dimensionality of examples."

"Often results in increased learning speed and better generalization"

Dimensionality reduction vs Feature selection: we analyze the properties of all existing features and remove those that, in our opinion, do not contribute much to the quality of the model. When we apply a dimensionality reduction technique to a dataset, we 
"replace all features in the original feature vector with a new vector, of lower dimensionality" 
and of synthetic features.

Some popular techniques:
1. Principal Component Analysis (PCA)
	a. The fastest
	b. Drawback: all data must fit into memory for PCA to work
	c. Variant: Incremental PCA,  which works with batches of data.
	d. The algorithm produces D so-called principal components, where D is the dimensionality of data.

	Usage: Using PCA as a step before model training to find the optimal value of the reduced dimensionality experimentally

2. In case of Visualization, where the goal is to produce 2D or 3D feature vectors, we can use techniques like:
	1. UMAP, which requires all data in memory
	2. autoencoder, which can run in batch


#### Scaling Features or feature scaling

Refers to:
"is bringing all your features to the same, or very similar, ranges of values or distributions"

"Can also increase the training speed of deep neural networks"

"Reduces the risk of numerical overflow"

Techniques include:
1. **Normalization**
	"process of converting an actual range of values, which a numerical feature can take, into a predefined and artificial range of values, typically in the interval [-1, 1] or [0, 1]."

	Possible drawback include: the min and max for calculating the range of values formula can be outliers. In which case, clipping can be used, which tries to choose the min and max to be values which aren't outliers.

2. **Standardization**

	"procedure during which the feature values are rescaled so that they have the properties of a standard normal distribution"


When to use what?

In theory, normalization would work better for uniformly distributed data, while standardization tends to work best for normally distributed data

Feature scaling is usually beneficial to most learning algorithms.



#### Features - storage and documentation

"It’s advised to design a schema file that provides a description of the features’ expected properties"

**Schema File** is a document that describes features. It is updated each time some changes are made to features in name of versioning. It usually contains:
i. name
ii. its type
iii. min and max values
iv. sample mean and variance
v. if zeroes are allowed
vi. if undefined values are allowed
vii. fraction of examples that can have this feature present (popularity)


#### Feature Store
A vault for storing documented, curated and access-controlled features within an organization. Each feature usually has:
1. Name
2. Description
3. Metadata
4. Definition


#### Feature Engineering Best Practices
1. Generate simple features in the beginning
	"A feature is simple when it doesn’t take significant time to code"
2. Reuse legacy systems
	"When replacing an old, non-machine-learning-based algorithm with a statistical model, use the output of the old algorithm as a feature for the new model."
3. Using IDs as features when needed, but reduce the cardinality when possible
	"Example - Using country name instead of zip code if every zip space is not what's bein"
	
4. Other ways to reduce cardinality
	Use feature hashing
	Group similar values
	Group long tail
	Remove the feature if all or almost all values are unique
	Verifying usage of count actually gives value as a feature
	
"Once the model is deployed in the production environment, and each time it is loaded, you must rerun feature extractor tests."

"The feature extractor has to throw an exception and die if any resource (API or database that the feature consumes) during feature extraction is unavailable"

"Avoid silent failures that may remain unnoticed for a long time with model performance degrading or becoming completely wrong."

"The version of the feature extraction code must be in sync with the model’s version and the data used to build it. All 3 should be deployed and rolled back together"

"Isolate Feature Extraction Code"

"Log the feature values extracted in production for a random sample of online examples. When  
you work on a new version of the model, these values will be useful to control the quality of the training data"


### Model Training

Points to consider before training the model:
1. Validating schema conformity.
	Ensure the schema in the data matches the schema file, previously created.
2. Setting up basic performance levels.
	Example - If input vector has high number of signals, we can expect near-zero errors.
3. Choosing a single performance metric
4. Choosing a right baseline
	"A baseline is a model or an algorithm that provides a reference point for comparison. It gets an input, and outputs a prediction."
5. Splitting data into 3 sets
	i. Validation and test sets must come from same statistical distribution, and should have mostly similar properties.
	ii. "Draw validation and test data from a distribution that looks much like the data you expect to observe once the model is deployed in production, which can be different from the distribution of the training data."
6. Convert examples into numerical feature vectors.
7. Engineer features and filled missed values using only the training data.


#### Common baseline algorithms
1. Random prediction algorithm
	"makes a prediction by randomly choosing a label from the collection of labels assigned to the training examples."

	For example - "In the regression problem it means selecting from all unique target values in the training data"
2. Zero rule algorithm
	zero rule algorithm strategy is to always predict the class most common in the training set, independently of the input value.


#### Distribution Shift (to read)


#### Algorithm Spot Checking

"Shortlisting candidate learning algorithms for a given problem"

Points to consider:
1. Select algorithms based on different principles, such as  
	instance-based algorithms
	kernel-based
	shallow learning
	deep learning
	ensembles
2. Trying each algorithm with 3 - 5 different values of the most sensitive hyperparameters
3. If the learning algorithm is not deterministic (such as the learning algorithms for neural  networks and random forests), run several experiments, and then average the results;
4. Use the same training/validation split for all experiments.

#### Building a pipeline
A sample pipeline can be:
1. Tokenizaton
2. Feature Extraction
3. Feature Selection
4. Feature Normalization
5. Model Training

"During the scoring, the input example passes through the entire pipeline and “becomes” an output"


#### Performance Metrics (2)

If model performs well on new/holdout data, we assume that it generalizes well. For this, a performance metric needs to be measured.

For Regression:

1. MSE / mean squared error
2. MAE / mean absolute error
3. ACPER / almost correct predictions error rate

For Classification:

1. Precision
2. Accuracy
3. Cost Sensitive Accuracy
4. Area under ROC


Before, Precision & Accuracy, must know Confusion Matrix


#### Confusion Matrix
||spam|not_spam|
|---|---|---|
|spam (actual)|23 (TP)| 1 (FN)|
|not_spam (actual)| 12 (FP) | 556(TN)


True Positive - Here 23 is what **were** **spam** and were **identified as spam**

False Negtive - Here 1 is what **was spam** but **wasn't identified as spam**

False Positive - Here 12 is what **were not spam** but **were identified as spam**

True Negative - Here 556 is what **were not spam** and were **not identified as spam**

#### Precision

Ratio of TP with total number of P i.e. TP and FP
TP / (TP+FP)

#### Recall

Ratio of TP with total number of actual positive examples i.e. TP and  FN
TP / (TP+FN)

Usually there's a precision-recall tradeoff, and have to settle for 1 depending on use-case.

Many ways to do this
Example - To increase precision (at the cost of a lower recall), we can decide that the prediction will be positive only if the score returned by the model is higher than 0.9.

In case of **multiclass classification**, First select a class for which you want to assess these metrics. Then you consider all examples of the selected class as positives and all examples of the remaining classes as negatives.

When comparing with other models or other performances of same model, **F-measure** or **F-score** can be used.


#### Accuracy

"the number of correctly classified examples, divided by the total number of classified examples"

In case of classification:

(TP+TN) / (TP+TN+FP+FN)


#### Cost-sensitive accuracy

Cases when different classes have different importance (i.e. TN and FP have different amount of relevance). Here:
1. Assign a cost (a positive number) to both types of mistakes.
2. Then after calculating values as usual, multiple values with their weights before calculating accuracy.

In cases of imbalanced dataset, a better metric than normal or cost-sensitive accuracy is per-class accuracy.

An overall better metric is **Cohen’s kappa statistic**

Also read: **ROC Curve**


Another metric for classification is
#### ROC Curve or receiver operating characteristic Curve

 - Uses TP rate and FP rate, to build summary picture of classicification performance.
 - TP rate = TP/(TP+FN), FP rate = FP/(FP+TN)
 - "The greater the area under the ROC curve (AUC), the better the classifier."
 - Are relatively simple to understand.


#### Hyperparameter Tuning
 - Some hyperparameters influence the speed of training.
 - Important hyperparameters control the two tradeoffs: bias-variance and precision-recall

Some areas where hyperparameter tuning can be used:

1. In data pre-processing, the hyperparameters could specify whether to use data-augmentation or using which technique to fill missing values.
2. In feature engineering, a hyperparameter could define which feature selection technique to apply.
3. When making predictions with a model that returns a score, a hyperparameter could specify the decision threshold for each class.

Popular Hyperparamater Tuning Techniques

#### Grid Search

- Used when the number of hyperparameters and their range is not too large.
- For example - 
```python
pipe = Pipeline([('dim_reduction', PCA()), ('model_training', SVC())])
param_grid = dict(dim_reduction__n_components=[2, 5, 10], model_training__C=[0.1, 10, 100])  
grid_search = GridSearchCV(pipe, param_grid=param_grid)
pipe.predict(new_example)
```
 - Step by step evaluation consists of:
	 1. Configuring a pipeline with a pair of hyperparameter values,
	 2. Applying the pipeline to the training data and training a model
	 3. Computing the performance metric for the model on the validation data


Another technique for this is

#### Random Search

Compared to Grid Search, here we don't provide set of values to explore for each hyperparamater.

"Instead, you provide a statistical distribution for each hyperparameter from which values are randomly sampled"


Combination of both techniques, results into another technique called

#### Coarse-to-fine Search (to read)

Also, important

#### Bayesian Techniques (to read)


#### Cross Validation
 - preferred to use when the data set isn't too big
 - steps to perform cross validation (For example - **five-fold cross validation**):
	 0. Fix hyperparameter values to evaluate.
	 1. Take the dataset.
	 2. Split it into 5 folds.
	 3. Train 5 models
		 i. For first model, use F1 as test set, and F2, F3, F4 and F5 as training set.
		 ii. For second model, use F2 as test set, and F1, F3, F4 and F5 as training set.
		 iii. And so on.
	 4. Average the five values of the metric to get the final value
	 5. Can also combine grid or any other hyperparamater tuning technique
	 6. Finally, assess the final model using the test set.


#### Shallow Model Training Strategy
1. Define a performance metric P.  
2. Shortlist learning algorithms.  
3. Choose a hyperparameter tuning strategy T.  
4. Pick a learning algorithm A.  
5. Pick a combination H of hyperparameter values for algorithm A using strategy T.  
6. Use the training set and train a model M using algorithm A parametrized with  
hyperparameter values H.  
7. Use the validation set and calculate the value of metric P for model M.  
8. Decide:  
	a. If there are still untested hyperparameter values, pick another combination H of hyperparameter values using strategy T and go back to step 6.  
	b. Otherwise, pick a different learning algorithm A and go back to step 5, or proceed to step 9 if there are no more learning algorithms to try.  
9. Return the model for which the value of metric P is maximized.


#### Saving and restoring the model
Usually done by Pickle for serialization and deserialization of the trained model for storing, then restoring the m


#### bias-variance (2) tradeoff

The second type of tradeoff, the first one being precision-recall, while choosing the hyperparameters.

Model having 
#### high-bias / Underfitting

A model performs well if it has low bias.

Vice-versa, when the model makes too many mistakes on training data only, its said to have underfitted.

**Reasons**
 - the model is too simple for the data (for example linear models often underfit);  
 - the features are not informative enough;  
 - too much regularization

**Solutions**
 - trying a more complex model,  
 - engineering features with higher predictive power,  
 - adding more training data, when possible, and  
 - reducing regularization


Model having
#### high variance / Overfitting

The model that overfits usually predicts the training data labels very well, but works poorly on the holdout data.

**Reasons**
 - The model is too complex for the data. Very tall decision trees or a very deep neural  
network often overfit;  
 - there are too many features and few training examples; and  
 - not enough regularization

**Solutions**
 - using simpler model
 - adding regularization.
 - reducign dimensionality or training with more data.


To deal with variance and bias, and choosing optimal values for hyperparameters. An approach is
#### Regularization (2)

"To create a regularized model, we modify the objective function"

"Regularization adds a penalizing term whose value is higher when the model is more complex"

#### L1 Regularization & L2 Regularization

Also called Lasso & Ridge Regularization


Other regularization techniques involve:
#### Elastic Net Regularization


#### Deep Learning Model Training Strategy

1. Define a performance metric P.  
	
	"define a metric that would allow comparing the performance of two models on the holdout data. Ex - F-score or Cohen’s kappa."
2. Define the cost function C.  
	
	"what our learning algorithm will optimize in order to train a model. Ex - for regression model, cost function is the mean squared error. For classification, categorical cross-entropy (for multiclass classification) or binary cross-entropy (for binary and multi-label classification)."
3. Pick a parameter-initialization strategy W.  
	
	"Before the training starts, the parameter values in all units are unknown. We must initialize them with some values. For neural networks, can consider algorithms like gradient descent to figure out these values."

	Can be strategies like:
	i. ones
	ii. zeros
	iii. random normal etc
4. Pick a cost function optimization algorithm A.  

	Mostly use: gradient descent and stochastic gradient descent. 

	Other ways to use Regularization in the process, is to use neural network-specific  
	regularizers:
	i. dropout
	ii. early stopping
	iii. batch normalization
5. Choose a hyperparameter tuning strategy T.  
6. Pick a combination H of hyperparameter values using the tuning strategy T.  
	
7. Train model M, using algorithm A, parametrized with hyperparameters H, to optimize  
cost function C.  
8. If there are still untested hyperparameter values, pick another combination H of  
hyperparameter values using strategy T, and repeat step 7.  
9. Return the model for which the metric P was optimized


Flowchart of creating and using models via neural networks:
![[Pasted image 20220813144012.png]]

"you start with some model, and then increase its size until it fits the training  
data well. If needed, we retrain the model"

If required to increase size of model, due to bad performance. One can:
1. Increase the size of individual layers
2. Add another layer

If after all 8 steps, the performance of the best model is still  
not satisfactory, try a different network architecture, add more labeled data, or try **transfer  
learning**


#### Gradient Descent (3) (to read)

#### Stochastic Gradient Descent (2) (to read)

#### Local and Global Minimum (2) (w.r.t cost function)

#### Learning rate decay 
1. Time based
2. Step based
3. Exponential

#### Dropout
"Each time you “run” a training example through the network, you temporarily exclude random some units."

"The dropout hyperparameter varies in the range [0, 1] and characterizes the fraction of units to randomly exclude from computation."

Most effective

#### Early Stopping
"trains a neural network by saving the preliminary model after every epoch. Models saved after each epoch are called checkpoints"

"By keeping a version of the model after each epoch, you can stop the training once you start observing a decreased performance on the validation set"

#### Batch Normalization / Batch Standardization
"standardizing the outputs of each layer before the next layer receives them as input"

#### Handling multiple input and multiple output

Bit difficult to do for shallow models.
In case of neural networks, one can have several sub-networks, based on input types.
For Example - In case of having an image and a text as an input, a CNN can read the image and the RNN can read the text. Both can have an embedding as their last layer.

Similarily for multiple outputs...


#### Transfer Learning
"using a pre-trained model to build a new model."
"The parameters learned by the pre-trained models can be useful for your task."

a pretrained model can be used in the following ways:

1. learned parameters can be used to initialize your own model
2. can be used as a feature extractor for your model


#### Stacking models

"Ensemble learning is training an ensemble model, which is a combination of several base models, each individually performing worse than the ensemble model"

Types:

Algorithms like Random Forest Learning and Gradient Boosting, which train 100s-1000s of weak models to get 1 strong model. This works because when several uncorrelated models agree, they are more likely to agree on the correct outcome. 

For this the models need to be uncorrelated, i.e. should be obtained by different feature/have different nature.

Ex - 
1. Combination of SVM and Random Forest

How to combine such models for stacking:

1. Averaging
	"applying all your base models to the input x, and then averaging the predictions"
2. Majority Vote
	"applying all your base models to the input x, and then returning the majority class among all predictions."

	"In the case of a tie, you can either randomly pick one of the classes, or return an error message if misclassifying would incur a significant loss for the business"
3. Model Stacking
	"an ensemble learning method that trains a strong model by inputting the outputs of other strong models"	

	"If some of your base models return a class plus a class score, you can use those scores as  additional input features for the stacked model."

**Dealing with possible data leakage in Model stacking**

1. Follow a process similar to cross-validation, i.e. first, split all training data into ten or more blocks. The more blocks the better, but the process of training the model will be slower.
2. Temporarily exclude one block from the training data, and train the base models on the  remaining blocks. 
3. Then apply the base models to the examples in the excluded block. Obtain the predictions, and build the synthetic training examples for the excluded block by using the predictions from the base models.  
4. Repeat the same process for each of the remaining blocks, and you will end up with the training set for the stacking model. The new synthetic training set will be of the same size as that of the original training set.


#### Distribution Shift (2)

"When the distributions of the training data and test data are not the same, we call it distribution shift"

Different types:

1. Covariate Shift
	"shift in the values of features" 
2. Prior Probability Shift
	"shift in the values of the target"
3. Concept Drift
	"shift in the relationship between the features and the label"


An approach to deal with problems which have large number of training examples, and few test examples can be:
#### Adversarial Validation


#### Imbalanced Data (2) (in learning)

Similar to previous section, some of the ways to deal with imbalanced data while learning are:

1. Class Weighting
	Some algorithms allow providing weights for each class. By providing greater weight to the minority class, it becomes harder for the learning algorithm, to disregard examples of the minority class, because it would result in much higher cost than without class weighting.
2. Ensembling of resampled data
	Transform imbalanced binary learning problem by chunking the examples of the majority class into four subsets, while copying the minority class equal number of times with each chunk from majority class


#### Model Calibration

Several reasons for poor model behavior

In case of underfitting, i.e, performing poor on training data

1. the model architecture or learning algorithm are not expressive enough
	"choose advanced algorithms"
2. too much regularization
	"reduce regularization"
3. suboptimal values for hyperparameters
	"tune hyperparameters"
4. engineered features don’t have enough predictive power
	"add informative features"
5. don’t have enough data for the model to generalize
	"try getting more data, using data augmentation or transfer learning"

In case of overfitting, i.e. performing poor on test/holdout data

1. don’t have enough data for generalization
	"add more data or use data augmentation"
2. model is under-regularized
	"add regularization"
3. training data distribution is different from the holdout data distribution
	"reduce distribution shift"
4. suboptimal values for hyperparameters
	"tune hyperparameters"
5. features have low predictive power
	"add informative features"


#### Tips on Error Analysis

Techniques to fix an existing error pattern in model predictions
 - preprocessing the input (e.g. image background removal, text spelling correction);  
 - data augmentation (e.g., blurring or cropping of images);  
 - labeling more training examples; and  
 - engineering new features that would allow the learning algorithm to distinguish between  “hard” cases.

In a system made up of multiple models, working together to give a final output of, lets say, 73% accuracy, it can be decided which model to focus on to improve the overall results. Accordingly the techniques mentioned above can be put to use.


In case of deep learning, below approach can be used:
![[Pasted image 20220814112604.png]]


#### Best Practices

1. 
2. 
3. Avoid correction cascades


### Model Evaluation
Some reasons to perform model evalution once its deployed to production:
1. Study properties of distribution of prod. data compared to training data, to detect distribution shift, and accordingly retrain the model.
2. Evaluate performance of the model. 
3. Monitoring the performance of the model, "It is important to be able to detect this and, either upgrade the model by adding new data, or train an entirely different model".

#### Offline Evaluation and Online Evaluation
"An offline model evaluation happens when the model is being trained by the  
analyst"

"The offline model evaluation reflects how well  the analyst succeeded in finding the right features, learning algorithm, model, and values of hyperparameters. In other words, the offline model evaluation reflects how good the model is from an engineering standpoint."

"Online evaluation, on the other hand, focuses on measuring business outcomes, such as  
customer satisfaction, average online time, open rate, and click-through rate. Such information may not be reflected in historical data."


Some popular types of online model evalutation include:
1. A/B Testing


#### A/B Testing
2 different versions of a model, usually one being old and one being new are deployed and served to different groups of users at the same time.

So, for models mA and mB two groups A and B will exist such that, group A traffic is routed to the old model (mA), while group B traffic is routed to the new model (mB).

By comparing the performance of the two models, a decision is made about whether the new  
model performs better than the old model. The performance is compared using **statistical  
hypothesis testing**.

#### Statistical Hypothesis Testing
In short, it states 2 hypothesis:
1. Null Hypothesis, that the new model doesn’t change the average value of the business metric.
2. Alternative Hypothesis, that the new model changes the average value of the metric

A/B Testing has several types of tests, with each type of test asking different type of questions:
#### G-Test
 - For metric that counts the answer to a “yes” or “no” question.
 - Example: "Whether the user bought the recommended article, Whether the user has spent more than $50 during a month"
 - Follows chi-square distribution, meaning if mA and mB were equal, we expect G to be small.
 - p-value: the probability of observing more than a specific amount of G, which is generally expected to be small, under null hypothesis. (Ex - 5% or 0.05). 
 - Impact of having low or high p-value? (to read)
 - Also possible to test more than two models (e.g. models A, B, and C) and more than two possible answers to the question that define our metric (e.g., “yes,” “no,” “maybe”).

#### Z-Test
 - when the question for each user is, “How many?” or, “How much?”.
 - Example: "How much time a user has spent on the website during a session, How much money a user has spent during a month"
 - Calculated value Z denotes that the larger Z, the more likely the difference between A and B is significant.
 - If the p-value is above or equal to 0.05, then we do not reject the null hypothesis


One drawback A/B testing suffers is that in order to reach conclusion, the number of times a user went through one of the model is very high, i.e. A lot of users going through a worse model to find a better model. 

"A significant portion of users routed to a suboptimal model would experience suboptimal behavior for a long time."

Other than A/B testing, another technique is:

#### Multi Armed Bandit
In probability, for MAB, "a fixed and limited set of resources must be allocated between competing choices in a way that maximizes the expected reward. Each choice’s properties are only partially known at the time of allocation, and may become better understood as time passes and we allocate resources to the choice."

"UCB1 (for Upper Confidence Bound) is a popular algorithm for solving the multi-armed  
bandit problem. The algorithm dynamically chooses an arm, based on the performance of that  
arm in the past, and how much the algorithm knows about it"


#### Evaluating model performance: Statistical Bounds

For Example - A 95% statistical interval indicates that there’s a 95% chance  
the parameter you’re estimating is between the intervals bounds.

In case of classifications, If the error ratio “err” for a classification model. Then, with probability 99%, “err” lies in the interval [err - δ, err + δ]


#### Bootstrapping Statistical Interval (to read)


#### Evaluating Test Set Adequacy (to read)


#### Evalutation of model properties
Other than accuracy and/or AUC, some other aspects of model's output to look at in terms of evaluating a model's performance:
1. Robustness
	"If the input example is perturbed by adding random noise, the performance of the model would degrade proportionally to the level of noise."

	"If you have several models that perform similarly according to the performance metric, you  would prefer to deploy in production a model that is δ-robust, when applied to the test data,  with the smallest δ"
2. Fairness
	"Equal opportunity means each group gets a positive prediction from the model at equal  rates, assuming that people in this group qualify for it."

	"The attributes that are sensitive and need protection from unfairness are called protected or sensitive attributes, like age, skin color, gender, religion"

