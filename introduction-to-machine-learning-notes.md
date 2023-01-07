### Supervised Machine Learning Algorithms
1. K-Nearest Neighbours
2. Linear Models
3. Naive Bayes Classifiers
4. Decision Trees
5. Random Forests
6. SVM


"""
the larger variety of data points your data‐set contains, the more complex a model you can use without overfitting.
"""

#### K-Nearest Neighbours
 - To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset.
 - Using no. of neigbor properties to make a simple>>>>complex model
 - The more no. of neighbours, the more complex model
 - To find out sweet spot between overfitting and underfitting no. of neighbors
 - Strength, is very easy to understand
 - Weakness, large number of dataset, may lead to slow prediction. 
 - Parameters, two important parameters, number of neighbors and how you measure distance between data points.

#### Linear Models
##### Linear Regression
 - has no parameters,which is a benefit, but it also has no way to control model complexity.
 - The “slope” parameters (w), also called weights or coefficients, are stored in the coef_ attribute, while the offset or intercept (b) is stored in the intercept_ attribute.
 - An alternative, which allows us to control complexity, and in-turn, control how much overfitting can exist in the model is Ridge Regression.

##### Ridge Regression
 - uses Regularization, i.e. explicitly restricting a model to avoid overfitting 
 - Ridge uses R2
 - The Ridge model makes a trade-off between the simplicity of the model (near-zero coefficients) and its performance on the training set.
 - Parameters, Alpha

##### Lasso Regression
 - Similar to Ridge, except it uses R1 regularization.
 - Here, some cofficients are exactly zero, so, causing some features to be ignored entirely by model.
 - Parameters, Alpha and max_iter

#### Linear Models for classification
##### Logistic Regression
 - Uses L2 by default.
 - Parameter, penalty, to decide what regularization to use
 - Parameter, C. The lower the value of C, the more algorithm will try to adjust to majority of data points and vicecersa.
##### Linear Support Vector Machines
 - Parameter, C. The lower the value of C, the more algorithm will try to adjust to majority of data points and vicecersa.

#### Linear models for multiclass classification
 - Common technique, one-vs.-rest approach. Here a binary model for each class vs all other classes is learned. While prediction, all binary classifiers run on point and one with highest score is preferred as result.

##### Strengths, Weakness, Parameters
 - Parameter, Alpha in regression models andc in linearSVC/logistic regression.
 - Parameter, whether to use L1 Regularization or L2 Regularization

#### Naive Bayes Classifiers
 - Faster than linear models
 - More generalized models than linear clasifiers, hence not as good performance.
 - Fast as they learn parameters by looking at each feature individually and collect simple per-class statistics from each feature.
 - Types like GaussianNB, BernoulliNB, MultinomialNB
 - Are great baseline models
  
 - Parameter, for Multi & Bernoully, take alpha for model complexity


#### Decision Trees
 - Learn a hierarchy of if/else question
 - Learning a decision tree means learning the sequence of if/else questions that gets us to the true answer most quickly.
 - Splitting the data based on tests (whether a point/value belongs to left [less than] or right [more than]  of something)
 - Example: X[1] <= 0.5 will draw a line, with let's say 32 points left [below the line] and 48 points right [above the line].
 - The process is repeated to keep drawing more such lines.
 - The recursive partitioning of the data is repeated until each region in the partition (each leaf in the decision tree) only contains a single target value.

#### Decision Tree for Regression
 - To make a prediction, we traverse the tree based on the tests in each node and find the leaf the new data point falls into.
 - Common to have overfitting. To prevent, meaning to stop the creation of tree early [called pre-pruning].
 - Another approach to prevent overfitting, is to build the tree then remove nodes that contain little information [called post-pruning].
 - Parameter, to control pre-pruning, is max_depth, to limit depth of tree. Other parameter for such concept include max_leaf_nodes and min_samples_leaf.
 - Feature Importance, summary to understand working of tree instead of looking at tree to its full depth.

#### Ensemble of decision trees: Random Forests
 - To overcome the drawback faced by single decision trees, i.e., overfitting.
 - To build many decision trees, and by injecting randomness in each of them to ensure they are different.
 - This is achieved by selecting the data points for built trees randomly, or by selecting the features in each split test.
 - For regression use-cases, averaging the results is considered, whereas for classification, soft voting strategy is used.
 - Parameter, n_jobs, to set no. of cores to be used
 - Parameter, max_depth
 - Paramet, max_features, the less the lesser the overfitting and vice-versa. Rule of thumb, sqrt(no of features)
 - Random forest can't perform well on high dimension data. Require more memory, slow to train.

#### Ensemble of decision trees: Gradient Boosted Trees
 - Compared to Random forests, this works by building trees in a serial man‐
ner, where each tree tries to correct the mistakes of the previous one.
 - Instead of randomization, by default, it uses pre-pruning to reduce overfitting.
 - Faster in terms of making predictions.
 - The main idea is to combine as many weak learners/simple models to iteratively improve performance.
 - Parameters include no. of trees (n_estimators) and pre-pruning and learning rate, which controls how strongly each tree will try to fix mistakes made by previous one.
 - Mostly used, XGBoost for this
 - Parameter, max_depth, to reduce complexity of each tree.
 - Way to decrease complexity, include applying stronger pre-pruning or limiting maximum depth.

