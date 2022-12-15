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
