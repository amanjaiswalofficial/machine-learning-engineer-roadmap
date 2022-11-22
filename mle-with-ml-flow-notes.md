### Machine Learning Engineering with MLFlow

#### Introduction to ML Flow

##### Built in modules in ML Flow
1. MLflow Tracking
2. MLflow projects
3. MLflow Models
4. MLflow Model Registry

##### Common types of problems solved by ML
1. Classification
2. Regression
3. Clustering
4. Generative model
5. Reinforcement learning

##### Components of a Data Science workbench
1. Tools/Frameworks
2. Experimentation Mgmt
3. Dependenct Mgmt
4. Deployment
5. Data Mgmt
6. Model Mgmt

##### Building Data Science workbench
1. Docker
2. Jupyterlab
3. MLflow
4. PostgreSQL

##### Components to manage models
1. Models
 - This module manages the format, library, and standards enforcement module on the platform. It supports a variety of the most used machine learning models: sklearn, XGBoost, TensorFlow, H20, fastai, and others. It has features to manage output and input schemas of models and to facilitate deployment.
2. Model Registry
 - This module handles a model life cycle, from registering and
tagging model metadata so it can be retrieved by relevant systems.

A MLflow model is bit like a Dockerfile for a model,
where you describe metadata of the model, and deployment tools upstream are able to
interact with the model based on the specification

##### Model flavors
1. mlflow.tensorflow
2. mlflow.h20
3. mlflow.spark