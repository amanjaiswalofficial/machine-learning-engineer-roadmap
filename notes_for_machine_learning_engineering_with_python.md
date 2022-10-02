Notes include lessons from:
[Machine Learning Engineering with Python](https://www.amazon.in/Machine-Learning-Engineering-Python-production-ebook/dp/B09CHHK2RJ) by [Andrew P. McMahon](https://www.amazon.in/s?i=digital-text&rh=p_27%3AAndrew+P.+McMahon&s=relevancerank&text=Andrew+P.+McMahon&ref=dp_byline_sr_ebooks_1)


### Text and their meanings
 - **Use-Case**
 - *Example*
 - `technical word`
 - ```python
    this = is_some_python_code()
    ```


### Machine Learning Development Cycle
4 major steps in any cycle:
1. Discover
2. Play
3. Develop
4. Deploy

#### Develop
To have a versioned model development, we use `mlflow`.  
Like, *Using mlfow as a wrapper on top of FBProphet models*
```python
class ProphetWrapper(mlflow...):
    def __init__(self, model):
        # model initialization code
    .
    .
    .
    def predict():
        # function to return prediction on future data
```

Using mlflow like tools gives functionalities like:
1. Logging performance metrics
2. Printing more information about the model runs
3. Then saving the model as:
   - pickle objects
   - joblib
   - JSON
   - etc..


#### Deploy

Typical deployments are done on:  
1. On-prem
2. Cloud using
    - IaaS
    - Paas

Also, using concepts of CI/CD like *Github Actions* are used.


### From Model to Model Factory
One important aspect is to design how the model will be trained and how will it provide the output.

2 ways to do this:
1. train-run process
![plot](./images/train.run.process.png)

2. train-persist process
![plot](./images/train.persist.process.png)

#### Identifying, when re-training is required

##### Drift
Drift is a term that covers a variety of reasons for your model's performance dropping
over time

##### Types of drift
1. Concept drift
   When there's some change in the relationships between the feature and the target predictions.
2. Data drift
   When there's a change in the statistical properties of the variables that are being used as features.

##### Detecting drift
*Using `alibi`*

General drift capture would look something like:
![plot](./images/drift.detection.with.training.png)
