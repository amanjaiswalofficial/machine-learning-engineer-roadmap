import pandas
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


pandas_df = pandas.read_csv("training_data.csv")
X=pandas_df.iloc[:,:-1]
Y=pandas_df.iloc[:,-1]
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.33, random_state=4284, stratify=Y)

"""
Prerequisite:
Set MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@localhost/mlflow in terminal
Open mlflow UI, create a new experiment
"""

# same name as experiment created on UI
mlflow.set_experiment("test_experiment_01")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="logistic regression_model_baseline") as run:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    y_preds = np.where(preds>0.5, 1, 0)
    f1 = f1_score(y_test, y_preds)
    mlflow.log_metric(key="f1_experiment_score", 
                      value=f1)
    
print(f1)