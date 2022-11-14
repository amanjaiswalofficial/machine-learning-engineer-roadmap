import mlflow
# creates a new experiment in the tracking server
mlflow.set_experiment('mlflow_experiments')
# starts and ends the mlflow experiment as a context manager
with mlflow.start_run():
    # logging some parameters
    mlflow.log_param("name","mlflow_test")
    mlflow.log_param("category","algorithm")
    mlflow.log_param("type","classification")
    # logging numeric values as metrics
    for i in range(5):
        mlflow.log_metric("i",i,step=i)
    # logging the entire file for traceability of model in future
    mlflow.log_artifact("mlflow_experiments.py")