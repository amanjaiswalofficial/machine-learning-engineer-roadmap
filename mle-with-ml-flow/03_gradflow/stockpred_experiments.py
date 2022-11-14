# Part 1, creating a model and then saving it
import random
import mlflow
from mlflow.pyfunc.model import PythonModel

class RandomPredictor(PythonModel):
      def __init__(self):
          pass
      
      def predict(self, context, model_input):
        return model_input.apply(lambda column: random.randint(0,1))

model_path = "randomizer_model"
r = RandomPredictor()
mlflow.pyfunc.save_model(path=model_path, python_model=r)

# --------

# Part 2, using the same model and then logging its predictions, metrics and artifacts 
import pandas as pd
# loading the model as a python function
loaded_model = mlflow.pyfunc.load_model(model_path)

model_input = pd.DataFrame([range(10)])

random_predictor = RandomPredictor()

mlflow.set_experiment('stockpred_experiment_day5_up')
with mlflow.start_run():
    model_output = loaded_model.predict(model_input)

    mlflow.log_metric("Days Up",model_output.sum())
    mlflow.log_artifact("stockpred_experiments.py")