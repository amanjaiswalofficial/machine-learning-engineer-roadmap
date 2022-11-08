import mlflow
import random

class RandomPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        pass
    
    def predict(self, context, model_input):
        return model_input.apply(lambda column: random.randint(0,1))

model_path = "random_model"
baseline_model = RandomPredictor()
mlflow.pyfunc.save_model(path=model_path, python_model=baseline_model)

