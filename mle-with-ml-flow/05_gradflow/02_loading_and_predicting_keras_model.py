import mlflow
logged_model = '/data/artifacts/1/132e6fa332f2412d85f3cb9e6d6bc933/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(X_test))