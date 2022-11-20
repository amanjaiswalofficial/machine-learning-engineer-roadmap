import pandas
import numpy as np
import mlflow
import tensorflow
from tensorflow import keras
import mlflow.keras
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.model_selection import train_test_split


pandas_df = pandas.read_csv("training_data.csv")
X=pandas_df.iloc[:,:-1]
Y=pandas_df.iloc[:,-1]
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.33, random_state=4284, stratify=Y)
    
mlflow.set_experiment("Baseline_Predictions_Mlflow_Check")
mlflow.tensorflow.autolog()


model = keras.Sequential([
  keras.layers.Dense(
    units=36,
    activation='relu',
    input_shape=(X_train.shape[-1],)
  ),
  keras.layers.BatchNormalization(),
  keras.layers.Dense(units=1, activation='sigmoid'),
])

model.compile(
  optimizer=keras.optimizers.Adam(lr=0.001),
  loss="binary_crossentropy",
  metrics="Accuracy"
)

with mlflow.start_run(run_name='keras_model_baseline') as run:
    model.fit(X_train, y_train, epochs=20, validation_split=0.05, shuffle=True,verbose=True)
    

    