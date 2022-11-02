import numpy as np
import datetime
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import mlflow.sklearn

def acquire_training_data():
    start = datetime.datetime(2022,9,1)
    end = datetime.datetime(2023,11, 1)
    df = web.DataReader("BTC-USD", "yahoo", start, end)
    return df

def digitize(n):
    if n > 0:
        return 1
    else:
        return 0
    
def rolling_window(a, window_len):
    """
    Ex - inp: [1,2,3,4,5,6], 4
         op: [[1,2,3,4], [2,3,4,5], [3,4,5,6]]
    """
    shape = a.shape[:-1] + (a.shape[-1] - window_len + 1)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def prepare_training_data(data):
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda d: digitize(d))
    return data

if __name__ == "__main__":
    with mlflow.start_run():
        training_data = acquire_training_data()
        prepared_training_data_df = \
            prepare_training_data(training_data)
        btc_mat = prepared_training_data_df.as_natrix()
        
        WINDOW_SIZE = 14
        X = rolling_window(btc_mat[:,7], WINDOW_SIZE[:-1])
        Y = prepared_training_data_df["to_predict"].as_matrix()[WINDOW_SIZE]
        X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=0.25, random_state=4284, stratify=Y)
        clf = \
            RandomForestClassifier(bootstrap=True,
                                   criterion='gini', 
                                   min_samples_split=2, 
                                   min_weight_fraction_leaf=0.0, 
                                   n_estimators=50, 
                                   random_state=4284,
                                   verbose=0)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        
        # to save the model upon training
        mlflow.sklearn.log_model(clf, "model_random_forest") 
        
        mlflow.log_metric("precision_label_0", precision_score(y_test, predicted, pos_label=0))
        
        mlflow.log_metric("recall_label_0", recall_score(y_test, predicted, pos_label=0))
        
        mlflow.log_metric("f1_score_label_0", f1_score(y_test, predicted, pos_label=0))
        
        mlflow.log_metric("precision_label_1", precision_score(y_test, predicted, pos_label=1))
        
        mlflow.log_metric("recall_label_1", recall_score(y_test, predicted, pos_label=1))
        
        mlflow.log_metric("f1_score_label_1", f1_score(y_test, predicted, pos_label=1))
        
        
          
    
    