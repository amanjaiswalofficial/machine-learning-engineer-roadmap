FROM continuumio/miniconda3:4.5.11


RUN pip install numpy==1.14.3 \
    && pip install scipy \
    && pip install pandas==0.22.0 \
    && pip install scikit-learn==0.20.4 \
    && pip install cloudpickle \
    && pip install pandas_datareader>=0.8.0

RUN pip install mlflow==1.9.0