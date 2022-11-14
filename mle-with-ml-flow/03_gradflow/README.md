### Setting up mlflow with Postgres as tracking server

Reference: [Setup MLFlow in production](https://pedro-munoz.tech/how-to-setup-mlflow-in-production/)

1. Run (USERNAME=postgres, PASSWORD=postgres)
   ```shell
    docker-compose up -d
   ```
2. Install psql and other dependencies
   ```shell
    sudo apt-get install postgresql-contrib postgresql-server-dev-all
   ```
3. Run psql
    ```shell
    psql postgresql://<USERNAME>:<PASSWORD>@localhost:5432/postgres
    ```
4. Run following commands in psql
    ```shell
    CREATE DATABASE mlflow;
    CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow';
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
    ```
5. Install dependencies in same environment as mlflow
    ```shell
    pip install psycopg2
    ```
6. Run mlflow from same env or from a docker-compose (here mlflow:mlflow is the USERNAME:PASSWORD for the mlflow user)
    ```shell
    mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow --default-artifact-root file://<YOUR_PATH_TO_MLRUNS_FOLDER> -h 0.0.0.0 -p 8000 --gunicorn-opts "--timeout 180"
    ```
7. Check on `localhost:8000` on browser

#### Running code

1. Set env variable (so that mlflow, while running can use the DB for storage, instead of local filesystem)
    ```shell
    export MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@localhost/mlflow
    ```
2. Run `python mlflow_expriments.py`
3. Visit `localhost:8000` to see a new experiment `mlflow_experiment`
4. Run `python stockpred_experiments.py`
5. Visit `localhost:8000` to see a new experiment `mlflow_experiment` with metric DaysUp as a prediction value from the model
6. To create more such runs, if needed, just run the code for 'Part 2' multiple times, for multiple prediction metrics
7. This is how we create experiment(s) and their metrics for loaded models (Part 2).