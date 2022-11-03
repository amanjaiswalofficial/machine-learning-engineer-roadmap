Steps to run
1. Setup environment with python3.8
2. Install requirements.txt
3. Install pyenv using [this](https://brain2life.hashnode.dev/how-to-install-pyenv-python-version-manager-on-ubuntu-2004)
4. Run
```shell
mlflow run .
```
5. Get UI with
```shell
mlflow ui
```
6. Serving prepared model (path: mlruns/0/UUID_HERE/artifacts/model_random_forest/)
```shell
mlflow models serve -m ./mlruns/0/acf78d1f8cb548f1af1c6a890964ffa8/artifacts/model_random_forest/
```
7. Testing deployed model with URL
```shell
 curl http://127.0.0.1:5000/invocations -H 'Content-Type:
application/json' -d '{"data":[[1,1,1,1,0,1,1,1,0,1,1,1,0,0]]
}' 
```