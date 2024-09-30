# EURAC-workshop-2024
The repository contains the hand-one exercises used during the EURAC Trustworthy AI workshop
# pytest, mlflow and DVC exercises 

## setup
Let's create a new venv (from any directory)
```sh
$python3 -m venv ./venv_corso
$source ./venv_corso/bin/activate                        (Linux)
$.\venv_corso\Scripts\activate.bat                 (Windows cmd)
```
Requirements installation
```sh
$pip install -e .
```
Let's train our 'toy' model and add it to github
```sh
$python ./model/model_training.py
$git add ./model.pkl
$git commit -m 'added model'
$git push
```

## Pytest
The file *test/test_data_and_model.py* contains a pytest example. To run pytest we can use the command
```sh
$python -m pytest
```
pytest will automatically look for function with the word test in their name.

Let's try to write new tests to:
* Check that the model is not a majority classifier
* Make sure that the model precision and recall are above a certain value
<details> 
  <summary>Possibible solution</summary>

    def test_model_metrics(adult_test_dataset):
        x, y, data_path = adult_test_dataset
        clf = joblib.load('./model.pkl')
        predictions = clf.predict(x)
        metrics = classification_report(y, predictions, output_dict=True)
    
        assert len(np.unique(predictions)) > 1
        assert metrics['>50K']['precision'] > 0.7 #fill here
        assert metrics['>50K']['recall'] > 0.1 #fill here
</details>


## GitHub Action
Let's create an automated testing workflow. We need to create a '.github/workflow' folder within the repo main directory. Github will look for all the .yaml files included in the workflow folder

Let's create a file 'CI.yaml' containing the following content
```yaml

name: Test

on: [push]

jobs:
    build:
        runs-on: ubuntu-latest
        strategy:
            matrix:
                python-version: [3.7, 3.8]

        steps:
        - uses: actions/checkout@v2

        - name: Set up Python ${{ matrix.python-version }}
          uses: actions/setup-python@v2
          with:
            python-version: ${{ matrix.python-version }}

        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            pip install -e .

        - name: Pytest
          run: |
            pytest -v --maxfail=3 --cache-clear
```
Next time we push the repo we will see a gitub action starting on github

## mlflow
An mlflow experiment is contained in the experiments directory. To run it we can use the following command 
```sh
$python experiments/run_grid_search
```
We can explore all the past experiments using the following command
```sh
$mlflow ui
```
The experiments can be queried using a SQL like query
```
metrics.precision > 0.6 and params.depth='3'
```

## DVC initialization 
DVC can be initialized for the repo using the following command
```sh
dvc init
```
We can add an external remote by adding
```sh
dvc remote add -d myremote s3://<bucket>/<key>
```
where s3:// is our remote address, in this case s3.

dvc init will create a folder to be added to our git repo. This folder contains the DVC settings
```sh
git add .dvc/config .dvc/.gitignore .dvcignore
git commit -m "first DVC init"
git push
```

## Adding a first dataset

First we need to remove our dataset from the git repo
```sh
git rm data/raw_data.csv 
git commit -m "removing data from git"
git push
```
Followed by:
```sh
dvc add data/raw_data.csv
```
The command creates a fike raw_data.csv.dvc used by git to track the file within the repo
```sh
git add data/raw_data.csv.dvc
git commit -m "added first DVC data"
git push
```
## Creation of a machine learning pipeline with DVC
To create a DVC pipeline we need to create a file dvc.yaml within the main folder of the repo. This file should contain the following content
```yaml
stages:
  prepare:
    cmd: python3 data/data_preparation.py
    deps:
    - data/data_preparation.py
    - data/raw_data.csv
    params:
    - prepare.test_size
    - prepare.undersampling
    - prepare.val_size
    outs:
    - holdout.csv
    - train.csv
    - val.csv
  train:
    cmd: python3 model/model_training.py
    deps:
    - model/model_training.py
    - train.csv
    - val.csv
    params:
    - train.criterion
    - train.max_depth
    - train.min_sample_leaf
    - train.n_estimators
    outs:
    - metrics.json
    - model.pkl
  test:
    cmd: python3 -m pytest
    deps:
    - model.pkl    
    - holdout.csv
    - test/test_data_and_model.py
    metrics:
    - rocauc.json:
        cache: true
    outs:
    - prc.json

plots:
  - Precision-Recall:
      template: simple
      x: recall
      y:
        prc.json: precision
```
The pipeline can be executed using the following command
```sh
dvc repro
```
We can visualize the model metrics and plots using the following command
```sh
dvc metrics show
dvc plots show
```
