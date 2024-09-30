import mlflow
from itertools import product
import warnings
import argparse
import json
import yaml
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import json
import joblib
from datetime import datetime


def data_loader(path: str):
    data_csv_path = path
    dataset = pd.read_csv(data_csv_path)

    target_column = 'income'
    y = dataset[target_column]
    x = dataset.drop(target_column, axis=1)
    return x, y


def train_model(save_model = False):
    # This function trains a random folder classifier using the data specified by datapath
    # If parameters are not specified as argument look for params.json file, otherwise create default values
    # if parameters is None:
    #     if os.path.exists('./params.json'):
    #         parameters = json.load(open("params.json", "r"))
    #     else:
    #         parameters = dict(n_estimators=100, max_depth=4, criterion='gini',
    #                           min_sample_leaf=10)
    parameters = yaml.safe_load(open("params.yaml"))["train"]

    x_training, y_training = data_loader('./train.csv')
    x_val, y_val = data_loader('./val.csv')

    # Scikit learn ColumnTransformer used to process ordinal and nominal data
    ordinal_features = x_training.select_dtypes(include="number").columns
    categorical_features = x_training.select_dtypes(include="object").columns

    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    x_encoder = ColumnTransformer(transformers=[('num', numerical_transformer, ordinal_features),
                                                ('cat', categorical_transformer, categorical_features)])

    print(parameters)
    rf_clf = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                            max_depth=parameters['max_depth'],
                            random_state=42)

    rf_pipeline = Pipeline(steps=[("preprocessing", x_encoder), ("rf_model", rf_clf)])
    rf_pipeline.fit(x_training, y_training)

    pred_val = rf_pipeline.predict(x_val)
    metrics = classification_report(y_val,pred_val, output_dict=True)
    # serialize model using joblib
    joblib.dump(rf_pipeline, 'model.pkl')
    json.dump(metrics, open("metrics.json", "w"))

    if save_model is True:
        signature = infer_signature(x_val,
                                    rf_pipeline.predict(x_val))

        input_example = {}
        for i in x_val.columns:
            input_example[i] = x_val[i].iloc[0]

        mlflow.sklearn.save_model(rf_pipeline, path='./bestmodel'+str(datetime.now().day+datetime.now().hour+datetime.now().minute),signature=signature, input_example=input_example)


    return rf_pipeline

def grid_search(name_experiment):

    # this function runs a grid search over the hyper-parameters specified below
    max_depth = [3, 6]
    criterion = ['gini', 'entropy']
    min_samples_leaf = [5, 10]
    n_estimators = [50, 100]

    parameters = product(max_depth, criterion, min_samples_leaf, n_estimators)
    parameters_list = list(parameters)

    print('Number of experiments:', len(parameters_list))

    # Hyperparameter search
    results = []
    best_param = None
    best_f1 = 0.0
    warnings.filterwarnings('ignore')

    for i, param in enumerate(parameters_list):
        print('Running experiment number ', i)
        with mlflow.start_run(run_name=name_experiment):
            # Tell mlflow to log the following parameters for the experiments dashboard
            mlflow.log_param('depth', param[0])
            mlflow.log_param('criterion', param[1])
            mlflow.log_param('minsamplesleaf', param[2])
            mlflow.log_param('nestimators', param[3])

            try:
                parameters = dict(n_estimators=param[3],
                                  max_depth=param[0],
                                  criterion=param[1],
                                  min_sample_leaf=param[2])
                
                data = {
                    'prepare': {
                        'test_size': 0.2,
                        'val_size': 0.2,
                        'undersampling': 0.6
                    },
                    'train': parameters
                }

                file_path = 'params.yaml'

                with open(file_path, 'w') as yaml_file:
                    yaml.dump(data, yaml_file, default_flow_style=False)
                clf = train_model()

                with open('./metrics.json', 'r') as json_file:
                    metrics = json.load(json_file)

                # Tell mlflow to log the following metrics
                mlflow.log_metric("precision", metrics['>50K']['precision'])
                mlflow.log_metric("F1", metrics['>50K']['f1-score'])

                # Store this artifact for each run
                mlflow.log_artifact('./metrics.json')
                print("F1-score: ", metrics['>50K']['f1-score'])
                # save the best experiment yet (in terms of precision)
                if metrics['>50K']['f1-score'] > best_f1:
                    best_param = parameters
                    best_f1 = metrics['>50K']['f1-score']

                results.append([param, metrics['>50K']['f1-score']])

            except ValueError:
                print('bad parameter combination:', param)
                continue

    mlflow.end_run()
    print('Best F1 was:', best_f1)
    print('Using the following parameters')

    print(best_param)

    data = {
            'prepare': {
            'test_size': 0.2,
            'val_size': 0.2,
            'undersampling': 0.6
            },
            'train': best_param
            }

    file_path = 'params.yaml'

    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    clf = train_model(save_model=True)    

    return results, best_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="experiment_name")
    args, leftovers = parser.parse_known_args()

    results, best_param = grid_search(args.name)
    json.dump(best_param, open("best_params.json", "w"))

