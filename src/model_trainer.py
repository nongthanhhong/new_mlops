import json
import mlflow
import time
import logging
import catboost 
import argparse
import numpy as np
import pandas as pd
from utils import *
import catboost as cb
import xgboost as xgb
from scipy.stats import entropy
from collections import Counter
from mlflow.models.signature import infer_signature
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from data_loader import train_data_loader
from problem_config import (ProblemConfig,
                            ProblemConst,
                            get_prob_config)


from utils import AppConfig

def load(**kwargs):
    return {k: load(**v) if isinstance(v, dict) else v for k, v in kwargs.items()}


class Models:
    def __init__(self, prob_config):
        self.EXPERIMENT_NAME = None
        self.config_folder = "./src/config_files/model_config"
        self.phase = prob_config.phase_id
        self.prob = prob_config.prob_id
        self.model = None
        self.params = None
        self.train = None
    
    def read_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return load(**config)
    
    def get_config(self, model_name):
        config_path = f"{self.config_folder}/{self.phase}/{self.prob}/{model_name}.json"
        config = self.read_config(config_path)
        self.EXPERIMENT_NAME = config["meta_data"]["model_name"]
        self.params = config["params"]
        self.train = config["train"]
        return config

    def xgb_classifier(self):
        config = self.get_config("xgb")
        self.model = xgb.XGBClassifier(**self.params)

    def catboost_classifier(self):
        config = self.get_config("catboost")
        self.model = catboost.CatBoostClassifier(**self.params)


def show_proportion(task, labels):
    counter = Counter(labels)
    logging.info(f"Data for {task}: ")
    for label in np.unique(labels):
        logging.info(f'label {label}: {counter[label]} - {100*counter[label]/sum(counter.values())}%')

def evaluate_model(model = None, dtest = None, test_x = None):

    probs_prediction = model.predict_proba(test_x)

    count = len(test_x[probs_prediction.max(axis=1) > 0.8])
    percentage = (count / len(probs_prediction)) * 100
    # avg_prob = np.mean(probs_prediction)

    start_time = time.time()
    predictions = model.predict(test_x)
    predict_time = round((time.time() - start_time) * 1000,2) #at ms
    
    # Calculate the ROC AUC score
    if len(np.unique(dtest.get_label()))>2:
        # Compute the ROC AUC score using the one-vs-one approach
        roc_auc = roc_auc_score(dtest.get_label(), probs_prediction, multi_class='ovo')

        # Compute the ROC AUC score using the one-vs-rest approach
        # roc_auc = roc_auc_score(dtest.get_label(), predictions, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(dtest.get_label(), predictions)
        
    log_loss_score = log_loss(dtest.get_label(), probs_prediction)

    acc_score = accuracy_score(dtest.get_label(), predictions)

    metrics = {"test_auc": roc_auc, "log_loss": log_loss_score, "test_acc": acc_score, "percent_prob": percentage, "predict_time_ms": predict_time/len(test_x)}
    
    logging.info(f"metrics: {metrics}")
    logging.info("\n" + classification_report(dtest.get_label(), predictions))
    logging.info("\n" + str(confusion_matrix(dtest.get_label(), predictions)))
    logging.info(f"\nPredict take {predict_time} ms for {len(test_x)} samples - AVG {predict_time/len(test_x)} ms each.")

    return metrics, predictions, probs_prediction


class ModelTrainer:

    @staticmethod
    def train_model(prob_config: ProblemConfig, add_captured_data=False):

        #define model
        class_model = Models(prob_config)
        class_model.catboost_classifier()

        logging.info("*****Start training phase*****")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{class_model.EXPERIMENT_NAME}"
        )

        logging.info("==============Load data==============")

        if add_captured_data:
            dtrain, dval, dtest, test_x, captured_x, train_x, train_y = train_data_loader(prob_config = prob_config, add_captured_data = add_captured_data)
        else:
            dtrain, dval, dtest, test_x = train_data_loader(prob_config = prob_config, add_captured_data = add_captured_data)

        print(f'Loaded {dtrain.shape[0]} Train samples, {dval.shape[0]} val samples , and {dtest.shape[0]} test samples!')
    
        show_proportion("training", dtrain.get_label())
        show_proportion("validating", dval.get_label())
        show_proportion("testing", dtest.get_label())
        
        logging.info("==============Training model==============")
        
        model = class_model.model
        model.fit(dtrain, 
                  eval_set=dval,
                  **class_model.train)
        
        logging.info("==============Testing model==============")

        metrics, predictions, probs_prediction = evaluate_model(model = model, dtest = dtest, test_x = test_x)

        logging.info("=================Retrain model with cost matrix======================")
        
        # calculate the confusion matrix
        cm = confusion_matrix(dtest.get_label(), predictions)

        # calculate the misclassification costs
        # create the cost matrix
        cost_matrix = np.zeros((cm.shape[0], cm.shape[1]))
        for i in range(cm.shape[0]):
            row_sum = sum(cm[i])
            for j in range(cm.shape[1]):
                if i != j:
                    cost_matrix[i][j] = cm[i][j] / row_sum

        weights_1 = np.mean(cost_matrix, axis=1)
        print(weights_1)

        classes = np.unique(dtrain.get_label())
        weights_2 = compute_class_weight(class_weight='balanced', classes=classes, y=dtrain.get_label())
        print(weights_2)

        weights = [x + y for x, y in zip(weights_1, weights_2)]
        
        # initialize the CatBoostClassifier with the cost matrix
        class_weights = dict(zip(classes, weights))
        
        key_to_exclude = 'auto_class_weights'

        new_params = {key: value for key, value in class_model.params.items() if key != key_to_exclude}
        model = cb.CatBoostClassifier(**new_params, class_weights=class_weights)

        # train the model
        model.fit(dtrain, 
                  eval_set=dval,
                  **class_model.train)
        
        logging.info("==============Testing new model==============")

        metrics, predictions, probs_prediction = evaluate_model(model = model, dtest = dtest, test_x = test_x)

        if add_captured_data:
            logging.info("==============Improve use Active learning==============")

            unlabeled_data = captured_x.copy()
            
            logging.disable(logging.CRITICAL)
            loop = 0

            while loop < 10 and len(unlabeled_data) != 0:
                probs_prediction = model.predict_proba(unlabeled_data)
                
                
                # Select the most informative unlabeled data points
                most_informative_data = unlabeled_data[(probs_prediction.max(axis=1) > 0.9) | (probs_prediction.min(axis=1) < 0.1)]
                most_informative_predictions = model.predict(most_informative_data)
                
                train_x = pd.concat([train_x, most_informative_data ])
                train_y = pd.concat([train_y, pd.DataFrame(most_informative_predictions) ])
                

                unlabeled_data.drop(index=most_informative_data.index, inplace=True)
                unlabeled_data.reset_index(drop=True, inplace=True)
                
                unlabeled_data.info()

                dtrain = cb.Pool(data=train_x, label=train_y)
                # model = class_model.model

                model.fit(dtrain, 
                  eval_set=dval,
                  **class_model.train)

            logging.disable(logging.NOTSET)   

            metrics, predictions, probs_prediction = evaluate_model(model = model, dtest = dtest, test_x = test_x) 

        logging.info("==============Improve use CalibratedClassifierCV model==============")

        # Assume that `X_test` and `y_test` are the test data and labels
        # Assume that `clf` is a trained binary classifier with a `predict_proba` method
        
        print(f"Previous log loss: {metrics['log_loss']:.3f}")

        # Calibrate the predicted probabilities using isotonic regression
        calibrator = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        calibrator.fit(test_x, dtest.get_label())
        
        model = calibrator
        
        metrics, predictions, probs_prediction = evaluate_model(model = model, dtest = dtest, test_x = test_x)
        
        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
        )
        mlflow.end_run()
        logging.info("finish train model")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)

    ModelTrainer.train_model(
        prob_config, add_captured_data=args.add_captured_data
    )
