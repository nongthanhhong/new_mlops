import json
import mlflow
import logging
import catboost 
import argparse
import numpy as np
import pandas as pd
from utils import *
import catboost as cb
import xgboost as xgb
from collections import Counter
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from data_loader import train_data_loader
from problem_config import (ProblemConfig,
                            ProblemConst,
                            get_prob_config)

from utils import AppConfig

def load(**kwargs):
    config = {}
    for k, v in kwargs.items():
      if type(v) == dict:
        v = load(**v)
      config[k] = v
    return config

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
            config = json.loads(f.read())

        return load(**config)

    def xgb_classifier(self):
        config =  self.read_config(self.config_folder+ "/" + self.phase + "/" + self.prob + "/xgb.json")
        print(config)
        self.EXPERIMENT_NAME = config["meta_data"]["model_name"]
        self.params = config["params"]
        self.train = config["train"]
        self.model = xgb.XGBClassifier(**self.params)

    def catboost_classifier(self):
        config =  self.read_config(self.config_folder+ "/" + self.phase + "/" + self.prob + "/catboost.json")
        self.EXPERIMENT_NAME = config["meta_data"]["model_name"]
        self.params = config["params"]
        self.train = config["train"]
        self.model = catboost.CatBoostClassifier(**self.params)

def show_proportion(task, labels):
    counter = Counter(labels)
    logging.info(f"Data for {task}: ")
    for label in np.unique(labels):
        logging.info(f'label {label}: {counter[label]} - {100*counter[label]/sum(counter.values())}%')

class ModelTrainer:

    @staticmethod
    def train_model(prob_config: ProblemConfig, add_captured_data=False):

        #define model
        class_model = Models(prob_config)
        class_model.catboost_classifier()

        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{class_model.EXPERIMENT_NAME}"
        )


        dtrain, dval, dtest, test_x = train_data_loader(prob_config = prob_config, add_captured_data = add_captured_data)

        print(f'Loaded {dtrain.shape[0]} Train samples, {dval.shape[0]} val samples , and {dtest.shape[0]} test samples!')
        
        logging.info("==============Training model==============")
    
        show_proportion("training", dtrain.get_label())

        show_proportion("validating", dval.get_label())
        
        model = class_model.model
        model.fit(dtrain, 
                  eval_set=dval,
                  **class_model.train)
        
        # evaluate

        logging.info("==============Testing model==============")
         
        show_proportion("testing", dtest.get_label())
        predictions = model.predict(dtest)
        
        if len(np.unique(dtest.get_label()))>2:
            # Compute the ROC AUC score using the one-vs-one approach
            auc_score = roc_auc_score(dtest.get_label(), model.predict_proba(dtest), multi_class='ovo')

            # Compute the ROC AUC score using the one-vs-rest approach
            # roc_auc = roc_auc_score(dtest.get_label(), predictions, multi_class='ovr')
        else:
            auc_score = roc_auc_score(dtest.get_label(), predictions)

        
        metrics = {"test_auc": auc_score}

        logging.info(f"metrics: {metrics}")
        logging.info("\n" + classification_report(dtest.get_label(), predictions))
        logging.info("\n" + str(confusion_matrix(dtest.get_label(), predictions)))


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

        # Plot the ROC curve
        # fpr, tpr, _ = roc_curve(dtest.get_label(), predictions)
        # plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend(loc='lower right')
        # plt.show()
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
