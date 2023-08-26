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
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, log_loss, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from data_loader import train_data_loader
from problem_config import (ProblemConfig,
                            ProblemConst,
                            get_prob_config)
from numpy import mean
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from data_loader import load_data



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

def evaluate_model(model = None, test_data = None, test_label = None):

    probs_prediction = model.predict_proba(test_data)

    count = len(test_data[probs_prediction.max(axis=1) > 0.8])
    percentage = (count / len(probs_prediction)) * 100
    # avg_prob = np.mean(probs_prediction)

    start_time = time.time()
    predictions = model.predict(test_data)
    predict_time = round((time.time() - start_time) * 1000,2) #at ms
    
    # Calculate the ROC AUC score
    if len(np.unique(test_label))>2:
        try:
            # Compute the ROC AUC score using the one-vs-one approach
            roc_auc = roc_auc_score(test_label, probs_prediction, multi_class='ovo')
        except:
            # Compute the ROC AUC score using the one-vs-rest approach
            roc_auc = roc_auc_score(test_label, predictions, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(test_label, predictions)
        
    log_loss_score = log_loss(test_label, probs_prediction)

    acc_score = accuracy_score(test_label, predictions)

    try:
        f1 = f1_score(test_label, predictions)
    except:
        f1 = f1_score(test_label, predictions, average='weighted')

    metrics = {"test_auc": roc_auc, "test_f1": f1, "log_loss": log_loss_score, "test_acc": acc_score, "percent_prob": percentage, "predict_time_ms": predict_time/len(test_data)}
    
    logging.info(f"metrics: {metrics}")
    logging.info("\n" + classification_report(test_label, predictions))
    logging.info("\n" + str(confusion_matrix(test_label, predictions)))
    logging.info(f"\nPredict take {predict_time} ms for {len(test_data)} samples - AVG {predict_time/len(test_data)} ms each.")

    return metrics, predictions, probs_prediction

class ModelTrainer:

    @staticmethod
    def train_model(prob_config: ProblemConfig, add_captured_data=False, run_name = None):

        #define model
        class_model = Models(prob_config)
        class_model.catboost_classifier()

        logging.info("*****Start training phase*****")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"{prob_config.phase_id}_{prob_config.prob_id}_{class_model.EXPERIMENT_NAME}")
        if run_name != "Not_set":
            mlflow.set_tag("mlflow.runName", run_name)

        logging.info("==============Load data==============")

        train_x, train_y, val_x, val_y, test_x, test_y = train_data_loader(prob_config = prob_config, add_captured_data = add_captured_data)

        dtrain = cb.Pool(data=train_x, label=train_y)
        dval =  cb.Pool(data=val_x, label=val_y)
        dtest =  cb.Pool(data=test_x, label=test_y)

        print(f'Loaded {len(train_x)} Train samples, {len(val_x)} val samples , and {len(test_x)} test samples!')
        show_proportion("training", train_y)
        show_proportion("validating", val_y)
        show_proportion("testing", test_y)

        
        logging.info("==============Training model==============")
        
        model = class_model.model
        
        
        model.fit(dtrain, eval_set=dval)
        
        evaluate_model(model = model, test_data=test_x, test_label=test_y)

        # define calibrated classifier
        clf = CalibratedClassifierCV(model, cv='prefit', method='isotonic')

        clf.fit(test_x, test_y)

        logging.info("==============Testing model==============")

        metrics, predictions, probs_prediction = evaluate_model(model = clf, test_data=test_x, test_label=test_y)

        model = clf
        
        # # test by cross validation
        # # define scoring metrics
        # scoring = {'roc_auc': 'roc_auc', 'accuracy': 'accuracy', 'f1': 'f1'}

        # # evaluate pipeline
        # cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=0)
        # scores = cross_validate(clf, data_x, data_y, scoring=scoring, cv=cv, n_jobs=-1)
        # best_model = cv.best_estimator_

        # # print results
        # for k,v in scores.items():
        #     print(f"{k} - {v}")
        #     print(f'Mean {k}: {mean(v)}')


        # if add_captured_data:
        #     logging.info("==============Improve use Active learning==============")

        #     unlabeled_data = captured_x.copy()
            
        #     logging.disable(logging.CRITICAL)
        #     loop = 0

        #     while loop < 10 and len(unlabeled_data) != 0:
        #         probs_prediction = model.predict_proba(unlabeled_data)
                
                
        #         # Select the most informative unlabeled data points
        #         most_informative_data = unlabeled_data[(probs_prediction.max(axis=1) > 0.9) | (probs_prediction.min(axis=1) < 0.1)]
        #         most_informative_predictions = model.predict(most_informative_data)
                
        #         train_x = pd.concat([train_x, most_informative_data ])
        #         train_y = pd.concat([train_y, pd.DataFrame(most_informative_predictions) ])
                

        #         unlabeled_data.drop(index=most_informative_data.index, inplace=True)
        #         unlabeled_data.reset_index(drop=True, inplace=True)
                
        #         unlabeled_data.info()

        #         dtrain = cb.Pool(data=train_x, label=train_y)
        #         # model = class_model.model

        #         model.fit(dtrain, 
        #           eval_set=dval,
        #           **class_model.train)

        #     logging.disable(logging.NOTSET)   

        #     metrics, predictions, probs_prediction = evaluate_model(model = model, dtest = dtest, test_x = test_x) 

        
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
        logging.info("=================== Finish train model ===================")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("--name-run", type=str, default="Not_set")
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)

    ModelTrainer.train_model(
        prob_config, add_captured_data=args.add_captured_data, run_name=args.name_run
    )
