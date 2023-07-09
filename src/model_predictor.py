
import os
import yaml
import time
import mlflow
import random
import logging
import uvicorn
import pickle
import argparse
import numpy as np
from utils import *
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Request
from utils import AppConfig, AppPath
from data_loader import deploy_data_loader
from problem_config import ProblemConst, create_prob_config

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):

        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        # mlflow.pyfunc.get_model_dependencies(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load model
        model_uri = os.path.join("models:/", self.config["model_name"], str(self.config["model_version"]))
        self.model = mlflow.pyfunc.load_model(model_uri)

        self.columns_to_keep = self.prob_config.categorical_cols + self.prob_config.numerical_cols

        if self.prob_config.prob_id == 'prob-2':
            save_path = f"./prob_resource/{self.prob_config.phase_id}/{self.prob_config.prob_id}/"
            with open(save_path + "label_mapping.pickle", 'rb') as f:
                label_mapping = pickle.load(f)
            self.inverse_label_mapping = {v: k for k, v in label_mapping.items()}


    def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        return random.choice([0, 1])

    def predict(self, data: Data):

        start_time = time.time()

        raw_data = pd.DataFrame(data.rows, columns=data.columns)
        # raw_data = pd.read_parquet("./data_warehouse/captured_data/phase-1/prob-1/3b236d80-417a-47b5-b3ba-0c6327dd27fe.parquet")

        feature_df = deploy_data_loader(prob_config = self.prob_config, raw_df = raw_data)
        
        prediction = self.model.predict(feature_df)

        if self.prob_config.prob_id == 'prob-2':
            '''
            
            '''
            prediction = [self.inverse_label_mapping[label] for label in prediction]

        is_drifted = self.detect_drift(feature_df)

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }


class PredictorApi:
    def __init__(self, predictor_1: ModelPredictor, predictor_2: ModelPredictor):
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-1/prob-1/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor_1.predict(data)
            self._log_response(response)
            return response

        @self.app.post("/phase-1/prob-2/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor_2.predict(data)
            self._log_response(response)
            return response

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    prob_1_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()

    prob_2_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE
        / ProblemConst.PROB2
        / "model-1.yaml"
    ).as_posix()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", nargs="+", default=[prob_1_config_path, prob_2_config_path])
    parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
    args = parser.parse_args()

    # predictor_1 = ModelPredictor(config_file_path=args.config_path[0])
    # predictor_2 = ModelPredictor(config_file_path=args.config_path[1])

    predictor_1 = ModelPredictor(config_file_path=prob_1_config_path)
    predictor_2 = ModelPredictor(config_file_path=prob_2_config_path)

    api = PredictorApi(predictor_1, predictor_2)
    api.run(port=args.port)
