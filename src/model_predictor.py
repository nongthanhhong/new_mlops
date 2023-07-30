
import os
import yaml
import time
import mlflow
import random
import json
import glob
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
from scipy.stats import wasserstein_distance, ks_2samp
from data_loader import deploy_data_loader, load_data
from problem_config import ProblemConst, create_prob_config
# import httpx
import concurrent.futures

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

        signature = self.model._model_meta.signature
        input_schema = signature.inputs

        feature_names = []
        for col_spec in input_schema:
            feature_names.append(col_spec.name)

        self.columns_to_keep = feature_names

        train_data, _ = load_data(self.prob_config)
        self.drift_column = train_data["feature19"]

        if self.prob_config.prob_id == 'prob-2' and self.prob_config.phase_id == "phase-2":
            save_path = f"./prob_resource/{self.prob_config.phase_id}/{self.prob_config.prob_id}/"
            with open(save_path + "label_mapping.pickle", 'rb') as f:
                label_mapping = pickle.load(f)
            self.inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        
        # submit_num = len(glob.glob(os.path.join(self.prob_config.captured_data_dir, '*/')))
        
        self.path_save_captured = (self.prob_config.captured_data_dir / f"{12}")
        os.makedirs(self.path_save_captured, exist_ok=True)

        with open("./src/config_files/data_config.json", 'r') as f:
            config = json.load(f)
        save_path = f"./prob_resource/{ self.prob_config.phase_id}/{ self.prob_config.prob_id}/"
        scaler_name = config['scale_data']['method']
        if not os.path.isfile(save_path + f"{scaler_name}_scaler.pkl"):
            raise ValueError(f"Not exist prefitted '{scaler_name}' scaler")
        # Load the saved scaler from the file
        with open(save_path + f"{scaler_name}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        save_path = f"./prob_resource/{self.prob_config.phase_id}/{self.prob_config.prob_id}/"
        with open(save_path + "encoder.pkl", 'rb') as f:
            self.encoder = pickle.load(f)
    


    def detect_drift(self, drift_feature) -> int:
        # watch drift between coming requests and training data
        ref_data = self.drift_column
        curr_data = drift_feature

        # _, p_value = ks_2samp(ref_data, curr_data)
        # return 1 if p_value < 0.9 else 0
    
        wasserstein = wasserstein_distance(ref_data, curr_data)
        return 1 if wasserstein > 0.33 else 0

    def predict(self, data: Data):

        start_time = time.time()


        data_time = time.time()
        raw_data = pd.DataFrame(data.rows, columns=data.columns)
        logging.info(f"Load data take {round((time.time() - data_time) * 1000, 0)} ms")

        process_data_time = time.time()
        feature_df = deploy_data_loader(prob_config = self.prob_config, raw_df = raw_data, captured_data_dir = self.path_save_captured, id = data.id, scaler=self.scaler, encoder=self.encoder)
        logging.info(f"Process data take {round((time.time() - process_data_time) * 1000, 0)} ms")

        predict_time = time.time()
        prediction = self.model.predict(feature_df[self.columns_to_keep])
        logging.info(f"Predict take {round((time.time() - predict_time) * 1000, 0)} ms")

        transform_predict_time = time.time()
        if self.prob_config.prob_id == 'prob-2' and self.prob_config.phase_id == "phase-2":
            '''
            transform numerical label to string label
            '''
            prediction_list = prediction.squeeze().tolist()
            prediction = [self.inverse_label_mapping[label] for label in prediction_list]
            prediction = np.array(prediction, dtype=str)
        logging.info(f"Transform predict take {round((time.time() - transform_predict_time) * 1000, 0)} ms")
            

        drift_detect_time = time.time()
        is_drifted = self.detect_drift(feature_df["feature19"])
        # is_drifted = 0
        logging.info(f"drift detect take {round((time.time() - drift_detect_time) * 1000, 0)} ms")



        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"total prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }


class PredictorApi:
    def __init__(self, predictor_1: ModelPredictor, predictor_2: ModelPredictor, phase_id:str):
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post(f"/{phase_id}/prob-1/predict")
        async def predict(data: Data, request: Request):

            # async with httpx.AsyncClient() as client:
            #     response = await client.get(f"https://api.example.com/data/{x}")
            #     data = response.json()

            self._log_request(request)

            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            #     response = executor.submit(self.predictor_1.predict, data).result()

            response = self.predictor_1.predict(data)

            self._log_response(response)
            return response

        @self.app.post(f"/{phase_id}/prob-2/predict")
        async def predict(data: Data, request: Request):

            self._log_request(request)

            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            #     response = executor.submit(self.predictor_2.predict, data).result()

            response =  self.predictor_2.predict(data)

            self._log_response(response)
            return response

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def get_app(self):
        return self.app

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)

app = None
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

    predictor_1 = ModelPredictor(config_file_path=args.config_path[0])
    predictor_2 = ModelPredictor(config_file_path=args.config_path[1])

    # predictor_1 = ModelPredictor(config_file_path=prob_1_config_path)
    # predictor_2 = ModelPredictor(config_file_path=prob_2_config_path)

    api = PredictorApi(predictor_1, predictor_2, phase_id = ProblemConst.PHASE)
    # api = PredictorApi(predictor_1, predictor_2, phase_id = "phase-1")
    # app = api.get_app()
    # uvicorn.run("model_predictor:app", host="0.0.0.0", port=args.port, workers=4)
    api.run(port=args.port)
