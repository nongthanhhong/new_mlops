import json
import os
from utils import AppPath
from utils import *


class ProblemConst:
    PHASE = "phase-3"
    PROB1 = "prob-1"
    PHASE = "phase-3"
    PROB2 = "prob-2"


class ProblemConfig:
    # required inputs
    phase_id: str
    prob_id: str
    test_size: float
    random_state: int

    prob_resource_path: str

    # for original data
    raw_data_path: str
    raw_feature_config_path: str
    processed_data_path: str
    processed_feature_config_path: str
    processed_x_path: str
    processed_y_path: str

    # ml-problem properties
    raw_categorical_cols: list
    target_col: str
    numerical_cols: list
    categorical_cols: list
    ml_type: str

    # for data captured from API
    captured_data_dir: str
    processed_captured_data_dir: str
    # processed captured data
    captured_x_path: str
    uncertain_y_path: str


def load_feature_configs_dict(config_path: str) -> dict:
    with open(config_path) as f:
        features_config = json.load(f)
    return features_config


def create_prob_config(phase_id: str, prob_id: str) -> ProblemConfig:
    prob_config = ProblemConfig()
    prob_config.prob_id = prob_id
    prob_config.phase_id = phase_id
    prob_config.test_size = 0.2
    prob_config.random_state = 45

    prob_config.prob_resource_path = f"./prob_resource/{prob_config.phase_id}/{prob_config.prob_id}/"

    # construct data paths for raw data
    prob_config.raw_data_path = (
            AppPath.RAW_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "raw_train.parquet" )

    prob_config.raw_feature_config_path = (
            AppPath.RAW_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "features_config.json")

    # data path train
    prob_config.processed_data_path = (
            AppPath.PROCESSED_DATA_DIR / f"{phase_id}" / f"{prob_id}")

    prob_config.processed_feature_config_path = (
            AppPath.PROCESSED_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "features_config.json")
    
    prob_config.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    prob_config.processed_x_path = prob_config.processed_data_path / "processed_data_x.parquet"
    prob_config.processed_y_path = prob_config.processed_data_path / "processed_data_y.parquet"







    raw_feature_configs = load_feature_configs_dict(prob_config.raw_feature_config_path)
    prob_config.raw_categorical_cols = raw_feature_configs.get("category_columns")


    # get properties of ml-problem in and out
    if os.path.isfile(prob_config.processed_feature_config_path):
        feature_configs = load_feature_configs_dict(prob_config.processed_feature_config_path)
    else:
        feature_configs = load_feature_configs_dict(prob_config.raw_feature_config_path)
    prob_config.target_col = feature_configs.get("target_column")
    prob_config.categorical_cols = feature_configs.get("category_columns")
    prob_config.numerical_cols = feature_configs.get("numeric_columns")
    prob_config.ml_type = feature_configs.get("ml_type")

    # construct data paths for API-captured data
    prob_config.captured_data_dir = (
        AppPath.CAPTURED_DATA_DIR / f"{phase_id}" / f"{prob_id}"
    )
    prob_config.captured_data_dir.mkdir(parents=True, exist_ok=True)
    prob_config.processed_captured_data_dir = (
        prob_config.captured_data_dir / "processed"
    )
    prob_config.processed_captured_data_dir.mkdir(parents=True, exist_ok=True)
    prob_config.captured_x_path = (
        prob_config.processed_captured_data_dir / "captured_x.parquet"
    )
    prob_config.uncertain_y_path = (
        prob_config.processed_captured_data_dir / "uncertain_y.parquet"
    )

    return prob_config


def get_prob_config(phase_id: str, prob_id: str):
    prob_config = create_prob_config(phase_id, prob_id)
    return prob_config
