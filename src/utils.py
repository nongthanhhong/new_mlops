import logging
import os
import hashlib
from pydantic import BaseModel

from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class AppPath:
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data_warehouse"
    # store raw data
    RAW_DATA_DIR = DATA_DIR / "raw_data"
    # store preprocessed data with EDA
    PROCESSED_DATA_DIR =  DATA_DIR / "processed_data"  
    # store captured data   
    CAPTURED_DATA_DIR = DATA_DIR / "captured_data"
    #store model config
    MODEL_CONFIG_DIR = ROOT_DIR / "src/config_files/model_config"


AppPath.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class AppConfig:
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    MLFLOW_MODEL_PREFIX = "model"


class Data(BaseModel):
    id: str
    rows: list
    columns: list


def generate_id(string):

    '''
    Create id for each record
    '''
    # Convert the string to bytes
    string_bytes = string.encode('utf-8')
    
    # Generate the hash object
    hash_object = hashlib.md5(string_bytes)
    
    # Get the hexadecimal representation of the hash
    hex_dig = hash_object.hexdigest()
    
    # Return the first 8 characters of the hexadecimal representation
    return hex_dig[:8]