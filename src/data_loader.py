import sys
sys.path.append('/mnt/e/mlops-marathon/new_mlops/utils')

import os
import glob
import pickle
import hashlib
import logging
import argparse
import numpy as np
import pandas as pd
import catboost as cb
from utils import *
from data_processor import *
from sklearn.model_selection import train_test_split
from problem_config import ProblemConfig, ProblemConst, get_prob_config, load_feature_configs_dict




def raw_data_process(prob_config: ProblemConfig):
    logging.info("------------Start process_raw_data------------")
    logging.info("Processing data from  %s %s", prob_config.phase_id, prob_config.prob_id)

    save_path = f"./prob_resource/{prob_config.phase_id}/{prob_config.prob_id}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    training_data = pd.read_parquet(prob_config.raw_data_path)
    

    logging.info("Encoding categorical columns...")
    encoded_data = train_encoder(prob_config = prob_config, df = training_data)

    dtype = encoded_data.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()
    with open(save_path + "types.json", 'w+') as f:
        json.dump(dtype, f)

    logging.info("Preprocessing data...")

    data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'train')
    
    # Export preprocessed data
    logging.info("Save data...")
    data_x = data.drop([prob_config.target_col], axis=1)
    data_y = data[[prob_config.target_col]]
    data_x.to_parquet(prob_config.processed_x_path, index=False)
    data_y.to_parquet(prob_config.processed_y_path, index=False)


    raw_config = json.load(open(prob_config.raw_feature_config_path))
    config ={}
    config['numeric_columns'] = []
    config['category_columns']= []
    for column in data.columns.drop(prob_config.target_col):
        if column in raw_config['category_columns']:
            config['category_columns'].append(column)
        else:
            config['numeric_columns'].append(column)

    config['target_column'] = raw_config['target_column'] 
    config['ml_type'] = raw_config['ml_type']


    with open(prob_config.processed_feature_config_path, 'w+') as f:
        json.dump(config, f, indent=4)
    
    logging.info(data.info())

    logging.info("Done!")
        
def load_capture_data(prob_config: ProblemConfig):
    captured_x_path = prob_config.captured_x_path
    captured_y_path = prob_config.uncertain_y_path
    captured_x = pd.read_parquet(captured_x_path)
    captured_y = pd.read_parquet(captured_y_path)
    return captured_x, captured_y[prob_config.target_col]

def load_data(prob_config: ProblemConfig):
    processed_x_path = prob_config.processed_x_path
    processed_y_path = prob_config.processed_y_path
    data_x = pd.read_parquet(processed_x_path)
    data_y = pd.read_parquet(processed_y_path)
    return data_x, data_y[prob_config.target_col]
    
def train_data_loader(prob_config: ProblemConfig, add_captured_data = False):

    """
    Load data for traning phase

    Args:
    prob_config (ProblemConfig): config for problem
    add_captured_data (bool): default is False, if True that mean add captured data that had been labeled

    Returns:

    data that suitable for traning phase

    """
     # load train data
    if add_captured_data:
        logging.info("Use captured data")

        data_x, data_y = load_data(prob_config)

        captured_x, captured_y = load_capture_data(prob_config)

        # Merge the labeled and unlabeled data
        all_data = pd.concat([data_x, captured_x], axis=0)
        all_labels = pd.concat([data_y, captured_y], axis=0)

        
        weight = int( len(train_y) / len(captured_y) ) if len(train_y)>len(captured_y) else 1
        weights = np.concatenate([np.ones(len(train_y)), np.ones(len(test_y)), np.ones(len(captured_y)) * weight])

        # split data into training, validation, and test sets
        train_x, test_x, train_y, test_y, train_weights, test_weights = train_test_split(all_data, all_labels, weights, 
                                                                                            test_size=0.2, 
                                                                                            random_state=42,
                                                                                            stratify= all_labels)
        train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(train_x, train_y, train_weights, 
                                                                                        test_size=0.25, 
                                                                                        random_state=42,
                                                                                        stratify= train_y)

        print('Train: old - new: ', np.unique(train_weights, return_counts=True))
        print('Val: old - new: ', np.unique(val_weights, return_counts=True))
        print('Test: old - new: ', np.unique(test_weights, return_counts=True))

        # create Pool objects for each set with weights
        dtrain = cb.Pool(data=train_x, label=train_y, weight=train_weights)
        dval = cb.Pool(data=val_x, label=val_y, weight=val_weights)
        dtest = cb.Pool(data=test_x, label=test_y, weight=test_weights)

    else:
        logging.info("Use original data")
        data_x, data_y = load_data(prob_config)
        

        # split data into training, validation, and test sets
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                            test_size=0.2, 
                                                            random_state=42,
                                                            stratify= data_y)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                            test_size=0.25, 
                                                            random_state=42,
                                                            stratify= train_y)

        dtrain = cb.Pool(data=train_x, label=train_y)
        dval =  cb.Pool(data=val_x, label=val_y)
        dtest =  cb.Pool(data=test_x, label=test_y)
        
    return dtrain, dval, dtest, test_x

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

def deploy_data_loader(prob_config: ProblemConfig, raw_df: pd.DataFrame):

    """
    Process data for deploy phase

    Args:
    prob_config (ProblemConfig): config for problem
    data (pandas.DataFrame): raw data frame input

    Returns:
    pandas.DataFrame: processed data for predict
    """

    
    
    columns_to_keep = prob_config.categorical_cols + prob_config.numerical_cols
    new_data = raw_df[columns_to_keep]

    encoded_data = transform_new_data(prob_config , new_data)

    new_data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'deploy')
    
    #generate id for save file
    parquet_files = glob.glob(os.path.join(prob_config.captured_data_dir, '*.parquet'))
    filename = generate_id(str(len(parquet_files)))

    # save request data for improving models
    output_file_path = os.path.join(prob_config.captured_data_dir, f"{filename}.parquet")
    raw_df.to_parquet(output_file_path, index=False)

    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)

    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    raw_data_process(prob_config)


