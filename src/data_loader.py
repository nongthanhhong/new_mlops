import sys
sys.path.append('/mnt/e/mlops-marathon/new_mlops/utils')

import os
import glob
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import catboost as cb
from utils import *
from data_processor import *
from scipy.stats import wasserstein_distance, ks_2samp
from adapt.instance_based import KLIEP
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from problem_config import ProblemConfig, ProblemConst, get_prob_config, load_feature_configs_dict




def raw_data_process(prob_config: ProblemConfig, flag = "new"):
    logging.info("------------Start process_raw_data------------")
    logging.info("Processing data from  %s %s", prob_config.phase_id, prob_config.prob_id)
    
    if os.path.exists(prob_config.prob_resource_path) and flag == "new":
        for file in os.listdir(prob_config.prob_resource_path):
            os.remove(os.path.join(prob_config.prob_resource_path, file))
        os.rmdir(prob_config.prob_resource_path)
    os.makedirs(os.path.dirname(prob_config.prob_resource_path), exist_ok=True)
    
    training_data = pd.read_parquet(prob_config.raw_data_path)

    # list_drop_1 = ['feature3', 'feature20', 'feature10', 'feature22', 'feature29', 'feature13', 'feature25', 'feature36', 'feature4', 'feature21', 'feature7', 'feature28', 'feature37', 'feature41', 'feature19', 'feature17', 'feature38', 'feature6', 'feature14', 'feature24', 'feature30', 'feature32', 'feature33', 'feature39', 'feature40']
    # list_drop_2 = ['feature3', 'feature20', 'feature10', 'feature22', 'feature29', 'feature13', 'feature25', 'feature36', 'feature4', 'feature21', 'feature7', 'feature28', 'feature37', 'feature41', 'feature19', 'feature17', 'feature38']
    # training_data =  training_data.drop(list_drop_2, axis = 1)

    # training_data = training_data.drop_duplicates().reset_index(drop=True)

    logging.info("Encoding categorical columns...")
    encoded_data = train_encoder(prob_config = prob_config, df = training_data)

    dtype = encoded_data.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()
    with open(prob_config.prob_resource_path + "types.json", 'w+') as f:
        json.dump(dtype, f)

    logging.info("Preprocessing data...")
    data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'train', flag = flag)
    
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
    # captured_y_path = prob_config.uncertain_y_path
    captured_x = pd.read_parquet(captured_x_path)
    # captured_y = pd.read_parquet(captured_y_path)
    return captured_x

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

    data_x, data_y = load_data(prob_config)
    

    if add_captured_data:
        
        logging.info("Use captured data and feature selection")
        captured_x = load_capture_data(prob_config)
        columns = data_x.columns

        #feature selection
        # logging.info("Selecting features...")

        # selected_columns = feature_selection(data_x = data_x, data_y = data_y, captured_x = captured_x)
        
        # data_x = data_x[selected_columns]
        # captured_x = captured_x[selected_columns]

        # # set weight
        # logging.info("Calculating sample_weight...")
        

        # split data into training, validation, and test sets
        # train_x, test_x, train_y, test_y, train_weights, test_weights = train_test_split(data_x, data_y, train_weight, 
        #                                                                                     test_size=0.1, 
        #                                                                                     random_state=42,
        #                                                                                     stratify= data_y)
        # train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(train_x, train_y, train_weights, 
        #                                                                                 test_size=0.25, 
        #                                                                                 random_state=42,
        #                                                                                 stratify= train_y)
        
        # dval = cb.Pool(data=val_x, label=val_y, weight=val_weights)
        # dtest = cb.Pool(data=test_x, label=test_y, weight=test_weights)

        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                            test_size=0.1, 
                                                            random_state=42,
                                                            stratify= data_y)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                            test_size=0.25, 
                                                            random_state=42,
                                                            stratify= train_y)
        
        # smote = SMOTE(sampling_strategy={2: desired_count}, random_state=0)
        smote = SMOTE(random_state=0)

        # X_resampled, y_resampled = smote.fit_resample(data_x, data_y)
        agr_train_x, agr_train_y = smote.fit_resample(train_x, train_y)

        logging.info("Checking distribution of new generated dataset...")

        total_dist = 0

        for col in train_x.columns:
            logging.info(col)
            # compute the Wasserstein distance
            wasserstein_dist = wasserstein_distance(train_x[col], agr_train_x[col])
            print("Wasserstein distance:", wasserstein_dist)

            # perform the two-sample Kolmogorov-Smirnov test
            statistic, pvalue = ks_2samp(train_x[col], agr_train_x[col])
            print("KS statistic:", statistic)
            print("KS p-value:", pvalue)

            total_dist += wasserstein_dist
            
        if total_dist < len(train_x.columns)*0.005:
            logging.info("Use oversampling for training dataset")
            # create Pool objects for each set with weights
            dtrain = cb.Pool(data=agr_train_x, label=agr_train_y)
            dval =  cb.Pool(data=val_x, label=val_y)
            dtest =  cb.Pool(data=test_x, label=test_y)
                
            return dtrain, dval, dtest, test_x, captured_x, agr_train_x, agr_train_y
        else:
            logging.info("Not use oversampling for training dataset")
            # create Pool objects for each set with weights
            dtrain = cb.Pool(data= train_x, label= train_y)
            dval =  cb.Pool(data=val_x, label=val_y)
            dtest =  cb.Pool(data=test_x, label=test_y)
                
            return dtrain, dval, dtest, test_x, captured_x, train_x, train_y
            

    else:
        logging.info("Use original data")
        # split data into training, validation, and test sets
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                            test_size=0.1, 
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
        
def deploy_data_loader(prob_config: ProblemConfig, raw_df: pd.DataFrame, captured_data_dir = None, id = None, scaler = None, encoder = None):

    """
    Process data for deploy phase

    Args:
    prob_config (ProblemConfig): config for problem
    data (pandas.DataFrame): raw data frame input

    Returns:
    pandas.DataFrame: processed data for predict
    """
    
    columns_to_keep = prob_config.categorical_cols + prob_config.numerical_cols
    new_data = raw_df[columns_to_keep].copy()

    transform_new_data_time = time.time()
    encoded_data = transform_new_data(prob_config , new_data, encoder = encoder)
    logging.info(f"transform_new_data_time take {round((time.time() - transform_new_data_time) * 1000, 0)} ms")

    preprocess_data_time = time.time()
    new_data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'deploy', deploy_scaler = scaler)
    logging.info(f"total preprocess_data take {round((time.time() - preprocess_data_time) * 1000, 0)} ms")

    #generate name for save file
    generate_id_time = time.time()
    filename = generate_id(id)
    logging.info(f"generate_name take {round((time.time() - generate_id_time) * 1000, 0)} ms")

    # save request data for improving models
    save_data_time = time.time()
    output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
    raw_df.to_parquet(output_file_path, index=False, engine='pyarrow', compression='snappy')
    logging.info(f"save_data take {round((time.time() - save_data_time) * 1000, 0)} ms")

    return new_data

def update_processor(prob_config: ProblemConfig, captured_data: pd.DataFrame):

    # Load the config file
    with open("./src/config_files/data_config.json", 'r') as f:
        config = json.load(f)
        
    
    # Load the saved scaler from disk
    if config['scale_data']:
        scaler_name = config['scale_data']['method']

    with open(prob_config.prob_resource_path + f"{scaler_name}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    #update with new captured data
    scaler.fit(captured_data)

    # Save the upgraded scaler 
    with open(prob_config.prob_resource_path + f"{scaler_name}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

def captured_data_loader(prob_config: ProblemConfig):

    columns_to_keep = prob_config.categorical_cols + prob_config.numerical_cols

    if os.path.isfile(prob_config.captured_data_dir / "total_data.parquet"):
        os.remove(prob_config.captured_data_dir / "total_data.parquet") 
    
    with open("./src/config_files/data_config.json", 'r') as f:
        config = json.load(f)
        
    scaler_name = config['scale_data']['method']
    
    with open(prob_config.prob_resource_path + "encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)

    captured_x = pd.DataFrame()

    # Disable all logging calls
    logging.disable(logging.CRITICAL)

    folder_names = [
                    d for d in os.listdir(prob_config.captured_data_dir)
                    if os.path.isdir(os.path.join(prob_config.captured_data_dir, d)) and d != "processed"
                    ]

    batchs = []
    for idx in folder_names:

        path_save = os.path.join(prob_config.captured_data_dir,str(idx))
        print(f"Processing {path_save}")

        for file_path in tqdm(glob.glob(path_save + "/*.parquet"), ncols=100, desc ="Loading...", unit ="file"):

            try:
                captured_data = pd.read_parquet(file_path)
                captured_x = pd.concat([captured_x, captured_data])
                filename = os.path.basename(file_path) #[:-len(".parquet")]
                batchs.append((captured_data,filename))

                # new_data = captured_data[columns_to_keep]
                # encoded_data = transform_new_data(prob_config , new_data)
                # new_data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'deploy')
                # new_data.to_parquet(prob_config.processed_captured_data_dir / file_path.name)
                # os.remove(file_path) 
            
            except:
                print(f"Error: Cannot open {file_path}, then remove it!")
                os.remove(file_path) 
    
    # Enable all logging calls
    logging.disable(logging.NOTSET)

    captured_x = captured_x.drop_duplicates().reset_index(drop=True)
    new_data = captured_x[columns_to_keep]
    encoded_data = transform_new_data(prob_config , new_data, encoder=encoder)

    update_processor(prob_config = prob_config, captured_data = encoded_data)

    raw_data_process(prob_config, flag = "update")
    
    if not os.path.isfile(prob_config.prob_resource_path + f"{scaler_name}_scaler.pkl"):
        raise ValueError(f"Not exist prefitted '{scaler_name}' scaler")
    # Load the saved scaler from the file
    with open(prob_config.prob_resource_path + f"{scaler_name}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)

    processed_new_data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'deploy', deploy_scaler=scaler)

    processed_new_data.to_parquet(prob_config.captured_x_path)
    processed_new_data.info()


    # Disable all logging calls
    logging.disable(logging.CRITICAL)
    #save batches
    for batch, file_name in tqdm(batchs, ncols=100, desc ="Saving batches...", unit ="file"):
        new_data = batch[columns_to_keep]
        encoded_data = transform_new_data(prob_config, new_data, encoder=encoder)
        new_data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'deploy', deploy_scaler=scaler)
        new_data.to_parquet(prob_config.processed_captured_data_dir / file_name)
        
    # Enable all logging calls
    logging.disable(logging.NOTSET)

    return processed_new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)

    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    raw_data_process(prob_config)


