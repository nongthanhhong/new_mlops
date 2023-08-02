
import sys
sys.path.append('/mnt/e/mlops-marathon/new_mlops/utils')

import os
import json
import pickle
import time
import logging
import numpy as np
import pandas as pd
import catboost as cb
from scipy import stats
from problem_config import ProblemConfig
from category_encoders import TargetEncoder
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def train_encoder(prob_config: ProblemConfig, df: pd.DataFrame):

    """
    Performing transform raw data with categorical columns to numerical:
    
    Args:
    prob_config (ProblemConfig): config for problem.
    df (pandas.DataFrame): raw data input
    
    Returns:
    pandas.DataFrame: Data that had been encoding using one-hot encoder
    """
    categorical_cols= prob_config.raw_categorical_cols

    # Create a OneHotEncoder object
    encoder = TargetEncoder()
    
    save_path = f"./prob_resource/{prob_config.phase_id}/{prob_config.prob_id}/"
    
    # Fit the encoder on the training data
    cat_columns = df[categorical_cols]
    if df[prob_config.target_col].dtypes == 'object':

        # Encode the fruit column using the factorize method
        labels, uniques = pd.factorize( df[prob_config.target_col])

        # Save the mapping between the original string labels and their encoded values
        label_mapping = dict(zip(uniques, range(len(uniques))))

        # Save the label mapping to disk using pickle
        
        with open(save_path + "label_mapping.pickle", 'wb') as f:
            pickle.dump(label_mapping, f)

        target_columns = labels
        df[prob_config.target_col] = labels

    else:
        target_columns = df[prob_config.target_col]

    encoder.fit(cat_columns, target_columns)
    
    # Transform the training data
    one_hot_df = pd.DataFrame(encoder.transform(df[categorical_cols]),
                              columns= categorical_cols)
    
    # Drop the original categorical column
    df = df.drop(categorical_cols, axis=1)

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    df = pd.concat([df, one_hot_df], axis=1)
    
    # Save the fitted encoder to disk
    with open(save_path + "encoder.pkl", 'wb') as f:
        pickle.dump(encoder, f)
    
    return df

def transform_new_data(prob_config: ProblemConfig, new_df: pd.DataFrame, encoder = None):
    """
    Performing transform new input data with categorical columns to numerical:
    
    Args:
    prob_config (ProblemConfig): config for problem.
    new_df (pandas.DataFrame): new raw data input
    
    Returns:
    pandas.DataFrame: Data that had been encoding using one-hot encoder
    """

    column = prob_config.raw_categorical_cols
    
    # Load the saved encoder from disk
    
    # Transform the new data using the fitted encoder
    one_hot_df = pd.DataFrame(encoder.transform(new_df[column]),
                              columns= column)
    
    # Drop the original categorical column
    drop_df = new_df.copy()
    drop_df.drop(column, axis=1, inplace=True)

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    new_df = pd.concat([drop_df, one_hot_df], axis=1)
    
    return new_df

def preprocess_data(prob_config: ProblemConfig, data: pd.DataFrame, mode='train', flag = "new", deploy_scaler = None):
        
        """
        Performing preprocessing data:

        1 - Handle missing values.

        2 - Scale data.

        3 - Handle wrong type data.
        
        Args:
        prob_config (ProblemConfig): config for problem.
        data (pandas.DataFrame): raw data input, that had been performed categorical encoding
        mode (str): default = 'train', 'train' for training phase, 'deploy' for deployment phase
        
        Returns:
        pandas.DataFrame: Data that had been preprocessed
        """

        # Load the config file
        with open("./src/config_files/data_config.json", 'r') as f:
            config = json.load(f)

        save_path = f"./prob_resource/{prob_config.phase_id}/{prob_config.prob_id}/"

        preprocess_missing_data_time = time.time()

        # Handle missing values
        if config['missing_values']['method'] == 'drop':
            data = data.dropna()
        elif config['missing_values']['method'] == 'fill':
            fill_value = config['missing_values']['fill_value']
            if fill_value == 'mean':
                data.fillna(np.mean(data), inplace=True)
            elif fill_value == 'median':
                data.fillna(np.median(data), inplace=True)
            else:
                data.fillna(fill_value, inplace=True)
            
        # Fill missing values only once, instead of checking for missing values after each fill method
        # data.fillna(method='ffill', inplace=True)
        # data.fillna(method='bfill', inplace=True)
        data.fillna(0, inplace=True)
            
        logging.info(f"preprocess_missing_data_time data take {round((time.time() - preprocess_missing_data_time) * 1000, 0)} ms")
        
        # Scale data

        Scale_data_time = time.time()
        if mode == 'train' :
            if config['scale_data']:
                scaler = None
                scaler_name = config['scale_data']['method']
                if scaler_name == 'standard':
                    scaler = StandardScaler()
                elif scaler_name == 'minmax':
                    scaler = MinMaxScaler()
                elif scaler_name == 'robust':
                    scaler = RobustScaler()
                    
            old_label = data[[prob_config.target_col]]
            old_features = data.drop([prob_config.target_col], axis=1)
            # if  flag == "new":
            scale_features = pd.DataFrame(scaler.fit_transform(old_features), columns=old_features.columns)
            # Save the scaler to a file for later use in deployment
            with open(save_path + f"{scaler_name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
                
        elif  flag == "update":
            if not os.path.isfile(save_path + f"{scaler_name}_scaler.pkl"):
                raise ValueError(f"Not exist prefitted '{scaler_name}' scaler")
            # Load the saved scaler from the file
            with open(save_path + f"{scaler_name}_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            scale_features = pd.DataFrame(scaler.transform(old_features), columns=old_features.columns)
            data = pd.concat([scale_features, old_label], axis=1)

        elif mode == 'deploy':
            data = pd.DataFrame(deploy_scaler.transform(data), columns=data.columns)
            
        logging.info(f"Scale_data_time data take {round((time.time() - Scale_data_time) * 1000, 0)} ms")
        
        # Handle the wrong datatype in each column
        wrong_data_type_time = time.time()

        # Use the apply() method to convert each column to the correct data type.
        data = data.apply(pd.to_numeric, errors='coerce')
        data.fillna(0, inplace=True)
            
        # with open(save_path + "types.json", 'r') as f:
            # dtype = json.load(f)
        # columns = data.columns if mode == 'deploy' else data.drop([prob_config.target_col], axis=1).columns
        # for column in columns:
        #     if column in dtype:
        #         data[column] = pd.to_numeric(data[column], errors='coerce')
        #         # Fill missing values only once per column, instead of checking for missing values after each fill method
        #         if data[column].isnull().any():
        #             data[column].fillna(method='ffill', inplace=True)
        #             data[column].fillna(method='bfill', inplace=True)
        #             data[column].fillna(0, inplace=True)
        #         data[column] = data[column].astype(dtype[column])
                        
        logging.info(f"wrong_data_type_time data take {round((time.time() - wrong_data_type_time) * 1000, 0)} ms")

        return data 

def feature_selection(data_x: pd.DataFrame, data_y: pd.DataFrame, captured_x: pd.DataFrame):

    model = cb.CatBoostClassifier(eval_metric = "AUC", silent=True)
    model.fit(data_x, data_y)

    # Get the feature importances
    feature_importances = model.get_feature_importance()
    keys = data_x.columns
    importances_dict = {keys[i]: feature_importances[i] for i in range(len(keys))}

    #Get drift score for all columns
    drift_score_dict = {}
    for column in data_x.columns:
        data1 = data_x[column].to_numpy()
        data2 = captured_x[column].to_numpy()
        drift_score_dict[column] = wasserstein_calculator(data1, data2)

    
    #calculate score for features

    score = {}
    alpha = 0.5
    beta = 0.5

    for column in data_x.columns: 
        score[column] = alpha * drift_score_dict[column] + beta * importances_dict[column]
    
    sorted_score = sorted(score.items(), key=lambda x: x[1])

    selected_columns = {k: v for k, v in sorted_score[:]}

    return selected_columns.keys()

def wasserstein_calculator(data1, data2):

    wasserstein = wasserstein_distance(data1, data2)
    # wasserstein = wasserstein / (np.max(data2) - np.min(data2))
    
    return wasserstein 