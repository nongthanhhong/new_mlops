
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
from category_encoders import TargetEncoder, BinaryEncoder
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def train_encoder(prob_config: ProblemConfig, train_x: pd.DataFrame, train_y: pd.DataFrame or pd.Series, val_x: pd.DataFrame, val_y: pd.DataFrame or pd.Series, test_x: pd.DataFrame, test_y: pd.DataFrame or pd.Series):

    """
    Performing transform raw data with categorical columns to numerical:
    
    Args:
    prob_config (ProblemConfig): config for problem.
    df (pandas.DataFrame): raw data input
    
    Returns:
    pandas.DataFrame: Data that had been encoding using one-hot encoder
    """
    
    categorical_cols =  list(set(prob_config.raw_categorical_cols) & set(train_x.columns))
    if len(categorical_cols) == 0:
        return train_x, train_y, val_x, test_x
    
    # define encoder
    logging.info(f"Encode : {categorical_cols}")
    # encoder = BinaryEncoder(return_df=False)
    encoder = TargetEncoder(return_df=False)
    
    # Fit the encoder on the training data
    cat_columns = train_x[categorical_cols].to_numpy()

    if train_y.dtypes == 'object':
        # Encode the fruit column using the factorize method
        train_labels, uniques = pd.factorize(train_y)

        # Save the mapping between the original string labels and their encoded values
        label_mapping = dict(zip(uniques, range(len(uniques))))

        # Save the label mapping to disk using pickle
        with open(prob_config.prob_resource_path + "label_mapping.pickle", 'wb') as f:
            pickle.dump(label_mapping, f)
       
        val_labels =  [label_mapping[label] for label in val_y]
        test_labels =  [label_mapping[label] for label in test_y]

        target_columns = train_labels

        cl = [prob_config.target_col]
        train_y = pd.DataFrame(train_labels, columns=cl)
        val_y = pd.DataFrame(val_labels, columns=cl)
        test_y = pd.DataFrame(test_labels, columns=cl)

    else:
        target_columns = train_y.to_numpy()

    encode_time = time.time()
    encoded_cols = encoder.fit_transform(cat_columns, target_columns)
    
    # Transform the training data
    encoded_categorical_cols = pd.DataFrame(encoded_cols, columns=categorical_cols)
    # Drop the original categorical column
    train_x = train_x.drop(categorical_cols, axis=1)
    # # Concatenate the original DataFrame with the one-hot encoded DataFrame
    train_x = pd.concat([train_x, encoded_categorical_cols], axis=1)
    
    # Save the fitted encoder to disk
    with open(prob_config.prob_resource_path + "encoder.pkl", 'wb') as f:
        pickle.dump(encoder, f)

    val_x = pd.concat([val_x.drop(categorical_cols, axis=1), pd.DataFrame(encoder.transform(val_x[categorical_cols].to_numpy()), columns=categorical_cols)], axis=1)
    test_x = pd.concat([test_x.drop(categorical_cols, axis=1), pd.DataFrame(encoder.transform(test_x[categorical_cols].to_numpy()), columns=categorical_cols)], axis=1)
    
    logging.info(f"Encode time take {round((time.time() - encode_time) * 1000, 0)} ms")    

    return train_x, train_y, val_x, val_y, test_x, test_y

def transform_new_data(prob_config: ProblemConfig, new_df: pd.DataFrame, encoder = None):
    """
    Performing transform new input data with categorical columns to numerical:
    
    Args:
    prob_config (ProblemConfig): config for problem.
    new_df (pandas.DataFrame): new raw data input
    
    Returns:
    pandas.DataFrame: Data that had been encoding using one-hot encoder
    """

    cat_columns = prob_config.categorical_cols
    if len(cat_columns) == 0:
        return new_df
    else:
        return pd.concat([new_df.drop(cat_columns, axis=1), pd.DataFrame(encoder.transform(new_df[cat_columns].to_numpy()), columns=cat_columns)], axis=1)

    # Load the saved encoder from disk

    # # Transform the new data using the fitted encoder
    # one_hot_df = pd.DataFrame(encoder.transform(new_df[cat_columns]), columns=cat_columns)

    # # Drop the original categorical column and concatenate the original DataFrame with the one-hot encoded DataFrame
    # new_df = pd.concat([new_df.drop(cat_columns, axis=1), one_hot_df], axis=1)

    # return new_df
    
def handle_missing_values_np(data, config):
    data_np = data.to_numpy()
    
    if config['missing_values']['method'] == 'drop':
        data_np = data_np[~np.isnan(data_np).any(axis=1)]
    elif config['missing_values']['method'] == 'fill':
        fill_value = config['missing_values']['fill_value']
        if fill_value == 'mean':
            fill_value = np.nanmean(data_np, axis=0)
            fill_value[np.isnan(fill_value)] = 0 # replace nan with 0
        elif fill_value == 'median':
            fill_value = np.nanmedian(data_np, axis=0)
            fill_value[np.isnan(fill_value)] = 0 # replace nan with 0
        
        mask = np.isnan(data_np)
        data_np[mask] = np.take(fill_value, mask.nonzero()[1])
    return pd.DataFrame(data_np, columns=data.columns)

def wrong_data_type(data):
    data_np = data.to_numpy()
    data_np = data_np.astype(np.float32, copy=False, casting='unsafe')
    data_np[np.isnan(data_np)] = 0
    return pd.DataFrame(data_np, columns=data.columns)

def preprocess_data(prob_config: ProblemConfig, data: pd.DataFrame, label: pd.Series = None, mode='train', flag = "new", deploy_scaler = None):
        
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
        

        preprocess_missing_data_time = time.time()
        data = handle_missing_values_np(data, config)
        logging.info(f"preprocess_missing_data data take {round((time.time() - preprocess_missing_data_time) * 1000, 0)} ms")
        
        # Scale data

        Scale_data_time = time.time()
        if mode == 'train' and flag == "new":
            if config['scale_data']:
                scaler = None
                scaler_name = config['scale_data']['method']
                if scaler_name == 'standard':
                    scaler = StandardScaler()
                elif scaler_name == 'minmax':
                    scaler = MinMaxScaler()
                elif scaler_name == 'robust':
                    scaler = RobustScaler()
                    
            old_label = label
            old_features = data
            data = pd.DataFrame(scaler.fit_transform(old_features), columns=old_features.columns)
            
            # Save the scaler to a file for later use in deployment
            with open(prob_config.prob_resource_path + f"{scaler_name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
                
        elif flag == "update":
            if not os.path.isfile(prob_config.prob_resource_path + f"{scaler_name}_scaler.pkl"):
                raise ValueError(f"Not exist prefitted '{scaler_name}' scaler")
            # Load the saved scaler from the file
            with open(prob_config.prob_resource_path + f"{scaler_name}_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
                    
            old_label = label
            old_features = data
            data = pd.DataFrame(scaler.transform(old_features), columns=old_features.columns)

        elif mode == 'deploy':
            data = pd.DataFrame(deploy_scaler.transform(data), columns=data.columns)
            
        logging.info(f"Scale_data data take {round((time.time() - Scale_data_time) * 1000, 0)} ms")


        convert_type = time.time()
        # Convert the dtype of the `data` DataFrame to `float32`.
        data = data.astype('float32')
        logging.info(f"convert_type data take {round((time.time() - convert_type) * 1000, 0)} ms")
        
        # Handle the wrong datatype in each column
        wrong_data_type_time = time.time()
        if mode == 'deploy':
            data = wrong_data_type(data)
        else:
            data = data.apply(pd.to_numeric, errors='coerce')
            # data.fillna(fill_value, inplace=True)
            data.fillna(0, inplace=True)
        logging.info(f"wrong_data_type data take {round((time.time() - wrong_data_type_time) * 1000, 0)} ms")

        if label is not None:
            label = label.astype('int8')
            return data, label
        else:
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