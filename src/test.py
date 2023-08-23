import pandas as pd
import pickle

# # Create a sample DataFrame
# data = {'fruit': ['apple', 'banana', 'orange', 'apple', 'banana']}
# df = pd.DataFrame(data)

# # Encode the fruit column using the factorize method
# labels, uniques = pd.factorize(df['fruit'])
# df['fruit_encoded'] = labels

# # Map the encoded values back to their original string labels
# df['fruit_decoded'] = df['fruit_encoded'].map(lambda x: uniques[x])

# # Display the resulting DataFrame
# print(df)

# save_path = f"./prob_resource/{'phase-2'}/{'prob-2'}/"
# with open(save_path + "label_mapping.pickle", 'rb') as f:
#     label_mapping = pickle.load(f)
# inverse_label_mapping = {v: k for k, v in label_mapping.items()}
# print(inverse_label_mapping[0].type())

# import numpy as np

# # assuming prediction is a list of integers
# prediction = [0, 1, 2, 3]

# # convert prediction to a tensor with dtype int64 and shape [-1, 1]
# prediction_tensor = np.array(prediction, dtype=np.int64).reshape(-1, 1)
# # prediction_tensor = torch.tensor(prediction, dtype=torch.int64).view(-1, 1)
# new_prediction_tensor = prediction_tensor.squeeze().tolist()
# new_prediction_tensor = np.array(new_prediction_tensor, dtype=np.int64).reshape(-1, 1)

# print( new_prediction_tensor== prediction_tensor)

# prints tensor([[0],
#                [1],
#                [2],
#                [3]])
# from fastapi import FastAPI
# from multiprocessing import Pool

# app = FastAPI()

# def cpu_bound_task(x):
#     # Perform some CPU-bound task on x
#     result = x * x
#     return result

# @app.get("/")
# async def root():
#     data = [1, 2, 3, 4]
#     with Pool() as p:
#         results = p.map(cpu_bound_task, data)
#         print(results)
#     return {"results": results}

# import pandas as pd

# file = pd.read_parquet("data_warehouse/raw_data/phase-2/prob-1/raw_train.parquet")

# path_csv = "data_warehouse/raw_data/phase-2/prob-1/raw_train.csv"
# file.to_csv(path_csv)

# import concurrent.futures
# import time

# def factorial(n):
#     if n == 0:
#         return 1
#     else:
#         return n * factorial(n - 1)

# if __name__ == "__main__":
#     start_time = time.time()
#     n = [n for n in range(10)]
#     with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#         for i in n :
#             futures = executor.submit(factorial,  i).result()
#             print(futures)
            
#     print("Time taken: {} seconds".format(time.time() - start_time))

import os
import glob
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
from data_processor import *
from scipy.stats import wasserstein_distance, ks_2samp
from problem_config import ProblemConfig, ProblemConst, get_prob_config, load_feature_configs_dict

# def read_total(prob_config:ProblemConfig, path):


#     with open("./src/config_files/data_config.json", 'r') as f:
#         config = json.load(f)
#     save_path = f"./prob_resource/{ prob_config.phase_id}/{prob_config.prob_id}/"
#     scaler_name = config['scale_data']['method']
#     if not os.path.isfile(save_path + f"{scaler_name}_scaler.pkl"):
#         raise ValueError(f"Not exist prefitted '{scaler_name}' scaler")
#     # Load the saved scaler from the file
#     with open(save_path + f"{scaler_name}_scaler.pkl", 'rb') as f:
#         scaler = pickle.load(f)
    
#     save_path = f"./prob_resource/{ prob_config.phase_id}/{ prob_config.prob_id}/"
#     with open(save_path + "encoder.pkl", 'rb') as f:
#         encoder = pickle.load(f)

#     captured_x = pd.DataFrame()
#     list_cap_x = []

#     columns_to_keep = prob_config.categorical_cols + prob_config.numerical_cols
    
#     logging.disable(logging.CRITICAL)
#     for file_path in glob.glob(path + "/*.parquet"):
        
#         captured_data = pd.read_parquet(file_path)
#         captured_data = captured_data[columns_to_keep]
#         encoded_data = transform_new_data(prob_config , captured_data, encoder=encoder)
#         new_data = preprocess_data(prob_config = prob_config, data = encoded_data, mode = 'deploy', deploy_scaler=scaler)
        
        
#         captured_x = pd.concat([captured_x, new_data])
#         list_cap_x.append(new_data)

#     print(f"done {path}")

#     return captured_x, list_cap_x

# prob_config = get_prob_config("phase-2", "prob-2")

# data_x = pd.read_parquet("/mnt/e/mlops-marathon/new_mlops/data_warehouse/processed_data/phase-2/prob-2/processed_data_x.parquet")

# data_x.info()

# captured_root = "/mnt/e/mlops-marathon/new_mlops/data_warehouse/captured_data/phase-2/prob-2"

# cap_0, list_cap_0 = read_total(prob_config, os.path.join(captured_root,"0"))
# # cap_1, list_cap_1 = read_total(prob_config, os.path.join(captured_root,"1"))
# cap_2, list_cap_2 = read_total(prob_config, os.path.join(captured_root,"3"))
# # cap_3, list_cap_3 = read_total(prob_config, os.path.join(captured_root,"3"))
# # cap_4, list_cap_4 = read_total(prob_config, os.path.join(captured_root,"4"))


# # sort dataframes by column 'A'
# df1 = cap_0.sort_values(by='feature1')
# df2 = cap_2.sort_values(by='feature1')

# # find common rows
# common_rows = pd.merge(df1[:100000], df2[:100000])

# # calculate percentage of common rows
# percentage = len(common_rows) / 100000 * 100

# print(f"{percentage:.2f}% of rows in df1 are also in df2")

# print(df1.info(), df2.info())


# rows_csv = []
# for data in list_cap_0:
#     print("cal drift for data: ", len(rows_csv))
#     columns = {}
#     # for col in data_x.columns:
#     #     # compute the Wasserstein distance
#     #     wasserstein_dist = wasserstein_distance(data_x[col], data[col])
#     #     # print("Wasserstein distance:", wasserstein_dist)

#     #     # perform the two-sample Kolmogorov-Smirnov test
#     #     statistic, pvalue = ks_2samp(data_x[col], data[col])
#     #     # print("KS statistic:", statistic)
#     #     # print("KS p-value:", pvalue)

#     #     columns[col] = f"{round(pvalue, 5)}"  # - {round(statistic, 5)} - {round(pvalue, 5)}"

#     wasserstein_dist = wasserstein_distance(data_x["feature19"], data["feature19"])
#     statistic, pvalue = ks_2samp(data_x["feature19"], data["feature19"])
    
#     columns["Wasserstein distance"] = f"{round(wasserstein_dist, 5)}"
#     columns["KS statistic"] = f"{round(statistic, 5)}"
#     columns["KS p-value"] = f"{round(pvalue, 5)}"  # - {round(statistic, 5)} - {round(pvalue, 5)}"

#     rows_csv.append(columns)





# df = pd.DataFrame(rows_csv)
# df.to_csv("output.csv", index=False)


import os
import pandas as pd

def split_parquet_file_with_pandas(parquet_file, output_folder, num_splits):
  """Splits a parquet file into num_splits parquet files using Pandas.

  Args:
    parquet_file: The path to the parquet file to split.
    output_folder: The path to the folder for saving the split files.
    num_splits: The number of splits to create.

  Returns:
    A list of the paths to the split parquet files.
  """
  os.makedirs(output_folder, exist_ok= True)

  df = pd.read_parquet(parquet_file)
  split_files = []
  for i in range(num_splits):
    split_file = os.path.join(output_folder, "split-" + str(i) + ".parquet")
    df_split = df.iloc[i::num_splits]
    df_split.to_parquet(split_file)
    print(len(df_split))
    split_files.append(split_file)

  return split_files


if __name__ == "__main__":
  parquet_file = "data_warehouse/raw_data/phase-3/prob-1/raw_train.parquet"
  output_folder = "load_test/phase-3/prob-1"

  num_splits = 95

  split_files = split_parquet_file_with_pandas(parquet_file, output_folder, num_splits)
  print("Split parquet file into", len(split_files), "files.")

  parquet_file = "data_warehouse/raw_data/phase-3/prob-2/raw_train.parquet"
  output_folder = "load_test/phase-3/prob-2"

  num_splits = 95

  split_files = split_parquet_file_with_pandas(parquet_file, output_folder, num_splits)
  print("Split parquet file into", len(split_files), "files.")



