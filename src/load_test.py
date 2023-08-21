import pyarrow.parquet as pq
import requests
import concurrent.futures
import time
import numpy as np
from tqdm import tqdm
import glob
import json
from threading import Thread

# Make sure to have the required libraries installed
# !pip install pyarrow requests

def read_parquet(file):
    data = pq.read_table(file).to_pandas()
    return data

def load_list_data(root_path, num_requests):
    id = 1
    list_data = []
    list_label = {}
    for file_path in tqdm(glob.glob(root_path), ncols=100, desc ="Loading...", unit ="file"):
            
            test_data = read_parquet(file_path)
            if "label" in test_data.columns:
              features = test_data.drop(["label"], axis =1)
              list_label[str(id)] = test_data["label"].to_numpy()
            else:
              features = test_data
              list_label[str(id)] = np.zeros(len(features))
               
            columns =  features.columns.to_list()
            list_data.append( {"id": str(id), "rows": features.to_numpy().tolist(), "columns": columns})
            
            if id == num_requests:
                break
            else: id+=1
    return list_data, list_label

def send_data_to_server(data, url):
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url, data=data_json, headers=headers)

    return response


def load_test(list_data, url, num_requests):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for data in tqdm(list_data, ncols=100, desc ="Requesting...", unit ="file"):
            future = executor.submit(send_data_to_server, data, url)
            futures.append(future)

        response_times = []
        drift = 0
        total_infe_time = 0
        predictions = {}
        for future in futures:
            response = future.result()
            response_times.append(response.elapsed.total_seconds())
            json_res = response.json()

            if json_res["drift"] == 1:
                 drift +=1
            
            predictions[json_res["id"]] = json_res["predictions"]
            # total_infe_time += json_res["inference time"]
        p95_response_time = sorted(response_times)[int(0.95 * num_requests)]

    return p95_response_time, drift, total_infe_time, predictions

def calculate_accuracy_for_each_id(truth_labels, predictions):
  """Calculates the average accuracy of the predictions for each id.

  Args:
    truth_labels: A dictionary of truth labels, where the keys are the ids and the
      values are the lists of truth labels.
    predictions: A dictionary of predictions, where the keys are the ids and the
      values are the lists of predictions.

  Returns:
    A dictionary of average accuracies, where the keys are the ids and the values
    are the average accuracies.
  """

  average_accuracies = {}

  for id in truth_labels:
    if id in predictions:
      correct_predictions = 0
      total_predictions = 0

      for truth_label, prediction in zip(truth_labels[id], predictions[id]):
        if truth_label == prediction:
          correct_predictions += 1
        total_predictions += 1

      if total_predictions > 0:
        average_accuracies[id] = correct_predictions / total_predictions
      else:
        average_accuracies[id] = 0.0

  return average_accuracies


def test_load(url, root_path, num_requests):

  list_data, list_label = load_list_data(root_path, num_requests)

  start_time = time.time()
  p95_response_time, drift, total_infe_time, predictions = load_test(list_data, url, num_requests)

  print(list_label.keys(),  f" - {predictions.keys()}")
  # print(list_label.values(),  f" - {predictions.values()}")

  print(f"Request take: {round((time.time() - start_time),2)} seconds")
  print(f"The P95 response time is {round(p95_response_time,2)} seconds.")
  print(f"The number of drift is {drift}.")
  # print(f"The total inference time is {round(total_infe_time/1000,2)} seconds.")
  # print(f"The AVG inference time is {total_infe_time/num_requests} ms.")
  average_accuracies = calculate_accuracy_for_each_id(list_label, predictions)
  print("Average accuracies:")
  avg = []
  for id, accuracy in average_accuracies.items():
    print(f"  {id}: {accuracy}")
    avg.append(accuracy)
  print(f"AVG: {np.mean(avg)}")

#take list file 
url = 'http://localhost:8000/phase-3/prob-1/predict'
root_path = "load_test/phase-3/prob-1/*.parquet"
# root_path = "data_warehouse/captured_data/phase-2//prob-1/13/*.parquet"
num_requests = 100
test_load(url, root_path, num_requests)

url = 'http://localhost:8000/phase-3/prob-2/predict'
root_path = "load_test/phase-3/prob-2/*.parquet"
# root_path = "data_warehouse/captured_data/phase-2/prob-2/13/*.parquet"
num_requests = 100
test_load(url, root_path, num_requests)
      


        


