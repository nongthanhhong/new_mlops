import pyarrow.parquet as pq
import requests
import concurrent.futures
import time
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
    for file_path in tqdm(glob.glob(root_path), ncols=100, desc ="Loading...", unit ="file"):
            
            features = read_parquet(file_path)
            columns =  features.columns.to_list()
            
            list_data.append( {"id": str(id), "rows": features.to_numpy().tolist(), "columns": columns})
            
            if id == num_requests:
                break
            else: id+=1
    return list_data

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
        for future in futures:
            response = future.result()
            response_times.append(response.elapsed.total_seconds())
            json_res = response.json()

            if json_res["drift"] == 1:
                 drift +=1
            total_infe_time += json_res["inference time"]

        p95_response_time = sorted(response_times)[int(0.95 * num_requests)]
        return p95_response_time, drift, total_infe_time


#take list file 

url = 'http://localhost:8000/phase-2/prob-2/predict'
root_path = "data_warehouse/captured_data/phase-2/prob-2/13/*.parquet"

num_requests = 20

list_data = load_list_data(root_path, num_requests)

start_time = time.time()
p95_response_time, drift, total_infe_time = load_test(list_data, url, num_requests)

print(f"Request take: {round((time.time() - start_time),2)} seconds")
print(f"The P95 response time is {round(p95_response_time,2)} seconds.")
print(f"The number of drift is {drift}.")
print(f"The total inference time is {round(total_infe_time/1000,2)} seconds.")
print(f"The AVG inference time is {total_infe_time/num_requests} ms.")


        


