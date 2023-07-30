import pyarrow.parquet as pq
import requests
from tqdm import tqdm
import glob
import json
from threading import Thread

# Make sure to have the required libraries installed
# !pip install pyarrow requests

def read_parquet(file):
    data = pq.read_table(file).to_pandas()
    return data

def send_data_to_server(data, url):
    headers = {'Content-type': 'application/json'}
    data_json = json.dumps(data)
    response = requests.post(url, data=data_json, headers=headers)

    return response

#take list file 

url = 'http://localhost:5040/phase-2/prob-2/predict'
root_path = "/mnt/e/mlops-marathon/new_mlops/data_warehouse/captured_data/phase-2/prob-2/0/*.parquet"

len = 0
drift = 0
for file_path in tqdm(glob.glob(root_path), ncols=100, desc ="Loading...", unit ="file"):

    # for i in range(100):
        
        id = len+1
        len += 1
        features = read_parquet(file_path)
        columns =  features.columns.to_list()
        
        data = {"id": str(id), "rows": features.to_numpy().tolist(), "columns": columns}
        # print(data)
        # break

        # t = Thread(target=send_data_to_server, args= (data, url))
        # t.start()
        
        response = send_data_to_server(data, url)

        # Check the response status code and content
        print(response.status_code)
        print(response.json()["id"])
        if response.json()["drift"] == 1:
             drift +=1

        if id == 10:
             break
print("num drift: ", drift)
        


