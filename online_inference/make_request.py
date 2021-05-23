import pandas as pd
import requests


DATA_FOR_PREDICTS_PATH = 'data/heart_for_predict.csv'
REQUEST_URL = 'http://127.0.0.1:8000/predict'

if __name__ == "__main__":
    data = pd.read_csv(DATA_FOR_PREDICTS_PATH).drop('target', axis=1)
    data.insert(0, 'id', range(len(data)))
    print(data.head())
    request_data = data.to_json(orient='records')
    print(f'Sample: \n{request_data}')
    response = requests.post(REQUEST_URL, data=request_data)
    print(f'Request status code {response.status_code}')
    print(f'First 10 request items: \n{response.json()}')
