import requests
import json
import time
import logging

# Setup logging ke file
logging.basicConfig(
    filename="crop_model_prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Endpoint REST API dari model Flask
API_ENDPOINT = "http://127.0.0.1:5001/predict"

# Payload input untuk model Crop Prediction
input_payload = {
    "0": 90.0,        # N
    "1": 42.0,        # P
    "2": 43.0,        # K
    "3": 20.8797,     # temperature
    "4": 82.0027,     # humidity
    "5": 6.5,         # ph
    "6": 202.9        # rainfall
}

headers = {"Content-Type": "application/json"}
json_data = json.dumps(input_payload)

def predict():
    start = time.time()

    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=json_data)
        duration = time.time() - start

        if response.ok:
            result = response.json()
            logging.info(f"Input: {input_payload} | Prediction: {result} | Time: {duration:.3f}s")
            print("Prediction:", result)
            print(f"Response Time: {duration:.3f} seconds")
        else:
            logging.error(f"Error {response.status_code}: {response.text}")
            print(f"Failed to get prediction. Status: {response.status_code}")
            print("Message:", response.text)

    except requests.exceptions.RequestException as e:
        logging.exception("Exception occurred during API call")
        print("Exception:", str(e))

if __name__ == "__main__":
    predict()
