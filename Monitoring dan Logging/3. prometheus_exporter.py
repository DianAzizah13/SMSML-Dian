from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, Gauge
import time
import mlflow.pyfunc
import psutil

print("[DEBUG] Running serve_model.py from:", __file__)

# Load model
model = mlflow.pyfunc.load_model("file:///D:/Laskar AI/crop_model")

app = Flask(__name__)

# === Prometheus Metrics ===
REQUEST_COUNT = Counter('model_inference_requests_total', 'Total inference requests')
REQUEST_LATENCY = Histogram('model_inference_latency_seconds', 'Inference latency in seconds')
ERROR_COUNT = Counter('model_inference_errors_total', 'Total inference errors')
SUCCESS_COUNT = Counter('model_inference_success_total', 'Total successful predictions')
INPUT_SIZE = Histogram('model_input_size_bytes', 'Size of input in bytes')
CPU_USAGE = Gauge('model_cpu_usage_percent', 'CPU usage percent')
MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage in bytes')
ACTIVE_DURATION = Gauge('model_active_duration_seconds', 'Duration since model loaded')
PREDICTION_CLASS_COUNT = Counter('model_prediction_class_count', 'Prediction class distribution', ['label'])
MODEL_LOADED_TIMESTAMP = Gauge('model_loaded_timestamp', 'Model loaded timestamp in UNIX seconds')

# Set waktu awal model dimuat
MODEL_LOADED_TIMESTAMP.set_to_current_time()
START_TIME = time.time()

@app.route("/", methods=["GET"])
def home():
    print("[DEBUG] Home route hit")
    return "Welcome to the Crop Prediction API"

@app.route("/predict", methods=["POST"])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc()
    try:
        input_data = request.get_json()
        print("[DEBUG] Original Input:", input_data)

        model_input = input_data
        print("[DEBUG] Converted Input for Model:", model_input)

        prediction = model.predict(model_input)
        print("[DEBUG] Prediction:", prediction)

        # Metrics tracking
        input_size = len(str(input_data).encode('utf-8'))
        INPUT_SIZE.observe(input_size)

        process = psutil.Process()
        CPU_USAGE.set(process.cpu_percent(interval=0.1))
        MEMORY_USAGE.set(process.memory_info().rss)
        ACTIVE_DURATION.set(time.time() - START_TIME)

        SUCCESS_COUNT.inc()
        for label in prediction.tolist():
            PREDICTION_CLASS_COUNT.labels(label=str(label)).inc()

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("[ERROR]", e)
        ERROR_COUNT.inc()
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}

if __name__ == "__main__":
    print("[INFO] Starting Flask server...")
    app.run(port=5001)
