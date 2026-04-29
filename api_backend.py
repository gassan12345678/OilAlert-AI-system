from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load assets
model = joblib.load('oil_degradation_rf_model.pkl.gz')
scaler = joblib.load('feature_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['map'], data['rpm'], data['thr'], data['tmp']]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    return jsonify({
        'oil_health': round(max(0, 100 - float(prediction)), 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)