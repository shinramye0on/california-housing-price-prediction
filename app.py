import numpy as np
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('california_housing_model_best_lgbm.joblib')

@app.route('/')
def home():
    return "API Model Prediksi Harga Rumah Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        features = [
            data['MedInc'],
            data['HouseAge'],
            data['AveRooms'],
            data['AveBedrms'],
            data['Population'],
            data['AveOccup'],
            data['Latitude'],
            data['Longitude']
        ]

        final_features = [np.array(features)]

        prediction = model.predict(final_features)

        output = prediction[0]

        return jsonify({'prediction': output})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)