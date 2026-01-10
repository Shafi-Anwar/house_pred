
import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import traceback
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'housing.pkl')
print(f"Base directory: {BASE_DIR}")
print(f"Looking for model at: {MODEL_PATH}")
model = None
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
    'Latitude', 'Longitude', 'Rooms_per_house', 'BedroomRatio',
    'PopulationPerHouse', 'DistToCoast', 'IncomePerOccupant'
]
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        

        print("\n Current directory contents:")
        for item in os.listdir(BASE_DIR):
            if os.path.isdir(item):
                print(f"  {item}/")
            else:
                print(f"   {item}")

        model_dir = os.path.join(BASE_DIR, 'model')
        if os.path.exists(model_dir):
            print(f"\n Contents of 'model' directory:")
            for item in os.listdir(model_dir):
                print(f"   {item}")
        else:
            print(f"\n'model' directory doesn't exist. Creating it...")
            os.makedirs(model_dir, exist_ok=True)
        
        return False
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Features expected: {len(feature_names)}")
        return True
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        traceback.print_exc()
        return False
def calculate_dist_to_coast(lat, lon):
    coast_lat, coast_lon = 36.8, -121.9  
    return np.sqrt((lat - coast_lat)**2 + (lon - coast_lon)**2)
def create_features(input_data):
    MedInc = float(input_data['MedInc'])
    HouseAge = float(input_data['HouseAge'])
    AveRooms = float(input_data['AveRooms'])
    AveBedrms = float(input_data['AveBedrms'])
    Population = float(input_data['Population'])
    AveOccup = float(input_data['AveOccup'])
    Latitude = float(input_data['Latitude'])
    Longitude = float(input_data['Longitude'])
    Rooms_per_house = AveRooms / AveOccup if AveOccup != 0 else 0
    BedroomRatio = AveBedrms / AveRooms if AveRooms != 0 else 0
    PopulationPerHouse = Population / AveOccup if AveOccup != 0 else 0
    DistToCoast = calculate_dist_to_coast(Latitude, Longitude)
    IncomePerOccupant = MedInc / AveOccup if AveOccup != 0 else 0
    return np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
        Latitude, Longitude, Rooms_per_house, BedroomRatio,
        PopulationPerHouse, DistToCoast, IncomePerOccupant
    ]])


load_model()
@app.route('/')
def home():
    return jsonify({
        'api': 'California Housing Price Predictor',
        'status': 'running',
        'model_loaded': model is not None,
        'features': len(feature_names),
        'endpoints': ['GET /', 'GET /health', 'POST /predict']
    })  
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model else 'unhealthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.json
        print("RECEIVED DATA:", data)  
        
        required = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                   'Population', 'AveOccup', 'Latitude', 'Longitude']

        missing = [field for field in required if field not in data]
        if missing:
            print(f"MISSING FIELDS: {missing}")  
            return jsonify({
                'error': 'Missing fields',
                'missing': missing,
                'received_data': data  
            }), 400
        

        features = create_features(data)

        log_pred = model.predict(features)[0]
        price = np.expm1(log_pred)  
        
        price_usd = price * 100000
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'value': round(float(price), 4),
                'us_dollars': round(float(price_usd), 2),
                'formatted': f"${price_usd:,.2f}"
            },
            'features_used': 13,
            'model_type': 'RandomForestRegressor'
        })

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500
if __name__ == '__main__':
    print("\nStarting Flask API...")
    print(f" Model status: {' LOADED' if model else 'NOT LOADED'}")
    print(f" Server: http://127.0.0.1:5000")
    print("=" * 60)
    
    if model is None:
        print("\n QUICK FIX:")
        print("1. Run this in your Jupyter notebook:")
        print("   joblib.dump(model_log, 'model/housing.pkl')")
        print("2. Restart this Flask app")
    
    app.run(host='0.0.0.0', port=5000, debug=True)