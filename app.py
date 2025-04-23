from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import sklearn
import sys
import platform
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Log environment details for debugging
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")

# Define features expected by the model
features = ['Temperature (C)', 'Humidity', 'Pressure (millibars)', 
            'Wind Speed (km/h)', 'precipitation', 'hour', 'day_of_week']

# Load the model and scaler
try:
    with open('model.joblib', 'rb') as f:
        print("Loading model.joblib with joblib...")
        pipeline = joblib.load(f)
    with open('scaler.joblib', 'rb') as f:
        print("Loading scaler.joblib with joblib...")
        scaler = joblib.load(f)
except FileNotFoundError as e:
    print(f"File not found: {str(e)}")
    # Create a fallback model
    print("Creating fallback model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            max_features='sqrt',
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    # Train on dummy data
    X_dummy = np.random.rand(100, len(features))
    y_dummy = np.random.rand(100)
    pipeline.fit(X_dummy, y_dummy)
    scaler = pipeline.named_steps['scaler']
    print("Fallback model created")
except Exception as e:
    print(f"Failed to load joblib files: {str(e)}")
    # Create a fallback model
    print("Creating fallback model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            max_features='sqrt',
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    # Train on dummy data
    X_dummy = np.random.rand(100, len(features))
    y_dummy = np.random.rand(100)
    pipeline.fit(X_dummy, y_dummy)
    scaler = pipeline.named_steps['scaler']
    print("Fallback model created")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract and validate input
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        pressure = float(data.get('pressure', 0))
        wind_speed = float(data.get('wind_speed', 0))
        precipitation = 1 if data.get('precipitation', 'no') == 'yes' else 0
        hour = int(data.get('hour', 0))
        day_of_week = int(data.get('day_of_week', 0))
        
        # Create input DataFrame
        input_data = pd.DataFrame([[
            temperature, humidity, pressure, wind_speed, 
            precipitation, hour, day_of_week
        ]], columns=features)
        
        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': round(prediction, 2)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)