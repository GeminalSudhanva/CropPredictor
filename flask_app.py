import json
import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from models import CropPredictor, FertilizerPredictor
from pymongo import MongoClient
import requests

app = Flask(__name__)

# Configure MongoDB
app.config['MONGO_URI'] = 'mongodb://localhost:27017/farmconnect_db'  # Replace with your MongoDB URI
mongo_client = MongoClient(app.config['MONGO_URI'])
db = mongo_client.get_database() # Get the default database

# Collections
recommendations_collection = db.recommendations
soil_data_collection = db.soil_data
irrigation_schedules_collection = db.irrigation_schedules

# Configure logging
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

import requests
import traceback
import pandas as pd

app.config['SECRET_KEY'] = 'your_secret_key'
# Read WeatherAPI key from environment variable to avoid hardcoding
app.config['WEATHERAPI_KEY'] = os.getenv('WEATHERAPI_KEY', '')

crop_predictor = CropPredictor(
    model_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/crop_recommendation_model.pkl',
    scaler_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/scaler.pkl',
    le_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/label_encoder.pkl'
)
app.logger.info(f"CropPredictor Feature Means: {crop_predictor.feature_means}")
app.logger.info(f"CropPredictor Feature Standard Deviations: {crop_predictor.feature_stds}")
fertilizer_predictor = FertilizerPredictor(
    model_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/fertilizer_model.pkl',
    scaler_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/fertilizer_scaler.pkl',
    le_soil_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/fertilizer_le_soil.pkl',
    le_crop_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/fertilizer_le_crop.pkl',
    le_fertilizer_url='https://raw.githubusercontent.com/GeminalSudhanva/CropPredictor/main/fertilizer_le_fertilizer.pkl'
)
app.logger.info(f"FertilizerPredictor Feature Means: {fertilizer_predictor.feature_means}")
app.logger.info(f"FertilizerPredictor Feature Standard Deviations: {fertilizer_predictor.feature_stds}")

def get_weather_data(city):
    api_key = app.config['WEATHERAPI_KEY']
    if not api_key:
        app.logger.error("WEATHERAPI_KEY is not set. Set the environment variable and restart.")
        return None

    base_url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": city, "aqi": "no"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json() if response.headers.get('Content-Type', '').startswith('application/json') else {}

        if response.status_code == 200 and 'current' in data:
            current = data["current"]
            temperature = current.get("temp_c")
            humidity = current.get("humidity")
            return {"temperature": temperature, "humidity": humidity}

        # WeatherAPI error structure: {"error": {"code": xxx, "message": "..."}}
        error_obj = data.get("error", {}) if isinstance(data, dict) else {}
        error_message = error_obj.get("message") or data.get('message', 'Unknown error')
        if response.status_code in (400, 401) and error_message and "api key" in error_message.lower():
            app.logger.error("WeatherAPI authentication failed: Invalid API key.")
        else:
            app.logger.error(f"Error fetching weather data ({response.status_code}): {error_message}")
        return None
    except requests.Timeout:
        app.logger.error("WeatherAPI request timed out.")
        return None
    except requests.RequestException as e:
        app.logger.error(f"WeatherAPI request error: {e}")
        return None

def load_crop_details():
    try:
        with open('crop_details.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: crop_details.json not found. Crop descriptions and images will be limited.")
        return {}

crop_details = load_crop_details()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop_recommendation')
def crop_recommendation():
    feature_info = {
        'means': crop_predictor.feature_means.tolist(),
        'stds': crop_predictor.feature_stds.tolist(),
        'feature_names': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    }
    return render_template('crop_recommendation.html', feature_info=feature_info)

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        print(f"Received crop prediction data: {data}")
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        print(f"Extracted crop features: N={N}, P={P}, K={K}, Temp={temperature}, Humidity={humidity}, PH={ph}, Rainfall={rainfall}")
        
        predicted_crop = crop_predictor.predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        print(f"Predicted crop: {predicted_crop}")

        # Save the recommendation to MongoDB
        recommendations_collection.insert_one({
            "type": "crop",
            "N": N, "P": P, "K": K,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall,
            "predicted_crop": predicted_crop,
            "timestamp": datetime.now()
        })

        # Save soil data to MongoDB
        soil_data_collection.insert_one({
            "N": N, "P": P, "K": K,
            "ph": ph,
            "timestamp": datetime.now()
        })

        return jsonify({'predicted_crop': predicted_crop})
    except Exception as e:
        print(f"Error during crop prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    default_city = "London"  # You can make this configurable or user-inputted
    weather_data = get_weather_data(default_city)
    irrigation_schedule = irrigation_schedules_collection.find_one(sort=[("timestamp", -1)])

    # Fetch latest recommendation from MongoDB
    latest_recommendation = recommendations_collection.find_one(
        {"type": "crop"}, sort=[("timestamp", -1)]
    )
    last_inputs = soil_data_collection.find_one(sort=[("timestamp", -1)])

    try:
        return render_template('dashboard.html', weather_data=weather_data, irrigation_schedule=irrigation_schedule, last_recommendation=latest_recommendation, last_inputs=last_inputs)
    except Exception as e:
        print(f"Error rendering dashboard: {e}")
        traceback.print_exc()
        return "Error loading dashboard", 500

@app.route('/recommendation')
def recommendation():
    feature_info = {
        'means': crop_predictor.feature_means.tolist(),
        'stds': crop_predictor.feature_stds.tolist(),
        'feature_names': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    }
    return render_template('crop_recommendation.html', feature_info=feature_info)

@app.route('/soil', methods=['GET', 'POST'])
def soil():
    if request.method == 'POST':
        data = request.json
        soil_data_collection.insert_one({
            "ph": data['ph'],
            "N": data['N'],
            "P": data['P'],
            "K": data['K'],
            "timestamp": datetime.now()
        })
        return jsonify({"status": "success"})
    last_inputs = soil_data_collection.find_one(sort=[("timestamp", -1)])
    return render_template('soil.html', last_inputs=last_inputs)

@app.route('/irrigation', methods=['GET', 'POST'])
def irrigation():
    if request.method == 'POST':
        data = request.json
        try:
            irrigation_schedules_collection.insert_one({
                "start_time": data['start_time'],
                "end_time": data['end_time'],
                "duration_minutes": data['duration_minutes'],
                "frequency": data['frequency'],
                "timestamp": datetime.now()
            })
            return jsonify({"status": "success", "message": "Irrigation schedule saved successfully!"})
        except Exception as e:
            print(f"Error saving irrigation schedule: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # For GET request, fetch the last saved schedule if any
    last_schedule = irrigation_schedules_collection.find_one(sort=[("timestamp", -1)])
    return render_template('irrigation.html', last_schedule=last_schedule)

@app.route('/weather', methods=['GET', 'POST'])
def weather():
    city = request.form.get('city')
    weather_data = None
    error = None

    if request.method == 'POST' and city:
        api_key = app.config['WEATHERAPI_KEY']
        if not api_key:
            error = "WeatherAPI key not configured. Set WEATHERAPI_KEY environment variable."
        else:
            base_url = "https://api.weatherapi.com/v1/current.json"
            params = {"key": api_key, "q": city, "aqi": "no"}
            try:
                response = requests.get(base_url, params=params, timeout=10)
                data = response.json() if response.headers.get('Content-Type', '').startswith('application/json') else {}

                if response.status_code == 200 and 'current' in data:
                    weather_data = data
                else:
                    # WeatherAPI error object
                    error_obj = data.get("error", {}) if isinstance(data, dict) else {}
                    message = error_obj.get("message") or data.get('message', 'Unknown error')
                    error = f"Error fetching weather data ({response.status_code}): {message}"
            except requests.Timeout:
                error = "Weather service timeout. Please try again."
            except requests.RequestException as e:
                error = f"Weather service error: {e}"
    elif request.method == 'POST' and not city:
        error = "Please enter a city name."

    return render_template('weather.html', weather=weather_data, city=city, error=error)

@app.route('/results')
def results():
    crop = request.args.get('crop')
    fertilizer_reco = request.args.get('fertilizer_reco')
    return render_template('results.html', crop=crop, fertilizer_reco=fertilizer_reco)

@app.route('/api/unique_crops')
def unique_crops():
    crop_details = load_crop_details()
    unique_crop_names = sorted(list(crop_details.keys()))
    return jsonify(unique_crop_names)

@app.route('/fertilizer_recommendation')
def fertilizer_recommendation():
    feature_info = {
        'means': fertilizer_predictor.feature_means.tolist(),
        'stds': fertilizer_predictor.feature_stds.tolist(),
        'feature_names': ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']
    }
    return render_template('fertilizer_recommendation.html', feature_info=feature_info)

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.json
        print(f"Received fertilizer prediction data: {data}")
        temp = data['Temperature']
        moisture = data['Moisture']
        rainfall = data['Rainfall']
        ph = data['PH']
        N = data['Nitrogen']
        P = data['Phosphorous']
        K = data['Potassium']
        carbon = data['Carbon']
        soil_type = data['Soil Type']
        crop_type = data['Crop Type']

        print(f"Extracted features: Temp={temp}, Moisture={moisture}, Rainfall={rainfall}, PH={ph}, N={N}, P={P}, K={K}, Carbon={carbon}, Soil Type={soil_type}, Crop Type={crop_type}")

        predicted_fertilizer = fertilizer_predictor.predict_fertilizer(temp, moisture, rainfall, ph, N, P, K, carbon, soil_type, crop_type)
        print(f"Predicted fertilizer: {predicted_fertilizer}")

        # Save the fertilizer recommendation to MongoDB
        recommendations_collection.insert_one({
            "type": "fertilizer",
            "Temperature": temp, "Moisture": moisture, "Rainfall": rainfall,
            "PH": ph, "Nitrogen": N, "Phosphorous": P, "Potassium": K,
            "Carbon": carbon, "Soil Type": soil_type, "Crop Type": crop_type,
            "predicted_fertilizer": predicted_fertilizer,
            "timestamp": datetime.now()
        })

        # Save soil data to MongoDB
        soil_data_collection.insert_one({
            "N": N, "P": P, "K": K,
            "ph": ph,
            "timestamp": datetime.now()
        })

        return jsonify({'predicted_fertilizer': predicted_fertilizer})
    except Exception as e:
        print(f"Error during fertilizer prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)