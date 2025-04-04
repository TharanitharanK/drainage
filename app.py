import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from flask import Flask, jsonify
app = Flask(__name__)
# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Sample Dataset (For Training)
data = {
    "gas": [300, 450, 500, 600, 700, 800, 400, 550, 620, 780],
    "water_speed": [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 0.7, 1.1, 1.6, 1.9],
    "water_level": [10, 15, 20, 25, 30, 35, 12, 18, 22, 28],
    "gps_location": [1, 2, 3, 4, 5, 6, 2, 3, 4, 5],  # Encoded locations
    "condition": ["Stable", "Stable", "Caution", "Caution", "Critical", "Critical", "Stable", "Caution", "Critical", "Critical"]
}

df = pd.DataFrame(data)
df["condition"] = df["condition"].map({"Stable": 0, "Caution": 1, "Critical": 2})

# Train-Test Split
X = df.drop(columns=["condition"])
y = df["condition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to Analyze Sensor Values
def analyze_sensor_values(gas, water_speed, water_level):
    conditions = {}

    # Gas Sensor
    if gas <= 500:
        conditions["Gas"] = f"✅ Stable ({gas} ppm)"
    elif 500 < gas <= 700:
        conditions["Gas"] = f"⚠️ Caution ({gas} ppm)"
    else:
        conditions["Gas"] = f"🚨 Critical ({gas} ppm - Potential Hazard)"

    # Water Speed
    if water_speed <= 1.0:
        conditions["Water Speed"] = f"✅ Stable ({water_speed} m/s)"
    elif 1.0 < water_speed <= 1.5:
        conditions["Water Speed"] = f"⚠️ Caution ({water_speed} m/s)"
    else:
        conditions["Water Speed"] = f"🚨 Critical ({water_speed} m/s - Risk of Flooding)"

    # Water Level
    if water_level <= 20:
        conditions["Water Level"] = f"✅ Stable ({water_level} cm)"
    elif 20 < water_level <= 30:
        conditions["Water Level"] = f"⚠️ Caution ({water_level} cm)"
    else:
        conditions["Water Level"] = f"🚨 Critical ({water_level} cm - Risk of Overflow)"

    return conditions

# Function to Fetch Sensor Data from Firebase
def get_sensor_data():
    doc_ref = db.collection("sensor data").document("sample values")
    doc = doc_ref.get()
    
    if doc.exists:
        data = doc.to_dict()
        return data
    else:
        print("No sensor data found!")
        return None

# Predict Function
def predict_drainage():
    print("\n🌊 **Drainage Prediction System** 🌊")

    # Fetch sensor values from Firebase
    sensor_data = get_sensor_data()
    if not sensor_data:
        return

    gas = sensor_data["gas"]
    water_speed = sensor_data["water_speed"]
    water_level = sensor_data["water_level"]
    gps_location = sensor_data["gps_location"]

    print(f"📡 Fetching Sensor Data...\n"
          f"🔹 Gas: {gas} ppm\n"
          f"🔹 GPS Location: {gps_location}\n"
          f"🔹 Water Level: {water_level} cm\n"
          f"🔹 Water Speed: {water_speed} m/s")
    print("--------------------------------------")

    # Prepare Data for Prediction
    feature_names = ["gas", "water_speed", "water_level", "gps_location"]
    input_data = pd.DataFrame([[gas, water_speed, water_level, gps_location]], columns=feature_names)

    # Make Prediction
    prediction = model.predict(input_data)[0]
    drainage_conditions = {0: "✅ Stable", 1: "⚠️ Caution", 2: "🚨 Critical"}

    # Get Sensor Analysis
    sensor_analysis = analyze_sensor_values(gas, water_speed, water_level)

    # Print Results
    print("\n📢 **Prediction Results** 📢\n")
    print(f"🔹 **Overall Drainage Condition:** {drainage_conditions[prediction]}\n")

    print("📊 **Sensor-wise Condition Report:**")
    for sensor, status in sensor_analysis.items():
        print(f"🔹 {sensor}: {status}")

    # Final Advice
    if prediction == 0:
        print("\n🟢 System is running smoothly. No action needed.")
    elif prediction == 1:
        print("\n🟡 Caution! Monitor the drainage system closely.")
    else:
        print("\n🔴 Critical Alert! Immediate maintenance required!")

# Run Prediction Periodically
while True:
    predict_drainage()
    time.sleep(60)  
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
