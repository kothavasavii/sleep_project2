from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pickle
import warnings
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and feature names
try:
    model = joblib.load('model.pkl')
    with open('model_features.pkl', 'rb') as f:
        model_features = pickle.load(f)
except FileNotFoundError as e:
    raise Exception(f"Model files not found: {str(e)}")

# Mappings
occupation_mapping = {
    'Software Engineer': 0, 'Doctor': 1, 'Sales Representative': 2, 'Teacher': 3,
    'Nurse': 4, 'Engineer': 5, 'Accountant': 6, 'Scientist': 7, 'Lawyer': 8,
    'Salesperson': 9, 'Manager': 10, 'Student': 11, 'Athlete': 12, 'Artist': 13
}
gender_mapping = {'Female': 0, 'Male': 1}

# Helper functions
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    return "Overweight"

def predict_sleep_disorder(score, duration):
    if score < 4 or (duration < 5):
        return "🚨 High risk of sleep disorder. Consult a specialist."
    elif score <= 6:
        return "⚠️ Moderate risk. Consider improving sleep hygiene."
    return "✅ Low risk. Maintain healthy sleep habits."

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prediction", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("prediction.html")
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    try:
        # Validate and extract inputs
        required_fields = [
            "age", "sleepDuration",
            "steps", "weight", "height", "systolic", "diastolic", "heartRate",
            "gender", "occupation"
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Convert and validate inputs
        try:
            age = int(data["age"])
            gender = gender_mapping[data["gender"]]
            occupation = occupation_mapping[data["occupation"]]
            sleep_duration = float(data["sleepDuration"])
         
            steps = int(data["steps"])
            weight = float(data["weight"])
            height = float(data["height"])
            systolic = int(data["systolic"])
            diastolic = int(data["diastolic"])
            heart_rate = int(data["heartRate"])
        except (ValueError, KeyError) as e:
            return jsonify({"error": f"Invalid input value: {str(e)}"}), 400

        # Calculate derived features
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        
        # Create input DataFrame
        input_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Systolic BP': systolic,
            'Diastolic BP': diastolic,
            'Heart Rate': heart_rate,
            'Daily Steps': steps   
        }
        
        # Ensure we only include features the model expects
        # filtered_input = {k: input_data[k] for k in model_features if k in input_data}
        # user_df = pd.DataFrame([filtered_input], columns=model_features)
        # Create input DataFrame with the correct order and column names
        user_df = pd.DataFrame([[input_data[feature] for feature in model_features]], columns=model_features)


        # Debug prints to help identify the mismatch
        print("Model expects these features (in order):", model.feature_names_in_)
        print("Input DataFrame columns:", user_df.columns.tolist())
        print("Input DataFrame values:", user_df.values)

        # Make prediction
        sleep_score = model.predict(user_df)[0]
        
        # Convert numpy types to native Python types
        sleep_score = convert_numpy_types(sleep_score)
        bmi = convert_numpy_types(bmi)

        response = {
            "bmi": round(bmi, 2),
            "bmi_category": bmi_category,
            "sleep_score": round(sleep_score, 1),
            "disorder_risk": predict_sleep_disorder(sleep_duration, sleep_duration)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
    