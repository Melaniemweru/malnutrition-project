from flask import Blueprint, request, render_template
import pandas as pd
import joblib
import os

print("[INFO] Loading routes...")

# Create Flask blueprint
main = Blueprint('main', __name__)

# Define paths
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'malnutrition_rf_model.pkl')

# Load model
model = joblib.load(model_path)
print("[INFO] Model loaded.")

# Define the expected feature columns used during training
expected_features = [
    'age_months', 'weight_kg', 'height_cm', 'muac_cm', 'recent_illness',
    'gender_Male', 'immunization_status_Partial'
]

# Home route (displays the form)
@main.route('/', methods=['GET'])
def home():
    return render_template('form.html')

# Predict route
@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate input fields
        required_fields = ['age_months', 'gender', 'weight_kg', 'height_cm', 'muac_cm', 'recent_illness', 'immunization_status']
        for field in required_fields:
            if not request.form.get(field):
                return render_template('form.html', prediction="Please fill all fields.")

        # Manual one-hot encoding for categorical variables
        input_data = {
            'age_months': int(request.form['age_months']),
            'weight_kg': float(request.form['weight_kg']),
            'height_cm': float(request.form['height_cm']),
            'muac_cm': float(request.form['muac_cm']),
            'recent_illness': int(request.form['recent_illness']),
            'gender_Male': 1 if request.form['gender'].strip().lower() == 'male' else 0,
            'immunization_status_Partial': 1 if request.form['immunization_status'].strip().lower() == 'partial' else 0
        }

        # Convert to DataFrame and align with expected features
        input_df = pd.DataFrame([input_data])
        input_encoded = input_df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_encoded)[0]

        # Map prediction to human-readable label
        label_map = {
            0: "Severely Malnourished",
            1: "Malnourished",
            2: "Normal"
        }
        prediction_label = label_map.get(int(prediction), "Unknown")

        return render_template("form.html", prediction=f"Predicted Nutrition Status: {prediction_label}")

    except Exception as e:
        return render_template("form.html", prediction=f"Error occurred: {str(e)}")










