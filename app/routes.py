from flask import Blueprint, request, render_template
import pandas as pd
import joblib
import os

print("✅ Loading routes...")  # This will confirm the file is being loaded

main = Blueprint('main', __name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'malnutrition_rf_model.pkl')
encoder_path = os.path.join(base_dir, '..', 'models', 'encoder.pkl')

model = joblib.load(model_path)
print("✅ Model loaded.")

encoder = joblib.load(encoder_path)
print("✅ Encoder loaded.")


@main.route('/', methods=['GET'])
def home():
    return render_template('form.html')


@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form input
        input_data = {
            'age_months': int(request.form['age_months']),
            'gender': request.form['gender'],
            'weight_kg': float(request.form['weight_kg']),
            'height_cm': float(request.form['height_cm']),
            'muac_cm': float(request.form['muac_cm']),
            'recent_illness': int(request.form['recent_illness']),
            'immunization_status': request.form['immunization_status']
        }

        print("📨 Received input:", input_data)

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Encode categorical variables
        encoded = encoder.transform(df)

        # Make prediction
        result = model.predict(encoded)[0]
        print("✅ Prediction result:", result)

        prediction_text = "Child is at risk of malnutrition" if result == 1 else "Child is NOT at risk"

        return render_template('form.html', prediction=prediction_text)

    except Exception as e:
        print("❌ Error:", str(e))
        return f"Error during prediction: {str(e)}"



