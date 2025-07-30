import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 1000

# Simulate features
age_months = np.random.randint(6, 60, size=n)
gender = np.random.choice(['Male', 'Female'], size=n)
weight_kg = np.round(np.random.normal(12, 2.5, size=n), 1).clip(5, 25)
height_cm = np.round(np.random.normal(85, 10, size=n), 1).clip(50, 120)
muac_cm = np.round(np.random.normal(13.5, 1.5, size=n), 1).clip(9, 18)
recent_illness = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
immunization_status = np.random.choice(['Complete', 'Partial', 'None'], size=n, p=[0.6, 0.3, 0.1])

# Define risk based on MUAC and weight thresholds
def classify(muac, weight):
    if muac < 11.5 or weight < 7:
        return 'Malnourished'
    elif muac < 13 or weight < 10:
        return 'At Risk'
    else:
        return 'Normal'

# Apply classification
risk_label = [classify(mu, w) for mu, w in zip(muac_cm, weight_kg)]

# Create DataFrame
df = pd.DataFrame({
    'age_months': age_months,
    'gender': gender,
    'weight_kg': weight_kg,
    'height_cm': height_cm,
    'muac_cm': muac_cm,
    'recent_illness': recent_illness,
    'immunization_status': immunization_status,
    'nutrition_status': risk_label
})

# Save to CSV
df.to_csv('simulated_child_malnutrition_data.csv', index=False)

print("✅ Simulated dataset saved as 'simulated_child_malnutrition_data.csv'")
