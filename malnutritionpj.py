import pandas as pd
import numpy as np

data=pd.read_csv('simulated_child_malnutrition_data.csv')
data = data.dropna()

# Encode the gender column
data = pd.get_dummies(data, columns=['gender','immunization_status'], drop_first=True)

print(data.head())
print(data.isnull().sum())
print(data.describe())
# The above code reads the simulated dataset and prints the first few rows, checks for missing values, and provides a summary of the dataset.


from sklearn.model_selection import train_test_split
x=data.drop('nutrition_status',axis=1)
y=data['nutrition_status']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# The above code splits the dataset into training and testing sets, trains a Random Forest classifier, and evaluates its performance using classification metrics.
# Example: Predict for a new child
# Suppose the columns after encoding are: age, height, weight, gender_male, immunization_status_partial

new_data = pd.DataFrame({
    'age_months': [44],
    'weight_kg': [18],
    'height_cm': [110],
    'muac_cm': [14.5],
    'recent_illness': [0],
    'gender_Male': [1],                
    'immunization_status_Partial': [0],
    
})

prediction = model.predict(new_data)
print("Predicted nutrition status:", prediction[0])

print("\nEnter details for prediction:")

age_months = int(input("Age (months): "))
weight_kg = float(input("Weight (kg): "))
height_cm = float(input("Height (cm): "))
muac_cm = float(input("MUAC (cm): "))
recent_illness = int(input("Recent illness (1 for Yes, 0 for No): "))
gender = input("Gender (male/female): ").strip().lower()
immunization_status = input("Immunization status (complete/partial): ").strip().lower()

# Encode user input to match one-hot encoding
gender_Male = 1 if gender == "male" else 0
immunization_status_Partial = 1 if immunization_status == "partial" else 0

user_data = pd.DataFrame({
    'age_months': [age_months],
    'weight_kg': [weight_kg],
    'height_cm': [height_cm],
    'muac_cm': [muac_cm],
    'recent_illness': [recent_illness],
    'gender_Male': [gender_Male],
    'immunization_status_Partial': [immunization_status_Partial]
})

prediction = model.predict(user_data)
print("Your child is :", prediction[0])