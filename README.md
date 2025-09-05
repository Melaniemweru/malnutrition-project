# Child Malnutrition Risk Prediction

Malnutrition is one of the leading causes of preventable child illness and mortality, yet many cases go undetected until it is too late for effective intervention. This project explores how data and machine learning can be applied to predict the risk of child malnutrition, providing a way to support earlier and more informed decision-making.  

From a technical perspective, the project demonstrates an end-to-end machine learning workflow:  
- **Exploratory Data Analysis (EDA)** and feature engineering in Jupyter Notebooks  
- **Model development and evaluation** using supervised learning techniques  
- **Deployment** of the trained model as a Flask web application for real-time predictions  

By integrating data analysis, predictive modeling, and deployment, the project highlights the ability to take a machine learning solution from concept to production-ready application.  


## Project Motivation / Background  

Child malnutrition remains a critical global health issue, with an estimated **149 million children under five stunted, 45 million wasted, and 37 million overweight in 2022** (WHO, UNICEF, PAHO). Nearly **half of all deaths among children under five** are linked to undernutrition, making early detection and intervention essential.  

We chose this project because it addresses a socially meaningful challenge while allowing us to apply and strengthen  skills in **data science, machine learning, and deployment**. By building a predictive model and deploying it with Flask, we explored how AI/ML solutions can contribute to data-driven decision-making in healthcare.  


## Exploratory Data Analysis (EDA)

The dataset contained **910 child health records** with both numerical and categorical features, including age, weight, height, mid-upper arm circumference (MUAC), immunization status, and recent illness history. The target variable was **`nutrition_status`**, categorized as *Normal*, *At Risk*, and *Malnourished*.  

### Data Cleaning
- Removed missing values (910 clean rows retained).  
- Encoded categorical features (`gender`, `immunization_status`) using one-hot encoding.  
- Saved processed data in `data/processed/malnutrition_clean.csv`.  

### Target Variable Distribution
The dataset showed an **imbalanced distribution** across nutrition status classes:  
- Normal: 471 samples  
- At Risk: 343 samples  
- Malnourished: 96 samples  

 *Insert bar chart of nutrition status distribution here*  

### Feature Insights
- **Age (months)** ranged from 6 to 59, with a mean of ~33 months.  
- **Weight** ranged from 5.0 kg to 19.9 kg; **Height** ranged from 55.7 cm to 117.6 cm.  
- **MUAC** values ranged between 9.0 and 17.8 cm, with lower values associated with malnourished cases.  

  <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/69c2ccf9-b631-41be-952f-056e9aa749b6" />


### Outlier Detection
Boxplots revealed a few outliers in weight, height, and MUAC, but most values were within expected biological ranges.  

 *Insert boxplot grid of numeric features here*  <img width="1489" height="664" alt="image" src="https://github.com/user-attachments/assets/b4e24758-8dbf-43d8-b750-d6c42e625dc1" />


### Correlation Analysis
- Strong positive correlation between **weight and height**.  
- MUAC showed meaningful correlation with both weight and nutrition status.  
- Recent illness showed weaker correlation with anthropometric measures.  

 *Insert heatmap of correlation matrix here*  <img width="1085" height="790" alt="image" src="https://github.com/user-attachments/assets/eed7c85f-ff2a-43b7-8c74-768d0d76b10e" />


---

This analysis guided feature selection and highlighted class imbalance as an important consideration during model training.  

## Model Training

The predictive model was built and evaluated in `notebooks/model_training.ipynb`. This workflow covered data preprocessing, handling class imbalance, model training, and saving reusable artifacts for deployment.

### Data Preparation
- Loaded cleaned dataset (`malnutrition_clean.csv`).
- One-hot encoded categorical variables: **gender** and **immunization_status**.
- Encoded the target variable (`nutrition_status`: Normal, At Risk, Malnourished) with `LabelEncoder`.
- Split data into **training (80%)** and **test (20%)** sets.

### Baseline Model
- Initial model: **Random Forest Classifier** (`scikit-learn`).
- Achieved **96% accuracy** on the test set.
- Evaluation:
  - Perfect classification for *Normal* cases.
  - Strong performance for *At Risk* children (Precision = 0.91, Recall = 1.00).
  - Malnourished children showed lower recall, indicating the class imbalance issue.
<img width="501" height="393" alt="image" src="https://github.com/user-attachments/assets/ce3dae28-c5b7-4253-996c-aa4a0885baec" />



Classification Report:
              precision    recall  f1-score   support

     At Risk       0.91      1.00      0.95        71
Malnourished       1.00      0.71      0.83        24
      Normal       1.00      1.00      1.00        87

    accuracy                           0.96       182
   macro avg       0.97      0.90      0.93       182
weighted avg       0.96      0.96      0.96       182

### Handling Class Imbalance
The dataset was imbalanced (many more *Normal* cases than *Malnourished*).  
To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data, balancing all classes before retraining.

- Balanced training set: 384 samples per class.
- Retrained Random Forest achieved:
  - High **precision and recall** across all three classes.
  - Improved detection of *Malnourished* cases, critical in healthcare use cases.

  precision    recall  f1-score   support

           0      0.972     0.972     0.972        71
           1      0.917     0.917     0.917        24
           2      1.000     1.000     1.000        87

    accuracy                          0.978       182
   macro avg      0.963     0.963     0.963       182
weighted avg      0.978     0.978     0.978       182

### Model Persistence
To support deployment:
- Trained model saved as **`models/malnutrition_rf_model.pkl`**.
- Label encoder saved as **`models/encoder.pkl`**.
- This allows consistent preprocessing and predictions without retraining.

### Making Predictions on New Data
The trained model can predict nutrition status from raw health inputs (age, gender, weight, height, MUAC, recent illness, immunization status).  
Pipeline ensures:
1. One-hot encoding matches training format.
2. Missing expected features are added with zeros.
3. Input features are reordered consistently.



---



## Tech Stack

**Languages & Core Tools**
- Python (data analysis, model training, deployment)
- Jupyter Notebooks (exploratory analysis, feature engineering, visualization)

**Libraries for Data & ML**
- pandas, numpy (data preprocessing, manipulation)
- matplotlib, seaborn (EDA, visualizations)
- scikit-learn (machine learning models, evaluation, preprocessing)
- imbalanced-learn (SMOTE for class imbalance handling)
- joblib (model persistence)

**Deployment & Backend**
- Flask (web application framework for deployment)


**Version Control & Collaboration**
- Git & GitHub (project tracking, collaboration, documentation)


Child-Malnutrition-Risk-Prediction/
├── app/                 # Flask application code (app logic, routes, views)
├── data/                # Datasets
│   └── processed/       # Cleaned data for modeling
├── docker/              # Docker configuration
├── models/              # Saved artifacts (.pkl, encoders)
├── notebooks/           # Jupyter notebooks (EDA, model training)
├── src/                 # Reusable Python modules (preprocessing, utils)
├── .gitignore           # Git ignore rules
├── Procfile             # Process definition for deployment
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── run.py               # App entry point (starts Flask server)


# How to Run the Project

## Prerequisites (One-time Setup)

Please install the following **once** on your computer:

1. **Git**  
   Download: [https://git-scm.com/downloads](https://git-scm.com/downloads)  
   → During install, choose **“Git Bash Here”** option for easy right-click access.  

2. **Python 3.9+**  
   Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)  
   → During installation, check **“Add Python to PATH”**.  

---

##  Daily Workflow – Step-by-Step

Open Git Bash and run:  


git clone https://github.com/Melaniemweru/malnutrition-project.git
cd malnutrition-project
python -m venv venv
Activate the virtual environment:

On Windows:


source venv/Scripts/activate

On Mac/Linux:


source venv/bin/activate
Then install all required packages:


pip install -r requirements.txt
This installs Flask, Pandas, Scikit-learn, Gunicorn, and other dependencies.

Now run the Flask app locally:


python run.py
You’ll see something like:


 * Running on http://127.0.0.1:5000
Open a browser and go to  http://127.0.0.1:5000 to test the nutrition status prediction form.

Next time you want to re-access the project, just do:

cd malnutrition-project
source venv/Scripts/activate   # or venv/bin/activate
python run.py





