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


