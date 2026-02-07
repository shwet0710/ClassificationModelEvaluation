# ClassificationModelEvaluation

## a. Problem statement
- Manually analyzing census data to identify high-income individuals is inefficient for large-scale economic planning. Economic indicators such as education, occupation, and capital gains are complex and interrelated, making it difficult to set clear thresholds for income classification without advanced analytics.
- The main objective is to develop and compare six machine learning classification models that predict whether an individual's annual income exceeds 50,000 based on demographic and employment attributes.
- Build a robust binary classifier (>50K vs. <=50K) that handles categorical encoding, missing data imputation (handling '?' values), and feature scaling. The final solution is deployed via a Streamlit web application for real-time model evaluation and performance comparison.

## b. Dataset description
- The Adult Census Income dataset ("UCI Adult" dataset) is a widely used for binary classification. It is derived from the 1994 U.S. Census database and is used to predict an individual's socio-economic status based on demographic and professional variables.
- The dataset consists of 32561 records and a total of 14 features including both discrete(categorical) as well as continous(numerical) features.
- Categorical features:
  - Workclass: Employment type(Private, Self-emp, State-gov etc.)
  - Education: Highest level of education (Bachelors, Masters, HS-grad, etc.)
  - Marital-Status: Relationship status (Married-civ-spouse, Divorced, Never-married, etc.)
  - Occupation: Professional role (Exec-managerial, Tech-support, Sales, etc.)
  - Relationship: Wife, Husband, Own-child etc.
  - Race: Ethnic background (White, Black etc.)
  - Sex: Male/Female
  - Native-country: Country of origin (United states, India, Mexico etc.)
- Numerical features:
  - Capital-gain: gain from investment sales
  - Capital-loss: loss from investment sales
  - fnlwgt: A calculation by the Census Bureau representing the number of people the census believes that row represents.
  - Age: individual age(17 to 90 years).
  - Education-num: years of education completed
  - Hours-per-week: Average number of hours worked per week.
- Target variable:
  - Income: A binary variable indicating whether an individual's annual income exceeds 50,000 or 50K.
  - Classes: >50K (High income) , <=50K(low income)
- Data Quality and preprocessing:
  - Several columns (Workclass, Occupation, Native-Country) contain missing values represented by a "?" string. Our pipeline treats these as NaN and removes them to ensure model stability.
  - The raw data often contains leading spaces (e.g., " Private" instead of "Private"). Our shared logic uses .str.strip() to clean these.

## c. Models used evaluation metrics comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8700 | 0.9200 | 0.7200 | 0.8600 | 0.8700 | 0.8600 |
| **Decision Tree** | 0.8500 | 0.9000 | 0.6800 | 0.8400 | 0.8500 | 0.8400 |
| **kNN** | 0.8200 | 0.7800 | 0.6200 | 0.8100 | 0.8200 | 0.8100 |
| **Naive Bayes**| 0.8200 | 0.8800 | 0.6100 | 0.8000 | 0.8200 | 0.8100 |
| **Random Forest(Ensemble)** | 0.8100 | 0.8500 | 0.5900 | 0.8000 | 0.8100 | 0.8000 |
| **XGBoost(Ensemble)** | 0.7800 | 0.8200 | 0.5400 | 0.7700 | 0.7800 | 0.7700 |
  
https://classificationmodelevaluation-mlassignment2.streamlit.app/
