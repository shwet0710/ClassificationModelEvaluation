# ClassificationModelEvaluation

## a. Problem statement
- Manually analyzing census data to identify high-income individuals is inefficient for large-scale economic planning. Economic indicators such as education, occupation, and capital gains are complex and interrelated, making it difficult to set clear thresholds for income classification without advanced analytics.
- The main objective is to develop and compare six machine learning classification models that predict whether an individual's annual income exceeds 50,000 based on demographic and employment attributes.
- Build a robust binary classifier (>50K vs. <=50K) that handles categorical encoding, missing data imputation (handling '?' values), and feature scaling. The final solution is deployed via a Streamlit web application for real-time model evaluation and performance comparison.

## b. Dataset description
- The Adult Census Income dataset ("UCI Adult" dataset) is a widely used for binary classification. It is derived from the 1994 U.S. Census database and is used to predict an individual's socio-economic status based on demographic and professional variables.
- The dataset consists of a total of 14 features including both discrete(categorical) as well as continous(numerical) features.
- Categorical features:
  Workclass: Employment type(Private, Self-emp, State-gov etc.)
  Education: Highest level of education (Bachelors, Masters, HS-grad, etc.)
  Marital-Status: Relationship status (Married-civ-spouse, Divorced, Never-married, etc.)
  Occupation: Professional role (Exec-managerial, Tech-support, Sales, etc.)
  Relationship: Wife, Husband, Own-child etc.
  Race: Ethnic background (White, Black etc.)
  Sex: Male/Female
  Native-country: Country of origin (United states, India, Mexico etc.)
- Numerical features:
  Capital-gain: gain from investment sales
  Capital-loss: loss from investment sales
  Age: individual age(17 to 90 years).
  Education-num: years of education completed
  Hours-per-week: Average number of hours worked per week.
- Target variable:
  Income: A binary variable indicating whether an individual's annual income exceeds 50,000 or 50K.
  Classes: >50K (High income) , <=50K(low income)

## c. Models used evaluation metrics comparison table

  
https://classificationmodelevaluation-mlassignment2.streamlit.app/
