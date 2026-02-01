#importing required libraries

import streamlit as smt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocesing import LabelEncoder
from sklearn.metrics import (
accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,matthews_corrcoef,confusion_matrix
)

#Setup of landing page

smt.set_page_config(page_title="Classification Models Evaluation", layout="wide")
smt.title("Classification Models Performance Evaluation")

#Preprocessing

def preprocess_data(data):
    #stripping whitespaces and removing '?'
    data = data.apply(lambda d: d.str.strip() if d.dtype=="object" else d)
    data.replace('?',np.nan,inplace=True)
    data.dropna(inplace=True)

    #handling target
    target = data.columns[-1];
    data.rename(columns={target: 'target'}, inplace=True)
    data['target']=data['target'].astype(str).str.replace('.','',regex=False)

    #doing encoding for categorical columns
    l = LabelEncoder()
    for d in data.columns:
        if data[d].dtype=="object":
            data[d] = l.fit_transform(data[d])

    return data

#url for test data from github
test_data_url = "https://raw.githubusercontent.com/shwet0710/ClassificationModelEvaluation/refs/heads/main/adult_test.csv"

#model selection dropdown logic
models = {
    "Logistic Regression": "LogisticRegression.pkl",
    "Decision Tree": "DecisionTree.pkl",
    "K Nearest Neighbor": "KNearestNeighbor.pkl",
    "Naive Bayes": "NaiveBayes.pkl",
    "Random Forest": "RandomForest.pkl",
    "XGBoost": "XGBoost.pkl"
}

model_opted = smt.sidebar.selectbox("Choose Model", list(model.keys()))

#final main logic

#loading test data
test_data = pd.read_csv(test_data_url)
smt.write("Test dataset")
smt.dataframe(test_data.head())

#download button for test data
smt.write("Click below to download the test dataset")
test_data_download = test_data.to_csv(index=False).encode('utf-8')
smt.download_button(
    label="Download test data",
    data = test_data_download,
    file_name = "adult_test.csv",
    mime="text/csv"
)

smt.write("        ")

if smt.button(f"Evaluate {model_opted}"):
    #preprocess data and segregate target and featurs on cleaned data
    test_data_cleaned = preprocess_data(test_data.copy())
    X_test = test_data_cleaned.drop('target',axis=1)
    y_actual = test_data_cleaned['target'].astype('int')

    #loading .pkl files from model folder
    model_pkl = f"model/{models[model_opted]}"
    m = joblib.load(model_pkl)

    #doing scaling for knn and logistic regression with the help of scaler.pkl
    if model_opted in ["Logistic Regression", "K Nearest Neighbor"]:
        sc = joblib.load('model/scaler.pkl')
        X_test = sc.transform(X_test)

    #doing predictions
    y_predicted = m.predict(X_test)
    y_probable = m.predict_proba(X_test)[:,1] if hasattr(m, "predict_proba") else y_predicted

    #finally displaying all the metrics calculated for the selected models
    smt.subheader(f"{model_opted} Evaluation Metrics")
    columns = smt.columns(6)

    eval_metrics = {
        "Accuracy": accuracy_score(y_actual, y_predicted),
        "AUC": roc_auc_score(y_actual, y_probable),
        "Precision": precision_score(y_actual, y_predicted, average='weighted'),
        "Recall": recall_score(y_actual, y_predicted, average='weighted'),
        "F1": f1_score(y_actual, y_predicted, average='weighted'),
        "MCC": matthews_corrcoef(y_actual, y_predicted)
    }

    for ind, (label,j) in enumerate(eval_metrics.items()):
        columns[ind].metric(label, f"{j: .4f}")

    smt.write("        ")

    #displaying confusion matrix
    smt.write("Confusion Matrix")
    cm = confusion_matrix(y_actual, y_predicted)

    #plotting visual for confusion matrix
    fix,ax = plt.subplots()
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax)
    smt.pyplot(fig)

