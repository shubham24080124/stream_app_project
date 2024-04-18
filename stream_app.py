import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from lime.lime_tabular import LimeTabularExplainer
import shap

# Load Data
@st.cache
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/scada_data.csv')
    status_df = pd.read_csv('/content/drive/MyDrive/status_data.csv')
    fault_df = pd.read_csv('/content/drive/MyDrive/fault_data.csv')
    return df, status_df, fault_df

df, status_df, fault_df = load_data()

# EDA
st.subheader('Exploratory Data Analysis (EDA)')
st.write("SCADA Data:")
st.write(df.head())
st.write("Status Data:")
st.write(status_df.head())
st.write("Fault Data:")
st.write(fault_df.head())

# Time Series Analysis
st.subheader('Time Series Analysis')
monthly_counts = fault_df.resample('M', on='DateTime').Fault.count()
monthly_counts.index = monthly_counts.index.strftime('%b %Y')
st.bar_chart(monthly_counts)

# Combine SCADA and Fault Data
st.subheader('Combine SCADA and Fault Data')
df_combine = df.merge(fault_df, on='Time', how='outer')
df_combine['Fault'] = df_combine['Fault'].replace(np.nan, 'NF')
st.write("Combined Data:")
st.write(df_combine.head())

# Data Preparation
st.subheader('Data Preparation')
lb = LabelEncoder()
df_combine['Fault'] = lb.fit_transform(df_combine['Fault'])
X = df_combine.drop(columns=['Fault'])
y = df_combine['Fault']

# Feature Engineering
st.subheader('Feature Engineering')
rf_regressor = RandomForestRegressor()
rfe = RFE(estimator=rf_regressor, n_features_to_select=5)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
X_selected = X[selected_features]

# Model Building
st.subheader('Model Building')
models = [
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('xgb', XGBClassifier())
]
pipelines = []
for model_name, model in models:
    pipeline = Pipeline([
        ('preprocessor', StandardScaler()),
        (model_name, model)
    ])
    pipelines.append((model_name, pipeline))

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
for model_name, pipeline in pipelines:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'{model_name} accuracy: {accuracy:.2f}')

# Hyperparameter Tuning
st.subheader('Hyperparameter Tuning')
models = [
    ('rf', RandomForestClassifier(), {'rf__n_estimators': [100, 200, 300], 'rf__max_depth': [None, 5, 10]}),
    ('lr', LogisticRegression(), {'lr__C': [1], 'lr__solver': ['lbfgs']}),
    ('dt', DecisionTreeClassifier(), {'dt__max_depth': [None, 5, 10], 'dt__min_samples_split': [2, 5, 10]}),
    ('xgb', XGBClassifier(), {'xgb__max_depth': [3, 5, 7], 'xgb__learning_rate': [0.1, 0.01]})
]
pipelines = []
for model_name, model, params in models:
    pipeline = Pipeline([
        ('preprocessor', StandardScaler()),
        (model_name, model)
    ])
    pipeline_cv = GridSearchCV(pipeline, params, cv=5)
    pipelines.append((model_name, pipeline_cv))

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
for model_name, pipeline_cv in pipelines:
    pipeline_cv.fit(X_train, y_train)
    y_pred = pipeline_cv.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'{model_name} best params: {pipeline_cv.best_params_}')
    st.write(f'{model_name} accuracy: {accuracy:.2f}')
    st.write(classification_report(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

# Model Interpretability
st.subheader('Model Interpretability')

# LIME
st.subheader('LIME')
explainer = LimeTabularExplainer(X.values, feature_names=X.columns, class_names=np.unique(y))
sample_idx = 10  # Change as needed
sample = X.iloc[[sample_idx]]
model = LogisticRegression()  # Change as needed
model.fit(X, y)
explanation = explainer.explain_instance(sample.values[0], model.predict_proba)
explanation.show_in_notebook()

# SHAP
st.subheader('SHAP')
model = LogisticRegression()  # Change as needed
model.fit(X, y)
explainer = shap.Explainer(model, X)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
