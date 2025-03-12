import os
import base64
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="ML Model Evaluator", layout="wide")
st.title("📊 ML Model Evaluator")
st.markdown(
    '<p style="color:red; font-size:18px; font-weight:bold;">'
    'This app evaluates multiple machine learning models on a given dataset assisting for classification problems only!'
    '</p>', unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

def process_data(file):
    """Load and preprocess data with validation."""
    df = pd.read_csv(file)
    df = df.dropna()
    
    if df.shape[0] < 5:
        st.error("Dataset too small (min 5 samples required)")
        return None, None
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = pd.get_dummies(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    class_counts = Counter(y)
    if len(class_counts) < 2:
        st.error("Need at least 2 classes for classification.")
        return None, None
    
    return X, y

def evaluate_models(X, y):
    """Evaluate ML models with cross-validation."""
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000), 
        {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}),
        ('KNN', KNeighborsClassifier(), 
        {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
        ('SVM', SVC(probability=True), 
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        ('Naive Bayes', GaussianNB(), {}),
        ('Decision Tree', DecisionTreeClassifier(), 
        {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}),
        ('Random Forest', RandomForestClassifier(), 
        {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}),
        ('Gradient Boosting', GradientBoostingClassifier(), 
        {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5]})
    ]
    
    results = []
    best_model = {'name': '', 'accuracy': 0}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    for name, model, params in models:
        try:
            grid = GridSearchCV(model, params, cv=5, n_jobs=-1, error_score='raise')
            grid.fit(X_train, y_train)
            best_model_tuned = grid.best_estimator_
            test_acc = best_model_tuned.score(X_test, y_test)
            cv_scores = cross_val_score(best_model_tuned, X_train, y_train, cv=5)
            
            model_results = {
                'model': name,
                'params': grid.best_params_,
                'test_acc': round(test_acc, 4),
                'cv_mean': round(np.mean(cv_scores), 4),
                'cv_scores': cv_scores.tolist()
            }
            
            results.append(model_results)
            
            if test_acc > best_model['accuracy']:
                best_model['name'] = name
                best_model['accuracy'] = test_acc
                
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
    
    return results, best_model

def create_plots(results):
    """Generate visualization plots."""
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='test_acc', y='model', data=df, ax=ax, palette='muted')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Models')
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=[res['cv_scores'] for res in results], ax=ax, palette='pastel')
    plt.xticks(ticks=range(len(results)), labels=[res['model'] for res in results], rotation=45)
    plt.title('Cross-Validation Scores Distribution')
    plt.xlabel('Models')
    plt.ylabel('CV Scores')
    st.pyplot(fig)

if uploaded_file:
    X, y = process_data(uploaded_file)
    if X is not None and y is not None:
        with st.spinner("Training models... Please wait."):
            results, best_model = evaluate_models(X, y)
        
        st.markdown("<h2 style='font-size:24px;'>Model Performance</h2>", unsafe_allow_html=True)
        st.write(pd.DataFrame(results))

        st.markdown("<h2 style='font-size:24px;'>Best Performing Model</h2>", unsafe_allow_html=True)
        st.write(f"**{best_model['name']}** with accuracy: **{best_model['accuracy']:.4f}**")
        
        st.markdown("<h2 style='font-size:24px;'>Visualizations</h2>", unsafe_allow_html=True)
        create_plots(results)
