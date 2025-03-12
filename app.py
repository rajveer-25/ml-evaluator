import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_data(file_path):
    """Load and preprocess data with enhanced validation"""
    df = pd.read_csv(file_path)
    df = df.dropna()
    
    if df.shape[0] < 5:
        raise ValueError("Dataset too small (min 5 samples required)")
        
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X = pd.get_dummies(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    class_counts = Counter(y)
    if len(class_counts) < 2:
        raise ValueError("Need at least 2 classes for classification")
        
    min_samples = min(class_counts.values())
    if min_samples < 2:
        raise ValueError(f"Class imbalance detected. Minimum samples per class: {min_samples} (need at least 2)")
    
    return X, y

def evaluate_models(X, y):
    """Evaluate machine learning models with adaptive cross-validation"""
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
    
    class_counts = Counter(y)
    min_samples = min(class_counts.values())
    n_splits = min(5, min_samples)
    
    # Determine cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=n_splits) if min_samples >= 2 else KFold(n_splits=n_splits)
    
    stratify = y if min_samples >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    for name, model, params in models:
        try:
            grid = GridSearchCV(
                model, 
                params, 
                cv=cv_strategy,
                n_jobs=-1,
                error_score='raise'
            )
            grid.fit(X_train, y_train)
            
            best_model_tuned = grid.best_estimator_
            test_acc = best_model_tuned.score(X_test, y_test)
            cv_scores = cross_val_score(best_model_tuned, X_train, y_train, cv=cv_strategy)
            
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
            print(f"Error training {name}: {str(e)}")
            continue
    
    if not results:
        raise RuntimeError("No models could be trained successfully")
    
    return results, best_model

# Rest of the code remains unchanged (create_plots, plot_to_base64, routes, etc.)

def create_plots(results):
    """Generate visualization plots with error handling"""
    try:
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(results)
        ax = sns.barplot(x='test_acc', y='model', data=df, palette='muted')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Accuracy Score')
        plt.ylabel('Models')
        accuracy_img = plot_to_base64(ax.get_figure())
        plt.close()
        
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=[res['cv_scores'] for res in results], palette='pastel')
        plt.xticks(ticks=range(len(results)), labels=[res['model'] for res in results])
        plt.title('Cross-Validation Scores Distribution')
        plt.xlabel('Models')
        plt.ylabel('CV Scores')
        plt.xticks(rotation=45)
        cv_img = plot_to_base64(ax.get_figure())
        plt.close()
        
        return accuracy_img, cv_img
        
    except Exception as e:
        print(f"Plotting error: {str(e)}")
        return None, None

def plot_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                X, y = process_data(file_path)
                results, best_model = evaluate_models(X, y)
                accuracy_img, cv_img = create_plots(results)
                os.remove(file_path)
                
                if not accuracy_img or not cv_img:
                    raise RuntimeError("Failed to generate visualizations")
                
                return render_template('index.html',
                                    results=results,
                                    best_model=best_model,
                                    accuracy_img=accuracy_img,
                                    cv_img=cv_img)
            
            except Exception as e:
                os.remove(file_path)
                error_msg = f"{str(e)}"
                if "n_splits" in str(e):
                    error_msg += "<br>Solution: Ensure your dataset has at least 3 samples per class"
                return render_template('index.html', error=error_msg)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)