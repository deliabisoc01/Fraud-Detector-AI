import sys
import os

# Add the parent directory (C:\AI\FraudDetector) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from data_preparation import load_and_preprocess_data
import joblib

def logistic_regression_model():
    X, y = load_and_preprocess_data()

    model_lr = LogisticRegression(multi_class='ovr', random_state=42, max_iter=200)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_lr, X, y, cv=skf, scoring='f1_weighted')
    print("\nCross-Validation F1 Scores (Logistic Regression):", cv_scores)
    print("Mean F1 Score (Logistic Regression):", np.mean(cv_scores))

    model_lr.fit(X, y)
    y_pred = model_lr.predict(X)
    report_lr = classification_report(y, y_pred, target_names=['Honest', 'Suspicious', 'Fraud'], labels=[0, 1, 2], zero_division=0)
    print("\nFinal Model Evaluation (Logistic Regression):")
    print(report_lr)

    # Save the trained logistic regression model
    joblib.dump(model_lr, 'models/logistic_regression_model.pkl')
    print("Logistic Regression model saved as 'models/logistic_regression_model.pkl'.")

if __name__ == '__main__':
    logistic_regression_model()
