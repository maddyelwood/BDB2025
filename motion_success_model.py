#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import RFE
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample

# load the unified data from all lane detected csv files
data = pd.read_csv('unified_data.csv')

# Define target variable for success (EPA, yardsGained, success metric...)
#target = data['expectedPointsAdded']
#target = data['binary_success']

features = [
    'absolute_change_area_snap',
    'percentage_change_area_snap',
    'width_of_lane_front_SNAP',
    'area_polygon_SNAP',
    'coverageType',
    'motion'
]


X = data[features]
y = data['binary_success']

X, y = X.dropna(), y[X.dropna().index]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define evaluation function
def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('ROC AUC Score:', roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# intialize log reg model
logistic = LogisticRegression(solver='liblinear', random_state=42)


# define hyperparameter grid for logistic regression
param_grid_lr = {
    'penalty': ['l1', 'l2'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300],
    'class_weight': [None, 'balanced']    
}

logistic = LogisticRegression(random_state=42)

grid_lr = GridSearchCV(estimator=logistic, param_grid=param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)

# print best hyperparamters
print('best hyperparameters for log reg:', grid_lr.best_params_)
print('best cross-val score:', grid_lr.best_score_)

# use best model found by grid search
best_lr_model = grid_lr.best_estimator_

eval_model(best_lr_model, X_test_scaled, y_test)

# adjust class weights manually for log reg
class_weights = {0:1, 1:3}
logistic_weighted_manual = LogisticRegression(solver='liblinear', class_weight=class_weights, random_state=42)
logistic_weighted_manual.fit(X_train_scaled, y_train)
eval_model(logistic_weighted_manual, X_test_scaled, y_test)

def eval_model_custom_threshold(y_pred_adjusted, y_test):
    print('confusion matrix:\n', confusion_matrix(y_test, y_pred_adjusted))
    print('classification report:\n', classification_report(y_test, y_pred_adjusted))
    print('ROC AUC Score:', roc_auc_score(y_test, y_pred_adjusted))
    
# predict probs and adjust threshold
y_pred_prob = best_lr_model.predict_proba(X_test_scaled)[:,1]
threshold = 0.3
y_pred_adjusted = (y_pred_prob >= threshold).astype(int)

# evaluate w adjusted threshold
eval_model_custom_threshold(y_pred_adjusted, y_test)

# alternative: random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]    
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)

print('best RF params:', grid_rf.best_params_)
print('best RF score:', grid_rf.best_score_)

best_rf_model = grid_rf.best_estimator_
eval_model(best_rf_model, X_test_scaled, y_test)

















