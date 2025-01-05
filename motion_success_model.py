#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# what features did caio use?
# add running back speed
# pff TE blocking v passing ability
# previous year receiving yards
# rushing yards per carry for the te motion plays

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

'''
# create RFE model and select top k features
rfe = RFE(estimator=logistic)

# we can test different numbers of features to select
n_features_to_select = [2, 3, 4, 5, 6]

# create grid search for feature selection
param_grid_rfe = {'n_features_to_select': n_features_to_select}

grid_rfe = GridSearchCV(estimator=rfe, param_grid=param_grid_rfe, cv=5, scoring='accuracy')
grid_rfe.fit(X_train_scaled, y_train)

print('best number of features for RFE:', grid_rfe.best_params_['n_features_to_select'])
print('best RFE score:', grid_rfe.best_score_)

# get best features
best_features_rfe = X_train.columns[grid_rfe.best_estimator_.support_]
print('best features selected by RFE:', best_features_rfe)
'''

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

## trying to balance data for better model representation
'''
# attempt 1: SMOTE
print('Oversampling w/ SMOTE:')
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
logistic_smote = logistic.fit(X_train_smote, y_train_smote)
eval_model(logistic_smote, X_test, y_test)

# attempt 2: undersampling
print('\n Undersampling:')
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resampmle(X_train, y_train)
logistic_under = logistic.fit(X_train_under, y_train_under)
eval_model(logistic_under, X_test, y_test)
'''
'''
# attempt 2.5: manual balancing
print('Manually Balanced Data:')
majority = data[data['binary_success'] == 0]
minority = data[data['binary_success'] == 1]

majority_downsampled = resample(majority, 
                                replace=False,
                                n_samples=len(minority),
                                random_state=42)

balanced_data = pd.concat([minority, majority_downsampled])
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Split data into train and test sets

target_b = balanced_data['binary_success']
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(balanced_data[features], target_b, test_size=0.3, random_state=42, stratify=target_b)

# standardize features
X_train_b_scaled = scaler.fit_transform(X_train_b)
X_test_b_scaled = scaler.transform(X_test_b)

X_train_b_scaled = pd.DataFrame(X_train_b_scaled).dropna()
y_train_b = y_train_b[X_train_b_scaled.index]

X_test_b_scaled = pd.DataFrame(X_test_b_scaled).dropna()
y_test_b = y_test_b[X_test_b_scaled.index]

X_train_b_scaled = X_train_b_scaled.reset_index(drop=True)
y_train_b = y_train_b.reset_index(drop=True)

X_test_b_scaled = X_test_b_scaled.reset_index(drop=True)
y_test_b = y_test_b.reset_index(drop=True)

logistic_b = logistic.fit(X_train_b_scaled, y_train_b)
eval_model(logistic_b, X_test_b_scaled, y_test_b)
'''

# attempt 3: class weight adjustment
#print('\n Class Weight Adjustment:')
#logistic_weighted = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
#logistic_weighted.fit(X_train_scaled, y_train)
#eval_model(logistic_weighted, X_test_scaled, y_test)


'''
# logistic regression with grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']    
}

log_reg = LogisticRegression()
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# best logistic regression model
best_log_reg = grid_search.best_estimator_

# evaluate logistic regression
y_pred = best_log_reg.predict(X_test_scaled)
y_pred_prob = best_log_reg.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression Performance:")
print(f"Best Parameters: {grid_search.best_params_}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob)}")
'''


'''
# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Linear Regression:')
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Display feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
print(feature_importance.sort_values(by='Coefficient', ascending=False))

# perform ridge regression with cross-validation
ridge= RidgeCV(alphas=np.logspace(-6,6,13), cv=5)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)

print('Ridge Regression:')
print(f'Best Alpha: {ridge.alpha_}')
print(f'Mean Squared Error: {mean_squared_error(y_test, ridge_preds)}')
print(f'R^2 Score: {r2_score(y_test, ridge_preds)}')

# coefficients
ridge_coefs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': ridge.coef_
})

print('\nRidge Coefficients:')
print(ridge_coefs)

# Perform Lasso Regression with cross-validation
lasso = LassoCV(alphas=np.logspace(-6, 6, 13), cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)

print("\nLasso Regression")
print(f"Best Alpha: {lasso.alpha_}")
print(f"Mean Squared Error: {mean_squared_error(y_test, lasso_preds)}")
print(f"R^2 Score: {r2_score(y_test, lasso_preds)}")

# Coefficients
lasso_coefs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
})
print("\nLasso Coefficients:")
print(lasso_coefs)
'''


