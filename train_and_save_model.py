# The first trained model is a logical regression model.
# Titanic Survival Prediction - Logistic Regression Model
# Author: Quantum parser
# Date: 01.02.2026
# Description: First ML model implementation with proper MLOps practices

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

train = 'titanic_df/train.csv'

df = pd.read_csv(train)

print(f'\nErrors: ')

if df.empty:
    raise ValueError("Dataset is empty!")
if 'Survived' not in df.columns:
    raise ValueError("Target column 'Survived' not found!")

print('\n' + '=' * 50)
print(f'\nColumns in train df {list(df.columns)}')

features = ['Pclass', 'Sex', 'Age', 'Fare']

print('\n' + '=' * 50)
print(f'\nChosen columns for train: {list(features)}')

# Why this columns?
# Pclass - social-economy status
# Sex - rescue policy 
# Age - age priority
# Fare - ticket price (complements Pclass)

x = df[features].copy()
y = df['Survived'].copy()

# # Fill missing Age with median (better than mean for Age)
print(f'\nMissed data in features before processing')
print(x.isnull().sum())

age_median = x['Age'].median()
print(f'\nAge_median: {age_median:.1f}')

# Count how many missing values I am filling
missing_age_count = x['Age'].isnull().sum()
print(f'\nFilling {missing_age_count} missing Age values with median')

x['Age'] = x['Age'].fillna(age_median)

# Also check and fill Fare if needed (though Titanic train.csv usually has no missing Fare)
if x['Fare'].isnull().any():
    fare_median = x["Fare"].median()
    missing_fare_count = x['Fare'].isnull().sum()
    print('\n')
    print(f'Filling {missing_fare_count} missing Fare values with median: {fare_median:.2f}')
    x['Fare'] = x['Fare'].fillna(fare_median)

print(f'\nMissed data in features after processing')
print(x.isnull().sum())

# Coding Sex (male = 0, female = 1)
x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})

print('\n' + '=' * 50)
print('\nSplitting data into training and test sets:')
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'  \nTraining set: {x_train.shape[0]} samples')
print(f'  \nTest set: {x_test.shape[0]} samples')

# Creating scaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print('\n' + '=' * 50)
print('\nScaling applied: ')
print(f'Before scaling (first 3 train lines):\n {x_train.iloc[:3]}')
print(f'After scaling (first 3 train lines):\n {x_train_scaled[:3]}')

# Train the Logistic Regression model
print('\n' + '=' * 50)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(x_train_scaled, y_train)
print(f'\nModel trained successfully on scaled data')

# Saving model
print('\n' + '=' * 50)
print('\nSaving model with joblib')
dump(model, 'titanic_model.joblib')
print('\nModel saved as titanic_model.joblib')

# Saving scaler
print('Saving scaler with joblib')
dump(scaler, 'titanic_scaler.joblib')
print('Scaler saved as titanic_scaler.joblib')

# Also saving features for checking (if necessary)
print('\nSaving features')
dump(features, 'titanic_features.joblib')
print('Featues saved as titanic_features.joblib')

# Message about saving
print('\n' + '=' * 50)
print('\nAll files saved successfully')
print('Files created: ')
print('1. titanic_model.joblib - trained model')
print('2. titanic_scaler.joblib - fitted scaler')
print('3. titanic_features.joblib - list of features used')

# Make predictions on scaled data and calculate accuracy
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print('\n' + '=' * 50)
print('\nRESULTS:')
print(f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

# Checks goal achivement
if accuracy >= 0.78:
    print(f'Goal achieved (78-80%)')
else:
    print(f'Below target (goal: 78-80%)')

print('\n' + '=' * 50)
print("\nPredictions excemples: ")
for i in range(15):
    print(f'Sample {i+1}: Actual={y_test.iloc[i]}, Predicted:{y_pred[i]}')