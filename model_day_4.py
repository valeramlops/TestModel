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
import warnings
warnings.filterwarnings('ignore')

train = 'titanic_df/train.csv'

df = pd.read_csv(train)

print(f'\n Checks: ')

if df.empty:
    raise ValueError("Dataset is empty!")
if 'Survived' not in df.columns:
    raise ValueError("Target column 'Survived' not found!")

print(f'\n')
print('=' * 50)
print(f'Columns in train df {list(df.columns)}')

features = ['Pclass', 'Sex', 'Age', 'Fare']

print('=' * 50)
print(f'\n Chosen columns for train: {list(features)}')

# Why this columns?
# Pclass - social-economy status
# Sex - rescue policy 
# Age - age priority
# Fare - ticket price (complements Pclass)

x = df[features].copy()
y = df['Survived'].copy()

# # Fill missing Age with median (better than mean for Age)
print(f'\n Missed data in features before processing')
print(x.isnull().sum())

age_median = x['Age'].median()
print(f'\n Age_median: {age_median:.1f}')

# Count how many missing values I am filling
missing_age_count = x['Age'].isnull().sum()
print(f'\n Filling {missing_age_count} missing Age values with median')

x['Age'] = x['Age'].fillna(age_median)

# Also check and fill Fare if needed (though Titanic train.csv usually has no missing Fare)
if x['Fare'].isnull().any():
    fare_median = x["Fare"].median()
    missing_fare_count = x['Fare'].isnull().sum()
    print('\n')
    print(f'Filling {missing_fare_count} missing Fare values with median: {fare_median:.2f}')
    x['Fare'] = x['Fare'].fillna(fare_median)

print(f'\n Missed data in features after processing')
print(x.isnull().sum())

# Coding Sex (male = 0, female = 1)
x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})

print('\n')
print(f'='*50)
print('\n Splitting data into training and test sets:')
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'  \n Training set: {x_train.shape[0]} samples')
print(f'  \n Test set: {x_test.shape[0]} samples')