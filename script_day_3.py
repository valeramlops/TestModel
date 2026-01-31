# Script to replace missing values in the age column with average age values
# I wrote this for self-study.

import pandas as pd
import numpy as np

df = pd.read_csv('titanic_df/test.csv')

average_age = round(float(df['Age'].mean()),1)
print(average_age)

was_missing = df['Age'].isnull()

df['Age'] = df['Age'].fillna(average_age)

df['Age_info'] = np.where(
    was_missing,
    f"{average_age} (avg)",
    df['Age'].astype(str)
)