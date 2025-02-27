import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression

'''
Importing the datasets
'''
df_train = pd.read_csv("hf://datasets/wwydmanski/wisconsin-breast-cancer/train.csv")
df_test = pd.read_csv("hf://datasets/wwydmanski/wisconsin-breast-cancer/test.csv")

'''
Preprocessing the test dataset
'''
# Dropping the first column (id column) from the test dataset
df_test.drop(columns=["Unnamed: 0"],inplace=True)

# Separating the diagnosis column (response) from the test dataset 
y_test = df_test['y']

# Covnerting diagnosis labels to binary: B -> 0, M -> 1
y_test = y_test.map({'B': 0, 'M': 1})

# Dropping the diagnosis column from the test dataset
X_test = df_test.drop(columns=['y'])

# Renaming columns to be descriptive
new_column_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]
X_test.columns = new_column_names
