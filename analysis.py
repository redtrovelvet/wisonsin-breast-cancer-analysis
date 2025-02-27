import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression

'''
Importing the datasets
'''
df_train = pd.read_csv("hf://datasets/wwydmanski/wisconsin-breast-cancer/train.csv")
df_test = pd.read_csv("hf://datasets/wwydmanski/wisconsin-breast-cancer/test.csv")


'''
Data Cleaning
'''
# Removing unwanted columns (id) from the train & test datasets
df_train.drop(columns=["Unnamed: 0"],inplace=True)
df_test.drop(columns=["Unnamed: 0"],inplace=True)

# Renaming columns of the train & test datasets to be more descriptive
new_column_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst','diagnosis'
]
df_train.columns = new_column_names
df_test.columns = new_column_names

# Covnerting diagnosis labels to binary: B -> 0, M -> 1
df_train['diagnosis'] = df_train['diagnosis'].map({'B': 0, 'M': 1})
df_test['diagnosis'] = df_test['diagnosis'].map({'B': 0, 'M': 1})


'''
Data Exploration
'''
# Creating a DataFrame with predictors and response for exploration
df_explore = df_train.copy()

# Summary statistics for each feature grouped by diagnosis
group_stats = df_explore.groupby('diagnosis').describe().T
print(group_stats)

# Heatmap of the correlation matrix of all features 
corr_matrix = df_explore.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Heatmap of All Features")
plt.show()


'''
Preprocessing the train & test dataset
'''
# Separating the diagnosis column (response) from the train & test datasets
y_train = df_train['diagnosis']
y_test = df_test['diagnosis']

# Removing the diagnosis column from the train & test datasets
X_train = df_train.drop(columns=['diagnosis'])
X_test = df_test.drop(columns=['diagnosis'])

# Scaling the datasets since they contain continuous features
scaler = StandardScaler()

# Fitting the scaler on the train dataset and transforming it
X_train_scaled = scaler.fit_transform(X_train)

# Transforming the test dataset using the same scale as the train dataset to avoid data leakage
X_test_scaled = scaler.transform(X_test)


