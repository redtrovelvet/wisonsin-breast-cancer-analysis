import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression


##############################
# 1. Importing the Datasets
##############################
df_train = pd.read_csv("hf://datasets/wwydmanski/wisconsin-breast-cancer/train.csv")
df_test = pd.read_csv("hf://datasets/wwydmanski/wisconsin-breast-cancer/test.csv")


##############################
# 2. Data Cleaning
##############################
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


##############################
# 3. Data Exploration on Raw Data
##############################
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



##############################
# 4. Preprocessing: Splitting & Scaling
##############################
# Separating the response column (response) from the features of the train & test datasets
y_train = df_train['diagnosis']
y_test = df_test['diagnosis']
X_train = df_train.drop(columns=['diagnosis'])
X_test = df_test.drop(columns=['diagnosis'])

# Scaling the datasets since they contain continuous features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


##############################
# 5. Visualizing Scaled Data with Box Plots
##############################
# Convert the scaled training data back to a DataFrame and reattach diagnosis, mapping the numeric values to its string label
df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
df_train_scaled['diagnosis'] = y_train.values
df_train_scaled['diagnosis'] = df_train_scaled['diagnosis'].map({0: "Benign (0)", 1: "Malignant (1)"})

# Decide how many features to plot per figure
features = X_train.columns.tolist()  # 30 features
n_features = len(features)
group_size = 10
n_groups = math.ceil(n_features / group_size)

# Loop over each group of features and create box plots
for i in range(n_groups):
    start = i * group_size
    end = min(start + group_size, n_features)
    subset_features = features[start:end]
    
    # Extract only these features plus the (now string-labeled) diagnosis column
    df_subset = df_train_scaled[subset_features + ['diagnosis']]
    
    # Melt the subset for Seaborn
    df_melted_subset = df_subset.melt(id_vars='diagnosis', var_name='feature', value_name='value')
    
    # Plot the grouped box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='feature', y='value', hue='diagnosis', data=df_melted_subset)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Box Plots of Scaled Features {start+1} to {end} by Diagnosis")
    plt.tight_layout()
    plt.show()


##############################
# 6. Feature Selection Based on Correlation
##############################
# Reattach column names by converting to DataFrame
df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
df_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Define your drop list based on your correlation analysis
drop_list_cor = [
    'perimeter_mean', 'radius_mean', 'compactness_mean', 'concave_points_mean',
    'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst',
    'compactness_worst', 'concave_points_worst', 'compactness_se', 'concave_points_se',
    'texture_worst', 'area_worst'
]

# Drop the columns from the scaled DataFrames
X_train_scaled_final = df_train_scaled.drop(columns=drop_list_cor)
X_test_scaled_final = df_test_scaled.drop(columns=drop_list_cor)

# Now, X_train_scaled_final and X_test_scaled_final contain only the selected features
print("Remaining columns:", X_train_scaled_final.columns.tolist())
