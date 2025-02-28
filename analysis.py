import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
'''
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
# 6. Feature Selection 
##############################

# Compting VIF values on the scaled training predictors to check for mullticollinearity
# Compute VIF on the scaled training predictors
df_X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

def compute_vif(df):
    vif_df = pd.DataFrame()
    vif_df["Feature"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_df

vif_data = compute_vif(df_X_train_scaled)
print("Initial VIF values:")
print(vif_data)

# Set a threshold (e.g., 5). Iteratively remove the feature with highest VIF until all VIFs are below the threshold.
threshold = 5
iteration = 1
while vif_data['VIF'].max() > threshold:
    max_feature = vif_data.sort_values('VIF', ascending=False)['Feature'].iloc[0]
    max_vif = vif_data.sort_values('VIF', ascending=False)['VIF'].iloc[0]
    print(f"Iteration {iteration}: Removing feature '{max_feature}' with VIF = {max_vif:.2f}")
    
    # Remove the feature with highest VIF
    df_X_train_scaled = df_X_train_scaled.drop(columns=[max_feature])
    
    # Recompute VIF values
    vif_data = compute_vif(df_X_train_scaled)
    print(vif_data, "\n")
    iteration += 1

print("Final feature set after VIF-based removal:")
print(df_X_train_scaled.columns.tolist())

