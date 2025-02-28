import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import sklearn.model_selection as skm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score



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
# Converting the scaled training data back to a DataFrame and reattach diagnosis, mapping the numeric values to its string label
df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
df_train_scaled['diagnosis'] = y_train.values
df_train_scaled['diagnosis'] = df_train_scaled['diagnosis'].map({0: "Benign (0)", 1: "Malignant (1)"})

# Deciding how many features to plot per figure
features = X_train.columns.tolist()  # 30 features
n_features = len(features)
group_size = 10
n_groups = math.ceil(n_features / group_size)

# Looping over each group of features and create box plots
for i in range(n_groups):
    start = i * group_size
    end = min(start + group_size, n_features)
    subset_features = features[start:end]
    
    # Extracting only these features plus the (now string-labeled) diagnosis column
    df_subset = df_train_scaled[subset_features + ['diagnosis']]
    
    # Melting the subset for Seaborn
    df_melted_subset = df_subset.melt(id_vars='diagnosis', var_name='feature', value_name='value')
    
    # Plotting the grouped box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='feature', y='value', hue='diagnosis', data=df_melted_subset)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Box Plots of Scaled Features {start+1} to {end} by Diagnosis")
    plt.tight_layout()
    plt.show()


##############################
# 6. Feature Selection Based on Correlation
##############################
# Reattaching column names by converting to DataFrame
df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
df_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Defining drop list based on correlation analysis
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


##############################
# Prepare the Data for Statsmodels
##############################
# Add a constant column (for the intercept) to the final scaled training and test data
X_train_sm = sm.add_constant(X_train_scaled_final, prepend=False)
X_test_sm = sm.add_constant(X_test_scaled_final, prepend=False)

##############################
# 7. K-Fold Cross-Validation using Statsmodels on the Training Set
##############################

kf = skm.KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# Loop through the folds
for train_index, val_index in kf.split(X_train_sm):
    # Split the training data into a training fold and a validation fold
    X_train_cv = X_train_sm.iloc[train_index]
    y_train_cv = y_train.iloc[train_index]
    X_val_cv = X_train_sm.iloc[val_index]
    y_val_cv = y_train.iloc[val_index]
    
    # Fit the model on the current training fold
    model_cv = sm.GLM(y_train_cv, X_train_cv, family=sm.families.Binomial())
    result_cv = model_cv.fit()
    
    # Predict on the validation fold
    y_val_pred_prob = result_cv.predict(X_val_cv)
    y_val_pred_class = (y_val_pred_prob >= 0.5).astype(int)
    
    # Compute and store the accuracy for the current fold
    fold_acc = accuracy_score(y_val_cv, y_val_pred_class)
    cv_scores.append(fold_acc)
    
print("K-Fold Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))


##############################
# 8. Final Model Building: Logistic Regression using Statsmodels
##############################

# Fit the logistic regression model on the entire training set
glm_binom = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
results = glm_binom.fit()

# Print the detailed summary (coefficients, std errors, z-values, p-values, etc.)
print(results.summary())

# Predict probabilities on the test set using the final model
y_pred_prob = results.predict(X_test_sm)  # Predicted probability for class "1" (Malignant)
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# Evaluate the predictions on the test set
cm = confusion_matrix(y_test, y_pred_class)
acc = accuracy_score(y_test, y_pred_class)

print("Confusion Matrix (Test Set):\n", cm)
print("Test Accuracy:", acc)


##############################
# 9. Data Visualization: Coefficient Bar Plot
##############################
coef = results.params[1:]  # excluding constant
odds_ratios = np.exp(coef)
features = X_train_scaled_final.columns

plt.figure(figsize=(10,6))
plt.bar(features, odds_ratios)
plt.yscale('log')  # Switch to log scale
plt.xticks(rotation=30, ha='right')
plt.ylabel("Odds Ratio (log scale)")
plt.title("Odds Ratios of Predictors from Logistic Regression")
plt.show()


##############################
# 9. Data Visualization: ROC Curve and AUC
##############################
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
