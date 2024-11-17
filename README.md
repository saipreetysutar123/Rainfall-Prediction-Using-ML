import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import xgboost as xgb
import os
print("Current working directory:", os.getcwd())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# Load the CSV file (update the path if the file is not in the same directory)
df = pd.read_csv('Rainfall.csv')
print(df.head())

# Checking the shape of the dataframe
print(df.shape)

# Summary of data information and statistics
print(df.info())
print(df.describe().T)

# Check for missing values
print(df.isnull().sum())

# Strip whitespace from column names if any
df.rename(columns=lambda x: x.strip(), inplace=True)
print("Updated columns:", df.columns)

# Fill missing values with column means
for col in df.columns:
    if df[col].isnull().sum() > 0:
        val = df[col].mean()
        df[col].fillna(val, inplace=True)

# Check again for missing values
print("Total missing values after filling:", df.isnull().sum().sum())

# Plot pie chart for 'rainfall' feature
plt.pie(df['rainfall'].value_counts().values,
        labels=df['rainfall'].value_counts().index,
        autopct='%1.1f%%')
plt.show()

# Group by 'rainfall' and get the mean of each group
print(df.groupby('rainfall').mean())

# Select numerical features
features = list(df.select_dtypes(include=np.number).columns)

# Remove 'day' if present as it seems not to be needed
if 'day' in features:
    features.remove('day')
print("Selected features:", features)

# Distribution plots for numerical features
plt.figure(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.show()

# Box plots for numerical features
plt.figure(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.boxplot(x=df[col])
plt.tight_layout()
plt.show()

# Replace categorical labels with binary values for 'yes'/'no'
df.replace({'yes': 1, 'no': 0}, inplace=True)

# Heatmap of correlations greater than 0.8
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

# Drop features with high correlation if necessary
df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

# Define features and target
features = df.drop(['day', 'rainfall'], axis=1)
target = df['rainfall']

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=2
)

# Handle class imbalance with RandomOverSampler
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Initialize models for comparison
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

# Train and evaluate models
for model in models:
    model.fit(X, Y)
    print(f'{model} :')

    train_preds = model.predict_proba(X)[:, 1]
    print('Training Accuracy:', metrics.roc_auc_score(Y, train_preds))

    val_preds = model.predict_proba(X_val)[:, 1]
    print('Validation Accuracy:', metrics.roc_auc_score(Y_val, val_preds))
    print()

# Confusion matrix for the last model (SVC in this case)
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[2], X_val, Y_val)
plt.show()

# Classification report
print(metrics.classification_report(Y_val, models[2].predict(X_val)))
