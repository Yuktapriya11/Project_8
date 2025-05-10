# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ Load Dataset
# Replace 'fraud_detection.csv' with your actual file path
df = pd.read_csv('fraud_detection.csv')

# Preview dataset
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# 2️⃣ Feature Engineering
# Example: Create a feature 'Amount_log' as log transformation
df['Amount_log'] = np.log1p(df['Amount'])

# Encode categorical 'Type' column
le = LabelEncoder()
df['Type_encoded'] = le.fit_transform(df['Type'])

# Drop irrelevant columns (if needed, e.g., Transaction ID)
df_model = df.drop(['Transaction ID', 'Type'], axis=1)

# 3️⃣ Define features and target
X = df_model.drop('Is Fraud', axis=1)
y = df_model['Is Fraud']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Hyperparameter Tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dtree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Best parameters
print("\nBest Parameters:", grid_search.best_params_)

# 5️⃣ Train Decision Tree with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)

# 6️⃣ Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Feature Importance
import matplotlib.pyplot as plt

feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance")
plt.show()

# ✅ Recommendations
print("\n✅ Consider using RandomForestClassifier or Anomaly Detection for improved accuracy.")
