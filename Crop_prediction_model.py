import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# --- 1. Load the dataset and show first 5 rows ---
print("Loading dataset...")
try:
    df = pd.read_csv('Crop_recommendation.csv')
    print("Dataset loaded successfully.")
    print("First 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: Crop_recommendation.csv not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2. EDA (Exploratory Data Analysis) ---
print("\n--- Performing EDA ---")

# Check dataset shape, columns, and missing values
print("\nDataset Shape:", df.shape)
print("\nDataset Columns:", df.columns)
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Distribution plots for features
print("\nGenerating distribution plots...")
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Count plot of crops
print("\nGenerating crop count plot...")
plt.figure(figsize=(12, 6))
sns.countplot(y='label', data=df, order = df['label'].value_counts().index)
plt.title('Count of Each Crop Type')
plt.xlabel('Count')
plt.ylabel('Crop Type')
plt.tight_layout()
plt.show()

# --- 3. Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Define features (X) and target (y)
X = df[features]
y = df['label']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Target labels encoded.")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Data split into training (shape: {X_train.shape}) and testing (shape: {X_test.shape}) sets.")

# StandardScaler for normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")

# --- 4. Model Training ---
print("\n--- Model Training ---")

# Decision Tree Classifier
print("Training Decision Tree Classifier...")
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)
y_pred_dt = dt_classifier.predict(X_test_scaled)
print("Decision Tree Classifier trained.")

# Random Forest Classifier
print("Training Random Forest Classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
y_pred_rf = rf_classifier.predict(X_test_scaled)
print("Random Forest Classifier trained.")

# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")

# Decision Tree Evaluation
print("\n--- Decision Tree Classifier Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt, target_names=le.classes_))

# Random Forest Evaluation
print("\n--- Random Forest Classifier Performance ---")
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Compare performance of both models
print("\n--- Model Comparison ---")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# --- 6. Save Best Model ---
print("\n--- Saving Best Model ---")
# In this case, Random Forest usually performs better, so we'll save it.
best_model = rf_classifier
model_filename = 'crop_recommendation_rf_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Best model (Random Forest) saved as {model_filename}")

# --- 7. Prediction Demo ---
print("\n--- Prediction Demo ---")
# Example: N=90, P=42, K=43, temp=20.87, humidity=82.0, pH=6.5, rainfall=202.93
example_data = np.array([[90, 42, 43, 20.87, 82.0, 6.5, 202.93]])
example_data_scaled = scaler.transform(example_data)

predicted_label_encoded = best_model.predict(example_data_scaled)
predicted_crop = le.inverse_transform(predicted_label_encoded)
print(f"Input: N=90, P=42, K=43, Temp=20.87, Humidity=82.0, pH=6.5, Rainfall=202.93")
print(f"Predicted Crop: {predicted_crop[0]}")

# --- 8. Plot Feature Importance from Random Forest ---
print("\n--- Plotting Feature Importance ---")
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- Final Output ---
print("\n--- Final Results ---")
print(f"Random Forest Model Accuracy: {rf_accuracy:.4f}")

print("Top 3 Most Important Features:")
for i in range(3):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")