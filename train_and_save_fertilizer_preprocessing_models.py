import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('FertilizerDatset/fertilizer_recommendation_dataset.csv')

# Identify categorical and numerical features for the fertilizer model
categorical_features = ['Soil', 'Crop', 'Fertilizer']
numerical_features = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']

# Create and save LabelEncoders for categorical features
fertilizer_le_soil = LabelEncoder()
df['Soil_encoded'] = fertilizer_le_soil.fit_transform(df['Soil'])
joblib.dump(fertilizer_le_soil, 'fertilizer_le_soil.pkl')
print("fertilizer_le_soil.pkl saved.")

fertilizer_le_crop = LabelEncoder()
df['Crop_encoded'] = fertilizer_le_crop.fit_transform(df['Crop'])
joblib.dump(fertilizer_le_crop, 'fertilizer_le_crop.pkl')
print("fertilizer_le_crop.pkl saved.")

fertilizer_le_fertilizer = LabelEncoder()
df['Fertilizer_encoded'] = fertilizer_le_fertilizer.fit_transform(df['Fertilizer'])
joblib.dump(fertilizer_le_fertilizer, 'fertilizer_le_fertilizer.pkl')
print("fertilizer_le_fertilizer.pkl saved.")

# Create and save StandardScaler for numerical features
fertilizer_scaler = StandardScaler()
df[numerical_features] = fertilizer_scaler.fit_transform(df[numerical_features])
joblib.dump(fertilizer_scaler, 'fertilizer_scaler.pkl')
print("fertilizer_scaler.pkl saved.")

print("Fertilizer preprocessing models created and saved successfully.")
