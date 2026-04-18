# rf_train.py

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("water_potability.csv")

print("Dataset Loaded:")
print(df.head())
print("Shape:", df.shape)

# =========================
# 2. Preprocessing
# =========================

# Features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Create Pipeline
# =========================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),   # handle missing values
    ("scaler", StandardScaler()),                  # scaling
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    ))
])

# =========================
# 5. Train Model
# =========================
pipeline.fit(X_train, y_train)

print("Model Training Completed!")

# =========================
# 6. Save Model
# =========================
with open("water_potability.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Model saved as water_potability.pkl")