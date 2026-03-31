# AI-Powered Particle Collision Analysis System
# BTech Level Mini Research Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Step 1: Generate Synthetic Particle Collision Dataset
# -----------------------------

np.random.seed(42)

n_samples = 5000

# Features simulating particle experiment outputs
energy = np.random.normal(loc=100, scale=20, size=n_samples)
momentum = np.random.normal(loc=50, scale=10, size=n_samples)
detector_signal = np.random.normal(loc=0.5, scale=0.1, size=n_samples)

# True event classification logic (hidden physics-like rule)
labels = ((energy > 110) & (momentum > 55)).astype(int)

data = pd.DataFrame({
    "energy": energy,
    "momentum": momentum,
    "detector_signal": detector_signal,
    "event": labels
})

print("Dataset Preview:")
print(data.head())

# -----------------------------
# Step 2: Traditional Rule-Based Filtering
# -----------------------------

def traditional_filter(df):
    return ((df["energy"] > 115) & (df["momentum"] > 60)).astype(int)

traditional_predictions = traditional_filter(data)

print("\nTraditional Method Accuracy:",
      accuracy_score(data["event"], traditional_predictions))

# -----------------------------
# Step 3: AI-Based Detection (Random Forest)
# -----------------------------

X = data[["energy", "momentum", "detector_signal"]]
y = data["event"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

ai_predictions = model.predict(X_test)

print("\nAI Model Accuracy:",
      accuracy_score(y_test, ai_predictions))

print("\nClassification Report:\n",
      classification_report(y_test, ai_predictions))

# -----------------------------
# Step 4: Confusion Matrix (False Detection Analysis)
# -----------------------------

cm = confusion_matrix(y_test, ai_predictions)
print("\nConfusion Matrix:\n", cm)

# Plot Confusion Matrix
plt.imshow(cm)
plt.title("Confusion Matrix (AI Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
