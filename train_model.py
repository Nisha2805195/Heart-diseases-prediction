import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/heart.csv")

print("Dataset shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
with open("model/heart_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save scaler
with open("model/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Save feature names
with open("model/features.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)

print("\nModel saved successfully!")
print("Files created:")
print("model/heart_model.pkl")
print("model/scaler.pkl")
print("model/features.pkl")