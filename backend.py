# cloudburst_prediction.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ===== Step 1: Load Data =====
data = pd.read_excel("DS tab.xlsx")

# ===== Step 2: Features & Target =====
X = data[["Latitude", "Longitude", "2m temperature", "Total precipitation"]]
y = data["Label"]

# ===== Step 3: Split Data =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Step 4: Train Model =====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===== Step 5: Evaluate =====
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===== Step 6: Save Model =====
joblib.dump(model, "cloudburst_model.pkl")

# ===== Step 7: Example Prediction =====
sample = pd.DataFrame({
    "Latitude": [29.7],
    "Longitude": [80.9],
    "2m temperature": [288],
    "Total precipitation": [0.004]
})

pred = model.predict(sample)
print("\n🌧 Predicted Cloudburst:", "YES" if pred[0]==1 else "NO")