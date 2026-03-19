# Spotify Song Like Prediction using SVM + Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/Indhuja12z/Machine_Learning/refs/heads/main/ASSIGNMENT/spotify.csv",
    index_col=0
)

print("Dataset loaded successfully.")
print("Shape:", data.shape)

# Step 2: Features & Target
X = data.drop(columns=["target", "song_title", "artist"])
y = data["target"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 4: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM MODEL

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\n===== SVM RESULTS =====")
print("Accuracy:", svm_accuracy)
print(classification_report(y_test, y_pred_svm))

# LOGISTIC REGRESSION

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
log_accuracy = accuracy_score(y_test, y_pred_log)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Accuracy:", log_accuracy)
print(classification_report(y_test, y_pred_log))

# COMPARISON

print("\n===== MODEL COMPARISON =====")
print("SVM Accuracy:", svm_accuracy)
print("Logistic Regression Accuracy:", log_accuracy)

# CONFUSION MATRICES


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(ax=ax[0])
ax[0].set_title("SVM Confusion Matrix")

# Logistic Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp_log.plot(ax=ax[1])
ax[1].set_title("Logistic Regression Confusion Matrix")

plt.tight_layout()
plt.show()

#test cases
# Select first 5 songs
test_songs = X.iloc[0:5]
actual_values = y.iloc[0:5]

# Scale inputs
test_songs_scaled = scaler.transform(test_songs)

# Predictions
svm_preds = svm_model.predict(test_songs_scaled)
log_preds = log_model.predict(test_songs_scaled)

# Convert to readable format
def label(val):
    return "Liked Song" if val == 1 else "Not Liked Song"

# Create result table
results = pd.DataFrame({
    "Song": data.iloc[0:5]["song_title"].values,
    "Artist": data.iloc[0:5]["artist"].values,
    "Actual": [label(v) for v in actual_values],
    "SVM Prediction": [label(v) for v in svm_preds],
    "Logistic Prediction": [label(v) for v in log_preds]
})

print("\n===== TEST CASE RESULTS (5 Songs) =====")
print(results)