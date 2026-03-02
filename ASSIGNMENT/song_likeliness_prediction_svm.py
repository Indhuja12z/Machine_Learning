# Spotify Song Like Prediction using SVM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Step 1: Load dataset (clean)
data = pd.read_csv("spotify.csv", index_col=0)

print("Dataset loaded successfully.")
print("Shape:", data.shape)
# Step 2: Select features and target
X = data.drop(columns=["target", "song_title", "artist"])

# Target column
y = data["target"]

print("\nFeatures used:")
print(X.columns)

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 4: Scale features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train SVM model

model = SVC(kernel='rbf')

model.fit(X_train_scaled, y_train)

print("\nModel trained successfully.")

# Step 6: Test model


y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Confusion Matrix Plot


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix - SVM Model")
plt.show()

# Step 8: Test Case Prediction


# Take real song from dataset
test_song = X.iloc[[0]]

# Actual value
actual = y.iloc[0]

# Scale
test_song_scaled = scaler.transform(test_song)

# Predict
prediction = model.predict(test_song_scaled)[0]

print("Test Case Prediction")

#print("Actual value:", actual)

if prediction == 1:
    print("Predicted: Liked Song ")
else:
    print("Predicted: Not Liked Song ")


# Step 9: Show song details

print("\nSong details:")
print("Song:", data.iloc[0]["song_title"])
print("Artist:", data.iloc[0]["artist"])
