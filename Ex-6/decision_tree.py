import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
df = pd.read_csv("diabetes.csv")
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Model Accuracy (Entropy): {accuracy_score(y_test, y_pred):.2%}")
test_patient = np.array([[2, 150, 70, 20, 100, 31.0, 0.45, 45]])
prediction = model.predict(test_patient)
prob = model.predict_proba(test_patient)
print("-" * 30)
print(f"Test Patient Prediction: {'DIABETIC' if prediction[0] == 1 else 'HEALTHY'}")
print(f"Confidence: {max(prob[0])*100:.2f}%")
print("-" * 30)
plt.figure(figsize=(20,10))
plot_tree(model,
          feature_names=X.columns,
          class_names=['Healthy', 'Diabetic'],
          filled=True, 
          rounded=True,
          fontsize=10) 
plt.title("Decision Tree using Entropy (Information Gain)")
plt.show()
