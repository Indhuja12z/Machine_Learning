import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset (Assumes pinterest_data.csv exists with Yes/No values)
df = pd.read_csv("pinterest.csv")

# 2. Encode Categorical Features and Target
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# 3. Define features (X) and target (y)
X = df.drop(columns=['Category'])
y = df['Category']

# 4. Split Data (using a small test size since dataset is only 12 points)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

test_case_raw = {'How_to': 'No', 'Delicious': 'Yes', 'Tutorial': 'No'}
test_case_encoded = [encoders[col].transform([test_case_raw[col]])[0] for col in ['How_to', 'Delicious', 'Tutorial']]

prediction = model.predict([test_case_encoded])
predicted_label = encoders['Category'].inverse_transform(prediction)[0]

# FINAL OUTPUT
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Test Case {test_case_raw} -> Result: {predicted_label}")
