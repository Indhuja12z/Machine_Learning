import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# 1. Read CSV

df = pd.read_csv("pinterest.csv")

X = df["text"]
y = df["category"]
# 2. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Text → numbers

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# 4. Naive Bayes-
model = MultinomialNB()
model.fit(X_train_vec, y_train)
# 5. Accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
# 6. Test case

#test_text = ["easy homemade craft ideas"]
test_text = ["Delicious Dinner Recipe"]
test_vec = vectorizer.transform(test_text)
print("Prediction:", model.predict(test_vec)[0])
