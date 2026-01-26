import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# read csv
df = pd.read_csv("pinterest.csv")

X = df["text"]
y = df["category"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# hugging face model (lightweight)
model = SentenceTransformer("all-MiniLM-L6-v2")

# text → vectors
X_train_vec = model.encode(X_train.tolist())
X_test_vec = model.encode(X_test.tolist())

# classifier
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# accuracy
pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# test case
#test = "easy homemade craft ideas"
test ="step by step dinner recipe"
print("Prediction:", clf.predict(model.encode([test]))[0])
