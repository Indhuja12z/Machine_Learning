import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib
import gradio as gr
dataset = load_dataset("mnemoraorg/spam-email-5k5")
df = dataset["train"].to_pandas()
print("Dataset shape:", df.shape)
df['label'] = df['Category'].map({
    'spam': 1,
    'ham': 0
})
X = df['Message']
y = df['label']
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
print("Model training complete")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\nEvaluation Metrics")
print("===================")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "svm_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel saved successfully")
model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
def predict_spam(email):
    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    spam_prob = prob[1]
    if pred == 1:
        return f"Spam (Confidence: {spam_prob:.2f})"
    else:
        return f"Not Spam (Confidence: {1-spam_prob:.2f})"
interface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=5, placeholder="Enter email text here"),
    outputs="text",
    title="SVM Spam Email Detector",
    description="Enter an email message to check if it is spam"
)
interface.launch()
