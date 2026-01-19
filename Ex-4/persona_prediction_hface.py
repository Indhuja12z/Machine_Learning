import pandas as pd
from transformers import pipeline

clf = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3"
)

data = pd.read_csv("user_interest.csv")

labels = data['Persona'].unique().tolist()

correct = 0

for i in range(len(data)):
    text = data.loc[i, 'Board_Name']
    actual = data.loc[i, 'Persona']

    pred = clf(text, labels)['labels'][0]

    if pred == actual:
        correct += 1

accuracy = (correct / len(data)) * 100

print("Accuracy:", accuracy, "%")

new_board = "Baking"
print("Prediction:", clf(new_board, labels)['labels'][0])
