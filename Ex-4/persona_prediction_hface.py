import pandas as pd
from transformers import pipeline

clf = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
data = pd.read_csv("user_interest.csv")

new_board = "Baking"
print(clf(new_board, data['Persona'].unique().tolist())['labels'][0])
