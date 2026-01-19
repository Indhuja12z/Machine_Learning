import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("user_interest.csv")


le_board = LabelEncoder()
le_persona = LabelEncoder()

data['Board_Name_Encoded'] = le_board.fit_transform(data['Board_Name'])
data['Persona_Encoded'] = le_persona.fit_transform(data['Persona'])


X = data[['Board_Name_Encoded']]
y = data['Persona_Encoded']


model = MultinomialNB()
model.fit(X, y)


y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
print("Model Accuracy:", accuracy * 100, "%")


new_board = "Baking"
new_board_encoded = le_board.transform([new_board])

prediction = model.predict([[new_board_encoded[0]]])
predicted_persona = le_persona.inverse_transform(prediction)

print("Board Name:", new_board)
print("Predicted Persona:", predicted_persona[0])
