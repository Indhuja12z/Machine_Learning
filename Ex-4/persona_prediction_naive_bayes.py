import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load dataset
data = pd.read_csv("user_interest.csv")

# Step 2: Encode categorical values
le_board = LabelEncoder()
le_persona = LabelEncoder()

data['Board_Name_Encoded'] = le_board.fit_transform(data['Board_Name'])
data['Persona_Encoded'] = le_persona.fit_transform(data['Persona'])

# Step 3: Define features and target
X = data[['Board_Name_Encoded']]
y = data['Persona_Encoded']

# Step 4: Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Step 5: Predict for a new board
new_board = "Baking"
new_board_encoded = le_board.transform([new_board])

prediction = model.predict([[new_board_encoded[0]]])
predicted_persona = le_persona.inverse_transform(prediction)

print("Board Name:", new_board)
print("Predicted Persona:", predicted_persona[0])
