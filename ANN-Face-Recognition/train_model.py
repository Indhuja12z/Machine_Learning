import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

dataset_path = "dataset"

faces = []
labels = []
names = []
label_id = 0

# Load dataset
for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    names.append(person)

    for image_name in os.listdir(person_path):

        img_path = os.path.join(person_path, image_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))

        faces.append(img.flatten())
        labels.append(label_id)

    label_id += 1

faces = np.array(faces) / 255.0
labels = np.array(labels)

print("Training model...")

model = MLPClassifier(
    hidden_layer_sizes=(256,128),
    max_iter=500,
    verbose=True,
    random_state=42
)

model.fit(faces, labels)

pickle.dump(model, open("face_model.pkl", "wb"))
pickle.dump(names, open("names.pkl", "wb"))

print("Model trained and saved successfully!")