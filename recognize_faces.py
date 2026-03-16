import cv2
import pickle
import numpy as np

model = pickle.load(open("face_model.pkl", "rb"))
names = pickle.load(open("names.pkl", "rb"))

camera = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(gray, (100,100))
    face = face.flatten().reshape(1,-1) / 255.0

    prediction = model.predict(face)[0]
    person_name = names[prediction]

    cv2.putText(frame,
                person_name,
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
