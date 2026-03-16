import cv2
import os

# Input person name
name = input("Enter person's name: ")

# Create dataset folder
os.makedirs("dataset", exist_ok=True)
path = os.path.join("dataset", name)
os.makedirs(path, exist_ok=True)

# Start webcam
camera = cv2.VideoCapture(0)
count = 0

print("Press SPACE to capture image")
print("Press ESC to exit")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        img_name = f"{path}/{count}.jpg"
        cv2.imwrite(img_name, gray)
        print("Saved:", img_name)
        count += 1

    elif key == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()