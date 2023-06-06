import cv2
import numpy as np
import os

# Train the face recognition model using the collected dataset
def train_model():
    data_path = 'data'
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    training_data = []
    labels = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('jpg'):
                path = os.path.join(root, file)
                label = int(path.split('.')[1])
                image = cv2.imread(path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face = face_classifier.detectMultiScale(gray, 1.3, 5)
                if face is not ():
                    for (x, y, w, h) in face:
                        cropped_face = gray[y:y + h, x:x + w]
                        training_data.append(cropped_face)
                        labels.append(label)

    labels = np.array(labels)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(training_data, labels)

    return model

# Implement the student attention monitoring system
def monitor_attention():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")
    model = train_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is not ():
            for (x, y, w, h) in faces:
                cropped_face = gray[y:y + h, x:x + w]
                label, confidence = model.predict(cropped_face)
                if confidence < 100:
                    eyes = eye_classifier.detectMultiScale(cropped_face)
                    if eyes is not ():
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                        cv2.putText(frame, "Attentive", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Inattentive", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Student Attention', frame)
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

monitor_attention()
