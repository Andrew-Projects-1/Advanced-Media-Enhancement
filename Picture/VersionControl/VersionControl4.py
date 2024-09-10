import cv2
import face_recognition
import numpy as np
from fer import FER
import os

# Initialize the FER object
fer = FER(mtcnn=True)

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file("faces/" + f)
                encoding = face_recognition.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded

def classify_face(im):
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_img = img[top:bottom, left:right]
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        emotions = fer.detect_emotions(face_img_rgb)

        if emotions:
            emotion, score = max(emotions[0]['emotions'].items(), key=lambda item: item[1])
            emotion_text = f"{emotion} {score:.2f}"
        else:
            emotion_text = "Unknown Emotion"

        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        info_text = f"{name}: {emotion_text}"
        cv2.putText(img, info_text, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

classify_face("test.jpg")
