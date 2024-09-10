from fer import FER
import cv2
import face_recognition as fr
import os
import numpy as np
import face_recognition

# Initialize the FER object
fer = FER(mtcnn=True)

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
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
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_img = img[top:bottom, left:right]
        emotions = fer.detect_emotions(face_img)
        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            emotion_score = emotions[0]['emotions'][dominant_emotion]
            emotion_text = f"{dominant_emotion} ({emotion_score:.2f})"
        else:
            emotion_text = "Emotion Unknown"

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name}: {emotion_text}"
        cv2.putText(img, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Video', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [(name, emotion_text) for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_face_encodings)]

print(classify_face("test.jpg"))
