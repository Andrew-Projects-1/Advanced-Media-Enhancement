from fer import FER
import cv2
import face_recognition as fr
import face_recognition
import os
import numpy as np

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

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    result = fer.detect_emotions(img)

    for i, emotion in enumerate(result):
        x1, y1, x2, y2 = emotion['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        max_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
        max_confidence = emotion['emotions'][max_emotion]
        cv2.putText(img, f"{max_emotion} ({max_confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
        cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+35), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text_scale = max(0.5, img.shape[1] / 1000)
        cv2.putText(img, name, (left -20, bottom + 15), font, text_scale, (255, 255, 255), 2)

    cv2.imshow('Video', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return face_names

print(classify_face("test.jpg"))