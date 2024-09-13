import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import threading

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded


def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]
    return encoding


def classify_face(im, faces, faces_encoded, known_face_names):
    cap = cv2.VideoCapture(im)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 1/4 of its original size
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        unknown_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left * 4, bottom * 4 - 15), (right * 4, bottom * 4 + 15), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left * 4 + 10, bottom * 4 + 10), font, 1.0, (255, 255, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Get the encoded faces
faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

# Create a thread for each video
threads = []
for video in ["Test.mp4"]:  
    t = threading.Thread(target=classify_face, args=(video, faces, faces_encoded, known_face_names))
    t.start()
    threads.append(t)

# Wait for all threads to complete
for t in threads:
    t.join()