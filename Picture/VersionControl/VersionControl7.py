import cv2
import face_recognition
import numpy as np
from fer import FER
import os

# Input file
INPUT_IMAGE_FILE = "test.jpg"

# Output file
OUTPUT_IMAGE_FILE = f"{INPUT_IMAGE_FILE.rsplit('.', 1)[0]}_FR_And_FER_Image.{INPUT_IMAGE_FILE.rsplit('.', 1)[-1]}"

# Check if input file exists
if not os.path.exists(INPUT_IMAGE_FILE):
    print("Input file does not exist.")
    exit()

# Remove output file if file already exists with same name
if os.path.exists(OUTPUT_IMAGE_FILE):
    os.remove(OUTPUT_IMAGE_FILE)

# Initialize the FER object
fer = FER(mtcnn=True)

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file("faces/" + f)
                encodings = face_recognition.face_encodings(face)
                # Check if face_encodings returned any faces, if not skip this photo
                if encodings:
                    encoding = encodings[0]  # Use the first encoding (face) found
                    encoded[f.split(".")[0]] = encoding
                else:
                    print(f"No faces found in image: {f}. Skipping...")
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

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)  # Green box
        info_text = f"{name}: {emotion_text}"
        cv2.rectangle(img, (left, bottom), (right, bottom + 20 + 10), (0, 255, 0), -1)  # Green background
        cv2.putText(img, info_text, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # Black text

    cv2.imwrite(OUTPUT_IMAGE_FILE, img)
    # cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

classify_face(INPUT_IMAGE_FILE)