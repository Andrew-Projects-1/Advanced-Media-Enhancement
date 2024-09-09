import cv2
import face_recognition
from fer import FER
import os

# Input file
INPUT_IMAGE_FILE = "SadNeutralHappy.jpg"

# Output file
OUTPUT_IMAGE_FILE = f"{INPUT_IMAGE_FILE.rsplit('.', 1)[0]}_FER_Picture.{INPUT_IMAGE_FILE.rsplit('.', 1)[-1]}"

# Check if input file exists
if not os.path.exists(INPUT_IMAGE_FILE):
    print("Input file does not exist.")
    exit()

# Remove output file if file already exists with same name
if os.path.exists(OUTPUT_IMAGE_FILE):
    os.remove(OUTPUT_IMAGE_FILE)

# Initialize the FER object
fer = FER(mtcnn=True)

# Load the image
img = cv2.imread(INPUT_IMAGE_FILE)

# Resize the image if necessary
img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)

# Use face_recognition to find the face locations
face_locations = face_recognition.face_locations(img)

# Process each face found
for face_location in face_locations:
    # Unpack the face location coordinates
    top, right, bottom, left = face_location

    # Crop the face from the image using the coordinates
    face_img = img[top:bottom, left:right]
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for FER

    # Detect emotion on the cropped face
    emotions = fer.detect_emotions(face_img_rgb)

    # Check if any emotion is detected
    if emotions:
        # Get the primary emotion and its confidence
        primary_emotion, confidence = max(emotions[0]['emotions'].items(), key=lambda item: item[1])
        emotion_text = f"{primary_emotion} ({confidence:.2f})"
    else:
        emotion_text = "No Emotion Detected"

    # Draw the face bounding box
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # Annotate the image with the identified emotion
    cv2.putText(img, emotion_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the new image
cv2.imwrite(OUTPUT_IMAGE_FILE, img)
