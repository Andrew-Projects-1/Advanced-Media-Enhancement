# Andrew Moskowitz
##
# This program combines facial recognition and emotion detection to identify both the identity and emotional state of individuals
# in an input image. It first detects faces and compares them to a database of known individuals stored in the 'Faces' directory, 
# with each subfolder representing a different person. If a match is found, the person's name is displayed on the image; if no match
# is found, the face is labeled as "Unknown." Simultaneously, the program analyzes facial expressions by examining key facial landmarks,
# classifying the emotion into categories such as happiness, sadness, anger, surprise, and more. The output is an annotated image that 
# displays both the individual's name and their detected emotion.
##

import cv2
import face_recognition
from fer import FER
import os
import numpy as np

# Input file
INPUT_IMAGE_FILE = "Test.jpg"

# Output file
OUTPUT_IMAGE_FILE = f"{INPUT_IMAGE_FILE.rsplit('.', 1)[0]}_FR_And_FER_Picture.{INPUT_IMAGE_FILE.rsplit('.', 1)[-1]}"

# Check if input file exists
if not os.path.exists(INPUT_IMAGE_FILE):
    print("Input file does not exist.")
    exit()

# Remove the output file if it already exists with the same name
if os.path.exists(OUTPUT_IMAGE_FILE):
    os.remove(OUTPUT_IMAGE_FILE)

# Initialize the FER object
fer = FER(mtcnn=True)

# Load the image
img = cv2.imread(INPUT_IMAGE_FILE)

# Resize the image if necessary
img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)

# Detect face locations using face_recognition
face_locations = face_recognition.face_locations(img)

# Function to encode known faces
def get_encoded_faces():
    # Initialize lists to store face encodings and corresponding person names
    encoded_faces = []
    face_names = []

    for dirpath, dnames, fnames in os.walk("./faces"):                      # Traverse the directory "./faces" and its subdirectories
        for f in fnames:                                                    # Loop through each file in directory
            if f.endswith(".jpg") or f.endswith(".png"):                    # Check if the file is an image with a .jpg or .png extension
                person_name = os.path.basename(dirpath)                     # Get the person's name from the directory name           
                face_image_path = os.path.join(dirpath, f)                  # Create the full path to the image file
                face = face_recognition.load_image_file(face_image_path)    # Load the image using face_recognition
                encodings = face_recognition.face_encodings(face)           # Try to extract the face encoding from the image
                if encodings:                                               # If a face encoding is found, store the encoding and person's name
                    encoding = encodings[0]                                 # Use the first face encoding found
                    encoded_faces.append(encoding)                          # Add the encoding to the list
                    face_names.append(person_name)                          # Associate the encoding with the person's name
                else:
                    print(f"No faces found in image: {f}. Skipping...")
    return encoded_faces, face_names                                        # Return the list of face encodings and the corresponding person names


# Function to classify faces in the image
def classify_face(img, face_locations):
    # Get face encodings and corresponding names from the faces directory
    faces_encoded, known_face_names = get_encoded_faces()
    
    # Detect face locations and encodings
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
    face_names = []

    for face_encoding in unknown_face_encodings:
        # Compute distances between the face encoding and all known encodings
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        # If there are any face distances calculated, find the best match
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)  # Get the index of the smallest distance (best match)
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]  # Assign the corresponding person's name from known_face_names
            else:
                name = "Unknown"  # If no good match, label as "Unknown"
        else:
            name = "Unknown"  # If no face distances are found, label the face as "Unknown"
        face_names.append(name)
    
    return face_names

# Check if faces are found
if len(face_locations) == 0:
    original_height, original_width = img.shape[:2]  # Get image dimensions
    
    # Font Settings
    font = cv2.FONT_HERSHEY_DUPLEX
    text_scale = original_width / 400   # Dynamically scale the text based on the image size
    thickness = 2                       # Set the thickness of text
    text = "No faces found"             # If no faces were found, display "No faces found" on the image
    
    # Get text size to manipulate position
    text_size = cv2.getTextSize(text, font, text_scale, thickness)[0]
    text_x = (original_width - text_size[0]) // 2  # Center the text horizontally
    text_y = (original_height + text_size[1]) // 2  # Center the text vertically
    
    # Renter the text and a background on the image
    cv2.rectangle(img, (text_x - 15, text_y + 15), (text_x + text_size[0] + 15, text_y - text_size[1] - 15), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, text, (text_x, text_y), font, text_scale, (0, 0, 0), thickness, cv2.LINE_AA)

else:
    # Get face names for recognition
    face_names = classify_face(img, face_locations)

    # Process each detected face for emotion and recognition
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Crop the face from the image using the coordinates
        face_img = img[top:bottom, left:right]
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) # Convert to RGB for FER

        # Detect emotion on the cropped face
        emotions = fer.detect_emotions(face_img_rgb)

        # Check if any emotion is detected
        if emotions:
            # Get the primary emotion and its confidence
            primary_emotion, confidence = max(emotions[0]['emotions'].items(), key=lambda item: item[1])
            emotion_text = f"{primary_emotion} ({confidence:.2f})"
        else:
            emotion_text = "No Emotion Detected"

        # Draw rectangle around the face (tighter fit, no extra padding)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Calculate font size and scale based on face height for better readability
        face_height = bottom - top
        text_scale = face_height / 250
        thickness = 2

        # Render the name text and background at the top of the bounding box
        name_text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
        cv2.rectangle(img, (left - 1, (top - 30 - name_text_size[1])), (left + name_text_size[0], top), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 2)


        # Render the emotion text and background at the bottom of the bounding box
        emotion_text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
        cv2.rectangle(img, (left - 1, bottom), (left + emotion_text_size[0], bottom + 30 + emotion_text_size[1]), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, emotion_text, (left, bottom + emotion_text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 2)

# Save the final output image with both face recognition and emotion detection
cv2.imwrite(OUTPUT_IMAGE_FILE, img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Image saved as {OUTPUT_IMAGE_FILE}")
