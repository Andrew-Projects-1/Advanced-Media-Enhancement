# Andrew Moskowitz
##
# This program is designed to identify individuals by comparing faces from an input image against a collection of known 
# faces stored in the 'Faces' directory. Each subfolder within the 'Faces' directory represents an individual, and the 
# program scans all images within these subfolders to extract facial encodings (numerical representations of faces). 
# When an input image is provided, the program detects faces within the image and compares their encodings to the known 
# faces, identifying matches. If a match is found, the name of the corresponding individual (derived from the subfolder name) 
# is displayed on the image; otherwise, the face is labeled as "Unknown". The output is an annotated version of the input image, 
# either identifying the individuals or indicating that no faces were recognized. If no faces are detected in the input image, 
# the program will print "No faces found" on the output image.
##

import face_recognition
import os
import cv2
import numpy as np

# Input file
INPUT_IMAGE_FILE = "Test.jpg"

# Output file
OUTPUT_IMAGE_FILE = f"{INPUT_IMAGE_FILE.rsplit('.', 1)[0]}_FR_Image.{INPUT_IMAGE_FILE.rsplit('.', 1)[-1]}"

# Check if input file exists
if not os.path.exists(INPUT_IMAGE_FILE):
    print("Input file does not exist.")
    exit()

# Remove output file if file already exists with same name
if os.path.exists(OUTPUT_IMAGE_FILE):
    os.remove(OUTPUT_IMAGE_FILE)


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

def classify_face(im):
    # Get face encodings and corresponding names from the faces directory
    faces_encoded, known_face_names = get_encoded_faces()

    img = cv2.imread(im, 1) # Read the input image from the file path 'im' in color mode
    original_height, original_width = img.shape[:2]  # Get original dimensions

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []

    if len(face_locations) == 0:
        # Font settings
        font = cv2.FONT_HERSHEY_DUPLEX
        text_scale = original_width / 1000  # Dynamically scale the text based on the image size
        thickness = 2  # Set the thickness of text
        text = "No faces found" # If no faces were found, display "No faces found" on the image

        # Get the text size to center it
        text_size = cv2.getTextSize(text, font, text_scale, thickness)[0]

        # Calculate the center position for the text
        text_x = (original_width - text_size[0]) // 2  # Center horizontally
        text_y = (original_height + text_size[1]) // 2  # Center vertically

        # Render the text on the image
        cv2.putText(img, text, (text_x, text_y), font, text_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    else:
        # If faces were found then proceed with face recognition
        for face_encoding in unknown_face_encodings:
            # Compute distances between the face encoding and all known encodings
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            # If there are any face distances calculated, find the best match
            if len(face_distances) > 0: 
                best_match_index = np.argmin(face_distances)  # Get the index of the smallest distance (best match)
                if face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index] # Assign the corresponding person's name from known_face_names
                else:
                    name = "Unknown" # If no good match, label as "Unknown"
            else:
                name = "Unknown" # If no face distances are found, label the face as "Unknown"

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a green box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (0, 255, 0), 2)

            # Adjust text size dynamically based on the height of the face box
            face_height = bottom - top
            text_scale = face_height / 150  # Adjust the text size based on the face height
            thickness = 2  # Thickness of the text box

            # Get the text size to adjust the background box accordingly
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, text_scale, thickness)[0]
            
            # Draw the background rectangle for the text
            cv2.rectangle(img, (left-20, bottom + 5), (left + text_size[0] + 20, bottom + text_size[1] + 25), (0, 255, 0), cv2.FILLED)
            
            # Render the name text on the image below the face box
            cv2.putText(img, name, (left - 20, bottom + text_size[1] + 15), cv2.FONT_HERSHEY_DUPLEX, text_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Save the modified image with the same size as the original
    cv2.imwrite(OUTPUT_IMAGE_FILE, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the names of the recognized faces (if any)
    return face_names

# Returns the names of recognized faces in the input image
print(classify_face(INPUT_IMAGE_FILE))
