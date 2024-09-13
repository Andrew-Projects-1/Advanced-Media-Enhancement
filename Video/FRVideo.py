# Andrew Moskowitz
##
# This program performs face recognition on a video file by detecting, identifying, and labeling known faces 
# within each frame of the video. It begins by loading a set of known face encodings and corresponding names 
# from a directory of images. It then processes the video frame by frame, using the face recognition library 
# to detect faces and compare them against the known faces. The detected faces are labeled with names 
# (or "Unknown" if no match is found) and outlined with rectangles. The program leverages multiprocessing to 
# speed up the frame processing, writing the labeled frames into an output video. After all frames are processed, 
# it uses FFmpeg to combine the processed video with the original audio. Additionally, it logs the names of all 
# recognized individuals into a text file and cleans up temporary files once the processing is complete.
##

import cv2
import face_recognition
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import subprocess
import time

# Input file
INPUT_VIDEO_FILE = "Test1.mp4"

# Output file
OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# List of found people
FOUND_PEOPLE_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Found_People.txt"

# Check if input file exists
if not os.path.exists(INPUT_VIDEO_FILE):
    print("Input file does not exist.")
    exit()

# Remove output files if they already exist
for file in [OUTPUT_VIDEO_FILE, FOUND_PEOPLE_FILE]:
    if os.path.exists(file):
        os.remove(file)

# Check if the input file is .avi or .mov
if INPUT_VIDEO_FILE.endswith('.avi') or INPUT_VIDEO_FILE.endswith('.mov'):
    # Convert .avi or .mov file to mp4
    subprocess.run(["ffmpeg", "-i", INPUT_VIDEO_FILE, "-vcodec", "h264", "-acodec", "aac", "-strict", "2", "-preset", "slow", "temp.mp4"])
    INPUT_VIDEO_FILE = "temp.mp4"
    OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

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

# Function to compare the input face_encoding to all known faces in faces_encoded
def compare_faces(face_encoding, faces_encoded, known_face_names):
    matches = face_recognition.compare_faces(faces_encoded, face_encoding)
    name = "Unknown" # Initialize a default name for unkown faces
    
    # Compute the face distance between the input face and all known faces
    face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
    # Find the index of the smalles distance (best match) in the face_distances array
    best_match_index = np.argmin(face_distances)
    # Check if the best match is a valid match - if the face encoding is close enough to be considered a match
    if matches[best_match_index]:
        # If it is a match, assign corresponding name from known_face_names
        name = known_face_names[best_match_index]
           
    return name # Return the name of the best match, or 'Unknown' if no good match is found

# Function to analyze each frame of video and detect the faces within it
def process_frame(args):
    frame, frame_count, faces_encoded, known_face_names = args      # Video frame, current frame number, known faces encodings, list of names
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)       # Resize frame for faster face recognition
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)        # Convert image from BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)     # Detect all face locations in current frame
    unknown_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations) # Encode the found faces at detected face locations
    
    face_names = []                                                 # List to hold the names of found people in current frame
    for face_encoding in unknown_face_encodings:                    # Loop over each face encoding found in the frame
        name = compare_faces(face_encoding, faces_encoded, known_face_names) # Compare the face encoding with known faces to get the name
        face_names.append(name)                                     # Append the recognized name (or "Unknown") to the list
    
    # Loop over the face locations and corresponding names to annotate the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):                        
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
        cv2.rectangle(frame, (left*4, bottom*4 - 35), (right*4, bottom*4), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left*4 + 6, bottom*4 - 6), font, 0.5, (0, 0, 0), 1)
    
    return frame_count, frame, face_names # Return the frame count for ordering, the processed frame, and the recognized names

# Function to remove duplicates from the found people file
def removeDuplicatesAndUnknown(names):
    unique_names = list(set(names))
    return [name for name in unique_names if name.lower() != 'unknown']

def main():
    # Record the start time to calculate total computation time
    start_time = time.time()

    # Get the encoded faces and their corresponding names 
    faces_encoded, known_face_names = get_encoded_faces()
    
    cap = cv2.VideoCapture(INPUT_VIDEO_FILE)    # Open the input video file for processing
    frame_rate = cap.get(cv2.CAP_PROP_FPS)      # Get the frame rate of the video
    delay_time = int(1000 / frame_rate)         # Calculate the delay time between frames
    frames_to_process = []                      # List to store the frames to be processed

    while True:                                 # Read frames from the video in a loop
        ret, frame = cap.read()                 # Read a single frame
        if not ret:                             # If no more frames are left, break the loop
            break
        # Get the current frame count to judge position in video
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Append the frame, frame count, and face data to the processing list
        frames_to_process.append((frame, frame_count, faces_encoded, known_face_names))

    # Initialize multiprocessing pool to process frames in parallel
    with Pool(processes=cpu_count()) as pool:
        # Process frames in parallel
        processed_frames = pool.map(process_frame, frames_to_process)
        # Sort the processed frames by their original frame count to maintain correct order
        processed_frames.sort(key=lambda x: x[0])

    # Define the output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Create a VideoWriter object to write the processed frames into an output video file
    out = cv2.VideoWriter('output_without_audio.mp4', cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, (frame_width, frame_height))

    # Loop through the processed frames and write each frame to the output video file
    for _, frame, _ in processed_frames:
        out.write(frame)                                # Write the frame to the output video
        if cv2.waitKey(delay_time) & 0xFF == ord('q'):  # Exit the loop if the 'q' key is pressed
            break

    # Release the video capture and writer objects, and close an OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Add audio to the output video
    subprocess.run(['ffmpeg', '-i', 'output_without_audio.mp4', '-i', INPUT_VIDEO_FILE, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', OUTPUT_VIDEO_FILE])

    # Collect all the names of the identified people from the processed frames
    found_names = [name for _, _, names in processed_frames for name in names]
    # Remove duplicates and unknown names from the list
    found_names = removeDuplicatesAndUnknown(found_names)
    
    # Write the unique found names to a text file
    with open(FOUND_PEOPLE_FILE, 'w') as f:
        for name in found_names:
            f.write(name + '\n')

    # Remove the temporary output video file with no audio
    os.remove('output_without_audio.mp4')
    if os.path.exists("temp.mp4"):
        os.remove("temp.mp4")

    # Record the end time to calculate total computation time
    end_time = time.time() 
    total_time = end_time - start_time
    print(f"Total computation time: {total_time:.2f} seconds")

    #  # Ask the user to input a face name to search for
    # face_name = input("Enter a face name to search for: ")

    # # Search for the face in the found names
    # if face_name in found_names:
    #     print("Face found!")
    #     # Display the frame where the face was found
    #     for frame_count, frame, _ in processed_frames:
    #         if face_name in frame:
    #             cv2.imshow('Face Found', frame)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #             break
    # else:
    #     print("Face not found!")

if __name__ == "__main__":
    main()
