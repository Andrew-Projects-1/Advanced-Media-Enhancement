import cv2
import face_recognition
import numpy as np
import os
import subprocess

# Input file
INPUT_VIDEO_FILE = "Test.mp4"

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
    OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FER_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"


def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):  # Traverse the directory tree
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):  # Check for image files
                face_image_path = os.path.join(dirpath, f)  # Create the full file path
                face = face_recognition.load_image_file(face_image_path)  # Load the image from the constructed full path
                encodings = face_recognition.face_encodings(face)
                if encodings:  # Check if face_encodings returned any faces, if not skip this photo
                    encoding = encodings[0]  # Use the first encoding (face) found
                    encoded[f.split(".")[0]] = encoding  # Use the file name without extension as the key for the encoding
                else:
                    print(f"No faces found in image: {f}. Skipping...")

    return encoded


def compare_faces(face_encoding, faces_encoded, known_face_names):
    matches = face_recognition.compare_faces(faces_encoded, face_encoding)
    name = "Unknown"

    face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    return name


def process_frame(frame, faces_encoded, known_face_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    unknown_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        name = compare_faces(face_encoding, faces_encoded, known_face_names)
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
        cv2.rectangle(frame, (left*4, bottom*4 - 35), (right*4, bottom*4), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left*4 + 6, bottom*4 - 6), font, 0.5, (0, 0, 0), 1)

    return frame, face_names


def removeDuplicatesAndUnknown(names):
    unique_names = list(set(names))
    return [name for name in unique_names if name.lower() != 'unknown']


def main():
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    cap = cv2.VideoCapture(INPUT_VIDEO_FILE)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    delay_time = int(1000 / frame_rate)

    # Define the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_without_audio.mp4', cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, (frame_width, frame_height))

    found_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame and apply face recognition
        processed_frame, face_names = process_frame(frame, faces_encoded, known_face_names)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Collect found names
        found_names.extend(face_names)

        if cv2.waitKey(delay_time) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Add audio to the output video
    subprocess.run(['ffmpeg', '-i', 'output_without_audio.mp4', '-i', INPUT_VIDEO_FILE, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', OUTPUT_VIDEO_FILE])

    # Remove duplicates and unknown names from the found names
    found_names = removeDuplicatesAndUnknown(found_names)

    with open(FOUND_PEOPLE_FILE, 'w') as f:
        for name in found_names:
            f.write(name + '\n')

    # Clean up
    os.remove('output_without_audio.mp4')
    if os.path.exists("temp.mp4"):
        os.remove("temp.mp4")


if __name__ == "__main__":
    main()
