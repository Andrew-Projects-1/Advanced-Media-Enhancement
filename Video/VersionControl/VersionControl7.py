import face_recognition
import os
import cv2
import numpy as np
import subprocess

# Input file
INPUT_VIDEO_FILE = "Test.mp4"

# Temporary output file (video without audio)
TEMP_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Video_no_audio.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# Final output file (video with audio)
OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# Check if input file exists
if not os.path.exists(INPUT_VIDEO_FILE):
    print("Input file does not exist.")
    exit()

# Remove output files if they already exist
if os.path.exists(TEMP_VIDEO_FILE):
    os.remove(TEMP_VIDEO_FILE)
if os.path.exists(OUTPUT_VIDEO_FILE):
    os.remove(OUTPUT_VIDEO_FILE)


def get_encoded_faces():
    # Initialize lists to store face encodings and corresponding person names
    encoded_faces = []
    face_names = []

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                person_name = os.path.basename(dirpath)
                face_image_path = os.path.join(dirpath, f)
                face = face_recognition.load_image_file(face_image_path)
                encodings = face_recognition.face_encodings(face)
                if encodings:
                    encoding = encodings[0]
                    encoded_faces.append(encoding)
                    face_names.append(person_name)
                else:
                    print(f"No faces found in image: {f}. Skipping...")
    return encoded_faces, face_names


def classify_faces_in_video(video_file):
    # Get face encodings and corresponding names from the faces directory
    faces_encoded, known_face_names = get_encoded_faces()

    # Open the input video file
    video_capture = cv2.VideoCapture(video_file)

    # Get the video frame width, height, and frames per second (fps)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the output video (without audio)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(TEMP_VIDEO_FILE, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Detect face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        unknown_face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []

        for face_encoding in unknown_face_encodings:
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (0, 255, 0), 2)
            face_height = bottom - top
            text_scale = face_height / 150
            thickness = 2
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, text_scale, thickness)[0]
            cv2.rectangle(frame, (left - 20, bottom + 5), (left + text_size[0] + 20, bottom + text_size[1] + 25), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left - 20, bottom + text_size[1] + 15), cv2.FONT_HERSHEY_DUPLEX, text_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        output_video.write(frame)

    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()

    print(f"Temporary video (no audio) saved as {TEMP_VIDEO_FILE}")


classify_faces_in_video(INPUT_VIDEO_FILE)

# Use ffmpeg to combine the video and audio
def add_audio_to_video(input_video, output_video, original_video):
    command = [
        'ffmpeg', '-i', input_video, '-i', original_video, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-shortest', output_video
    ]
    subprocess.run(command, check=True)
    print(f"Final video with audio saved as {OUTPUT_VIDEO_FILE}")


# Add the original audio to the output video
add_audio_to_video(TEMP_VIDEO_FILE, OUTPUT_VIDEO_FILE, INPUT_VIDEO_FILE)
