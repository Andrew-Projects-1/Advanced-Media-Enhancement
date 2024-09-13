import face_recognition
import os
import cv2
import numpy as np
import subprocess
from multiprocessing import Pool, cpu_count
import time

# Input file
INPUT_VIDEO_FILE = "Test.mp4"

# Temporary output file (video without audio)
TEMP_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Video_no_audio.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# Final output file (video with audio)
OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# List of found people
# FOUND_PEOPLE_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FR_Found_People.txt"

# Check if input file exists
if not os.path.exists(INPUT_VIDEO_FILE):
    print("Input file does not exist.")
    exit()

# Remove output files if they already exist
# for file in [OUTPUT_VIDEO_FILE, TEMP_VIDEO_FILE, FOUND_PEOPLE_FILE]:
for file in [OUTPUT_VIDEO_FILE, TEMP_VIDEO_FILE]:
    if os.path.exists(file):
        os.remove(file)

def get_encoded_faces():
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

def process_frame(frame_data):
    frame, frame_count, faces_encoded, known_face_names = frame_data

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

    return frame_count, frame

def classify_faces_in_video_parallel(video_file):
    faces_encoded, known_face_names = get_encoded_faces()

    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frames_to_process = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_to_process.append((frame, frame_count, faces_encoded, known_face_names))

    # Initialize multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        processed_frames = pool.map(process_frame, frames_to_process)
        processed_frames.sort(key=lambda x: x[0])  # Ensure frames are sorted by their original order

    # Define the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(TEMP_VIDEO_FILE, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    for _, frame in processed_frames:
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Temporary video (no audio) saved as {TEMP_VIDEO_FILE}")

def add_audio_to_video(input_video, output_video, original_video):
    command = [
        'ffmpeg', '-i', input_video, '-i', original_video, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-shortest', output_video
    ]
    subprocess.run(command, check=True)
    os.remove(TEMP_VIDEO_FILE)
    print(f"Final video with audio saved as {OUTPUT_VIDEO_FILE}")

def main():
    start_time = time.time()
    classify_faces_in_video_parallel(INPUT_VIDEO_FILE)
    add_audio_to_video(TEMP_VIDEO_FILE, OUTPUT_VIDEO_FILE, INPUT_VIDEO_FILE)

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total computation time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
