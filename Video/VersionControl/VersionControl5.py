import cv2
import face_recognition
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from functools import partial

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file("faces/" + f)
                encoding = face_recognition.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded

def process_frame(frame, frame_count, faces_encoded, known_face_names):
    # Resize the frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    unknown_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
        face_names.append(name)
    
    # Write the names to a file
    with open('found_people.txt', 'a') as f:
        for name in face_names:
            f.write(name + '\n')
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
        cv2.rectangle(frame, (left*4, bottom*4 - 35), (right*4, bottom*4), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left*4 + 6, bottom*4 - 6), font, 0.5, (0, 0, 0), 1)
    
    return frame_count, frame

def removeDuplicatesAndUnknown():
    with open('found_people.txt', 'r') as f:
        lines = f.readlines()

    # Remove duplicates
    lines = list(set(lines))
    lines = [line for line in lines if 'unknown' not in line.lower()]

    # Write the file again
    with open('found_people.txt', 'w') as f:
        f.writelines(lines)


def main():
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    
    cap = cv2.VideoCapture("TandB.mp4")
    frame_count = 0
    frames_to_process = []
    pool = Pool(processes=cpu_count())
    process_frame_partial = partial(process_frame, faces_encoded=faces_encoded, known_face_names=known_face_names)
    
    # Get the frame rate of the video
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_to_process.append((frame, frame_count))
        frame_count += 1
    
    # Use pool.map to process frames in parallel
    processed_frames = pool.starmap(process_frame_partial, frames_to_process)
    processed_frames.sort(key=lambda x: x[0])  # Sort frames by their original order
    
    for _, frame in processed_frames:
        cv2.imshow('Video', frame)
        
        # Calculate the delay time to match the frame rate of the video
        delay_time = int(1000 / frame_rate)
        
        if cv2.waitKey(delay_time) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pool.close()
    pool.join()

    removeDuplicatesAndUnknown()


if __name__ == "__main__":
    main()