import face_recognition
import os
import cv2
import numpy as np
import threading

def get_encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file("faces/" + f)
                encoding = face_recognition.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

def classify_face(im, faces, faces_encoded, known_face_names):

    cap = cv2.VideoCapture(im)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if CUDA is available and use CUDA operations
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Perform CUDA-accelerated resizing
            resized_gpu_frame = cv2.cuda.resize(gpu_frame, (0, 0), fx=0.25, fy=0.25)

            # Perform CUDA-accelerated color conversion from BGR to RGB
            rgb_gpu_frame = cv2.cuda.cvtColor(resized_gpu_frame, cv2.COLOR_BGR2RGB)

            # Download frame from GPU to CPU for face_recognition processing
            rgb_frame = rgb_gpu_frame.download()
        else:
            # Fallback to CPU operations if CUDA is not available
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
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

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom + 30), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

threads = []
for video in ["Test.mp4"]:  
    t = threading.Thread(target=classify_face, args=(video, faces, faces_encoded, known_face_names))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
