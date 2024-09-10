import cv2
import subprocess
from fer import FER
import os

video_path = "ThisVideo5.mp4"

# Clean up previous outputs if exist
for file in ["output_with_emotion_and_audio.mp4", "audio.mp3", "output.mp4"]:
    if os.path.exists(file):
        os.remove(file)

# Extract audio first
subprocess.run(["ffmpeg", "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", "audio.mp3"])

# Initialize FER with MTCNN
detector = FER(mtcnn=True)

# Video Capture
cap = cv2.VideoCapture(video_path)

# Get the resolution and FPS of the input video
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

frame_skip = 10  # Skip every 5 frames to speed up the process
frame_count = 0

last_detected_faces = []  # Store the last detected faces and emotions

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update detection only on specific frames
    if frame_count % frame_skip == 0:
        last_detected_faces = detector.detect_emotions(frame)
    # Draw bounding boxes and emotions using the last detected information
    for detected_face in last_detected_faces:
        x, y, w, h = detected_face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        dominant_emotion = max(detected_face['emotions'], key=detected_face['emotions'].get)
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Merge audio and video
subprocess.run(['ffmpeg', '-i', 'output.mp4', '-i', 'audio.mp3', '-shortest', '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', "output_with_emotion_and_audio.mp4"])

# Final cleanup
for file in ["audio.mp3", "output.mp4"]:
    if os.path.exists(file):
        os.remove(file)
