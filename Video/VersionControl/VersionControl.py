import cv2
import subprocess
from fer import FER
import os


if os.path.exists("output_with_emotion_and_audio.mp4"):
    os.remove("output_with_emotion_and_audio.mp4")

# Initialize FER
detector = FER(mtcnn=True)  # Using MTCNN for face detection

# Video Capture
video_path = "ThisVideo1.mp4"  
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect emotions
    result = detector.detect_emotions(frame)

    # Draw bounding boxes and emotions
    for detected_face in result:
        x, y, w, h = detected_face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        dominant_emotion, score = detector.top_emotion(frame)
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

subprocess.run(["ffmpeg", "-i", "ThisVideo1.mp4", "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", "audio.mp3"])

subprocess.run(['ffmpeg', '-i', 'output.mp4', '-i', 'audio.mp3', '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', "output_with_emotion_and_audio.mp4"])

if os.path.exists("audio.mp3"):
    os.remove("audio.mp3")
if os.path.exists("output.mp4"):
    os.remove("output.mp4")
