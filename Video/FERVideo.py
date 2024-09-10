# Andrew Moskowitz
##
# This program processes a video file by analyzing and detecting emotions on human faces in each frame 
# using the FER (Facial Expression Recognition) library. It reads the video frame by frame, detects faces, 
# and identifies the primary emotion (e.g., happy, sad, angry) along with its confidence level. The detected 
# emotions are displayed in a green box alongside each face. The program ensures that the display updates at a 
# consistent time interval, regardless of the video's frame rate (FPS), by calculating how many frames to skip 
# between updates. The resulting video, including the emotion detections, is then saved. Additionally, 
# it merges the original video's audio back into the output if audio is present.
##

import cv2
import subprocess
from fer import FER
import os

# Input file
INPUT_VIDEO_FILE = "Test1.mp4" 

# Output file
OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FER_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# Check if input file exists
if not os.path.exists(INPUT_VIDEO_FILE):
    print("Input file does not exist.")
    exit()

# Remove the output file if it already exists
if os.path.exists(OUTPUT_VIDEO_FILE):
    os.remove(OUTPUT_VIDEO_FILE)

# Check if the input file is .avi or .mov
if INPUT_VIDEO_FILE.endswith('.avi') or INPUT_VIDEO_FILE.endswith('.mov'):
    # If so, convert .avi or .mov file to mp4
    subprocess.run(["ffmpeg", "-i", INPUT_VIDEO_FILE, "-vcodec", "h264", "-acodec", "aac", "-strict", "2", "-preset", "slow", "temp.mp4"])
    INPUT_VIDEO_FILE = "temp.mp4"
    OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FER_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# Extract audio if audio track exists
result = subprocess.run(['ffmpeg', '-i', INPUT_VIDEO_FILE], stderr=subprocess.PIPE, universal_newlines=True)
if "Stream #0:1" in result.stderr and "Audio" in result.stderr:
    subprocess.run(["ffmpeg", "-i", INPUT_VIDEO_FILE, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", "audio.mp3"])
    audio_exists = True
else:
    print("No audio track found in the video.")
    audio_exists = False

# Initialize FER with MTCNN
fer = FER(mtcnn=True)

# Video Capture
cap = cv2.VideoCapture(INPUT_VIDEO_FILE)

# Get the resolution and FPS of the input video
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set the desired update interval
update_interval_sec = 0.5  # Update every 0.5 seconds

# Calculate the number of frames to skip based on the video's FPS
frame_skip = int(fps * update_interval_sec)  # Number of frames to skip to achieve constant update speed

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

frame_count = 0 # Track how many frames have been processed
last_detected_faces = []  # Store the last detected faces and emotions

while True:
    # Capture and read frames from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions for the current frame
    if frame_count % frame_skip == 0:
        last_detected_faces = fer.detect_emotions(frame)

    # Process each detected face for emotion
    for detected_face in last_detected_faces:
        x, y, w, h = detected_face['box'] # Get dimensions of box around face

        # Get the primary emotion and its confidence
        primary_emotion, confidence = max(detected_face['emotions'].items(), key=lambda item: item[1])
        emotion_text = f"{primary_emotion} ({confidence:.2f})"

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

         # Calculate the text scale based on the size of the green box
        text_scale = 1  # Start with a default scale

        # Calculate maximum possible text size that fits inside the green box
        max_text_width = w  # The width of the green box
        max_text_height = h * 0.3  # Text should take up 30% of the height of the green box

        # Find the maximum scale where the text still fits within the green box
        while True:
            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)[0]
            if text_size[0] <= max_text_width and text_size[1] <= max_text_height:
                break
            text_scale -= 0.1  # Reduce text scale until it fits

        # Render the emotion text and background below the bounding box
        emotion_text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)[0]
        text_x = x + (w - emotion_text_size[0]) // 2

        # Draw the filled rectangle under face box
        cv2.rectangle(frame, (x - 1, y + h), (x + w + 1, y + h + 15 + emotion_text_size[1]), (0, 255, 0), cv2.FILLED)
        
        # Put the centered text
        cv2.putText(frame, emotion_text, (text_x, y + h + emotion_text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 1)

    # Write the processed frame to the output video
    out.write(frame)
    frame_count += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Merge audio and video
if audio_exists:
    subprocess.run(['ffmpeg', '-i', 'output.mp4', '-i', 'audio.mp3', '-shortest', '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', OUTPUT_VIDEO_FILE])
else:
    # Just rename the output video if there's no audio
    os.rename('output.mp4', OUTPUT_VIDEO_FILE)

# Final cleanup
for file in ["audio.mp3", "output.mp4", "temp.mp4"]:
    if os.path.exists(file):
        os.remove(file)
