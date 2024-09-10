import cv2
from fer import FER

# Initialize FER
detector = FER(mtcnn=True)  # Using MTCNN for face detection

# Video Capture
video_path = "ThisVideo5.mp4"  
cap = cv2.VideoCapture(video_path)

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

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
