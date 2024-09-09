from fer import FER
import cv2

# Initialize the FER object
fer = FER(mtcnn=True)

# Load the image
img = cv2.imread("Bill_Gates.jpg")

# Detect and recognize the emotion
result = fer.detect_emotions(img)

# Loop over the results
for i, emotion in enumerate(result):
    # Get the coordinates of the bounding box for the face
    x1, y1, x2, y2 = emotion['box']

    # Draw the bounding box on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Get the emotion with the highest confidence
    max_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
    max_confidence = emotion['emotions'][max_emotion]

    # Draw the emotion and confidence on the image
    cv2.putText(img, f"{max_emotion} ({max_confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the new image
cv2.imwrite("image_with_emotions.jpg", img)