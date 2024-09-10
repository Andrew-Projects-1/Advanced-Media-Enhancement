import cv2
from fer import FER
import multiprocessing as mp
import time

def process_frame(frame_queue, results_queue):
    detector = FER(mtcnn=True)
    while True:
        item = frame_queue.get()
        if item is None:
            break
        frame, frame_id = item
        
        result = detector.detect_emotions(frame)
        # Include the original frame in the result for display
        results_queue.put((frame, result))
        
def display_frame(results_queue, desired_fps=30):
    prev_frame_time = 0
    while True:
        item = results_queue.get()
        if item is None:
            break
        frame, result = item
        
        for detected_face in result:
            x, y, w, h = detected_face['box']
            dominant_emotion = detected_face['emotions']
            dominant_emotion = max(dominant_emotion, key=dominant_emotion.get)  # Get the emotion with the highest score

            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Sleep to maintain desired FPS
        time.sleep(max(1./desired_fps - (new_frame_time - prev_frame_time), 0))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    video_path = "ThisVideo5.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frame_queue = mp.Queue()
    results_queue = mp.Queue()

    num_consumers = mp.cpu_count()
    consumers = [mp.Process(target=process_frame, args=(frame_queue, results_queue)) for _ in range(num_consumers)]
    for c in consumers:
        c.start()
        
    display_process = mp.Process(target=display_frame, args=(results_queue,))
    display_process.start()

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Add a frame identifier if needed for tracking
        frame_queue.put((frame.copy(), frame_id))  
        frame_id += 1

    for _ in consumers:
        frame_queue.put(None)
    for c in consumers:
        c.join()
        
    results_queue.put(None)
    display_process.join()
    
    cap.release()
    cv2.destroyAllWindows()
