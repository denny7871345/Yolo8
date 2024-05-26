import numpy as np
import cv2
import pprint
import pydirectinput
from ultralytics import YOLO

model = YOLO('yolov8m.pt')


def webcam_grab():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return None

    # Read frame from webcam
    ret, frame = cap.read()

    # Release webcam
    cap.release()

    return frame


def detect_objects(frame):
    results = model(frame)
    return results


if __name__ == '__main__':
    # Set parameters
    detection_interval = 5  # 每隔5幀進行一次物體檢測
    cap = cv2.VideoCapture(0)
    while True:
        # Grab frame from webcam
        frame = webcam_grab()

        # Perform object detection
        if frame is not None:
            # Resize frame to reduce processing time
            resized_frame = cv2.resize(frame, (640, 480))  # 將畫面縮小為640x480

            # Perform object detection every 'detection_interval' frames
            if detection_interval == 0:
                results = detect_objects(resized_frame)

                # Display results
                image = results[0].plot()
                print('v8 results:', results[0].names)
                boxes = results[0].boxes.data
                print(boxes)
                formatted_boxes = [[float(f"{num:.2f}") for num in box.tolist()] for box in boxes]
                pprint.pprint(formatted_boxes)

                # Show image with detected objects
                cv2.imshow('result', image)
                cv2.setWindowProperty('result', cv2.WND_PROP_TOPMOST, 1)

                # Reset detection interval
                detection_interval = 5
            else:
                # Decrement detection interval
                detection_interval -= 1

            # Get mouse position
            x, y = pydirectinput.position()
            print("mouse:", x, y)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
