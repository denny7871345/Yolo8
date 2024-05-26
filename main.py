import numpy as np
from PIL import Image, ImageGrab
import cv2
import pprint
import keyboard
import win32api
import pydirectinput
from ultralytics import YOLO

modol = YOLO('yolov8n.pt')

def pil_grab(target):
    
    game_frame_np = cv2.cvtColor(np.array(target),cv2.COLOR_RGB2BGR)
    results = modol(game_frame_np)
    image = results[0].plot()
    print('v8 results:' ,results[0].names)

    boxes = results[0].boxes.data
    print(boxes)
    formatted_boxes = []
    for box in boxes:
        formatted_box = [float(f"{num:.2f}") for num in box.tolist()]
        formatted_boxes.append(formatted_box)
    pprint.pprint(formatted_boxes)

    return image,formatted_boxes

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        exit()

    output_width = 800
    output_height = 800

    window_x = 0
    window_y = 0

    while(1):

        ret, frame = cap.read()
        frame = cv2.resize(frame, (output_width, output_height))
        image,formatted_boxes = pil_grab(frame)

        cv2.namedWindow('result',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result',800,800)
        cv2.imshow('result',image)
        cv2.setWindowProperty('result',cv2.WND_PROP_TOPMOST,1)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()