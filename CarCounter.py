import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *
import numpy as np
import csv
from datetime import datetime
import os

# Initialize the video capture
cap = cv2.VideoCapture('Cars.mp4')
cap.set(3, 1280)
cap.set(4, 720)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Create directories if they do not exist
os.makedirs('Detected Ingoing', exist_ok=True)
os.makedirs('Detected Outgoing', exist_ok=True)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('processed_video.mp4', fourcc, frame_rate, (frame_width, frame_height))

classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load the YOLO model
model = YOLO("./YOLO_Weights/yolov8n-seg.pt")
mask = cv2.imread("./Mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define limits for the counting line
limits = [0, 350, 1280, 350]

# Initialize counts
totalCountIn = []
totalCountOut = []

# Open the CSV file for writing
with open('./Detection_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'Ingoing Count', 'Outgoing Count', 'Frame Number'])

    frame_number = 0

    while True:
        success, image = cap.read()
        if not success:
            break

        imageRegion = cv2.bitwise_and(image, mask)
        results = model(imageRegion, stream=True)
        detections = np.empty((0, 5))

        # Draw the counting line
        line = cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().numpy()
                score = math.ceil(box.conf[0] * 100) / 100
                classIndex = int(box.cls[0])
                currentClass = classNames[classIndex]
                if currentClass in ['car', 'truck', 'bus', 'motorbike']:
                    currentArray = np.array([x1, y1, x2, y2, score])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, ID = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2

            cvzone.cornerRect(image, (x1, y1, w, h), l=9, rt=2)
            cvzone.putTextRect(image, f'{int(ID)}', (max(x1, 0), max(35, y1)), scale=2, thickness=3, offset=10)

            if limits[0] < cx < limits[2] and limits[1] - 25 < cy < limits[1] + 25:
                crop_img = image[y1:y2, x1:x2]  # Crop the detected car region
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                if cx < (limits[2] // 2):  
                    # Ingoing (left half)
                    if totalCountIn.count(ID) == 0:
                        totalCountIn.append(ID)
                        #Saving the detected car image
                        cv2.imwrite(f'Detected Ingoing/{current_time}_{int(ID)}.jpg', crop_img)
                        line = cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
                else:  
                    # Outgoing (right half)
                    if totalCountOut.count(ID) == 0:
                        totalCountOut.append(ID)
                        #Saving the detected car image
                        cv2.imwrite(f'Detected Outgoing/{current_time}_{int(ID)}.jpg', crop_img)
                        line = cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)

        # Get the current time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log the data to CSV
        writer.writerow([current_time, len(totalCountIn), len(totalCountOut), frame_number])

        # Display counts
        cvzone.putTextRect(image, f'In going: {len(totalCountIn)}', (50, 50))
        cvzone.putTextRect(image, f'Out going: {len(totalCountOut)}', (900, 50))

        # Write the processed frame to the video
        out.write(image)

        # Display the image with bounding boxes and counts
        cv2.imshow('Image', image)
        
        frame_number += 1
        
        # Keep displaying processed frames until q is pressed 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video writer and capture objects
out.release()
cap.release()
cv2.destroyAllWindows()
