from ultralytics import YOLO
import numpy as np
import cv2 as cv
import random as rand

# Load class names
with open("utils/coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Generate random colors for each class
detection_colors = [
    (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
    for _ in range(len(class_list))
]

# Load the YOLOv8 model
model = YOLO("weights/yolov8n.pt")

# Resize video frame
frame_width = 640
frame_height = 480

# Open the video file
cap = cv.VideoCapture("inference/videos/afriq1.MP4")

if not cap.isOpened():
    print("Unable to Open Video File")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to receive frame... Exiting...")
        break

    # Resize the frame for faster processing
    frame = cv.resize(frame, (frame_width, frame_height))

    # Make predictions
    detection_params = model(frame)

    # Access the first prediction result (assuming single frame prediction)
    results = detection_params[0]

    if results and len(results.boxes) > 0:
        for box in results.boxes:
            # Extract bounding box coordinates (x_min, y_min, x_max, y_max)
            bb = box.xyxy[0].cpu().numpy().astype(int)

            # Extract class ID and confidence
            classID = int(box.cls.cpu().numpy()[0])
            confidence = box.conf.cpu().numpy()[0]

            # Draw the bounding box
            cv.rectangle(
                frame,
                (bb[0], bb[1]), (bb[2], bb[3]),
                detection_colors[classID], 2
            )

           
            label = f"{class_list[classID]}: {round(confidence * 100, 2)}%"
            cv.putText(
                frame, label,
                (bb[0], bb[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.8,
                
            detection_colors[classID], 2
        )

    # Display the frame with object detection
    cv.imshow("Object Detection", frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv.destroyAllWindows()
