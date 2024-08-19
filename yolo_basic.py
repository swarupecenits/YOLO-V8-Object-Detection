from ultralytics import YOLO
import numpy

model = YOLO("weights/yolov8n.pt")  


detection_output = model.predict(source="inference/images/img0.JPG", conf=0.25, save=True)

# Display the detection results
print(detection_output)

print(detection_output[0].numpy())
