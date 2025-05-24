# Unified Car Detection, Tracking, and Classification Pipeline using Fast R-CNN

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from collections import deque

from pathlib import Path

car_object_detection = Path('C:\Users\kvika\Desktop\Main Paper\Source Code/car-object-detection')
Car_tracking = Path('C:\Users\kvika\Desktop\Main Paper\Source Code/car-tracking')
car_classification= Path('C:\Users\kvika\Desktop\Main Paper\Source Code/car-classification')
# Load Fast R-CNN model (Faster R-CNN)
detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.eval()

# Load classification model (DenseNet + MultiMask CNN)
class CarClassifier(torch.nn.Module):
    def __init__(self):
        super(CarClassifier, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 3)  # Sedan, SUV, Truck

    def forward(self, x):
        return self.model(x)

classifier_model = CarClassifier()
classifier_model.eval()

# Preprocessing for classification
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tracking setup
track_history = {}
id_counter = 0
max_history = 50

# Initialize video capture
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    img_tensor = transforms.ToTensor()(frame).unsqueeze(0)

    # Detection
    with torch.no_grad():
        detections = detection_model(img_tensor)[0]

    new_track_history = {}

    for i in range(len(detections['boxes'])):
        score = detections['scores'][i].item()
        label_id = detections['labels'][i].item()

        # Only detect cars (label_id 3 for car in COCO)
        if label_id == 3 and score > 0.6:
            box = detections['boxes'][i].int().tolist()
            x1, y1, x2, y2 = box
            car_crop = frame[y1:y2, x1:x2]

            # Classification
            img = Image.fromarray(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = classifier_model(input_tensor)
                pred_class = torch.argmax(output, 1).item()
                label = ['Sedan', 'SUV', 'Truck'][pred_class]

            # Draw detection + classification
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Track based on centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            matched = False
            for tid, points in track_history.items():
                if abs(points[-1][0] - cx) < 30 and abs(points[-1][1] - cy) < 30:
                    points.append((cx, cy))
                    new_track_history[tid] = points[-max_history:]
                    matched = True
                    break
            if not matched:
                new_track_history[id_counter] = deque([(cx, cy)], maxlen=max_history)
                id_counter += 1

    track_history = new_track_history

    # Draw tracks
    for tid, points in track_history.items():
        for j in range(1, len(points)):
            cv2.line(frame, points[j - 1], points[j], (0, 0, 255), 2)

    cv2.imshow('Car Detection + Tracking + Classification (Fast R-CNN)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
