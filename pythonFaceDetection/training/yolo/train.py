import os
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")


model.train(
    data="/Users/puja/IdeaProjects/charamelFaceDetection/data/set1_26_june",
    epochs=300,
    patience=30,
    project="/Users/puja/IdeaProjects/charamelFaceDetection/data/set1_26_june/Results"

)