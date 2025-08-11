import cv2
from ultralytics import YOLO
import os

# Load YOLO model once
model = YOLO(os.path.join("model", "best (4).pt"))

def run_detection(image_path, output_path):
    results = model(image_path, save=True, conf=0.25)
    
    # YOLO saves output inside 'runs/detect/predict...'
    # Move it to our desired output folder
    latest_folder = sorted(os.listdir("runs/detect"), key=lambda x: os.path.getmtime(os.path.join("runs/detect", x)))[-1]
    latest_path = os.path.join("runs/detect", latest_folder, os.path.basename(image_path))
    
    os.rename(latest_path, output_path)
    return output_path
