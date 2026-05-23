from ultralytics import YOLO
import cv2
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")
print("YOLO loaded successfully")
