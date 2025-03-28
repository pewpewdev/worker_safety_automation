from ultralytics import YOLO
import cv2

model = YOLO("weights/spillDetectionModel.pt")
video = "videos/spill_1.mp4"

cap = cv2.VideoCapture(video)

