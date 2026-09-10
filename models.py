"""AI Models: MiDaS Monocular Depth & YOLOv8 Object Detector."""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import config

class PerceptionPipeline:
    def __init__(self):
        # Choose Apple Silicon MPS GPU if available, else CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load MiDaS Small
        print(f"[AI] Loading MiDaS depth estimator on {self.device}...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        # Load YOLOv8 Nano
        print("[AI] Loading YOLOv8n object detector...")
        self.yolo = YOLO("yolov8n.pt")

    @torch.no_grad()
    def estimate_depth(self, rgb_frame):
        img_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        prediction = self.midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_raw = prediction.cpu().numpy()
        # Invert & normalize inverse relative depth to metric approximation
        depth_norm = cv2.normalize(depth_raw, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        depth_metric = (1.0 - depth_norm) * (config.MAX_DEPTH_METERS - config.MIN_DEPTH_METERS) + config.MIN_DEPTH_METERS
        return depth_metric.astype(np.float32)

    def detect_objects(self, rgb_frame):
        results = self.yolo(rgb_frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            label = results.names[cls_id]
            conf = float(box.conf[0].item())
            detections.append({"bbox": (x1, y1, x2, y2), "label": label, "conf": conf})
        return detections
