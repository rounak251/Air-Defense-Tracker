# tracker.py
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtCore import QTimer, Qt

class TargetTracker:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        # Using a lightweight nano model for real-time edge processing constraints
        self.model = YOLO("yolov8n.pt") 
        # Background subtractor to isolate small targets against cloud/sky clutter
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    def process_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, []

        # 1. Apply image processing to reduce noise (Sky Clutter reduction)
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # 2. Frame Inference 
        results = self.model(frame, verbose=False)[0]
        detected_targets = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            
            # Filter specifically for drone-like targets (birds or airplanes)
            if label in ['bird', 'airplane']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Math Logic: Calculate simulated distance (Range) and Azimuth angle relative to camera sensor center
                img_h, img_w, _ = frame.shape
                norm_x = (cx - (img_w / 2)) / (img_w / 2)
                norm_y = ((img_h / 2) - cy) / (img_h / 2)
                
                azimuth = np.degrees(np.arctan2(norm_x, 1))
                simulated_range = np.sqrt(norm_x**2 + norm_y**2) * 5000  # Map to 5km max range
                
                detected_targets.append({
                    "class": "Drone Target" if label == "airplane" else "Unknown Wildlife",
                    "azimuth": round(azimuth, 2),
                    "range_meters": round(simulated_range, 2),
                    "bbox": (x1, y1, x2, y2)
                })
                
        return frame, detected_targets


class RadarDisplayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.sweep_angle = 0
        self.tracked_targets = []
        
        # UI Refresh Timer matching standard defense display update loops
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_radar_sweep)
        self.timer.start(30) # ~33 FPS

    def update_radar_sweep(self):
        self.sweep_angle = (self.sweep_angle + 2) % 360
        self.update() # Triggers paintEvent

    def update_targets(self, targets):
        self.tracked_targets = targets

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw tactical dark theme background
        width, height = self.width(), self.height()
        center_x, center_y = width // 2, height // 2
        radius = min(center_x, center_y) - 20
        
        painter.fillRect(self.rect(), QColor(10, 15, 10))
        
        # Draw Concentric Radar Rings
        pen = QPen(QColor(0, 100, 0), 1, Qt.DotLine)
        painter.setPen(pen)
        for r in range(radius // 4, radius + 1, radius // 4):
            painter.drawEllipse(center_x - r, center_y - r, r * 2, r * 2)

        # Draw Spinning Radar Sweep Line
        sweep_rad = np.radians(self.sweep_angle)
        end_x = int(center_x + radius * np.sin(sweep_rad))
        end_y = int(center_y - radius * np.cos(sweep_rad))
        painter.setPen(QPen(QColor(0, 255, 0, 150), 2))
        painter.drawLine(center_x, center_y, end_x, end_y)

        # Plot Active Detected Targets
        painter.setFont(QFont("Consolas", 9))
        for target in self.tracked_targets:
            # Polar-to-Cartesian conversion for UI mapping
            t_rad = np.radians(target['azimuth'])
            t_dist = (target['range_meters'] / 5000) * radius
            
            tx = int(center_x + t_dist * np.sin(t_rad))
            ty = int(center_y - t_dist * np.cos(t_rad))
            
            # Draw tactical target blip
            painter.setPen(QPen(QColor(255, 50, 50), 3))
            painter.setBrush(QColor(255, 50, 50, 100))
            painter.drawRect(tx - 5, ty - 5, 10, 10)
            
            # Text telemetry tagging
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(tx + 12, ty, f"{target['class']}")
            painter.drawText(tx + 12, ty + 15, f"R: {target['range_meters']}m | Az: {target['azimuth']}°")