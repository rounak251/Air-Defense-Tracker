# Air-Defense Tactical Tracking & Radar GUI Simulator

An advanced multi-threaded Python application designed for real-time target detection, classification (e.g., UAVs/Drones), and radar coordinate visualization. Built specifically for defense-software research tracking applications.

## 🛠️ Features
- **Computer Vision Pipeline:** Frame processing utilizing OpenCV background isolation alongside YOLOv8 neural network inference models.
- **Dynamic Coordinate Conversion:** Generates automated Range and Azimuth vectors from normalized bounding-box center weights.
- **Defense Dashboard GUI:** Real-time tactical Plan Position Indicator (PPI) radar display engineered via PyQt5.

## 🚀 Setup Instructions
1. Clone the repository.
2. Install dependencies: `pip install ultralytics opencv-python numpy PyQt5`
3. Execute the launcher pipeline: `python main.py`