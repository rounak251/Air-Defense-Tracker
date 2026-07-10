# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
from tracker import TargetTracker, RadarDisplayWidget

class TrackingThread(QThread):
    # Custom signal to safely pass data from background thread to Main UI Thread
    frame_processed = pyqtSignal(object, list)

    def __init__(self, video_source):
        super().__init__()
        # Use 0 for your laptop webcam, or a path to a video file (e.g., 'drone.mp4')
        self.tracker = TargetTracker(video_source) 
        self.running = True

    def run(self):
        while self.running:
            frame, targets = self.tracker.process_next_frame()
            if frame is not None:
                self.frame_processed.emit(frame, targets)
            else:
                break

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UTS Tactical Air-Defense Radar & Tracker")
        self.setGeometry(100, 100, 1200, 600)

        # Main Layout split horizontally: Left = Video, Right = Radar UI
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        
        self.video_label = QLabel("Loading Camera Stream...")
        self.radar_widget = RadarDisplayWidget()
        
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.radar_widget, stretch=1)
        self.setCentralWidget(main_widget)

        # Start background processing thread
        self.thread = TrackingThread(0) # 0 uses webcam
        self.thread.frame_processed.connect(self.update_ui)
        self.thread.start()

    def update_ui(self, frame, targets):
        # Update Radar screen coordinates
        self.radar_widget.update_targets(targets)
        
        # Convert OpenCV BGR image matrix to PyQt QPixmap to display camera feed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(600, 500))

    def closeEvent(self, event):
        self.thread.running = False
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())