# Human Detection & Liveness Verification System

A high-performance pipeline for human detection, persistent tracking, and pose-based liveness verification. This system integrates SOTA computer vision models with temporal analysis to distinguish between live humans and static objects.

## 🚀 Overview

This project provides a robust solution for environments requiring more than just simple object detection. By combining **YOLOv11+** for precision, **ByteTrack** for identity persistence, and **MediaPipe Pose** for liveness verification, the system can reliably filter out non-human subjects and static representations (photos, posters, statues).

### Key Features
- **Real-time Detection**: Optimized YOLO inference for high-accuracy human detection.
- **Persistent Tracking**: Multi-object tracking (ByteTrack) to maintain IDs across frames.
- **Liveness Verification**: Temporal variance analysis of skeletal landmarks to verify biological movement.
- **Modular Design**: Decoupled core logic (Detector, Tracker, Verifier) for easy integration into external applications.

## 🛠 Tech Stack

- **Detection**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (S-variant for crowd accuracy)
- **Tracking**: ByteTrack Algorithm for consistent ID mapping.
- **Verification**: [MediaPipe](https://github.com/google/mediapipe) Pose estimation for biomechanical analysis.
- **Core Engine**: Python 3.x, OpenCV, NumPy.

## 📦 Project Structure

```text
HumanDetector/
├── core/               # Engine components
│   ├── detector.py     # YOLO-based human detection
│   ├── tracker.py      # ByteTrack implementation
│   └── verifier.py     # Liveness verification logic
├── assets/             # Test videos and images
├── main.py             # System orchestrator
├── check_setup.py      # Environment verification script
├── export_model.py     # ONNX export utility
└── requirements.txt    # Project dependencies
```

## 🛠 Installation & Setup

### 1. Prerequisite: Python Environment
It is recommended to use a virtual environment to manage dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🖥 Usage

### Run the System
Execute the main orchestrator to start the full detection and verification pipeline:
```bash
python main.py
```

### Verification Scripts
Use these to test specific components:
- **Baseline Detection**: `python check_setup.py` (Tests detection on a static image)
- **Tracking Accuracy**: `python check_tracking.py` (Tests ID persistence)

### Model Export
Export the trained weights to ONNX for cross-platform deployment:
```bash
python export_model.py
```

## 🔧 Modular Integration

The `core/` folder is designed to be plug-and-play. To use the `HumanDetector` in your own project:

```python
from core.detector import HumanDetector

detector = HumanDetector(model_path="yolo26s.pt")
results = detector.detect(base_frame)
```

## 📈 Future Improvements
- [ ] **GPU Optimization**: Full TensorRT integration for 60+ FPS performance.
- [ ] **Advanced Liveness**: Multi-landmark breath/blink detection for higher security.
- [ ] **Edge Deployment**: Pre-defined exports for TFLite and CoreML.

---
Developed as a professional-grade human monitoring solution.
