from ultralytics import YOLO

class HumanDetector:
    def __init__(self, model_path="yolo26s.pt"):
        """
        Initializes the YOLO detector with the specified model.
        """
        print(f"Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        """
        Executes inference on a single frame, filtering for class 0 (Human).
        """
        # conf=0.6: Minimum confidence threshold
        results = self.model(frame, conf=0.6, classes=[0], verbose=False)
        return results[0]
