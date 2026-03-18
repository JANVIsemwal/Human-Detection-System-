class HumanTracker:
    def __init__(self):
        """
        Initializes the tracker logic using ByteTrack.
        """
        self.tracking_active = True

    def update(self, detector, frame):
        """
        Updates tracking state for the current frame.
        """
        # imgsz=1280: High resolution inference for distant detection
        results = detector.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.15,
            imgsz=1280,
            tracker="bytetrack.yaml",
            verbose=False
        )
        return results[0]
