import cv2
import os
from core.detector import HumanDetector
from core.tracker import HumanTracker
from core.verifier import HumanVerifier

class HumanDetectionSystem:
    def __init__(self, video_source="assets/new_video2.mp4"):
        """
        Orchestrator: Connects Detector, Tracker, and Verifier components.
        """
        print("\n--- Starting Human Detection System ---")
        self.detector = HumanDetector()
        self.tracker = HumanTracker()
        self.verifier = HumanVerifier()
        
        # Check if file exists, else use webcam
        source = video_source if os.path.exists(video_source) else 0
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open video source: {source}")

    def run(self):
        print("System Active. Processing... Press 'q' to stop.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. Detection & Tracking
            # Processes detection and maintains track persistence
            results = self.tracker.update(self.detector, frame)
            
            if results.boxes is not None:
                for box in results.boxes:
                    # Basic Info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # ID (Required for liveness history)
                    if box.id is None:
                        continue
                    track_id = int(box.id[0].cpu().numpy())
                    
                    # 2. Liveness Verification
                    is_live, dance_score = self.verifier.verify(frame, track_id, [x1, y1, x2, y2])
                    
                    # 3. Visualization Logic
                    if is_live:
                        color = (0, 255, 0) # Green for verified
                        label = f"VERIFIED HUMAN #{track_id} ({conf:.2f})"
                    else:
                        color = (0, 255, 255) # Yellow for unverified
                        label = f"SCANNING ID #{track_id}..."

                    # 4. Visualization
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(
                        frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
            
            # Display Output
            cv2.imshow("Human Detection System (YOLO + ByteTrack + MediaPipe)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = HumanDetectionSystem()
    system.run()
