import cv2
import os
from core.detector import HumanDetector
from core.tracker import HumanTracker

def test_tracking(video_source=0):
    """
    Tests tracking. source can be 0 (webcam) or a path to a video file.
    """
    print(f"\n--- Initializing YOLO26 Tracking System ---")
    print(f"Source: {'Webcam' if video_source == 0 else video_source}")
    
    detector = HumanDetector()
    tracker = HumanTracker()
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {video_source}")
        return

    print("Tracking started! Press 'q' in the window to EXIT.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera disconnected.")
            break
            
        # Run Tracker
        results = tracker.update(detector, frame)
        
        # Draw Results
        if results.boxes is not None:
            for box in results.boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # ID
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else "Detecting..."
                
                # Confidence
                conf = float(box.conf[0].cpu().numpy())
                
                # Draw
                color = (0, 255, 0) # Green
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame, f"Human #{track_id} ({conf:.2f})", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
        
        cv2.imshow("Tracking Verification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If there is a file called assets/my_video.mp4, use it. Otherwise, use webcam.
    video_file = "assets/new_video2.mp4" # You can rename your phone video to this
    
    if os.path.exists(video_file):
        test_tracking(video_file)
    else:
        test_tracking(0)
