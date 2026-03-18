import cv2
from core.detector import HumanDetector

def test_system():
    # 1. Initialize detector
    print("--- Initializing Detector ---")
    detector = HumanDetector()
    
    # 2. Load the test image you downloaded
    image_path = "assets/test_person.jpg"
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not find image at {image_path}")
        return

    # 3. Run detection
    print(f"--- Running Detection on {image_path} ---")
    results = detector.detect(frame)
    
    # 4. Analyze results
    num_humans = len(results.boxes)
    print(f"Detection Complete! Found {num_humans} human(s).")
    
    # 5. Visualization
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Human {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection Test", frame)
    print("Test complete. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_system()
