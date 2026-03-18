# integration_example.py

from core.detector import HumanDetector

def start_new_app():
    print("--- Starting External App ---")
    
    # 1. Initialize your detector
    # (It will automatically load yolo26s.pt)
    my_detector = HumanDetector()
    
    # 2. In your new app, you just need to do this:
    # results = my_detector.model.predict("some_image.jpg")
    
    print("\n✅ Success! Your 'HumanDetector' is now a reusable library.")
    print("Just copy the 'core' folder to any new project to use it.")

if __name__ == "__main__":
    start_new_app()
