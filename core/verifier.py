import mediapipe as mp
import numpy as np
from collections import deque

import mediapipe as mp

class HumanVerifier:
    def __init__(self, history_size=15, threshold=0.0005):
        """
        Initializes the Pose-based Liveness Verifier.
        """
        # Standard MediaPipe Solutions API
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # 2. History of movements for each Track ID
        # Format: { track_id: deque([(x,y), (x,y), ...]) }
        self.history = {}
        self.history_size = history_size
        self.threshold = threshold

    def verify(self, frame, track_id, bbox):
        """
        Verifies if a specific Tracked ID is a "Live Human".
        
        Args:
            frame: The current image frame
            track_id: The ID from the tracker
            bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 1. Crop the frame to the person to save CPU
        crop = frame[max(0, y1):min(frame.shape[0], y2), 
                    max(0, x1):min(frame.shape[1], x2)]
        
        if crop.size == 0:
            return False, 0
            
        # 2. Run Pose Detection on the crop
        results = self.pose.process(crop)
        
        if not results.pose_landmarks:
            return False, 0 # No skeleton found
            
        # 3. Landmark Extraction
        # Extracts key landmarks for liveness verification
        nose = results.pose_landmarks.landmark[0]
        pos = (nose.x, nose.y)
        
        # 4. History Update
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_size)
        
        self.history[track_id].append(pos)
        
        # 5. Variance Calculation
        if len(self.history[track_id]) < self.history_size:
            # Buffer history before verification
            return False, 0 
            
        # Convert history to a numpy array to calculate variance
        coords = np.array(self.history[track_id])
        variance = np.var(coords, axis=0).mean()
        
        # 6. Final Verdict
        # Increased threshold to 0.001 to handle minor hand shake/noise
        is_live = variance > 0.001
        
        return is_live, variance
