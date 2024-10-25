import mediapipe as mp
from mediapipe.tasks import python as mp_python
import cv2
import numpy as np
import time
import pickle


MP_TASK_FILE = "face_landmarker_v2_with_blendshapes.task"

class FaceMeshDetector:
    def __init__(self):
        # Load the pre-trained model
        with open('lm_bs_rf_model.p', 'rb') as f:
            self.classifier = pickle.load(f)
            
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.mp_callback)
        self.model = mp_python.vision.FaceLandmarker.create_from_options(options)

        self.landmarks = None
        self.blendshapes = None
        self.latest_time_ms = 0

    def mp_callback(self, mp_result, output_image, timestamp_ms: int):
        if len(mp_result.face_landmarks) >= 1 and len(mp_result.face_blendshapes) >= 1:
            self.landmarks = mp_result.face_landmarks[0]
            self.blendshapes = [b.score for b in mp_result.face_blendshapes[0]]

    def update(self, frame):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            return

        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.model.detect_async(frame_mp, t_ms)
        self.latest_time_ms = t_ms

    def get_results(self):
        return self.landmarks, self.blendshapes

    def draw_bounding_box(self, frame, landmarks):
        if landmarks is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Get face bounding box coordinates
        x_coordinates = [landmark.x * w for landmark in landmarks]
        y_coordinates = [landmark.y * h for landmark in landmarks]
        
        x_min = int(min(x_coordinates))
        x_max = int(max(x_coordinates))
        y_min = int(min(y_coordinates))
        y_max = int(max(y_coordinates))
        
        # Add padding to the bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Draw the bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return frame, (x_min, y_min, x_max, y_max)

    def predict_expression(self, landmark_features):
        if landmark_features is None:
            return None, None
        model = self.classifier['model']
        prediction = model.predict([landmark_features])[0]
        confidence = np.max(model.predict_proba([landmark_features]))  # Use model. instead of self.classifier.
        return prediction, confidence

def main():
    facemesh_detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Update face detection
        facemesh_detector.update(frame)
        landmarks, blendshapes = facemesh_detector.get_results()
        
        if landmarks is not None and blendshapes is not None:
            # Draw bounding box
            frame, bbox = facemesh_detector.draw_bounding_box(frame, landmarks)
            
            landmarks_features = []
            for landmark in landmarks:
                landmarks_features.extend([landmark.x, landmark.y, landmark.z])
            face_feature = np.concatenate(landmarks_features, blendshapes)
            # Get prediction
            prediction, confidence = facemesh_detector.predict_expression(face_feature)
            
            if prediction is not None:
                # Display prediction and confidence
                x_min, y_min, _, _ = bbox
                text_position = (x_min, y_min - 10)
                
                # Add background rectangle for text
                text = f"{prediction} ({confidence:.2f})"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, 
                            (text_position[0] - 5, text_position[1] - text_h - 5),
                            (text_position[0] + text_w + 5, text_position[1] + 5),
                            (0, 255, 0), -1)
                
                # Add text
                cv2.putText(frame, text, text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow('Face Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()