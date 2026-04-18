import numpy as np

class GeometricEmotionDetector:
    def __init__(self):
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291

        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374

        self.LEFT_BROW_TOP = 52
        self.RIGHT_BROW_TOP = 282

    def analyze(self, landmarks):
        if landmarks is None or len(landmarks) < 400:
            return None

        mouth_height = np.linalg.norm(np.array(landmarks[self.MOUTH_TOP][:2]) - np.array(landmarks[self.MOUTH_BOTTOM][:2]))
        mouth_width = np.linalg.norm(np.array(landmarks[self.MOUTH_LEFT][:2]) - np.array(landmarks[self.MOUTH_RIGHT][:2]))
        mar = mouth_height / (mouth_width + 1e-6)

        mouth_center_y = (landmarks[self.MOUTH_TOP][1] + landmarks[self.MOUTH_BOTTOM][1]) / 2
        avg_corner_y = (landmarks[self.MOUTH_LEFT][1] + landmarks[self.MOUTH_RIGHT][1]) / 2
        lip_curvature = avg_corner_y - mouth_center_y

        left_eye_h = np.linalg.norm(np.array(landmarks[self.LEFT_EYE_TOP][:2]) - np.array(landmarks[self.LEFT_EYE_BOTTOM][:2]))
        right_eye_h = np.linalg.norm(np.array(landmarks[self.RIGHT_EYE_TOP][:2]) - np.array(landmarks[self.RIGHT_EYE_BOTTOM][:2]))
        eye_openness = (left_eye_h + right_eye_h) / 2

        left_brow_dist = np.linalg.norm(np.array(landmarks[self.LEFT_BROW_TOP][:2]) - np.array(landmarks[self.LEFT_EYE_TOP][:2]))
        right_brow_dist = np.linalg.norm(np.array(landmarks[self.RIGHT_BROW_TOP][:2]) - np.array(landmarks[self.RIGHT_EYE_TOP][:2]))
        brow_height = (left_brow_dist + right_brow_dist) / 2

        geo_guess = "Neutral"
        confidence = 0.0

        is_eyes_wide = eye_openness > 0.038
        is_mouth_open_narrow = (mar > 0.15 and mar < 0.4)

        if mar > 0.5:
            geo_guess = "Surprise"
            confidence = min(1.0, (mar - 0.5) / 0.2)
        elif is_eyes_wide and is_mouth_open_narrow:
            geo_guess = "Fear"
            confidence = min(1.0, (eye_openness - 0.038) / 0.01)
        elif lip_curvature < -0.015:
            geo_guess = "Happy"
            confidence = min(1.0, (abs(lip_curvature) - 0.015) / 0.015)
        elif lip_curvature > 0.005:
            geo_guess = "Sad"
            confidence = min(1.0, (lip_curvature - 0.005) / 0.01)
        elif brow_height < 0.025:
            geo_guess = "Angry"
            confidence = min(1.0, (0.025 - brow_height) / 0.015)
        elif eye_openness > 0.045:
            geo_guess = "Fear"
            confidence = min(1.0, (eye_openness - 0.045) / 0.01)
            
        return {
            "guess": geo_guess,
            "confidence": confidence * 100,
            "mar": mar,
            "curvature": lip_curvature,
            "eye_open": eye_openness,
            "brow_h": brow_height
        }
