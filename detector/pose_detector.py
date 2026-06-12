import cv2
import mediapipe as mp


class PoseDetector:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

    def detect_pose(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)

        return results

    def draw_landmarks(self, frame, results):

        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

    def get_landmarks(self, frame, results):

        landmarks_positions = {}

        if results.pose_landmarks:

            h, w, _ = frame.shape

            landmarks = results.pose_landmarks.landmark

            important_points = {
                "nose": self.mp_pose.PoseLandmark.NOSE,
                "left_shoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                "right_shoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP,
                "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP,
            }

            for name, landmark_id in important_points.items():

                landmark = landmarks[landmark_id]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                landmarks_positions[name] = (x, y)

        return landmarks_positions