import cv2

from detector.pose_detector import PoseDetector
from detector.posture_analyzer import PostureAnalyzer

from utils.drawing_utils import draw_points


# Initialisation
pose_detector = PoseDetector()

posture_analyzer = PostureAnalyzer()


# Webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Détection posture
    results = pose_detector.detect_pose(frame)

    # Dessiner squelette
    pose_detector.draw_landmarks(frame, results)

    # Récupérer coordonnées
    landmarks = pose_detector.get_landmarks(frame, results)

    # Dessiner points
    draw_points(frame, landmarks)

    # Analyse posture
    posture = posture_analyzer.analyze_posture(landmarks)

    # Afficher posture
    posture_analyzer.draw_posture_text(frame, posture)

    # Affichage final
    cv2.imshow("Smart Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()