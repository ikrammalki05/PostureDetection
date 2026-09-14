```python
"""Application locale de détection de posture avec webcam."""

import cv2

from detector.pose_detector import PoseDetector
from detector.posture_analyzer import PostureAnalyzer
from utils.drawing_utils import draw_points


def main():
    """Lance la détection de posture avec la webcam."""

    # Initialisation
    pose_detector = PoseDetector()
    posture_analyzer = PostureAnalyzer()

    # Ouverture de la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    print("Webcam démarrée.")
    print("Appuyez sur Q pour quitter.")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Erreur : impossible de lire la webcam.")
                break

            # Détection de la pose
            results = pose_detector.detect_pose(frame)

            # Dessiner le squelette
            pose_detector.draw_landmarks(
                frame,
                results
            )

            # Récupérer les coordonnées
            landmarks = pose_detector.get_landmarks(
                frame,
                results
            )

            # Dessiner les points
            draw_points(
                frame,
                landmarks
            )

            # Analyse de la posture
            posture = posture_analyzer.analyze_posture(
                landmarks
            )

            # Afficher le résultat
            posture_analyzer.draw_posture_text(
                frame,
                posture
            )

            # Afficher la vidéo
            cv2.imshow(
                "Smart Posture Detection",
                frame
            )

            # Quitter avec Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()

        print("Webcam arrêtée.")


if __name__ == "__main__":
    main()
```
