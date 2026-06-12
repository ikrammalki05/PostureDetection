import cv2
import pickle
import os


class PostureAnalyzer:

    def __init__(self, model_path="posture_model.pkl"):
        self.model = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"[INFO] Modèle de posture ML chargé depuis {model_path}.")
            except Exception as e:
                print(f"[ERREUR] Impossible de charger le modèle : {e}. Utilisation du mode par défaut.")
        else:
            print("[INFO] Aucun modèle ML trouvé. Utilisation de la règle heuristique par défaut.")

    def analyze_posture(self, landmarks):

        if not landmarks:
            return "No Detection"

        # Si le modèle ML est chargé, on l'utilise pour prédire
        if self.model is not None:
            features = []
            points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            
            # Vérifier que tous les points nécessaires sont présents
            all_present = True
            for point in points:
                if point in landmarks:
                    features.extend([landmarks[point][0], landmarks[point][1]])
                else:
                    all_present = False
                    break
                    
            if all_present:
                prediction = self.model.predict([features])[0]
                return "Bad Posture" if prediction == 1 else "Good Posture"
            else:
                # Si des points manquent pour l'IA, on bascule sur l'heuristique
                pass

        # Mode de repli : l'ancienne règle heuristique (STABILITÉ)
        nose_x, nose_y = landmarks.get("nose", (0, 0))

        left_shoulder_x, left_shoulder_y = landmarks.get("left_shoulder", (0, 0))
        right_shoulder_x, right_shoulder_y = landmarks.get("right_shoulder", (0, 0))

        if (left_shoulder_x, left_shoulder_y) == (0, 0) or (right_shoulder_x, right_shoulder_y) == (0, 0) or (nose_x, nose_y) == (0, 0):
             return "No Detection"

        shoulder_center_x = (left_shoulder_x + right_shoulder_x) // 2

        # Distance tête / épaules
        difference = abs(nose_x - shoulder_center_x)

        if difference > 40:
            return "Bad Posture"

        return "Good Posture"

    def draw_posture_text(self, frame, posture):

        color = (0, 255, 0)

        if posture == "Bad Posture":
            color = (0, 0, 255)

        cv2.putText(
            frame,
            posture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )