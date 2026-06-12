import os
import cv2
import csv
from detector.pose_detector import PoseDetector

def extract_features_from_dataset(dataset_dir="dataset", output_csv="posture_data.csv"):
    """
    Parcourt le dossier dataset, extrait les landmarks pour chaque image,
    et les sauvegarde dans un fichier CSV.
    Structure attendue :
    dataset/
      ├── good/ (images avec label 0)
      └── bad/  (images avec label 1)
    """
    
    if not os.path.exists(dataset_dir):
        print(f"Le dossier '{dataset_dir}' n'existe pas. Veuillez le créer et y placer vos images (sous-dossiers 'good' et 'bad').")
        return

    pose_detector = PoseDetector()

    # Points clés extraits par get_landmarks dans pose_detector.py
    points_of_interest = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    
    # En-tête du CSV
    header = ["filename", "label"]
    for point in points_of_interest:
        header.extend([f"{point}_x", f"{point}_y"])

    categories = {"good": 0, "bad": 1}
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for category, label in categories.items():
            category_path = os.path.join(dataset_dir, category)
            if not os.path.exists(category_path):
                print(f"Attention: Le sous-dossier '{category}' est manquant dans '{dataset_dir}'.")
                continue

            for filename in os.listdir(category_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                filepath = os.path.join(category_path, filename)
                frame = cv2.imread(filepath)
                
                if frame is None:
                    print(f"Impossible de lire l'image: {filepath}")
                    continue

                # Détection
                results = pose_detector.detect_pose(frame)
                landmarks = pose_detector.get_landmarks(frame, results)

                if not landmarks:
                    print(f"Aucune posture détectée dans {filename}")
                    continue

                # Préparation de la ligne CSV
                row = [filename, label]
                
                # Vérification que tous les points sont présents
                all_points_found = True
                for point in points_of_interest:
                    if point in landmarks:
                        row.extend([landmarks[point][0], landmarks[point][1]])
                    else:
                        all_points_found = False
                        break
                
                if all_points_found:
                    writer.writerow(row)
                    print(f"Traité avec succès: {filename} (Label: {category})")
                else:
                    print(f"Points manquants dans {filename}")

    print(f"Extraction terminée ! Données sauvegardées dans '{output_csv}'")

if __name__ == "__main__":
    print("Démarrage de l'extraction des caractéristiques...")
    extract_features_from_dataset()
