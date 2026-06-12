import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_posture_model(csv_file="posture_data.csv", model_output="posture_model.pkl"):
    """
    Lit le fichier CSV, entraîne un modèle Random Forest et le sauvegarde.
    """
    if not os.path.exists(csv_file):
        print(f"Le fichier de données '{csv_file}' est introuvable.")
        print("Veuillez d'abord lancer 'extract_features.py' pour générer les données.")
        return

    print("Chargement des données...")
    # Lire le CSV avec pandas
    df = pd.read_csv(csv_file)

    if df.empty:
        print("Le fichier CSV est vide. L'entraînement est annulé.")
        return

    # Séparer les features (X) et le label (y)
    # On ignore la colonne 'filename' et 'label' pour X
    X = df.drop(columns=['filename', 'label'])
    y = df['label']

    print(f"Nombre d'échantillons: {len(X)}")

    # Diviser les données en entraînement et test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entraînement du modèle (Random Forest)...")
    # Initialiser et entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluer le modèle
    print("Évaluation sur les données de test...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPrécision (Accuracy): {accuracy * 100:.2f}%")
    print("\nRapport de Classification :")
    print(classification_report(y_test, y_pred, target_names=["Good (0)", "Bad (1)"]))

    # Sauvegarder le modèle
    print(f"Sauvegarde du modèle dans '{model_output}'...")
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    
    print("Entraînement et sauvegarde terminés avec succès !")

if __name__ == "__main__":
    train_posture_model()
