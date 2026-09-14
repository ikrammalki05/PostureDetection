# Smart Posture Detection

Application Web Streamlit qui analyse une photo prise par la webcam du navigateur et indique si la posture détectée est bonne ou à corriger.

## Lancer en local

Utilisez Python 3.10 ou 3.11, puis installez les dépendances :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Ouvrez ensuite l'adresse indiquée par Streamlit (habituellement `http://localhost:8501`).

## Déployer sur Streamlit Community Cloud

1. Créez un dépôt GitHub et envoyez-y tous les fichiers, y compris `posture_model.pkl`.
2. Connectez-vous à [Streamlit Community Cloud](https://share.streamlit.io/).
3. Cliquez sur **Create app**, sélectionnez le dépôt et la branche, puis choisissez `app.py` comme fichier principal.
4. Cliquez sur **Deploy**. La plateforme fournit une URL publique et sécurisée ; le navigateur demandera l'autorisation d'utiliser la caméra.

> Le fichier `posture_model.pkl` est requis pour la prédiction ML. Sans lui, l'application emploie uniquement la règle de secours intégrée.
