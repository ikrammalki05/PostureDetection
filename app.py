"""Interface Web Streamlit pour la détection de posture.

L'image est prise depuis la caméra du navigateur, puis analysée sur le
serveur Streamlit. Aucune vidéo n'est conservée par l'application.
"""

import cv2
import numpy as np
import streamlit as st

from detector.pose_detector import PoseDetector
from detector.posture_analyzer import PostureAnalyzer
from utils.drawing_utils import draw_points


st.set_page_config(page_title="Smart Posture Detection", layout="wide")


@st.cache_resource
def load_analyzers():
    """Charge les modèles une seule fois par instance Streamlit."""
    return PoseDetector(), PostureAnalyzer()


def analyze_image(image_bytes: bytes):
    """Détecte la posture dans une image JPEG/PNG envoyée par le navigateur."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("L'image envoyée par la caméra est illisible.")

    pose_detector, posture_analyzer = load_analyzers()
    results = pose_detector.detect_pose(frame)
    pose_detector.draw_landmarks(frame, results)
    landmarks = pose_detector.get_landmarks(frame, results)
    draw_points(frame, landmarks)
    posture = posture_analyzer.analyze_posture(landmarks)
    posture_analyzer.draw_posture_text(frame, posture)

    # OpenCV utilise BGR ; Streamlit attend RGB.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), posture


st.title("🧍 Smart Posture Detection")
st.write("Prenez une photo face à la caméra pour analyser votre posture.")

with st.sidebar:
    st.header("Conseils")
    st.write("Cadrez votre tête, vos épaules et vos hanches.")
    st.write("La caméra est utilisée uniquement lorsque vous prenez une photo.")

photo = st.camera_input("Prendre une photo")

if photo is None:
    st.info("Autorisez l'accès à la caméra, puis prenez une photo.")
else:
    try:
        annotated_image, posture = analyze_image(photo.getvalue())
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error("Une erreur est survenue pendant l'analyse.")
        st.exception(error)
    else:
        left, right = st.columns(2)
        with left:
            st.image(photo, caption="Photo capturée", use_container_width=True)
        with right:
            st.image(annotated_image, caption="Analyse de posture", use_container_width=True)

        if posture == "Good Posture":
            st.success("Bonne posture détectée.")
        elif posture == "Bad Posture":
            st.warning("Posture à corriger détectée.")
        else:
            st.info("Aucune posture détectée. Essayez de mieux vous cadrer.")
