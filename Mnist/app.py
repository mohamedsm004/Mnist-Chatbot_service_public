import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Projet MNIST", layout="wide")
st.title("🔢 Projet MNIST - Reconnaissance")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_keras_model():
    try:
        model = load_model("best_model.h5")
        return model
    except:
        st.error("ERREUR : 'best_model.h5' introuvable.")
        return None

MODEL = load_keras_model()
LABELS = {i: str(i) for i in range(10)}

# --- FONCTION DE PRÉTRAITEMENT ---
def process_drawing(img_gray):
    """
    Prépare l'image dessinée pour le modèle :
    Recadrage, ajout de marges et redimensionnement.
    """
    coords = cv2.findNonZero(img_gray)
    if coords is None:
        return None
        
    # Définition de la zone de dessin réelle (bounding box)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Ajout d'une petite marge (5 pixels)
    boundary = 5
    h_img, w_img = img_gray.shape
    
    x_min, y_min = max(x - boundary, 0), max(y - boundary, 0)
    x_max, y_max = min(x + w + boundary, w_img), min(y + h + boundary, h_img)
    
    # Extraction et redimensionnement
    img_cropped = img_gray[y_min:y_max, x_min:x_max]
    if img_cropped.size == 0: return None
    
    # Redimensionnement avec padding pour correspondre au style MNIST
    image = cv2.resize(img_cropped, (28, 28))
    image = np.pad(image, (10, 10), 'constant', constant_values=0)
    image = cv2.resize(image, (28, 28)) / 255.0
    
    return image

# --- INTERFACE ---
tab1, tab2 = st.tabs(["✍️ 1. Mode Dessin", "📂 2. Soumettre une image"])

# === MODE DESSIN (Streamlit Canvas) ===
with tab1:
    col_draw, col_pred = st.columns([1, 1])
    
    with col_draw:
        st.write("**Dessinez un chiffre ci-dessous :**")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas_mnist"
        )

    with col_pred:
        if canvas_result.image_data is not None:
            # Conversion de l'image du canvas en gris
            img_raw = canvas_result.image_data.astype('uint8')
            img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGBA2GRAY)
            
            final_input = process_drawing(img_gray)
            
            if final_input is not None and MODEL:
                pred = MODEL.predict(final_input.reshape(1, 28, 28, 1), verbose=0)
                label = np.argmax(pred)
                conf = np.max(pred) * 100
                
                st.metric(label="Chiffre détecté", value=label, delta=f"{conf:.1f}% confiance")
                
                with st.expander("Voir l'image traitée par l'IA"):
                    st.image(final_input, width=100)
            else:
                st.info("Dessinez quelque chose pour voir la prédiction.")

# === MODE FICHIER ===
with tab2:
    uploaded_file = st.file_uploader("Choisissez une image (fond noir ou blanc)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and MODEL:
        image_pil = Image.open(uploaded_file).convert('L')
        img_arr = np.array(image_pil)
        
        # Inversion si fond blanc
        if np.mean(img_arr) > 127:
            img_arr = cv2.bitwise_not(img_arr)
            
        img_resized = cv2.resize(img_arr, (28, 28)) / 255.0
        img_final = img_resized.reshape(1, 28, 28, 1)
        
        pred = MODEL.predict(img_final, verbose=0)
        st.header(f"Résultat : {np.argmax(pred)}")
        st.image(img_arr, width=150, caption="Image analysée")