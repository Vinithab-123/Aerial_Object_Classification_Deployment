import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path

# --- Configuration & Model Loading ---

# 🚨 FINAL PATH FIX: Use .resolve() for a guaranteed absolute path 🚨
from pathlib import Path

# 1. Get the directory of the current script (src/)
BASE_DIR = Path(__file__).resolve().parent
# 2. Construct the full path using Path operators (BASE_DIR / ".." / "models" / ...)
# 3. Use .resolve() to clean up the '..' and get the absolute path
CLASSIFICATION_MODEL_PATH = (BASE_DIR / ".." / "models" / "best_mobilenet_model.keras").resolve()

# We convert to str() when passing to load_model, which is the correct approach
# ...

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Bird', 'Drone'] 

# Use caching to load the model only once
@st.cache_resource
def load_classification_model():
    """Loads the best Keras classification model."""
    try:
        # Convert Path object to string for load_model
        model = load_model(str(CLASSIFICATION_MODEL_PATH))
        return model
    except Exception as e:
        # Updated error message to reflect the correct structure
        st.error(f"Error loading Classification Model. Please ensure the model file exists at the correct path: {CLASSIFICATION_MODEL_PATH}")
        st.exception(e)
        return None

# Load the model globally
clf_model = load_classification_model()

# --- Prediction Function ---

def predict_classification(img, model):
    """Preprocesses image and predicts Bird/Drone class and confidence."""
    if model is None:
        return "Model Not Loaded", 0.0
        
    img = img.convert('RGB').resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Prediction is a single probability (Drone probability)
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Determine class and confidence
    if prediction > 0.5:
        predicted_class = CLASS_NAMES[1].upper() # DRONE
        confidence = prediction * 100
    else:
        predicted_class = CLASS_NAMES[0].upper() # BIRD
        confidence = (1 - prediction) * 100
        
    return predicted_class, confidence

# --- Streamlit UI ---

st.set_page_config(page_title="🦅 Bird/Drone Classification App", layout="centered")

st.title("🦅 Aerial Object Classification AI")
st.markdown("A deep learning solution to classify aerial images as either a **Bird** or a **Drone**.")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Read the image
    image = Image.open(uploaded_file)
    # Use the correct argument for better compatibility
    st.sidebar.image(image, caption='Uploaded Image', use_container_width=True) 
    
    st.header("Classification Results")
    
    if clf_model:
        # 2. Run Classification
        with st.spinner('Running Classification Model...'):
            predicted_class, confidence = predict_classification(image, clf_model)
        
        # 3. Display Results
        
        if confidence >= 90:
            st.success(f"## **Prediction:** {predicted_class} ✅")
        elif confidence >= 70:
            st.warning(f"## **Prediction:** {predicted_class} ⚠️")
        else:
            st.info(f"## **Prediction:** {predicted_class} ❓")

        st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
        
    else:
        st.error("The classification model could not be loaded. Please check model path and file existence.")

# --- How to Run (Updated Instructions) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ How to Run the App")
st.sidebar.markdown("1. Ensure **`tensorflow`**, **`streamlit`**, **`pillow`**, and **`numpy`** are installed (via `requirements.txt`).")
st.sidebar.markdown("2. Ensure your project structure includes a `src/` folder with `app.py` and a `models/` folder with `best_mobilenet_model.keras`.")
st.sidebar.markdown("3. Navigate to the project root directory and run:")
st.sidebar.code("streamlit run src/app.py")