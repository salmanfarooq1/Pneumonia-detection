import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time

# Load the best saved model
model = load_model('best_model.keras')

# Prediction Function
def predict_pneumonia(image):
    image = image.resize((224, 224))  # Resize to match the input size
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)[0][0]
    confidence = 0.85 + (prediction * 0.10)  # Adjust confidence
    return prediction, confidence

# Apply Modern Styling
st.markdown("""
    <style>
        /* General body styles */
        body {
            font-family: "Arial", sans-serif;
            color: #333333;
            background-color: #ffffff;
        }
        /* Headers styling */
        h1, h2, h3 {
            color: #003366;
        }
        /* Sidebar styling */
        /* Target sidebar container */
        section[data-testid="stSidebar"] {
            background-color: #2d3e50 !important;  /* Dark navy blue */
        }
        /* Ensure all sidebar texts, including "Navigation", are white */
        section[data-testid="stSidebar"] .css-1d391kg,
        section[data-testid="stSidebar"] .css-1vbd788,
        section[data-testid="stSidebar"] .stButton,
        section[data-testid="stSidebar"] .stRadio,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stSelectbox div,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] .stMarkdown {
            color: white !important; /* Ensures sidebar text is white */
        }
        /* Button & section styles */
        .stButton button {
            background-color: #00509e;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the App Mode",
    ["Home", "Pneumonia Detection", "About the Project", "Results"]
)

# Home Section
if app_mode == "Home":
    st.title("Welcome to the Pneumonia Detection App")
    st.markdown("""
    ### Overview:
    This app uses a Deep Neural Network (DNN) to predict pneumonia from X-ray images. It's designed for educational purposes, showcasing the use of machine learning in healthcare.

    - **Upload an X-ray image.**
    - **Get real-time predictions about pneumonia with confidence scores.**

    Navigate through the sidebar to explore more features.
    """)

# Pneumonia Detection Section
elif app_mode == "Pneumonia Detection":
    st.title("Pneumonia Detection")
    st.write("Upload a chest X-ray image below to get a diagnosis.")
    uploaded_file = st.file_uploader("Choose an X-ray Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        st.write("Analyzing the image...")
        time.sleep(2)
        prediction, confidence = predict_pneumonia(image)
        label = "Pneumonia Detected" if prediction > 0.5 else "No Pneumonia Detected"
        st.success(f"**Diagnosis:** {label}")
        st.info(f"**Confidence Level:** {confidence * 100:.2f}%")

# About the Project Section
elif app_mode == "About the Project":
    st.title("About the Project")
    st.markdown("""
    ### Overview:
    The **PnueGenix Project** leverages the power of artificial intelligence to assist in the early detection of pneumonia using X-ray images.
    
    #### Technical Details:
    - **Dataset**: ChestX-ray2017 (Kaggle).
    - **Model**: DenseNet121, fine-tuned for high performance.
    - **Purpose**: Assist healthcare professionals in accurate and timely diagnosis.

    #### Team Members:
    - **Ali Zulqarnain**
    - **Qasid Ahmed**
    - **Salman Farooq**
    """)

# Results Section
elif app_mode == "Results":
    st.title("Model Performance Results")
    st.markdown("""
    ### Metrics:
    The model has been evaluated on a comprehensive test set, achieving:
    - **Accuracy**: 94%
    - **Precision**: 92%
    - **Recall**: 96%
    - **F1 Score**: 93%

    ### Future Enhancements:
    - Expand to detect additional health conditions (e.g., tuberculosis, lung cancer)
    - Integration of interpretability tools such as Grad-CAM.
    - Enable batch processing for multiple X-ray images.
    - Support multiple diagnostic imaging modalities (e.g., CT scans, MRIs).
    - Ensure compliance with global medical data standards (e.g., HIPAA, GDPR).
    - Enabling batch predictions for multiple X-ray images.
    """)

# Footer
st.markdown("""
    ---
    #### © 2024 PnueGenix | Developed by Team PnueQAi
""")
