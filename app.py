import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64
import os
from io import BytesIO

# === Set background ===
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 10px;
        }}
        .centered-img {{
            display: flex;
            justify-content: center;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}
        .bordered-img {{
            border: 4px solid white;
            border-radius: 10px;
            width: 200px;
            height: auto;
        }}
        .button-container {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
        }}
        .stButton>button {{
            background-color: white !important;
            color: black !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: 2px solid black !important;
            font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# === Load model ===
@st.cache_resource
def load_model_cached():
    return load_model("best_model.h5")

model = load_model_cached()
class_names = ["Mask", "No Mask"]

# === Set up app ===
st.set_page_config(page_title="😷 Face Mask Detector", layout="centered")
set_bg_from_local("static/covid-bg.jpg")

st.markdown("<h1 style='color:white; text-align:center; font-size:2rem;'>😷 Face Mask Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:white; text-align:center; font-size:0.95rem;'>Upload a face image to check for mask usage.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((200, int(200 * image.height / image.width)))

    # Save uploaded image permanently to 'static/upload/'
    upload_dir = "static/upload"
    os.makedirs(upload_dir, exist_ok=True)
    image_save_path = os.path.join(upload_dir, uploaded_file.name)
    image.save(image_save_path)

    # Encode image in memory for display
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode()

    # Show image
    st.markdown(f"""
        <div class='centered-img'>
            <img src='data:image/png;base64,{encoded_img}' class='bordered-img' />
        </div>
    """, unsafe_allow_html=True)

    # Detect Button
    if st.button("Detect", use_container_width=True):
        with st.spinner("Analyzing..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_input = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_input, verbose=0)[0]
            label_index = np.argmax(prediction)
            label = class_names[label_index]
            confidence = round(prediction[label_index] * 100, 2)
            emoji = "🟢" if label == "Mask" else "🔴"

            st.markdown(
                f"<h3 style='color:white; text-align:center;'>{emoji} {label}</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:white; text-align:center;'>Confidence: <strong>{confidence}%</strong></p>",
                unsafe_allow_html=True)
