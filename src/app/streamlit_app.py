import streamlit as st
import requests
from PIL import Image
import io

# ---------------------------
# FastAPI URL
# ---------------------------
API_URL = "http://localhost:8000/predict"

st.title("🌿 Plant Disease Classification")
st.write("Upload an image of a leaf and get predictions from the FastAPI backend.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send to FastAPI
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            img_bytes = uploaded_file.read()
            response = requests.post(
                API_URL,
                files={"file": img_bytes}
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction complete!")
                st.write(f"### 🌱 Predicted Class: **{result['class_name']}**")
                st.write(f"### 🔢 Confidence: **{result['confidence']:.4f}**")

            else:
                st.error("❌ API Error. Make sure FastAPI is running.")