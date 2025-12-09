import streamlit as st
import requests
from PIL import Image
import io
import mimetypes

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
            uploaded_file.seek(0)  # Reset pointer to start

            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)
            if mime_type is None:
                mime_type = "image/jpeg"  # fallback

            files = {"file": (uploaded_file.name, uploaded_file, mime_type)}

            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction complete!")
                    st.write(f"### 🌱 Predicted Class: **{result['class_name']}**")
                    st.write(f"### 🔢 Confidence: **{result['confidence']:.4f}**")
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Could not reach API. {e}")