import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import json
import io
import os

# ---------------------------
# Base directory (project root)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------
# Load class names
# ---------------------------
class_names_path = os.path.join(BASE_DIR, "configs", "class_names.json")
with open(class_names_path, "r") as f:
    CLASS_NAMES = json.load(f)

# ---------------------------
# Load model once at startup
# ---------------------------
model_path = os.path.join(BASE_DIR, "models", "resnet50_best.h5")
model = load_model(model_path)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Plant Disease API", version="1.0")

# Allow Streamlit calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(image)

    preds = model.predict(input_tensor)
    class_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    class_name = CLASS_NAMES[class_index]

    return {
        "class_name": class_name,
        "class_index": class_index,
        "confidence": confidence
    }

# ---------------------------
# Run the API
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)