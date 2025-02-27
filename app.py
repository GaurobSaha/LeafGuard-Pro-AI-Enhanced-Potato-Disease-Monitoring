from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model from the local project folder
MODEL_PATH = "1.keras"  # Ensure this matches the filename you downloaded
model = load_model(MODEL_PATH)

# Define class names based on your dataset (Modify if needed)
CLASS_NAMES = ["Potato___healthy", "Potato___Late_blight", "Potato___Early_blight"]

# Define image size (Must match training size)
IMAGE_SIZE = (256, 256)

# Set up Jinja2Templates to serve HTML files
templates = Jinja2Templates(directory="templates")

# Mount the static directory for serving CSS/JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_image(image_file):
    """
    Preprocesses the uploaded image to match the model's expected input format.
    """
    img = Image.open(io.BytesIO(image_file)).convert("RGB")
    img = img.resize(IMAGE_SIZE)  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.get("/")
async def home(request: Request):
    """
    Serve the main upload page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Handle image upload, process it, and return the predicted class.
    """
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    # Perform inference
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    return {"prediction": predicted_class}

