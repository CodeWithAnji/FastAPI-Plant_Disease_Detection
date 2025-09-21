from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

# Initialize FastAPI
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load trained model
model = load_model("plant_disease_model.h5")

# Define class labels (replace with your actual labels)
class_labels = ["Healthy", "Powdery Mildew", "Rust", "Blight"]

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((128, 128))  # resize to model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    result = class_labels[class_index]

    return {"prediction": result}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
