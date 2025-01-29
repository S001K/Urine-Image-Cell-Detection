from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import pandas
import torch
import io
from fastapi.middleware.cors import CORSMiddleware

# uvicorn api_script:app --host 127.0.0.1 --port 8000 --reload 

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Change to specific domains in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the model
model = YOLO("model.pt") 

# Define a response schema
class Prediction(BaseModel):
    label: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

@app.post("/predict/", response_model=list[Prediction])
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Perform inference
        results = model(img)

        response = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy[0]
                    response.append({
                        "label": result.names[int(box.cls.item())],
                        "confidence": float(box.conf.item()),
                        "xmin": float(bbox[0].item()),
                        "ymin": float(bbox[1].item()),
                        "xmax": float(bbox[2].item()),
                        "ymax": float(bbox[3].item()),
                    })

        if not response:
            return JSONResponse(
                content={"detail": "No objects detected"},
                status_code=200
            )

        return JSONResponse(content=response)

    except Exception as e:
        # Log the exception and return a 500 response with the error message
        print(f"Error during prediction: {e}")
        return JSONResponse(
            content={"detail": f"An error occurred: {str(e)}"},
            status_code=500
        )

@app.get("/")
async def root():
    return {"message": "YOLO Model API is up and running!"}
