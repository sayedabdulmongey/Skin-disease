from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from inference import prediction
from data import calculate_mean_std

import io
import uuid

from PIL import Image

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        file.filename = f"{uuid.uuid4()}.jpg"
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        class_name = prediction(image)

    
        return JSONResponse(content={"class_name": class_name})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
