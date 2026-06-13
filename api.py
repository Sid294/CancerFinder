from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import io

from main import load_trained_model, make_eval_transform, ROOT

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when the API starts
MODEL_PATH = ROOT / "cinder_model.pth"

model, device = load_trained_model(MODEL_PATH)
transform = make_eval_transform()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()

        # Open image and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply same preprocessing as training
        tensor = transform(image).unsqueeze(0).to(device)

        # Run the model
        with torch.no_grad():
            output = model(tensor)

            # Convert logits to probabilities
            probs = torch.softmax(output, dim=1)[0]

            # Get predicted class index
            prediction_index = torch.argmax(probs).item()

            # Confidence is the probability of the predicted class
            confidence = float(probs[prediction_index])

        # Return data formatted for React frontend
        return {
            "prediction": (
                "cancerous"
                if prediction_index == 1
                else "non-cancerous"
            ),
            "confidence": confidence,
            "probabilities": {
                "non-cancerous": float(probs[0]),
                "cancerous": float(probs[1])
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )