"""
Step 2: Serve the ONNX model behind a FastAPI HTTP endpoint.

This is the "service" part of deployment: something listens on a port,
accepts an image, runs inference via onnxruntime (NOT PyTorch - notice we
never import torch here, which is the point of exporting to ONNX), and
returns a prediction as JSON.

Run locally:  uvicorn app:app --host 0.0.0.0 --port 8000
Then POST an image to http://localhost:8000/predict
"""
from fastapi import FastAPI, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="ResNet18 ONNX Inference Service")

# Same instrumentation pattern as the YOLO service - exposes GET /metrics
# with request count, latency, and in-progress request metrics, ready for
# Prometheus to scrape on its own schedule.
Instrumentator().instrument(app).expose(app)

# Load the ONNX model once at startup, not per-request - this is a common
# mistake beginners make (loading the model inside the endpoint function
# reloads it on every single request, which is slow and wasteful).
session = ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Human-readable ImageNet class names, in the same order the model was
# trained on (index 0 = "tench", index 1 = "goldfish", etc.). Without this,
# you only get a meaningless integer back - the model has no idea about
# words, it only knows 1000 output neurons.
with open("imagenet_classes.txt") as f:
    IMAGENET_CLASSES = [line.strip() for line in f.readlines()]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr

@app.get("/health")
def health():
    # Every real deployment needs a health check - load balancers and
    # orchestrators (Docker/Kubernetes) use this to know if the service is alive.
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess(image)

    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    predicted_class = int(np.argmax(logits))

    return {
        "predicted_class_index": predicted_class,
        "predicted_label": IMAGENET_CLASSES[predicted_class],
        "confidence": float(np.exp(logits[predicted_class]) / np.sum(np.exp(logits))),
    }
