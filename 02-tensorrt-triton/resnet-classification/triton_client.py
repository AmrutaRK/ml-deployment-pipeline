"""
Client for a running Triton server serving the ResNet18 TensorRT engine.

Triton's protocol differs from plain HTTP+JSON (what curl/FastAPI use) -
it needs tensor shapes, names, and datatypes specified explicitly. The
tritonclient library handles that formatting.

Requires: pip install tritonclient[http] pillow numpy

Usage:
    python triton_client.py --image test.jpg
"""
import argparse
import numpy as np
from PIL import Image
import tritonclient.http as httpclient

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default="resnet18")
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=args.url)

    input_tensor = preprocess(args.image)
    inputs = [httpclient.InferInput("input", input_tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_tensor)
    outputs = [httpclient.InferRequestedOutput("output")]

    result = client.infer(model_name=args.model, inputs=inputs, outputs=outputs)
    logits = result.as_numpy("output")[0]

    predicted_class = int(np.argmax(logits))
    with open("../../01-onnx-fastapi-docker-gcp/resnet-classification/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    print(f"Predicted: {classes[predicted_class]} (class {predicted_class})")

if __name__ == "__main__":
    main()
