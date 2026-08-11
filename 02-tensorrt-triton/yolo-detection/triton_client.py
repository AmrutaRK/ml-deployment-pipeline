"""
Client for a running Triton server serving the YOLOv8 TensorRT engine.

Notice this file is meaningfully more code than the ResNet stage's
Triton client. That's not accidental complexity - it reflects a real
architectural point: Triton (and the TensorRT engine it serves) only
runs the raw neural network forward pass. All the same post-processing
logic that lived in app.py for the FastAPI service - letterboxing,
decoding YOLO's (84, 8400) output, NMS, mapping boxes back to the
original image - still has to happen SOMEWHERE. It doesn't disappear
just because Triton is doing the serving; it just moves from server-side
(inside app.py) to client-side (here), since none of this logic is baked
into the TensorRT engine itself.

Requires: pip install tritonclient[http] pillow numpy

Usage:
    python triton_client.py --image ../../01-onnx-fastapi-docker-gcp/yolo-detection/test.jpg
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
import tritonclient.http as httpclient

INPUT_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45


def letterbox(image: Image.Image, size: int = INPUT_SIZE):
    # Identical logic to app.py's letterbox() - preprocessing must match
    # exactly whatever the model was trained/exported expecting, regardless
    # of which serving system runs the actual inference.
    orig_w, orig_h = image.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    # Identical to app.py's NMS - see that file's comments for the full
    # explanation of why this lives in client/serving code, not the model.
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        order = order[1:][iou <= iou_threshold]
    return keep


def postprocess(output: np.ndarray, scale: float, pad_x: int, pad_y: int, classes: list):
    predictions = output[0].T  # (84, 8400) -> (8400, 84)
    boxes_raw = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences > CONF_THRESHOLD
    boxes_raw, class_ids, confidences = boxes_raw[mask], class_ids[mask], confidences[mask]
    if len(boxes_raw) == 0:
        return []

    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep_idx = non_max_suppression(boxes_xyxy, confidences, IOU_THRESHOLD)

    results = []
    for i in keep_idx:
        bx1, by1, bx2, by2 = boxes_xyxy[i]
        bx1, by1 = (bx1 - pad_x) / scale, (by1 - pad_y) / scale
        bx2, by2 = (bx2 - pad_x) / scale, (by2 - pad_y) / scale
        results.append({
            "box": [float(bx1), float(by1), float(bx2), float(by2)],
            "label": classes[class_ids[i]],
            "confidence": float(confidences[i]),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default="yolov8n")
    args = parser.parse_args()

    with open("../../01-onnx-fastapi-docker-gcp/yolo-detection/coco_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    image = Image.open(args.image).convert("RGB")
    padded, scale, pad_x, pad_y = letterbox(image)
    arr = np.array(padded).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    input_tensor = np.expand_dims(arr, axis=0)

    client = httpclient.InferenceServerClient(url=args.url)

    inputs = [httpclient.InferInput("images", input_tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_tensor)
    outputs = [httpclient.InferRequestedOutput("output0")]

    result = client.infer(model_name=args.model, inputs=inputs, outputs=outputs)
    raw_output = result.as_numpy("output0")

    detections = postprocess(raw_output, scale, pad_x, pad_y, classes)

    print(f"Found {len(detections)} object(s):")
    for det in detections:
        print(f"  {det['label']} ({det['confidence']:.0%}) at {det['box']}")

    # Draw and save, same visual output style as the FastAPI service
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1 + 2, y1 - 12), f'{det["label"]} {det["confidence"]:.0%}', fill="lime")
    image.save("triton_result.png")
    print("Saved annotated image to triton_result.png")


if __name__ == "__main__":
    main()
