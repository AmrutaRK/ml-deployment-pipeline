"""
Serve the YOLOv8 ONNX model behind FastAPI, returning an IMAGE with boxes
drawn on it (not just JSON) - this is a real difference from stage 01's
classification service, where a text label was the whole answer. Here the
"answer" is inherently visual, so the response is a rendered image.

Run: uvicorn app:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 in a browser and upload an image.
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

app = FastAPI(title="YOLOv8 ONNX Object Detection Service")

# Auto-instruments every route on this app with Prometheus metrics -
# request count, latency histograms, in-progress requests - and exposes
# them at GET /metrics in the plain-text format Prometheus expects.
# This is a PULL-based system: Prometheus itself will periodically visit
# /metrics and scrape whatever's there - the app doesn't push anything
# anywhere, it just passively exposes current numbers when asked.
Instrumentator().instrument(app).expose(app)

session = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
INPUT_SIZE = 640

with open("coco_classes.txt") as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]

CONF_THRESHOLD = 0.35   # discard low-confidence candidate boxes early
IOU_THRESHOLD = 0.45    # overlap threshold for Non-Max Suppression


def letterbox(image: Image.Image, size: int = INPUT_SIZE):
    """
    Resize the image to fit inside a size x size square WITHOUT distorting
    its aspect ratio, padding the leftover space with gray. Naively
    stretching a non-square image to 640x640 would warp objects and hurt
    accuracy - letterboxing is the standard fix, and YOLO was trained
    expecting exactly this kind of preprocessing.
    Returns the padded square image, plus the scale and padding used, so
    box coordinates can be mapped back to the ORIGINAL image size later.
    """
    orig_w, orig_h = image.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h))

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def preprocess(image: Image.Image):
    padded, scale, pad_x, pad_y = letterbox(image)
    arr = np.array(padded).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)
    return arr, scale, pad_x, pad_y


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """
    Plain numpy NMS - no opencv/torch dependency, keeping this stage as
    lightweight as stage 01. Greedily keeps the highest-confidence box,
    removes/suppresses other boxes that overlap it too much (measured by
    IoU - Intersection over Union), repeats. This is exactly the kind of
    input-dependent looping logic that doesn't translate well into a
    static ONNX graph - which is why it lives here in the serving code
    instead of inside the exported model.
    """
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


def postprocess(output: np.ndarray, scale: float, pad_x: int, pad_y: int):
    """
    Raw YOLOv8 output shape: (1, 84, 8400) -> 4 box coords + 80 class
    scores, for 8400 candidate detections. This function turns that into
    a clean list of (box, class_name, confidence) - filtering junk boxes,
    running NMS, and mapping coordinates back to the ORIGINAL image size
    using the scale/padding from letterboxing.
    """
    predictions = output[0].T  # -> (8400, 84)

    boxes_raw = predictions[:, :4]           # cx, cy, w, h (in 640x640 space)
    class_scores = predictions[:, 4:]        # 80 class confidence scores
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences > CONF_THRESHOLD
    boxes_raw, class_ids, confidences = boxes_raw[mask], class_ids[mask], confidences[mask]

    if len(boxes_raw) == 0:
        return []

    # Convert center-x/center-y/width/height -> x1,y1,x2,y2 (corners)
    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep_idx = non_max_suppression(boxes_xyxy, confidences, IOU_THRESHOLD)

    results = []
    for i in keep_idx:
        bx1, by1, bx2, by2 = boxes_xyxy[i]
        # Undo letterbox padding + scaling to map back to the ORIGINAL
        # (pre-resize) image coordinates
        bx1 = (bx1 - pad_x) / scale
        by1 = (by1 - pad_y) / scale
        bx2 = (bx2 - pad_x) / scale
        by2 = (by2 - pad_y) / scale
        results.append({
            "box": [float(bx1), float(by1), float(bx2), float(by2)],
            "label": COCO_CLASSES[class_ids[i]],
            "confidence": float(confidences[i]),
        })
    return results


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["label"]} {det["confidence"]:.0%}'
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(
            [text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill="lime",
        )
        draw.text((x1 + 2, y1 - 2), label, fill="black")
    return image


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 Object Detection</title>
        <style>
            body { font-family: sans-serif; max-width: 640px; margin: 40px auto; text-align: center; }
            #resultImg { max-width: 100%; margin-top: 20px; border-radius: 8px; }
            button { padding: 8px 20px; margin-top: 12px; cursor: pointer; }
            #status { margin-top: 10px; color: #555; }
        </style>
    </head>
    <body>
        <h2>YOLOv8 Object Detection</h2>
        <input type="file" id="fileInput" accept="image/*"><br>
        <button onclick="predict()">Detect Objects</button>
        <div id="status"></div>
        <img id="resultImg" style="display:none;">

        <script>
            const fileInput = document.getElementById('fileInput');
            const resultImg = document.getElementById('resultImg');
            const status = document.getElementById('status');

            async function predict() {
                if (!fileInput.files[0]) {
                    status.textContent = 'Choose an image first.';
                    return;
                }
                status.textContent = 'Running detection...';
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const response = await fetch('/predict', { method: 'POST', body: formData });
                if (!response.ok) {
                    status.textContent = 'Error running detection.';
                    return;
                }
                const blob = await response.blob();
                resultImg.src = URL.createObjectURL(blob);
                resultImg.style.display = 'block';
                status.textContent = 'Done.';
            }
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor, scale, pad_x, pad_y = preprocess(image)
    outputs = session.run(None, {input_name: input_tensor})
    detections = postprocess(outputs[0], scale, pad_x, pad_y)

    result_image = draw_detections(image.copy(), detections)

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
