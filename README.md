# ML Model Deployment

Production-style implementations of the steps between "a trained model" and "a model something else can actually call" — export, serving, containerization, and cloud deployment. More deployment patterns (GPU-optimized serving, orchestration, monitoring) are being added over time.

## `01-onnx-fastapi-docker-gcp/` — ONNX export → FastAPI service → Docker → GCP Cloud Run

The same deployment pattern applied to two different model types, to show what stays the same and what actually changes when the task changes:

- [`resnet-classification/`](01-onnx-fastapi-docker-gcp/resnet-classification/) — ResNet18 → ONNX → FastAPI `/predict` (JSON response) → Docker → deployed to GCP Cloud Run.
- [`yolo-detection/`](01-onnx-fastapi-docker-gcp/yolo-detection/) — YOLOv8 → ONNX → FastAPI `/predict` (image response, with letterboxing + NMS in the serving code) → Docker → deployed to GCP Cloud Run, reusing the same GCP project/Artifact Registry setup as the ResNet service.

Each subfolder has its own README with the full deployment walkthrough, including the actual gotchas hit along the way.

## What's next

More stages — GPU-optimized inference serving (TensorRT/Triton), multi-service orchestration, and production monitoring/autoscaling — are in progress and will be added as they're ready.
