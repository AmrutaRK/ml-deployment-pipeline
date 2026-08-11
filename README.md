# ML Model Deployment

Production-style implementations of the steps between "a trained model" and "a model something else can actually call" — export, serving, containerization, orchestration, CI/CD, and observability.

## `01-onnx-fastapi-docker-gcp/` — ONNX export → FastAPI service → Docker → GCP Cloud Run

The same deployment pattern applied to two different model types, to show what stays the same and what actually changes when the task changes:

- [`resnet-classification/`](01-onnx-fastapi-docker-gcp/resnet-classification/) — ResNet18 → ONNX → FastAPI `/predict` (JSON response) → Docker → deployed to GCP Cloud Run.
- [`yolo-detection/`](01-onnx-fastapi-docker-gcp/yolo-detection/) — YOLOv8 → ONNX → FastAPI `/predict` (image response, with letterboxing + NMS in the serving code) → Docker → deployed to GCP Cloud Run, reusing the same GCP project/Artifact Registry setup as the ResNet service.

Each subfolder has its own README with the full deployment walkthrough, including the actual gotchas hit along the way.

## `02-tensorrt-triton/` — GPU-optimized inference serving

Converts each model's ONNX export into a TensorRT engine and serves it via NVIDIA Triton Inference Server instead of hand-rolled FastAPI:

- [`resnet-classification/`](02-tensorrt-triton/resnet-classification/) — dynamic-batching TensorRT engine + Triton config.
- [`yolo-detection/`](02-tensorrt-triton/yolo-detection/) — fixed-shape TensorRT engine (YOLOv8's ONNX export has no dynamic batch axis, handled explicitly rather than silently); NMS/post-processing logic verified independently on CPU via ONNX Runtime.

Both READMEs are upfront about verification status: written against documented TensorRT/Triton patterns, but not run end-to-end against a live GPU (a GCP T4 GPU VM was attempted across multiple zones and consistently hit `ZONE_RESOURCE_POOL_EXHAUSTED`, a known capacity constraint on smaller accounts, not a configuration issue).

## `03-kubernetes-orchestration/` — running both services under Kubernetes

Runs the ResNet and YOLO services together in one GKE cluster — each with its own Deployment (replica count + self-healing), Service (stable external IP), and HorizontalPodAutoscaler (scales on real CPU load) — to demonstrate what Kubernetes adds on top of a single-container Cloud Run deployment: coordinating multiple independent, differently-scaled workloads on shared infrastructure.

See [`03-kubernetes-orchestration/README.md`](03-kubernetes-orchestration/README.md) for the full concepts writeup and how to run it on GKE.

## `04-CICD/` — automated build, test, and deploy with GitHub Actions

GitHub Actions workflows that replace the manual `docker build` / `push` / `kubectl` sequence: on every push to `main` that touches a given service, the workflow builds the image, smoke-tests it for real before anything ships, pushes to Artifact Registry, and rolls out the update to the corresponding GKE deployment from stage 03 — verifying the rollout actually succeeded rather than assuming it did.

See [`04-CICD/README.md`](04-CICD/README.md) for the full walkthrough, including GCP service-account setup with least-privilege IAM roles.

## `05-monitoring/` — observability with Prometheus + Grafana

Adds real observability to the ResNet and YOLO services running on Kubernetes: both are instrumented with `prometheus-fastapi-instrumentator` to expose a `/metrics` endpoint, Prometheus scrapes both on a 15s interval via internal cluster DNS, and Grafana visualizes request rate, latency, and in-flight requests on top of it. The Grafana admin password is provisioned as a Kubernetes Secret at deploy time, never committed to the repo.

See [`05-monitoring/README.md`](05-monitoring/README.md) for the full walkthrough, PromQL examples, and setup steps.
