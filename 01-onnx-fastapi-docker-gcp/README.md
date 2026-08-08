# Stage 01: ONNX + FastAPI + Docker + GCP Cloud Run

The minimum viable production deployment path — export a model, wrap it in an HTTP API, containerize it, ship it — applied to two different model types on purpose, to separate what's generic about this pattern from what's specific to the task:

| | [`resnet-classification/`](resnet-classification/) | [`yolo-detection/`](yolo-detection/) |
|---|---|---|
| Task | Image classification | Object detection |
| Output | JSON label + confidence | Image with boxes drawn (`StreamingResponse`) |
| Preprocessing | Resize to 224×224 | Letterboxing (preserves aspect ratio) |
| Post-processing | `argmax` | Non-max suppression (NMS) |
| GCP setup | Created project, billing, Artifact Registry repo | Reused all of it — same project/billing/repo, just a new image |

Both are deployed to GCP Cloud Run as real, publicly reachable endpoints. See each subfolder's README for the full walkthrough and the specific gotchas hit for that model.
