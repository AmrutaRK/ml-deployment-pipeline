# Stage 01 (v2): YOLOv8 Object Detection

## Why this exists as a separate folder from the classification version

Same deployment pattern (ONNX + FastAPI + Docker), but object detection is
a meaningfully different task than classification, and it's worth seeing
what actually changes and what stays the same.

## What's different from the classification stage

**Output shape and meaning.** Classification: one label for the whole
image. Detection: potentially many boxes, each with its own label and
confidence - and the count varies per image. This is why post-processing
(`postprocess()` in `app.py`) is far more involved here than classification's
simple "argmax and done."

**Preprocessing: letterboxing, not naive resize.** Classification just
squished the image to 224x224 - fine, since we only cared about *what*
was in the image, not precise spatial positions. Detection needs accurate
box coordinates, so naive resizing (which distorts aspect ratio) would
throw off box positions. Letterboxing resizes while preserving aspect
ratio and pads the rest - see `letterbox()` in `app.py`.

**NMS (Non-Max Suppression) lives in the serving code, not the model.**
Raw YOLO output has thousands of overlapping candidate boxes per object.
NMS filters duplicates down to one confident box per real object. This
kind of input-dependent, loopy logic is exactly what doesn't export
cleanly into a static ONNX graph - which is also why Mask R-CNN (which
needs similar dynamic logic throughout, not just at the end) is so much
harder to export than YOLO.

**The response is an image, not JSON.** Classification's answer was a
label - JSON was the natural format. Detection's answer is inherently
visual (boxes overlaid on the image), so `/predict` returns a PNG
directly via `StreamingResponse`, and the web UI displays it as an
image rather than parsing text.

# Deployment Log: Local → Docker → GCP Cloud Run (YOLOv8 Detection)

This documents the actual steps to go from the YOLOv8 detection model to
a publicly reachable API. Same overall path as the ResNet classification
service, with a few detection-specific differences and some new gotchas
hit along the way.

## 1. Local setup (no Docker yet)

```bash
# Confirm Python 3.11 specifically - see gotcha below on why this matters
python3.11 --version

# If missing:
brew install python@3.11

# Create an isolated virtual environment, explicitly using 3.11
cd 01-yolo-detection
python3.11 -m venv venv
source venv/bin/activate      # must re-run this every new terminal session

# Install dependencies
pip install --upgrade pip
pip install ultralytics       # only needed for the export step
pip install -r requirements.txt
```

**Gotcha hit:** running `python3 -m venv venv` (no version pin) created a
venv tied to whatever `python3` resolved to on PATH - which turned out to
be Python 3.14, not the intended 3.11. This caused two follow-on failures:

- `pip install onnxruntime==1.19.2` → `No matching distribution found` -
  no pre-built wheel exists for that pinned version on 3.14.
- `pip install pillow` (unpinned) → wheel build failure - pip fell back
  to compiling Pillow from source, which needs system libraries
  (libjpeg, zlib) not set up on this Mac by default.

**Fix:** delete the venv and recreate it explicitly with `python3.11 -m
venv venv` instead of the ambiguous `python3`. Once on 3.11, both
`onnxruntime` and `pillow` installed cleanly from pre-built wheels, no
compilation needed. Lesson: always pin the venv creation command to a
specific installed version if the machine has multiple Python versions
available - don't rely on `python3` resolving to the one you expect.

## 2. Export the model to ONNX

```bash
python export_to_onnx.py
```

Downloads pretrained `yolov8n.pt` and produces `yolov8n.onnx` in the same
folder - the detection model converted into a framework-independent
format, same idea as the ResNet ONNX export, different architecture.

## 3. Run the FastAPI service locally (still no Docker)

```bash
uvicorn app:app --reload
```

Test it - note the response here is a PNG image, not JSON, since
detection results are drawn directly onto the image:
```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/predict -o result.png
```
Or just open `http://localhost:8000` in a browser and use the upload UI.

At this point the service only works on the machine it's running on -
`localhost` means "this computer," nobody else can reach it.

## 4. Containerize with Docker

Docker Desktop for Mac should already be installed/running from the
ResNet stage.

```bash
docker build -t yolo-detection-service .
docker run -p 8000:8000 yolo-detection-service
```

Same test as step 3 now hits the containerized version - proves the
container is self-contained and reproducible.

## 5. GCP setup

Already done once for the ResNet service - same project, same billing
account, same region. No repeat setup needed here, just reused:

```bash
gcloud config set run/region europe-west3
gcloud config set artifacts/location europe-west3
```

## 6. Enable required GCP services

Already enabled during the ResNet deployment - `run.googleapis.com` and
`artifactregistry.googleapis.com` don't need re-enabling per-service.

## 7. Artifact Registry repo

Reusing the same `mlops-repo` repository created for the ResNet image -
Artifact Registry repos can hold multiple differently-named images, no
need for a separate repo per model.

```bash
gcloud auth configure-docker europe-west3-docker.pkg.dev
```

## 8. Build the image for the correct architecture

Same ARM64 vs AMD64 gotcha as the ResNet deployment applies here too -
Apple Silicon builds ARM64 by default, Cloud Run needs AMD64:

```bash
docker build --platform linux/amd64 \
  -t europe-west3-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/yolo-detection-service .
```

## 9. Push the image to Artifact Registry

```bash
docker push europe-west3-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/yolo-detection-service
```

## 10. Deploy to Cloud Run

```bash
gcloud run deploy yolo-detection-service \
  --image europe-west3-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/yolo-detection-service \
  --region europe-west3 \
  --allow-unauthenticated \
  --port 8000
```

Returns a public URL, e.g.:
```
https://yolo-detection-service-xxxxx.europe-west3.run.app
```

## 11. Test from anywhere (not just localhost)

```bash
curl https://yolo-detection-service-xxxxx.europe-west3.run.app/health
```

Or, better for detection specifically - open the URL directly in a
browser, upload an image, click "Detect Objects," and see boxes drawn on
the result in the UI rather than piping a curl response to a file.

## Key takeaways, detection-specific

- **Response type changes what "testing it" looks like.** The ResNet
  service returned JSON, easy to eyeball in a terminal. This service
  returns an image - curl still works but needs `-o result.png` plus
  actually opening the file; the browser UI is the more natural way to
  verify results for a visual task like this.
- **Reusing GCP infrastructure across services is normal and expected.**
  Project, billing, region, and Artifact Registry repo were all set up
  once for ResNet and reused as-is here - only the image name and Cloud
  Run service name changed.
- **Pin the Python version explicitly when creating a venv**, especially
  on a machine with multiple Python versions installed - `python3` is
  not guaranteed to resolve to the version you intend, and the resulting
  errors (failed wheel builds, missing distributions) can look unrelated
  to the actual root cause at first glance.
- Same ARM64/AMD64 and billing-account gotchas from the ResNet deployment
  apply identically here - infra gotchas tend to be per-machine/per-account,
  not per-model.
