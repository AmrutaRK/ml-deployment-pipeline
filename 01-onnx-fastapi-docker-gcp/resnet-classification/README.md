# Deployment Log: Local → Docker → GCP Cloud Run

This documents the actual steps taken to go from a trained model to a
publicly reachable API. Written from real experience deploying this on
macOS (Apple Silicon) to Google Cloud Run, including the gotchas hit
along the way, not just the happy path.

## 1. Local setup (no Docker yet)

```bash
# Install Python via Homebrew if needed
brew install python@3.11

# Create an isolated virtual environment
cd 01-onnx-fastapi
python3 -m venv venv
source venv/bin/activate      # must re-run this every new terminal session

# Install dependencies
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

## 2. Export the model to ONNX

```bash
python export_to_onnx.py
```

Produces `resnet18.onnx` in the same folder. This is the trained model
converted into a framework-independent format.

## 3. Run the FastAPI service locally (still no Docker)

```bash
uvicorn app:app --reload
```

Test it:
```bash
curl -X POST -F "file=@fish.png" http://localhost:8000/predict
```

At this point the service only works on the machine it's running on —
`localhost` means "this computer," nobody else can reach it.

## 4. Containerize with Docker

Install Docker Desktop for Mac first (docker.com), make sure it's running.

```bash
docker build -t resnet18-service .
docker run -p 8000:8000 resnet18-service
```

Same curl command as step 3 now hits the containerized version instead
of the raw local environment, proves the container is self-contained
and reproducible.

**Gotcha hit:** none at this stage, but this is the step that matters most
for stage 05 — the exact image built here is what gets shipped to the cloud
unchanged.

## 5. Set up GCP

```bash
brew install --cask google-cloud-sdk
gcloud init                          # logs in, picks/creates a project

# Set a consistent default region (Frankfurt, closest to Prague)
gcloud config set run/region europe-west3
gcloud config set artifacts/location europe-west3
```

**Gotcha hit:** `gcloud services enable ...` failed with
`FAILED_PRECONDITION: Billing account ... not found`. GCP requires a
billing account linked to the project even to use free-tier services.
Fixed by adding a billing account at
console.cloud.google.com/billing and linking it to the project.

## 6. Enable required GCP services

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

## 7. Create an Artifact Registry repo (where the Docker image gets stored)

```bash
gcloud artifacts repositories create mlops-repo \
  --repository-format=docker \
  --location=europe-west3 \
  --description="MLOps journey images"

gcloud auth configure-docker europe-west3-docker.pkg.dev
```

## 8. Build the image for the correct architecture

**Gotcha hit:** deploy failed with
`Container manifest type ... must support amd64/linux`.
Apple Silicon Macs build ARM64 images by default; Cloud Run's machines
run AMD64. Fixed by explicitly targeting the platform:

```bash
docker build --platform linux/amd64 \
  -t europe-west3-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/resnet18-service .
```

## 9. Push the image to Artifact Registry

```bash
docker push europe-west3-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/resnet18-service
```

## 10. Deploy to Cloud Run

```bash
gcloud run deploy resnet18-service \
  --image europe-west3-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/resnet18-service \
  --region europe-west3 \
  --allow-unauthenticated \
  --port 8000
```
Returns a public URL, e.g.:
```
https://resnet18-service-xxxxx.europe-west3.run.app
```

## 11. Test from anywhere (not just localhost)

```bash
curl https://resnet18-service-xxxxx.europe-west3.run.app/health
curl -X POST -F "file=@fish.png" https://resnet18-service-xxxxx.europe-west3.run.app/predict
```

**Gotcha hit:** first curl attempt used the root URL with no path, got
`{"detail":"Not Found"}` — that's FastAPI correctly reporting no route
exists at `/`. Needed the explicit `/predict` path.

## Key takeaways

- **Local → Docker → Cloud** each step should change *where* something
  runs, not *what* it does. If behavior differs between stages, that's
  a bug to chase down, not something to shrug off.
- **Serverless (Cloud Run) means no manual "keep it running."** It scales
  to zero when idle (no cost) and spins up on request (small "cold start"
  delay on the first hit after idle time).
- **Architecture mismatches (ARM64 vs AMD64)** are a common but easy-to-fix
  gotcha specifically for Apple Silicon users deploying to standard cloud
  infrastructure — `--platform linux/amd64` is the fix to remember.
- **Billing must be linked** before most GCP services will activate, even
  ones with a free tier.
