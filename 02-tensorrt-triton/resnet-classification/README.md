# Stage 02: TensorRT + Triton - ResNet18 Classification

## Status: written, not executed

Requires an NVIDIA GPU with CUDA + TensorRT installed - unavailable in
the environment this was built in. Everything here follows standard,
documented TensorRT/Triton patterns, but has not been run end-to-end
against a live GPU. Treat this as a solid starting point, not
verified-working code - real issues (version mismatches, shape errors)
are likely on first run.

## What's here

- `export_to_tensorrt.py` - converts the ONNX model (from the ResNet
  classification stage) into a TensorRT `.engine` file
- `model_repository/resnet18/config.pbtxt` - tells Triton how to serve
  the model (input/output shapes, dynamic batching)
- `triton_client.py` - sends a prediction request to a running Triton
  server and prints the predicted label

## How this differs from FastAPI serving

**No custom Dockerfile.** NVIDIA publishes a pre-built Triton server
image (`nvcr.io/nvidia/tritonserver`) - there's no `docker build` step
here, just mounting a model repository folder into that image.

**Client and server are separate processes.** The Triton container only
runs the server. `triton_client.py` is a separate script, run outside
the container, playing the same role curl played for the FastAPI
service - sending requests in, not something the container executes
itself.

**Three ports, not one.** Triton exposes 8000 (HTTP), 8001 (gRPC), and
8002 (metrics) by default, versus FastAPI's single port.

**GPU is non-negotiable.** `docker run --gpus=1` is required - no CPU
fallback exists for a TensorRT engine, unlike ONNX Runtime.

**Dynamic batching works here** because the ONNX export used a dynamic
batch axis (`dynamic_axes` in `export_to_onnx.py`), so
`max_batch_size: 16` with Triton's automatic batching applies cleanly.

## Deployment to GCP: not attempted

A GCP GPU VM (T4) was attempted across multiple zones
(europe-west3, europe-west4, us-central1) and consistently returned
`ZONE_RESOURCE_POOL_EXHAUSTED` - a documented GCP behavior where newer
or smaller accounts can be deprioritized for scarce GPU capacity,
confirmed by multiple independent reports of the same symptom (failing
in every zone, working fine on established accounts). Given that, this
stage focuses on documenting the deployment pattern accurately rather
than working around a capacity allocation issue outside the account's
control.

## How to run this (on a GPU machine, once available)

```bash
pip install tensorrt
cd ../../01-onnx-fastapi-docker-gcp/resnet-classification && python export_to_onnx.py && cd -
python export_to_tensorrt.py

docker run --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models

pip install tritonclient[http] pillow numpy
python triton_client.py --image test.jpg
```

Note the NGC container tag (`24.09-py3`) may need updating to a current
version - check NVIDIA's NGC catalog when running this.
