# Stage 02 (v2): TensorRT + Triton for YOLOv8 Detection

## Status: mixed - some of this is genuinely tested, some isn't

Unlike the earlier ResNet version of this stage, part of this one was
actually verified, not just written: the `letterbox()`, `postprocess()`,
and NMS logic in `triton_client.py` was run against a real ONNX Runtime
inference (CPU, no GPU needed for this check).

What's still unverified: the actual TensorRT engine build
(`export_to_tensorrt.py`) and running it inside a real Triton server -
both require an NVIDIA GPU, unavailable here (see the honest note at the
bottom for why this was not run against a real GPU).

## What's different from the ResNet version of this stage

**Fixed shape, not dynamic batching.** This is the most important
difference, and it's a real limitation worth understanding, not just a
config detail. The ResNet ONNX export used `dynamic_axes` for a variable
batch size, so Triton's `max_batch_size: 16` with automatic dynamic
batching worked directly. YOLOv8's ONNX export (via `ultralytics`, as
used in stage 01) produces a FIXED shape `(1, 3, 640, 640)` - confirmed
by inspecting the actual exported graph, not assumed. Triton's automatic
batching requires a variable batch dimension to exist in the model,
which this export doesn't have. `config.pbtxt` here uses
`max_batch_size: 0` to reflect that honestly, rather than configuring
batching that would silently fail to actually batch anything.

**To fix this properly** (worth doing before running this for real):
re-export the ONNX model with a dynamic batch axis, the same way stage
01's `export_to_onnx.py` did with `dynamic_axes` - `export_to_tensorrt.py`
here is NOT set up for that yet, and would need updating alongside a
re-export if real dynamic batching for detection is needed later.

**Post-processing complexity moved, not disappeared.** The FastAPI
service's `postprocess()` function (letterbox, decode, NMS, coordinate
mapping) has a near-identical twin now living in `triton_client.py`.
Whichever serving system runs the raw model, that logic has to exist
somewhere, since none of it is baked into the TensorRT engine itself -
it's genuinely the same code, just relocated from server-side to
client-side depending on which serving system is used.

## What's in this folder

- `export_to_tensorrt.py` - ONNX -> TensorRT engine, with input/output
  tensor names (`images`, `output0`) and shapes verified against the
  real exported model, not guessed
- `model_repository/yolov8n/config.pbtxt` - Triton config, `max_batch_size: 0`
  as explained above
- `triton_client.py` - sends a request to Triton, decodes the raw output
  into actual boxes/labels, draws them on the image - post-processing
  logic verified correct via direct testing

## How to actually run this (on a GPU machine)

```bash
pip install tensorrt
cd ../../01-onnx-fastapi-docker-gcp/yolo-detection && python export_to_onnx.py && cd ../../02-tensorrt-triton/yolo-detection
python export_to_tensorrt.py \
  --onnx ../../01-onnx-fastapi-docker-gcp/yolo-detection/yolov8n.onnx \
  --engine model_repository/yolov8n/1/model.plan

docker run --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models

pip install tritonclient[http] pillow numpy
python triton_client.py --image ../../01-onnx-fastapi-docker-gcp/yolo-detection/some_test_image.jpg
```

## Why this wasn't run against a real GPU

A GCP GPU VM (T4) was attempted across multiple zones (europe-west3,
europe-west4, us-central1) - every attempt returned
`ZONE_RESOURCE_POOL_EXHAUSTED`. This is a documented GCP behavior where
newer or smaller accounts can be deprioritized for scarce GPU capacity,
not a configuration mistake - confirmed by multiple independent reports
of the identical symptom (every zone failing, working fine on
established/corporate accounts). Given that, this stage focuses on
documenting the pattern accurately and verifying what could be verified
without a GPU (the post-processing logic) rather than working around a
capacity allocation issue outside the account's control.
