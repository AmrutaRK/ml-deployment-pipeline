"""
Step 1: Convert the YOLOv8 ONNX model -> TensorRT engine.

REQUIRES: an NVIDIA GPU with CUDA + TensorRT installed. No CPU fallback,
same hard requirement as the ResNet version of this stage.

Unlike ResNet18's single (1000,) classification output, YOLO's output is
(1, 84, 8400) - 4 box coordinates + 80 class scores, for 8400 candidate
detections. TensorRT doesn't care about that difference in MEANING - it's
still just a fixed-shape tensor to optimize - but it matters for whoever
reads the output later (see the Triton client), since raw TensorRT output
still needs the same NMS/decoding logic the FastAPI service already does.

Usage (on a GPU machine, with TensorRT's Python bindings installed):
    python export_to_tensorrt.py \
        --onnx ../../01-onnx-fastapi-docker-gcp/yolo-detection/yolov8n.onnx \
        --engine model_repository/yolov8n/1/model.plan
"""
import argparse
import tensorrt as trt

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    # YOLOv8's ONNX export (via ultralytics) uses a FIXED input size
    # (640x640, batch size 1) - confirmed by inspecting the actual
    # exported graph (onnx.load + checking model.graph.input[0]) rather
    # than assumed. Input tensor name is "images", output is "output0" -
    # both verified this way too. If a newer ultralytics version is used
    # to re-export, these names should be re-verified rather than assumed
    # unchanged.
    profile = builder.create_optimization_profile()
    profile.set_shape("images", min=(1, 3, 640, 640), opt=(1, 3, 640, 640), max=(1, 3, 640, 640))
    config.add_optimization_profile(profile)

    print("Building TensorRT engine - this can take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved engine to {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="../../01-onnx-fastapi-docker-gcp/yolo-detection/yolov8n.onnx")
    parser.add_argument("--engine", default="model_repository/yolov8n/1/model.plan")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()
    build_engine(args.onnx, args.engine, args.fp16)
