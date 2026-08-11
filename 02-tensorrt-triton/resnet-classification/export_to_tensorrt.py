"""
Convert the ResNet18 ONNX model into a TensorRT engine.

Requires an NVIDIA GPU with CUDA + TensorRT installed - no CPU fallback
exists for TensorRT, unlike ONNX Runtime.

TensorRT reads the ONNX graph and rebuilds it as an optimized "engine" -
fusing layers, picking the fastest kernels for the exact GPU it's built
on, optionally reducing precision (fp32 -> fp16) for speed. The resulting
engine is tied to that specific GPU architecture - not portable the way
ONNX is.

Usage:
    python export_to_tensorrt.py \
        --onnx ../../01-onnx-fastapi-docker-gcp/resnet-classification/resnet18.onnx \
        --engine model_repository/resnet18/1/model.plan
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

    # Matches the dynamic_axes set during ONNX export (input_names=["input"],
    # dynamic_axes={"input": {0: "batch_size"}}) - variable batch size,
    # fixed 224x224 spatial dimensions.
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min=(1, 3, 224, 224), opt=(4, 3, 224, 224), max=(16, 3, 224, 224))
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
    parser.add_argument("--onnx", default="../../01-onnx-fastapi-docker-gcp/resnet-classification/resnet18.onnx")
    parser.add_argument("--engine", default="model_repository/resnet18/1/model.plan")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()
    build_engine(args.onnx, args.engine, args.fp16)
