"""
Step 1: Export a trained PyTorch model to ONNX.

ONNX (Open Neural Network Exchange) is a format that represents a model as a
static computation graph instead of live Python/PyTorch code. Why bother?
- Framework-independent: can be run by onnxruntime, TensorRT, etc. without PyTorch installed.
- Usually faster for inference (graph-level optimizations, no Python overhead).
- A necessary intermediate step before further optimization (e.g. TensorRT in step 02).

Run: python export_to_onnx.py
"""
import torch
import torchvision.models as models

def main():
    # Using ResNet18 pretrained on ImageNet as our toy model.
    # In your real work you'd load your own trained checkpoint here instead.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()  # IMPORTANT: switches off dropout/batchnorm training behavior

    # ONNX export needs a sample input to trace the computation graph.
    # Shape: (batch_size, channels, height, width) - standard ImageNet input.
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "resnet18.onnx",
        export_params=True,        # store the trained weights inside the file
        opset_version=17,          # ONNX operator set version - 17 is broadly compatible as of 2025/2026
        do_constant_folding=True,  # optimization: precompute constant subgraphs
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={              # allow variable batch size at inference time
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,  # use the stable TorchScript-based exporter. As of torch 2.9+,
                        # the new dynamo-based exporter is default but needs the extra
                        # 'onnxscript' package - dynamo=False avoids that dependency
                        # for this toy example. Worth revisiting dynamo=True later,
                        # it's the direction PyTorch is heading.
    )
    print("Exported resnet18.onnx")

if __name__ == "__main__":
    main()
