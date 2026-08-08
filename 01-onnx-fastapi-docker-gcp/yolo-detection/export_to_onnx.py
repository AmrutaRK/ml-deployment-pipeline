"""
Step 1: Export a pretrained YOLOv8 model to ONNX.

Note the output is NOT yet "boxes and labels" - it's raw numbers that still
need decoding (see app.py). The model doesn't do Non-Max Suppression (NMS)
internally in this export - filtering overlapping duplicate boxes happens
in the serving code instead, since NMS involves the kind of dynamic,
input-dependent logic ONNX handles poorly baked into the graph itself.

Run: python export_to_onnx.py
"""
from ultralytics import YOLO

def main():
    # yolov8n = "nano", the smallest/fastest YOLOv8 variant - good for a
    # toy/demo service. Pretrained on COCO (80 everyday object classes:
    # person, car, dog, etc.) - not fine-tuned on anything specific here.
    model = YOLO("yolov8n.pt")

    model.export(
        format="onnx",
        imgsz=640,       # YOLO's standard input resolution
        simplify=True,   # runs onnxslim to clean up/optimize the graph
        opset=17,
    )
    print("Exported yolov8n.onnx")

if __name__ == "__main__":
    main()
