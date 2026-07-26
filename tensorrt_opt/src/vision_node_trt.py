#!/usr/bin/env python3
"""
vision_node_trt.py — TensorRT / PyTorch switchable edge vision node
"""

import os
import time
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration via environment variables (same style as original vision_node)
# ---------------------------------------------------------------------------
MODEL_PATH     = os.getenv("MODEL_PATH", "models/yolo11n.engine")
VIDEO_SOURCE   = os.getenv("VIDEO_SOURCE", "0")          # 0 = camera, or path to video
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IMGSZ          = int(os.getenv("IMGSZ", "640"))

def main():
    print("=" * 60)
    print("Edge Vision Node (TensorRT / PyTorch)")
    print(f"  MODEL_PATH     : {MODEL_PATH}")
    print(f"  VIDEO_SOURCE   : {VIDEO_SOURCE}")
    print(f"  CONF_THRESHOLD : {CONF_THRESHOLD}")
    print(f"  IMGSZ          : {IMGSZ}")
    print("=" * 60)

    # Load model
    print(f"Loading model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH, task="detect")
    print("Model loaded successfully")

    # Open video source
    source = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source {VIDEO_SOURCE}")
        return

    # FPS tracking
    frame_count = 0
    start_time = time.perf_counter()
    fps = 0.0

    print("Starting live inference. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or camera error")
                break

            # Inference
            results = model(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)

            # Annotate
            annotated = results[0].plot()

            # Update FPS every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed

            # Overlay FPS
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Edge Vision (TRT)", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final average FPS: {fps:.1f}")
        print("Done.")

if __name__ == "__main__":
    main()