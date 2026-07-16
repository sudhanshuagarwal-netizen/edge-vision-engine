#!/usr/bin/env python3
"""Tactical Edge Vision System - Containerized Vision Node
YOLO11n real-time object detection with configurable camera source.
Env: VIDEO_SOURCE (default 0), CONF_THRESHOLD (0.45), MODEL_PATH (yolo11n.pt)
"""
import os
import sys
import cv2
from ultralytics import YOLO

def get_camera_source():
    source = os.environ.get("VIDEO_SOURCE", "0").strip()
    if source.isdigit():
        return int(source)
    return source

def main():
    print("=" * 60)
    print("  Tactical Edge Vision System - Vision Node")
    print("  Multi-architecture • Offline-ready • Containerized")
    print("=" * 60)
    model_path = os.environ.get("MODEL_PATH", "yolo11n.pt")
    conf_threshold = float(os.environ.get("CONF_THRESHOLD", "0.45"))
    video_source = get_camera_source()
    print(f"[CONFIG] VIDEO_SOURCE   = {video_source}")
    print(f"[CONFIG] MODEL_PATH     = {model_path}")
    print(f"[CONFIG] CONF_THRESHOLD = {conf_threshold}")
    print("-" * 60)
    print("[INIT] Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print(f"[INIT] Model loaded: {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    print(f"[INIT] Opening camera: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {video_source}")
        print("  USB: VIDEO_SOURCE=0 or /dev/video0")
        print("  DroidCam: VIDEO_SOURCE=http://PHONE_IP:4747/video")
        print("  Docker: --network host and/or --device /dev/video0")
        sys.exit(1)
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        cap.release()
        sys.exit(1)
    height, width = test_frame.shape[:2]
    cx, cy = width // 2, height // 2
    print(f"[INIT] Resolution: {width}x{height} | Center: ({cx}, {cy})")
    print("[INIT] System Operational. Press q to quit.")
    print("-" * 60)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARN] Frame read failed. Retrying...")
                continue
            results = model(frame, stream=True, conf=conf_threshold, verbose=False)
            target_found = False
            tx, ty = cx, cy
            tname = "SEARCHING"
            annotated = frame.copy()
            for r in results:
                annotated = r.plot()
                for box in r.boxes:
                    if not target_found:
                        coords = box.xywh[0]
                        tx = int(coords[0])
                        ty = int(coords[1])
                        cid = int(box.cls[0])
                        tname = model.names[cid].upper()
                        target_found = True
                        break
            if target_found:
                ox = tx - cx
                oy = ty - cy
                print(f"TRACKING: {tname:12} | Coords: ({tx:4d},{ty:4d}) | Offset: {ox:+5d},{oy:+5d}px")
            else:
                print("STATUS: Airspace Clear - Sweeping Passive Scanning Pattern...")
            cv2.circle(annotated, (cx, cy), 10, (0, 0, 255), 2)
            cv2.line(annotated, (cx-20, cy), (cx+20, cy), (0, 0, 255), 1)
            cv2.line(annotated, (cx, cy-20), (cx, cy+20), (0, 0, 255), 1)
            cv2.imshow("Tactical Edge Vision System", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[EXIT] User quit.")
                break
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[CLEANUP] Offline.")

if __name__ == "__main__":
    main()