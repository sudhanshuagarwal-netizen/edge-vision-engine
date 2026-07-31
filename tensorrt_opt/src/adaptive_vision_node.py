#!/usr/bin/env python3
"""
adaptive_vision_node.py
Adaptive TensorRT YOLO node with temperature + latency based frame skipping.
Includes warmup and hysteresis for stable behavior.
"""

import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from jtop import jtop


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive TensorRT vision node")
    p.add_argument("--model", default="models/yolo11n.engine")
    p.add_argument("--source", default="videos/Benchmarking-video.mp4")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to run")
    p.add_argument("--temp-enter", type=float, default=52.0,
                   help="GPU temp (°C) to ENTER stress mode")
    p.add_argument("--temp-exit", type=float, default=50.5,
                   help="GPU temp (°C) to EXIT stress mode")
    p.add_argument("--latency-enter", type=float, default=40.0,
                   help="Latency (ms) to ENTER stress mode")
    p.add_argument("--latency-exit", type=float, default=32.0,
                   help="Latency (ms) to EXIT stress mode")
    p.add_argument("--window", type=int, default=12,
                   help="Moving window size for averaging")
    p.add_argument("--skip-rate", type=int, default=2,
                   help="Process every Nth frame when under stress")
    p.add_argument("--warmup", type=int, default=20,
                   help="Number of frames to ignore before controller starts")
    return p.parse_args()


class AdaptiveController:
    """Decides whether to process or skip frames based on recent temp + latency.
    Uses hysteresis to avoid rapid on/off switching.
    """

    def __init__(self, temp_enter, temp_exit, latency_enter, latency_exit,
                 window, skip_rate):
        self.temp_enter = temp_enter
        self.temp_exit = temp_exit
        self.latency_enter = latency_enter
        self.latency_exit = latency_exit
        self.window = window
        self.skip_rate = skip_rate

        self.temp_history = deque(maxlen=window)
        self.latency_history = deque(maxlen=window)

        self.in_stress = False
        self.current_skip = 1
        self.frame_counter = 0
        self.decisions = []

    def update(self, temp_c, latency_ms):
        self.temp_history.append(temp_c)
        if latency_ms > 0:          # only record real inference latencies
            self.latency_history.append(latency_ms)

        avg_temp = float(np.mean(self.temp_history)) if self.temp_history else 0.0
        avg_lat = float(np.mean(self.latency_history)) if self.latency_history else 0.0

        # Hysteresis logic
        if not self.in_stress:
            # Currently normal → check enter conditions
            if (avg_temp >= self.temp_enter) or (avg_lat >= self.latency_enter):
                self.in_stress = True
                reason = []
                if avg_temp >= self.temp_enter:
                    reason.append(f"temp {avg_temp:.1f}°C ≥ {self.temp_enter}°C")
                if avg_lat >= self.latency_enter:
                    reason.append(f"latency {avg_lat:.1f}ms ≥ {self.latency_enter}ms")
                reason_str = " + ".join(reason)
                action = f"SKIP every {self.skip_rate}"
                print(f"  → Adaptive decision: {action}  ({reason_str})")
                self.decisions.append({
                    "t": time.time(),
                    "action": action,
                    "reason": reason_str,
                    "avg_temp": round(avg_temp, 1),
                    "avg_latency": round(avg_lat, 1)
                })
                self.current_skip = self.skip_rate
        else:
            # Currently in stress → check exit conditions
            if (avg_temp <= self.temp_exit) and (avg_lat <= self.latency_exit):
                self.in_stress = False
                reason_str = f"temp {avg_temp:.1f}°C ≤ {self.temp_exit}°C and latency {avg_lat:.1f}ms ≤ {self.latency_exit}ms"
                action = "FULL rate"
                print(f"  → Adaptive decision: {action}  ({reason_str})")
                self.decisions.append({
                    "t": time.time(),
                    "action": action,
                    "reason": reason_str,
                    "avg_temp": round(avg_temp, 1),
                    "avg_latency": round(avg_lat, 1)
                })
                self.current_skip = 1

        return avg_temp, avg_lat

    def should_process(self):
        self.frame_counter += 1
        return (self.frame_counter % self.current_skip) == 0


def main():
    args = parse_args()

    print("=" * 70)
    print("ADAPTIVE VISION NODE (with warmup + hysteresis)")
    print(f"  Model                : {args.model}")
    print(f"  Source               : {args.source}")
    print(f"  imgsz                : {args.imgsz}")
    print(f"  Temp enter / exit    : {args.temp_enter} / {args.temp_exit} °C")
    print(f"  Latency enter / exit : {args.latency_enter} / {args.latency_exit} ms")
    print(f"  Skip rate (stress)   : every {args.skip_rate} frames")
    print(f"  Warmup frames        : {args.warmup}")
    print(f"  Duration             : {args.duration} s")
    print("=" * 70)

    print("Loading model...")
    model = YOLO(args.model, task="detect")
    print("Model loaded.")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    controller = AdaptiveController(
        temp_enter=args.temp_enter,
        temp_exit=args.temp_exit,
        latency_enter=args.latency_enter,
        latency_exit=args.latency_exit,
        window=args.window,
        skip_rate=args.skip_rate
    )

    jetson = jtop()
    jetson.start()
    time.sleep(1.0)

    print(f"\nWarming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            _ = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    print("Running adaptive loop... (Ctrl+C to stop early)\n")

    processed = 0
    skipped = 0
    latencies = []
    start = time.perf_counter()

    try:
        while time.perf_counter() - start < args.duration:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            stats = jetson.stats
            temp_gpu = float(stats.get("Temp gpu", 0.0))

            if controller.should_process():
                t0 = time.perf_counter()
                _ = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
                latency = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency)
                processed += 1
                controller.update(temp_gpu, latency)
            else:
                skipped += 1
                # Still update temperature so the controller can exit stress
                controller.update(temp_gpu, 0.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        jetson.close()
        cap.release()

    total_frames = processed + skipped
    elapsed = time.perf_counter() - start
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Runtime              : {elapsed:.1f} s")
    print(f"  Frames seen          : {total_frames}")
    print(f"  Frames processed     : {processed}")
    print(f"  Frames skipped       : {skipped}")
    print(f"  Mean latency (proc)  : {mean_lat:.2f} ms")
    print(f"  Adaptive decisions   : {len(controller.decisions)}")
    print("=" * 70)

    if controller.decisions:
        print("\nDecision log:")
        for d in controller.decisions:
            print(f"  {d['action']:18s} | {d['reason']}")

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"adaptive_decisions_{ts}.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["t", "action", "reason", "avg_temp", "avg_latency"]
        )
        writer.writeheader()
        writer.writerows(controller.decisions)

    print(f"\nDecision log saved to: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()