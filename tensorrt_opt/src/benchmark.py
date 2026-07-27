#!/usr/bin/env python3
"""
benchmark.py — Fair comparison between PyTorch (.pt) and TensorRT (.engine)
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Optional power/thermal sampling via jtop
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO TensorRT vs PyTorch benchmark")
    parser.add_argument("--model", required=True, help="Path to .pt or .engine model")
    parser.add_argument("--source", default="0", help="Camera index or path to video file")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run after warmup")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup frames")
    parser.add_argument("--no-power", action="store_true", help="Disable jtop power/thermal sampling")
    return parser.parse_args()


def get_source(source_str):
    if source_str.isdigit():
        return int(source_str)
    return source_str


def run_benchmark(args):
    model_path = args.model
    source = get_source(args.source)

    print("=" * 70)
    print("BENCHMARK CONFIG")
    print(f"  Model     : {model_path}")
    print(f"  Source    : {args.source}")
    print(f"  imgsz     : {args.imgsz}")
    print(f"  conf      : {args.conf}")
    print(f"  Warmup    : {args.warmup} frames")
    print(f"  Duration  : {args.duration} s")
    print(f"  Power     : {'disabled' if args.no_power or not JTOP_AVAILABLE else 'jtop enabled'}")
    print("=" * 70)

    # Load model
    print("Loading model...")
    model = YOLO(model_path, task="detect")
    print("Model loaded.")

    # Open source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    # Warmup
    print(f"Warming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video if needed
            ret, frame = cap.read()
        if ret:
            _ = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    # Prepare timing + optional power sampling
    latencies = []
    power_samples = []
    temp_samples = []

    use_jtop = JTOP_AVAILABLE and not args.no_power
    jtop_proc = None
    if use_jtop:
        jtop_proc = jtop()
        jtop_proc.start()

    print(f"Running timed benchmark for {args.duration} seconds...")
    start_time = time.perf_counter()
    frames = 0

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.duration:
                break

            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            t0 = time.perf_counter()
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)
            frames += 1

            if use_jtop and jtop_proc.ok:
                stats = jtop_proc.stats
                # Power in mW → convert to W
                power = stats.get("Power TOT", stats.get("power", 0)) / 1000.0
                temp = stats.get("Temp GPU", stats.get("temp", 0))
                power_samples.append(power)
                temp_samples.append(temp)

    finally:
        if jtop_proc is not None:
            jtop_proc.close()
        cap.release()

    # Statistics
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    avg_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    avg_power = np.mean(power_samples) if power_samples else None
    avg_temp = np.mean(temp_samples) if temp_samples else None

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print(f"  Frames processed : {frames}")
    print(f"  Mean latency     : {mean_lat:.2f} ms")
    print(f"  Median latency   : {median_lat:.2f} ms")
    print(f"  P95 latency      : {p95_lat:.2f} ms")
    print(f"  Average FPS      : {avg_fps:.2f}")
    if avg_power is not None:
        print(f"  Avg Power        : {avg_power:.2f} W")
        print(f"  Avg GPU Temp     : {avg_temp:.1f} °C")
    print("=" * 70)

    # Save CSV
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem
    csv_path = results_dir / f"bench_{model_name}_{args.imgsz}_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "model", "source", "imgsz", "conf",
            "frames", "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
            "avg_fps", "avg_power_w", "avg_gpu_temp_c"
        ])
        writer.writerow([
            timestamp, model_path, args.source, args.imgsz, args.conf,
            frames, f"{mean_lat:.3f}", f"{median_lat:.3f}", f"{p95_lat:.3f}",
            f"{avg_fps:.3f}",
            f"{avg_power:.3f}" if avg_power is not None else "",
            f"{avg_temp:.1f}" if avg_temp is not None else ""
        ])

    print(f"\nResults saved to: {csv_path}")
    return {
        "mean_latency": mean_lat,
        "median_latency": median_lat,
        "p95_latency": p95_lat,
        "avg_fps": avg_fps,
        "avg_power": avg_power,
        "avg_temp": avg_temp,
        "csv": str(csv_path)
    }


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
