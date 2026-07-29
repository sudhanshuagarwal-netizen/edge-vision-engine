#!/usr/bin/env python3
"""
profile_power_thermal.py
Idle + sustained TensorRT load power/thermal time-series profiling.
Corrected for Jetson Orin Nano / current jtop key names.
"""
import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from jtop import jtop


def parse_args():
    p = argparse.ArgumentParser(description="Power/Thermal profiling for TensorRT YOLO")
    p.add_argument("--model", default="models/yolo11n.engine")
    p.add_argument("--source", default="videos/Benchmarking-video.mp4")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--idle-s", type=float, default=30.0, help="Idle baseline seconds")
    p.add_argument("--load-s", type=float, default=90.0, help="Sustained load seconds")
    p.add_argument("--sample-hz", type=float, default=2.0, help="jtop sampling rate")
    p.add_argument("--warmup", type=int, default=15, help="Warmup frames before timed load")
    return p.parse_args()


class Sampler:
    """Background jtop sampler at fixed rate. Uses correct key names for this Jetson."""
    def __init__(self, hz=2.0):
        self.hz = hz
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.phase = "init"
        self.jtop = jtop()

    def start(self):
        self.jtop.start()
        # Give jtop a moment to populate stats
        time.sleep(1.0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.jtop.close()

    def set_phase(self, phase: str):
        self.phase = phase

    def _extract(self, stats):
        # Power is reported in mW
        power_tot = float(stats.get("Power TOT", 0)) / 1000.0
        power_cpu_gpu = float(stats.get("Power VDD_CPU_GPU_CV", 0)) / 1000.0
        power_soc = float(stats.get("Power VDD_SOC", 0)) / 1000.0

        # Temperature keys are lowercase on this system
        temp_gpu = float(stats.get("Temp gpu", 0.0))
        temp_cpu = float(stats.get("Temp cpu", 0.0))
        temp_tj  = float(stats.get("Temp tj", 0.0))

        return {
            "power_tot_w": round(power_tot, 3),
            "power_cpu_gpu_w": round(power_cpu_gpu, 3),
            "power_soc_w": round(power_soc, 3),
            "temp_gpu_c": round(temp_gpu, 1),
            "temp_cpu_c": round(temp_cpu, 1),
            "temp_tj_c": round(temp_tj, 1),
        }

    def _loop(self):
        interval = 1.0 / self.hz
        t0 = time.perf_counter()
        while not self._stop.is_set():
            if self.jtop.ok:
                stats = self.jtop.stats
                row = {
                    "t_rel": round(time.perf_counter() - t0, 3),
                    "phase": self.phase,
                    **self._extract(stats),
                    "latency_ms": "",
                }
                self.samples.append(row)
            time.sleep(interval)


def run_inference_loop(model, cap, imgsz, conf, duration_s, sampler, latencies):
    start = time.perf_counter()
    frames = 0
    while time.perf_counter() - start < duration_s:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, conf=conf, verbose=False)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        frames += 1
        if sampler.samples:
            sampler.samples[-1]["latency_ms"] = round(lat, 2)
    return frames


def main():
    args = parse_args()
    print("=" * 70)
    print("POWER / THERMAL PROFILING (corrected keys)")
    print(f"  Model      : {args.model}")
    print(f"  Source     : {args.source}")
    print(f"  imgsz      : {args.imgsz}")
    print(f"  Idle       : {args.idle_s} s")
    print(f"  Load       : {args.load_s} s")
    print(f"  Sample rate: {args.sample_hz} Hz")
    print("=" * 70)

    print("Loading model...")
    model = YOLO(args.model, task="detect")
    print("Model loaded.")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    print(f"Warm-up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            _ = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    sampler = Sampler(hz=args.sample_hz)
    sampler.start()

    # ---------- IDLE ----------
    print(f"\n>>> IDLE baseline for {args.idle_s} s ...")
    sampler.set_phase("idle")
    time.sleep(args.idle_s)

    # ---------- LOAD ----------
    print(f">>> LOAD (TensorRT {args.imgsz}) for {args.load_s} s ...")
    sampler.set_phase("load")
    latencies = []
    frames = run_inference_loop(
        model, cap, args.imgsz, args.conf, args.load_s, sampler, latencies
    )

    sampler.stop()
    cap.release()

    # ---------- Summary ----------
    samples = [s for s in sampler.samples if s["phase"] in ("idle", "load")]
    idle = [s for s in samples if s["phase"] == "idle"]
    load = [s for s in samples if s["phase"] == "load"]

    def avg(key, rows):
        vals = [s[key] for s in rows if isinstance(s.get(key), (int, float)) and s[key] != ""]
        return float(np.mean(vals)) if vals else None

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Total usable samples : {len(samples)}")
    print(f"  Idle samples         : {len(idle)}")
    print(f"  Load samples         : {len(load)}")
    print(f"  Frames processed     : {frames}")
    if latencies:
        mean_lat = float(np.mean(latencies))
        print(f"  Mean latency         : {mean_lat:.2f} ms")
        print(f"  Avg FPS              : {1000.0 / mean_lat:.2f}")
    print(f"  Idle  Power TOT      : {avg('power_tot_w', idle):.2f} W")
    print(f"  Load  Power TOT      : {avg('power_tot_w', load):.2f} W")
    print(f"  Idle  GPU Temp       : {avg('temp_gpu_c', idle):.1f} °C")
    print(f"  Load  GPU Temp       : {avg('temp_gpu_c', load):.1f} °C")
    print(f"  Idle  CPU Temp       : {avg('temp_cpu_c', idle):.1f} °C")
    print(f"  Load  CPU Temp       : {avg('temp_cpu_c', load):.1f} °C")
    print(f"  Idle  TJ Temp        : {avg('temp_tj_c', idle):.1f} °C")
    print(f"  Load  TJ Temp        : {avg('temp_tj_c', load):.1f} °C")
    print("=" * 70)

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"power_thermal_640_{ts}.csv"

    fieldnames = [
        "t_rel", "phase",
        "power_tot_w", "power_cpu_gpu_w", "power_soc_w",
        "temp_gpu_c", "temp_cpu_c", "temp_tj_c",
        "latency_ms"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nTime-series saved to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()