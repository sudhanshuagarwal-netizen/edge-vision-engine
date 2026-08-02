#!/usr/bin/env python3
"""
orbital_sim.py — Phase 3 Orbital Edge Data Processing Pipeline Simulation

Combines:
  - TensorRT YOLO inference
  - Phase 2 AdaptiveController (temp + latency + hysteresis + frame skip)
  - Simulated power regimes (sunlight / eclipse)
  - Detection priority scoring (household-focused)
  - Downlink bandwidth model with optional keyframes
  - Detailed keep/discard reasons + live orbital status line
  - Time-series + decision logging

Designed to run on Jetson Orin Nano (same environment as Phase 1 & 2).
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


# ---------------------------------------------------------------------------
# Phase 2 AdaptiveController (self-contained)
# ---------------------------------------------------------------------------
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
        if latency_ms > 0:
            self.latency_history.append(latency_ms)

        avg_temp = float(np.mean(self.temp_history)) if self.temp_history else 0.0
        avg_lat = float(np.mean(self.latency_history)) if self.latency_history else 0.0

        if not self.in_stress:
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
            if (avg_temp <= self.temp_exit) and (avg_lat <= self.latency_exit):
                self.in_stress = False
                reason_str = (f"temp {avg_temp:.1f}°C ≤ {self.temp_exit}°C and "
                              f"latency {avg_lat:.1f}ms ≤ {self.latency_exit}ms")
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


# ---------------------------------------------------------------------------
# Household-focused class weights (COCO names)
# ---------------------------------------------------------------------------
CLASS_WEIGHTS = {
    # High value (1.8 – 2.0)
    "laptop": 2.0, "cell phone": 2.0, "tv": 1.9, "remote": 1.8,
    "keyboard": 1.8, "mouse": 1.8, "book": 1.8, "clock": 1.7,
    "bottle": 1.7, "cup": 1.7, "bowl": 1.6, "vase": 1.6,
    "potted plant": 1.6, "chair": 1.7, "couch": 1.8, "bed": 1.7,
    "dining table": 1.6, "refrigerator": 1.7, "microwave": 1.6,
    "oven": 1.5, "sink": 1.5, "toilet": 1.5, "toothbrush": 1.4,
    "hair drier": 1.4, "scissors": 1.4, "teddy bear": 1.5,

    # Medium
    "person": 1.2,
    "backpack": 1.0, "handbag": 1.0, "suitcase": 1.0,
    "umbrella": 0.9,
}

DEFAULT_WEIGHT = 0.5


def get_class_weight(name: str) -> float:
    return CLASS_WEIGHTS.get(name.lower(), DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# Power model (Phase 2 measurements, 25 W mode)
# ---------------------------------------------------------------------------
IDLE_POWER_W = 5.33
FULL_DETECT_DELTA_W = 1.79          # → ~7.12 W under load
CLOCKS_ON_IDLE_W = 7.32
CLOCKS_ON_LOAD_DELTA_W = 2.17       # → ~9.49 W


def estimate_power(processing: bool, regime: str, in_stress: bool) -> float:
    """Simple power estimate based on Phase 2 numbers."""
    if regime == "sunlight":
        base = CLOCKS_ON_IDLE_W if not in_stress else IDLE_POWER_W
        delta = CLOCKS_ON_LOAD_DELTA_W if processing else 0.0
    else:  # eclipse
        base = IDLE_POWER_W
        delta = FULL_DETECT_DELTA_W if processing else 0.0
        if in_stress:
            delta *= 0.6
    return base + delta


# ---------------------------------------------------------------------------
# Downlink / bandwidth model
# ---------------------------------------------------------------------------
DETECTION_PAYLOAD_KB = 3.0
KEYFRAME_KB = 80.0


class BandwidthTracker:
    def __init__(self, budget_kb_per_min: float):
        self.budget_kb_per_min = budget_kb_per_min
        self.window_start = time.time()
        self.used_kb = 0.0
        self.total_sent_kb = 0.0
        self.dropped = 0
        self.keyframes_sent = 0

    def can_send(self, size_kb: float) -> bool:
        now = time.time()
        if now - self.window_start >= 60.0:
            self.window_start = now
            self.used_kb = 0.0
        return (self.used_kb + size_kb) <= self.budget_kb_per_min

    def send(self, size_kb: float) -> bool:
        if self.can_send(size_kb):
            self.used_kb += size_kb
            self.total_sent_kb += size_kb
            return True
        self.dropped += 1
        return False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 Orbital Edge Simulation")
    p.add_argument("--model", default="models/yolo11n.engine")
    p.add_argument("--source", default="videos/Benchmarking-video.mp4")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--regime", choices=["sunlight", "eclipse"], default="sunlight",
                   help="Power / downlink regime")
    p.add_argument("--temp-enter", type=float, default=52.0)
    p.add_argument("--temp-exit", type=float, default=50.5)
    p.add_argument("--latency-enter", type=float, default=40.0)
    p.add_argument("--latency-exit", type=float, default=32.0)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--skip-rate", type=int, default=2)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max-dets-per-frame", type=int, default=5,
                   help="Keep at most this many highest-scoring detections (sunlight)")
    p.add_argument("--min-score", type=float, default=0.35,
                   help="Minimum score (conf × weight) to keep a detection (sunlight)")
    p.add_argument("--keyframe-every", type=int, default=30,
                   help="Send a keyframe every N processed frames (0 = disable)")
    p.add_argument("--status-every", type=float, default=5.0,
                   help="Print orbital status line every N seconds")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Regime → downlink budget + selectivity
    if args.regime == "sunlight":
        downlink_budget = 2500.0          # KB / min ≈ 2.5 MB/min
        effective_min_score = args.min_score
        effective_max_dets = args.max_dets_per_frame
    else:
        downlink_budget = 800.0           # KB / min ≈ 0.8 MB/min
        # More selective under eclipse
        effective_min_score = max(args.min_score, 0.45)
        effective_max_dets = min(args.max_dets_per_frame, 2)

    print("=" * 72)
    print("PHASE 3 — ORBITAL EDGE DATA PROCESSING PIPELINE SIMULATION")
    print(f"  Model                : {args.model}")
    print(f"  Source               : {args.source}")
    print(f"  Regime               : {args.regime.upper()}")
    print(f"  Downlink budget      : {downlink_budget:.0f} KB/min")
    print(f"  Min score / max dets : {effective_min_score:.2f} / {effective_max_dets}")
    print(f"  Keyframe every       : {args.keyframe_every} processed frames"
          + (" (disabled)" if args.keyframe_every <= 0 else ""))
    print(f"  Temp enter / exit    : {args.temp_enter} / {args.temp_exit} °C")
    print(f"  Latency enter / exit : {args.latency_enter} / {args.latency_exit} ms")
    print(f"  Skip rate (stress)   : every {args.skip_rate} frames")
    print(f"  Duration             : {args.duration} s")
    print("=" * 72)

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

    bw = BandwidthTracker(downlink_budget)

    jetson = jtop()
    jetson.start()
    time.sleep(1.0)

    # Warm-up
    print(f"\nWarming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            _ = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    print("Running orbital simulation loop... (Ctrl+C to stop early)\n")

    # Counters
    processed = 0
    skipped = 0
    total_dets_seen = 0
    total_dets_kept = 0
    total_dets_discarded = 0
    discard_reasons = {"low_score": 0, "top_n": 0, "bandwidth": 0}
    latencies = []
    power_samples = []
    timeseries = []
    keep_discard_log = []          # detailed per-detection decisions

    start = time.perf_counter()
    frame_idx = 0
    last_status_t = 0.0
    keyframes_attempted = 0

    try:
        while time.perf_counter() - start < args.duration:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            frame_idx += 1
            t_now = time.perf_counter() - start
            stats = jetson.stats
            temp_gpu = float(stats.get("Temp gpu", 0.0))

            do_process = controller.should_process()
            latency = 0.0
            dets_this_frame = []
            kept_this_frame = 0
            discarded_this_frame = 0
            bytes_sent = 0.0
            keyframe_sent = False

            if do_process:
                t0 = time.perf_counter()
                results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
                latency = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency)
                processed += 1
                controller.update(temp_gpu, latency)

                # --- Priority scoring ---
                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        names = r.names
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = names.get(cls_id, "unknown")
                            weight = get_class_weight(name)
                            score = conf * weight
                            dets_this_frame.append({
                                "name": name,
                                "conf": conf,
                                "weight": weight,
                                "score": score
                            })

                total_dets_seen += len(dets_this_frame)

                # Sort by score descending
                dets_this_frame.sort(key=lambda d: d["score"], reverse=True)

                keep_list = []
                for i, d in enumerate(dets_this_frame):
                    reason = None
                    if d["score"] < effective_min_score:
                        reason = "low_score"
                        discard_reasons["low_score"] += 1
                    elif len(keep_list) >= effective_max_dets:
                        reason = "top_n"
                        discard_reasons["top_n"] += 1
                    else:
                        # Candidate for keep — still subject to bandwidth
                        if bw.send(DETECTION_PAYLOAD_KB):
                            keep_list.append(d)
                            bytes_sent += DETECTION_PAYLOAD_KB
                            keep_discard_log.append({
                                "t": round(t_now, 3),
                                "frame": frame_idx,
                                "name": d["name"],
                                "conf": round(d["conf"], 3),
                                "score": round(d["score"], 3),
                                "decision": "KEEP",
                                "reason": "priority+bandwidth_ok"
                            })
                        else:
                            reason = "bandwidth"
                            discard_reasons["bandwidth"] += 1

                    if reason is not None:
                        discarded_this_frame += 1
                        keep_discard_log.append({
                            "t": round(t_now, 3),
                            "frame": frame_idx,
                            "name": d["name"],
                            "conf": round(d["conf"], 3),
                            "score": round(d["score"], 3),
                            "decision": "DISCARD",
                            "reason": reason
                        })

                kept_this_frame = len(keep_list)
                total_dets_kept += kept_this_frame
                total_dets_discarded += discarded_this_frame

                # --- Keyframe (occasional full verification image) ---
                if args.keyframe_every > 0 and (processed % args.keyframe_every == 0):
                    keyframes_attempted += 1
                    if bw.send(KEYFRAME_KB):
                        bytes_sent += KEYFRAME_KB
                        bw.keyframes_sent += 1
                        keyframe_sent = True
                        print(f"  → Keyframe sent at t={t_now:.1f}s "
                              f"(processed frame #{processed})")

            else:
                skipped += 1
                controller.update(temp_gpu, 0.0)

            # Power estimate
            est_power = estimate_power(
                processing=do_process,
                regime=args.regime,
                in_stress=controller.in_stress
            )
            power_samples.append(est_power)

            # Live orbital status line
            if t_now - last_status_t >= args.status_every:
                status = (
                    f"[t={t_now:5.1f}s] {args.regime.upper():8s} | "
                    f"power≈{est_power:4.1f} W | "
                    f"skip={controller.current_skip} | "
                    f"dets {kept_this_frame}/{len(dets_this_frame)} | "
                    f"bw {bw.used_kb:6.0f}/{downlink_budget:.0f} KB | "
                    f"temp {temp_gpu:4.1f}°C"
                )
                if keyframe_sent:
                    status += " | KEYFRAME"
                print(status)
                last_status_t = t_now

            # Time-series row
            timeseries.append({
                "t": round(t_now, 3),
                "frame": frame_idx,
                "processed": int(do_process),
                "skip_rate": controller.current_skip,
                "in_stress": int(controller.in_stress),
                "temp_gpu": round(temp_gpu, 1),
                "latency_ms": round(latency, 2),
                "est_power_w": round(est_power, 2),
                "dets_seen": len(dets_this_frame),
                "dets_kept": kept_this_frame,
                "dets_discarded": discarded_this_frame,
                "bw_sent_kb": round(bytes_sent, 2),
                "bw_used_window_kb": round(bw.used_kb, 2),
                "keyframe": int(keyframe_sent),
            })

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        jetson.close()
        cap.release()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start
    total_frames = processed + skipped
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    mean_power = float(np.mean(power_samples)) if power_samples else 0.0
    raw_video_estimate_mb = (total_frames * 0.15)   # rough 150 KB/frame @ 640

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  Regime                  : {args.regime.upper()}")
    print(f"  Runtime                 : {elapsed:.1f} s")
    print(f"  Frames seen             : {total_frames}")
    print(f"  Frames processed        : {processed}  ({100*processed/max(1,total_frames):.1f}%)")
    print(f"  Frames skipped          : {skipped}  ({100*skipped/max(1,total_frames):.1f}%)")
    print(f"  Mean latency (proc)     : {mean_lat:.2f} ms")
    print(f"  Mean estimated power    : {mean_power:.2f} W")
    print(f"  Detections seen         : {total_dets_seen}")
    print(f"  Detections kept         : {total_dets_kept}")
    print(f"  Detections discarded    : {total_dets_discarded}")
    print(f"    └─ low_score          : {discard_reasons['low_score']}")
    print(f"    └─ top_n cutoff       : {discard_reasons['top_n']}")
    print(f"    └─ bandwidth full     : {discard_reasons['bandwidth']}")
    print(f"  Keyframes sent          : {bw.keyframes_sent} / {keyframes_attempted} attempted")
    print(f"  Bandwidth sent          : {bw.total_sent_kb:.1f} KB  ({bw.total_sent_kb/1024:.2f} MB)")
    print(f"  Bandwidth drops         : {bw.dropped}")
    print(f"  Raw video (est.)        : {raw_video_estimate_mb:.1f} MB")
    print(f"  Adaptive decisions      : {len(controller.decisions)}")
    print("=" * 72)

    if controller.decisions:
        print("\nAdaptive decision log:")
        for d in controller.decisions:
            print(f"  {d['action']:18s} | {d['reason']}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Time-series CSV
    ts_path = out_dir / f"orbital_sim_{args.regime}_{ts}.csv"
    fieldnames = list(timeseries[0].keys()) if timeseries else []
    with open(ts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeseries)
    print(f"\nTime-series saved to: {ts_path}")

    # Adaptive decision log
    if controller.decisions:
        dec_path = out_dir / f"orbital_decisions_{args.regime}_{ts}.csv"
        with open(dec_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["t", "action", "reason", "avg_temp", "avg_latency"]
            )
            writer.writeheader()
            writer.writerows(controller.decisions)
        print(f"Adaptive decisions saved to: {dec_path}")

    # Detailed keep/discard log (sample of decisions — useful for narrative)
    if keep_discard_log:
        kd_path = out_dir / f"orbital_keep_discard_{args.regime}_{ts}.csv"
        with open(kd_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["t", "frame", "name", "conf", "score", "decision", "reason"]
            )
            writer.writeheader()
            writer.writerows(keep_discard_log)
        print(f"Keep/discard log saved to: {kd_path}")

    # Summary text
    summary_path = out_dir / f"orbital_summary_{args.regime}_{ts}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Phase 3 Orbital Simulation — {args.regime.upper()}\n")
        f.write(f"Runtime: {elapsed:.1f}s\n")
        f.write(f"Processed: {processed}/{total_frames} "
                f"({100*processed/max(1,total_frames):.1f}%)\n")
        f.write(f"Mean power: {mean_power:.2f} W\n")
        f.write(f"Dets kept/seen: {total_dets_kept}/{total_dets_seen}\n")
        f.write(f"Discard reasons — low_score: {discard_reasons['low_score']}, "
                f"top_n: {discard_reasons['top_n']}, "
                f"bandwidth: {discard_reasons['bandwidth']}\n")
        f.write(f"Keyframes sent: {bw.keyframes_sent}\n")
        f.write(f"Bandwidth sent: {bw.total_sent_kb:.1f} KB\n")
    print(f"Summary saved to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()