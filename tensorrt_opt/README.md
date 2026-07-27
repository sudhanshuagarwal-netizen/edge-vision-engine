# TensorRT Optimization & Benchmarking (Phase 1)

**Status:** Core milestone achieved (July 27, 2026)  
**Hardware:** NVIDIA Jetson Orin Nano Super (JetPack 6.2.1, TensorRT 10.3)  
**Model:** YOLO11n (FP16 TensorRT engine built via ONNX + native `trtexec`)

## Goal
Convert the native YOLO pipeline to TensorRT, integrate it into a clean runtime, and produce repeatable performance numbers under real edge constraints.

## Key Results

| Backend              | Input Size | Mean Latency | Avg FPS  | Notes                                      |
|----------------------|------------|--------------|----------|--------------------------------------------|
| **TensorRT (.engine)** | 640×640   | **27.1 ms**  | **36.9** | Successful sustained run on recorded video |
| PyTorch (.pt)        | 416×416    | 39.6 ms      | 25.2     | Highest resolution that fit in memory      |
| PyTorch (.pt)        | 640×640    | —            | —        | CUDA OOM on Orin Nano (8 GB shared)        |

**Takeaway:** At the target 640 resolution the TensorRT engine runs cleanly at ~37 FPS while the full PyTorch model cannot fit in device memory. This is exactly the kind of efficiency gain needed for power- and memory-constrained edge / orbital platforms.

## Deliverables Completed
- [x] Valid FP16 TensorRT engine (`yolo11n.engine`) built with native `trtexec`
- [x] Minimal engine load test (`src/test_engine_load.py`)
- [x] Production-style switchable node (`src/vision_node_trt.py`) – supports both `.engine` and `.pt` via `MODEL_PATH`
- [x] Repeatable benchmarking harness (`src/benchmark.py`)
- [x] Head-to-head numbers on a fixed 44 s recorded video source
- [x] Results CSVs under `benchmarks/results/`

## How to Reproduce

```bash
# TensorRT (640)
python3 src/benchmark.py \
  --model models/yolo11n.engine \
  --source videos/Benchmarking-video.mp4 \
  --imgsz 640 --duration 30 --warmup 10 --no-power

# PyTorch (416 – highest that fits)
python3 src/benchmark.py \
  --model models/yolo11n.pt \
  --source videos/Benchmarking-video.mp4 \
  --imgsz 416 --duration 30 --warmup 10 --no-power