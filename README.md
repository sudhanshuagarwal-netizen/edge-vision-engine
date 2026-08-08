# Edge Vision Engine

**Real-time Object Detection on the Edge**  
Portfolio project supporting Georgia Tech OMSCS (Computing Systems) application

## Overview

This project builds a fully local, offline-capable real-time computer vision system and shows its progression from a laptop prototype to a true edge deployment under real-world constraints of limited power, no internet, and mobile operation.

The system uses YOLO11n for object detection and runs entirely on-device. No cloud inference is used at any stage.

The work is organized into three phases.

## Project Phases

### Phase 1 — Desktop Prototype

**Goal:** Establish a working real-time detection pipeline with minimal hardware friction.

**Setup**  
- Host: Laptop  
- Camera: Android phone streaming via DroidCam over Wi-Fi  
- Model: YOLO11n  
- Runtime: Python, Ultralytics, OpenCV

**Demonstrated**  
Live indoor detection of people, a dog, furniture, and common household objects with real-time bounding boxes and confidence scores.

This phase validated the core detection loop and camera abstraction before moving to dedicated edge hardware.

**Demo video:** https://youtu.be/3HD6DjMObGs

### Phase 2 — Fully Off-Grid Edge Deployment

**Goal:** Run the complete system on dedicated edge hardware with zero external dependencies (no laptop, no phone, no Wi-Fi, no cloud).

**Hardware**  
- NVIDIA Jetson Orin Nano Developer Kit  
- Powered by Tesla Model 3 twelve-volt cigarette lighter  
- USB webcam  
- Local display in the vehicle

**Demonstrated**  
Fully local inference while the vehicle was in motion. Real-time detection of cars, trucks, stop signs, and birds with no network connection. The system ran solely on vehicle power.

This phase proved the pipeline could operate independently under mobile, power-constrained, offline conditions.

**Demo video:** https://youtube.com/shorts/D7kQWPCcZAI

### Phase 3 — Optimization and Orbital-Style Edge Efficiency

This phase improves performance and demonstrates resource-aware behaviour on the Jetson Orin Nano.

**3.1 TensorRT Optimization**  
The model was converted to a TensorRT FP16 engine (8.4 MB). At 640 resolution the optimized pipeline achieves a mean latency of 27.1 ms and approximately 36.9 frames per second. Native PyTorch cannot load the same 640 resolution on the Orin Nano due to CUDA out-of-memory errors; the highest resolution PyTorch can run is 416, where it reaches only 25.2 frames per second. TensorRT therefore enables full-resolution inference that would otherwise be impossible on this hardware.

**3.2 Power, Thermal Profiling, and Adaptive Control**  
Under the 25 W power mode:  
- Idle power approximately 5.3 W  
- Full-quality detection approximately 7.1 W (extra cost of only 1.8 W)  
- Temperature rise under load approximately 1 °C  

Enabling maximum performance clocks raises power to about 9.5 W and increases speed to approximately 59 frames per second.

An adaptive controller monitors GPU temperature and latency. When stress is detected it automatically skips frames. In a controlled test it skipped about 26 percent of frames while keeping average latency near 27 ms. When conditions improve it returns to full rate.

**3.3 Orbital Edge Data Processing Simulation**  
The optimized adaptive pipeline was placed under orbital-style constraints using two regimes on the same recorded video:

- Sunlight regime (higher budget of 2500 KB per minute)  
- Eclipse regime (tighter budget of 800 KB per minute)

Key 90-second results:

**Sunlight**  
- 50 percent of frames processed (adaptive skip active)  
- Mean estimated power 6.42 W  
- 746 detections kept  

**Eclipse**  
- 100 percent of frames processed  
- Mean estimated power 7.12 W  
- Only 40 detections kept (tight bandwidth forced high selectivity)

The system successfully combined efficient inference, adaptive frame skipping, priority scoring of detections, and a rolling bandwidth limit. The simulation shows how an onboard node can stay within power and data-volume budgets while deciding what is worth transmitting.

## Technical Stack

- Detection model: YOLO11n (Ultralytics) with TensorRT FP16 engine  
- Runtime: Python 3, OpenCV, PyTorch, TensorRT  
- Edge platform: NVIDIA Jetson Orin Nano Developer Kit  
- Camera sources: USB webcam or DroidCam IP stream  
- Power (Phase 2): Vehicle 12 V

## How to Run

Install dependencies:  
```bash
pip install -r requirements.txt
```

Run the main node:  
```bash
python vision_node.py
```

Optional environment variables:  
```bash
VIDEO_SOURCE=0                  # USB camera
VIDEO_SOURCE=http://PHONE_IP:4747/video   # DroidCam
CONF_THRESHOLD=0.45
MODEL_PATH=yolo11n.pt
```

Press `q` to quit.

## Repository Structure

```
vision_node.py              # main real-time detection loop
vision_core.py              # earlier version with PID scaffolding
vision_node_trt.py          # TensorRT-optimized node
adaptive_vision_node.py     # adaptive frame-skipping controller
orbital_sim.py              # orbital constraint simulation
yolo11n.pt / yolo11n.engine # model weights (engines are Jetson-specific)
requirements.txt
README.md
```

## Key Design Notes

- Fully local inference from the beginning  
- Camera source abstracted early so the same code works with USB or IP streams  
- Progressive move from laptop to dedicated edge hardware  
- Real vehicle power and mobility used in Phase 2  
- TensorRT conversion enables full 640 resolution that native PyTorch cannot load  
- Measured power and thermal data added for efficiency  
- Adaptive control introduced so the system can trade a small amount of performance for lower resource use when needed  
- Priority-aware downlink simulation added to show behaviour under bandwidth limits  
- Docker containerization was explored and later dropped in favour of a clean native Python path

## Author

Sudhanshu Agarwal  
Hardware: NVIDIA Jetson Orin Nano Developer Kit  
Status: All three phases complete (Phase 1 desktop, Phase 2 off-grid vehicle, Phase 3 optimisation and orbital simulation)
