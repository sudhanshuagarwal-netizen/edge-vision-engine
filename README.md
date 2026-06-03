# Building a Local Wireless Video Ingestion Engine

This is Phase 1 of a personal project to get comfortable with edge AI hardware constraints before deploying models onto an NVIDIA Jetson Orin Nano. 

The goal here was simple but tricky: capture a live, uncompressed video matrix from my Google Pixel 10, stream it over my local home Wi-Fi network, and unpack the raw frames in real time using Python and OpenCV on my Mac. 

## The Constraints (Why Python 3.10?)
When building for edge devices like the Jetson architecture (running JetPack 6.x / Ubuntu 22.04), environment mismatch is a nightmare. I intentionally locked this local environment to **Python 3.10** and **OpenCV 4.x** right from the start. Taking the extra five minutes to configure this virtual environment now saves me from massive PyTorch wheel compilation errors when I flash the Jetson hardware later.

## What Broke (and How I Fixed It)

### 1. The Invisible Network Wall (`No route to host`)
Once the Python environment was live, my initial script kept timing out trying to grab the stream. 
* **The Problem:** The phone was jumping between my home Wi-Fi network and a cellular 5G carrier IP, which effectively blocked my Mac from finding it.
* **The Fix:** Forced the Pixel 10 to lock onto the local subnet, verified they shared the same access point, and routed the OpenCV `VideoCapture` object explicitly through port `4747` (DroidCam's raw MJPEG server endpoint). 

### 2. Stripping the Matrix Weight (Canny Pre-processing)
Streaming raw `(480, 640, 3)` color frames into an edge device can choke memory quickly. To prepare the stream for real-time model inference, I added an active pre-processing step:
1. Converted incoming frames to grayscale to drop the data weight by 66% (getting rid of the blue, green, and red channels).
2. Applied a Canny edge-detection algorithm to isolate physical boundaries.

I manually tuned the algorithm thresholds to **(30, 100)**. This range is strict enough to wipe out standard indoor background shadows while cleanly capturing sharp object borders.

## Current Setup & Live Diagnostics
If you run `python vision_core.py`, it will initialize the dual-window matrix pipeline. You'll see the raw frame comparison alongside the live mathematical edge map, printing out the current matrix frame arrays directly to the terminal loop.

## What's Next
Next up is importing a lightweight object detection model (like YOLO11n) and calculating the exact CPU inference latency and frame-rate drops.