#!/usr/bin/env python3
from ultralytics import YOLO
import time, cv2
print('Loading TensorRT engine...')
model = YOLO('models/yolo11n.engine')
print('Engine loaded successfully')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Camera failed'); exit(1)
print('Running 30 frames for timing...')
times = []
for i in range(30):
    ret, frame = cap.read()
    if not ret: continue
    t0 = time.perf_counter()
    results = model(frame, verbose=False)
    times.append((time.perf_counter() - t0) * 1000)
cap.release()
print(f'Avg latency: {sum(times)/len(times):.1f} ms')
print(f'Approx FPS: {1000 / (sum(times)/len(times)):.1f}')
print('SUCCESS — engine is usable')