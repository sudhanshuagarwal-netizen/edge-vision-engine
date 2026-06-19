import time
import cv2
from ultralytics import YOLO
import torch

print("Initializing AI Vision Engine on Jetson Nano...")

# ================== MODEL SETUP ==================
# Load once at startup
model = YOLO('yolo11n.pt')  # We'll optimize to TensorRT later

# ================== CAMERA SETUP (Logitech C920x) ==================
cap = cv2.VideoCapture(0)  # 0 = default USB camera

# Force 640x480 resolution (good balance for Jetson Nano)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open Logitech webcam!")
    print("Make sure the camera is plugged in and not used by another program.")
    exit(1)

print("✅ Logitech C920x connected successfully.")
print("System Operational. Tracking Loop Live.")

# Frame center for 640x480
FRAME_CENTER_X = 320
FRAME_CENTER_Y = 240

# ========== PID CONTROLLERS (kept for future) ===========
class PIDController:
    def __init__(self, kp, ki, kd, max_output=30.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output # Software clamping threshold to protect motors
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

    def calculate(self, current_pos, target_pos):
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.001 # Prevent division by zero errors if frames match timing exactly

        # 1. Calculate raw error (the spatial gap)
        error = target_pos - current_pos

        # 2. Proportional Term (The Gas)
        p_out = self.kp * error

        # 3. Integral Term (The Memory)
        self.integral += error * dt
        i_out = self.ki * self.integral

        # 4. Derivative Term (The Brake)
        d_out = self.kd * (error - self.prev_error) / dt

        # 5. Total Combined Control Effort
        output = p_out + i_out + d_out

        # 6. SOFTWARE CLAMPING (Safety Cage)
        # Prevents wild mathematical spikes from burning out physical hardware
        output = max(-self.max_output, min(self.max_output, output))

        # Save states for the next loop iteration
        self.prev_error = error
        self.prev_time = current_time

        return output, error

# Instantiate independent PID controllers for Pan (Left/Right) and Tilt (Up/Down)
# Starting with safe, conservative baseline gains based on our tuning analysis
pan_pid = PIDController(kp=0.12, ki=0.001, kd=0.08, max_output=25.0)
tilt_pid = PIDController(kp=0.12, ki=0.001, kd=0.08, max_output=25.0)


while True:
    # Grab the current frame from the camera
    success, frame = cap.read()
    if not success:
        print("Warning: Failed to grab frame. Retrying...")
        time.sleep(0.1)
        continue

    # Feed the frame into the AI (confidence threshold set to 45%)
    # stream=True optimizes memory by not storing historical frames
    # Running YOLO inference (Jetson-friendly setting)
    results = model(frame, stream=True, conf=0.45, verbose=False)

    # Track only the primary detected target per frame to avoid multi-target confusion
    target_found = False
    target_x, target_y = FRAME_CENTER_X, FRAME_CENTER_Y
    target_name = "SEARCHING"

    annotated_frame = None

    # Extract the data and draw the bounding boxes
    for r in results:
        # Use Ultralytics built-in tool to draw boxes on the frame
        annotated_frame = r.plot()

        # --- ROBOTICS LOGIC: Extracting the Tensors ---
        # r.boxes contains all the mathematical tensors for every object detected
        for box in r.boxes:
            if not target_found: # Lock onto the first object meeting confidence threshold
                coords = box.xywh[0]
                target_x = int(coords[0])
                target_y = int(coords[1])
                class_id = int(box.cls[0])
                target_name = model.names[class_id].upper()
                target_found = True

    # --- THE HARDWARE ABSTRACTION LAYER (HAL) ---
    # Calculate PID efforts (for future hardware)
    pan_effort, pan_error = pan_pid.calculate(target_x, FRAME_CENTER_X)
    tilt_effort, tilt_error = tilt_pid.calculate(target_y, FRAME_CENTER_Y)

    status = f"TRACKING {target_name}" if target_found else "SCANNING"
    print(f"{status} | Pan: {pan_effort:+.1f} Tilt: {tilt_effort:+.1f} | FPS ~ live")

    # Display
    if annotated_frame is not None:
        cv2.circle(annotated_frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 12, (0, 0, 255), 2)
        cv2.imshow('Edge Vision Engine - Jetson', annotated_frame)

    # Safety switch to break the loop if you press 'q'
    # time.sleep(0.5) # pauses the loop for half second (slow down to 2 frames/sec) to read terminal for debugging
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the windows when done
cap.release()
cv2.destroyAllWindows()