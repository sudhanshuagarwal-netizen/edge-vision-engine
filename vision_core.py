import time
import cv2
from ultralytics import YOLO

# Sprint 3 - Adding PIDController class
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
        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative

        # 5. Total Combined Control Effort
        total_output = p_out + i_out + d_out

        # 6. SOFTWARE CLAMPING (Safety Cage)
        # Prevents wild mathematical spikes from burning out physical hardware
        if total_output > self.max_output:
            total_output = self.max_output
        elif total_output < -self.max_output:
            total_output = -self.max_output

        # Save states for the next loop iteration
        self.prev_error = error
        self.prev_time = current_time

        return total_output, error

# 1. Initializing / loading the AI Brain called YOLO (You only look once)
print("Initializing AI Vision Engine...")
model = YOLO('yolo11n.pt')

# Frame dimensions are 640x480. Dead center is (320, 240)
FRAME_CENTER_X = 320
FRAME_CENTER_Y = 240

# Instantiate independent PID controllers for Pan (Left/Right) and Tilt (Up/Down)
# Starting with safe, conservative baseline gains based on our tuning analysis
pan_pid = PIDController(kp=0.08, ki=0.002, kd=0.02, max_output=25.0)
tilt_pid = PIDController(kp=0.08, ki=0.002, kd=0.02, max_output=25.0)

# 2. Connect to my phone's camera stream
stream_url = 'http://192.168.1.231:4747/video' 
cap = cv2.VideoCapture(stream_url)

print("System Operational. Tracking Loop Live.")

while True:
    # Grab the current frame from the camera
    success, frame = cap.read()
    if not success:
        print("Failed to connect to the camera stream.")
        break

    # 3. Feed the frame into the AI (confidence threshold set to 45%)
    # stream=True optimizes memory by not storing historical frames
    results = model(frame, stream=True, conf=0.45)

    # Track only the primary detected target per frame to avoid multi-target confusion
    target_found = False
    target_x, target_y = FRAME_CENTER_X, FRAME_CENTER_Y
    target_name = "SEARCHING"

    # 4. Extract the data and draw the bounding boxes
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
    # We pass the current AI target position vs where we want it to be (the center)
    pan_effort, pan_error = pan_pid.calculate(current_pos=target_x, target_pos=FRAME_CENTER_X)
    tilt_effort, tilt_error = tilt_pid.calculate(current_pos=target_y, target_pos=FRAME_CENTER_Y)

    if target_found:
        print(f"TRACKING: {target_name} | "
              f"ErrorX: {pan_error:4d} px -> Pan Command: {pan_effort:+.2f}°/s | "
              f"ErrorY: {tilt_error:4d} px -> Tilt Command: {tilt_effort:+.2f}°/s")
    else:
        print("STATUS: Airspace Clear - Sweeping Passive Scanning Pattern...")

    # Draw tracking crosshairs on your display interface
    cv2.circle(annotated_frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 10, (0, 0, 255), 2)
    cv2.imshow('Tactical Edge Vision System', annotated_frame)

    # 6. Safety switch to break the loop if you press 'q'
    #time.sleep(0.5) # pauses the loop for half second (slow down to 2 frames/sec) to read terminal for debugging
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the windows when done
cap.release()
cv2.destroyAllWindows()