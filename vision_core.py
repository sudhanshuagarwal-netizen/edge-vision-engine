import cv2
import numpy as np

# 1. Connect to your phone's wireless camera stream
stream_url = 'http://192.168.1.231:4747/video' 
cap = cv2.VideoCapture(stream_url)

print("Connecting to the vision matrix... Press 'q' to quit.")

# 2. Start the continuous video loop
while True:
    # Grab the current frame from the camera
    success, frame = cap.read()
    
    if not success:
        print("Failed to connect to the camera stream. Is DroidCam running?")
        break

    # 3. The Matrix Test! Print the dimensions of the color image
    print(f"Current Matrix Shape: {frame.shape}")

    # 4. Display the video feed on your Mac
    cv2.imshow('Edge Vision Engine - Live Feed', frame)

    # 5. Safety switch to break the loop if you press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the windows when done
cap.release()
cv2.destroyAllWindows()