import cv2
import numpy as np

# 1. Connecting to my phone's camera & starting video stream
stream_url = 'http://192.168.1.231:4747/video' 
cap = cv2.VideoCapture(stream_url)

print("Connecting to the vision matrix... Press 'q' to quit.")

while True:
    # Grab the current frame from the camera
    success, frame = cap.read()
    if not success:
        print("Failed to connect to the camera stream.")
        break

    # 2. Convert the frame to Grayscale (drops memory weight by 66%)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Apply the Canny Edge Detection Algorithm
    # This traces lines around objects where contrast changes sharply
    edge_matrix = cv2.Canny(gray_frame, threshold1=30, threshold2=100)

    # 4. Display BOTH streams simultaneously
    cv2.imshow('Original Live Feed', frame)
    cv2.imshow('Edge Detection Matrix (AI Pre-Processor)', edge_matrix)

    # 5. Safety switch to break the loop if you press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the windows when done
cap.release()
cv2.destroyAllWindows()