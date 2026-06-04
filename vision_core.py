import cv2
from ultralytics import YOLO

# 1. Load the AI Brain called YOLO (You only look once)
print("Loading YOLO11 Nano model...")
model = YOLO('yolo11n.pt')

# 2. Connect to my phone's camera stream
stream_url = 'http://192.168.1.231:4747/video' 
cap = cv2.VideoCapture(stream_url)

print("Connecting to the vision matrix... Press 'q' to quit.")

while True:
    # Grab the current frame from the camera
    success, frame = cap.read()
    if not success:
        print("Failed to connect to the camera stream.")
        break

    # 3. Feed the frame into the AI (confidence threshold set to 45%)
    # stream=True optimizes memory by not storing historical frames
    results = model(frame, stream=True, conf=0.45)

    # 4. Extract the data and draw the bounding boxes
    for r in results:
        # Use Ultralytics built-in tool to draw boxes on the frame
        annotated_frame = r.plot()

        # --- ROBOTICS LOGIC: Extracting the Tensors ---
        # r.boxes contains all the mathematical tensors for every object detected
        for box in r.boxes:
            # Extract the X/Y center coordinates (xywh = X_center, Y_center, width, height)
            coords = box.xywh[0] 
            x_center = int(coords[0])
            y_center = int(coords[1])
            
            # Get the object's class ID and look up its English name
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            
            # Print the exact location to terminal
            print(f"Target Acquired: {object_name.upper()} at [ X:{x_center}, Y:{y_center} ]")
        # ----------------------------------------------

    # 5. Display the live AI feed
    cv2.imshow('Edge Vision Engine - Object Detection', annotated_frame)

    # 6. Safety switch to break the loop if you press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close the windows when done
cap.release()
cv2.destroyAllWindows()