import cv2
import time

# Initialize both cameras
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(1)  # Second camera

# Try to set resolution and frame rate for both cameras
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 240)  # May be ignored on macOS

# Get properties for first camera
frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps1 = cap1.get(cv2.CAP_PROP_FPS)
print(f"Camera 1 - Actual resolution: {frame_width1}x{frame_height1}")
print(f"Camera 1 - Actual FPS: {actual_fps1}")

# Get properties for second camera
frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps2 = cap2.get(cv2.CAP_PROP_FPS)
print(f"Camera 2 - Actual resolution: {frame_width2}x{frame_height2}")
print(f"Camera 2 - Actual FPS: {actual_fps2}")

resolution_text1 = f"Cam1 Res: {frame_width1}x{frame_height1}"
resolution_text2 = f"Cam2 Res: {frame_width2}x{frame_height2}"
fps_text = "FPS: Calculating..."

prev_frame_time = 0
frame_count = 0
start_time = time.time()

while True:
    # Read from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Failed to grab frame from one or both cameras")
        break

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        fps_text = f"FPS: {fps:.2f}"
        frame_count = 0
        start_time = current_time

    # Add text to frames
    cv2.putText(frame1, resolution_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame1, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame2, resolution_text2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame2, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display both frames
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release both cameras
cap1.release()
cap2.release()
cv2.destroyAllWindows()