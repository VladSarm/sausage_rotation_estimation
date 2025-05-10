import cv2
import time

cap = cv2.VideoCapture(0)  # macOS default uses AVFoundation

# Try to set resolution and frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*2)
cap.set(cv2.CAP_PROP_FPS, 240)  # May be ignored on macOS

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual resolution: {frame_width}x{frame_height}")
print(f"Actual FPS: {actual_fps}")

resolution_text = f"Res: {frame_width}x{frame_height}"
fps_text = "FPS: Calculating..."

prev_frame_time = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        fps_text = f"FPS: {fps:.2f}"
        frame_count = 0
        start_time = current_time

    cv2.putText(frame, resolution_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()