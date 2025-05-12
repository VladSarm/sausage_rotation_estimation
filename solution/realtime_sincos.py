
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from ultralytics import YOLO
import math
from torchvision import datasets, transforms
import os
import time
import sys
import numpy as np

neighbor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'angle_recognition'))
sys.path.append(neighbor_path)
from sincos_model import SinCosResModel

CROP_SIZE = 224

img_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CROP_SIZE,CROP_SIZE)),
    # transforms.RandomRotation(degrees=(180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_angle_cv2(bgr_img, model):
    """
    bgr_img: H×W×3 uint8 array from OpenCV (cv2.imread or frame)
    returns: angle in degrees in [0,360)
    """
    # convert BGR→RGB
    rgb = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

    # apply transforms → tensor of shape (3,H,W)
    t = img_tf(rgb)

    # add batch dim → (1,3,IMG_SIZE,IMG_SIZE)
    t = t.unsqueeze(0).to(device)

    # forward pass → (1,2) unit vector
    with torch.no_grad():
        vec = model(t)

    # extract back to CPU & numpy
    vec = vec[0].cpu().numpy()    # [cosθ, sinθ]

    # reconstruct angle
    angle_rad = math.atan2(vec[1], vec[0])        # (–π,π]
    angle_deg = (angle_rad * 180 / math.pi) % 360

    return angle_deg
    
IMG_SIZE = 640

def process_frame(frame, model, yolo_model):
    w, h = frame.shape[1], frame.shape[0]
    image = frame[h//2 - IMG_SIZE//2:h//2 + IMG_SIZE//2, w//2 - IMG_SIZE//2:w//2 + IMG_SIZE//2]
    results = yolo_model(image, verbose=False)
    detections = results[0]  # Extract detections

    if len(detections) == 0:
        return -1, None
    
    det = detections[0]

    x1, y1, x2, y2 = det.boxes.xyxy[0].tolist()
    conf = det.boxes.conf[0].item()
    if conf < 0.3:  # Confidence threshold
        return -1, None
    
    # Crop the detected region
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    padding = 0.1
    x1 = x1 - int((x2 - x1) * padding)
    y1 = y1 - int((y2 - y1) * padding)
    x2 = x2 + int((x2 - x1) * padding)
    y2 = y2 + int((y2 - y1) * padding)
    cropped = image[y1:y2, x1:x2]

    return predict_angle_cv2(cropped, model), (x1, y1, x2, y2)

def camera(cap, name="camera"):
    global t_prev
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        exit()

    result, box = process_frame(frame, model, yolo_model)
    result_color = (0, 255, 0) if abs(result-180)%360 <= ERROR_TRESHOLD else (0, 0, 255)

    image = cv.putText(frame, f"Rotation: {result:.1f} deg", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    fps = 1 / (time.time() - t_prev)
    t_prev = time.time()
    image = cv.putText(image, f"FPS : {int(fps)}", (10, 200), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    image = cv.putText(image, f"Inference : {int(1/fps*1000)} ms", (10, 320), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    w, h = frame.shape[1], frame.shape[0]
    image = cv.rectangle(image, (w//2 - IMG_SIZE//2, h//2 - IMG_SIZE//2), (w//2 + IMG_SIZE//2, h//2 + IMG_SIZE//2), (0, 255, 0), 6)
    if not box is None:
        x1, y1, x2, y2 = box
        x1 += w//2 - IMG_SIZE//2
        y1 += h//2 - IMG_SIZE//2
        x2 += w//2 - IMG_SIZE//2
        y2 += h//2 - IMG_SIZE//2
        image = cv.rectangle(image, (x1, y1), (x2, y2), result_color, 2)
    cv.imshow(name, image)
    return result

def circular_average_deg(angle1, angle2):
    # Convert degrees to radians
    a1_rad = math.radians(angle1)
    a2_rad = math.radians(angle2)

    # Compute average using vector summation
    x = math.cos(a1_rad) + math.cos(a2_rad)
    y = math.sin(a1_rad) + math.sin(a2_rad)

    avg_rad = math.atan2(y, x)
    avg_deg = math.degrees(avg_rad) % 360

    return avg_deg

USE_WEBCAM = True
DATA_FOLDER = "/Users/vlad.sarm/Documents/sausage_rotation_estimation/data/test_dataset"
ERROR_TRESHOLD = 10

if __name__ == "__main__":
    yolo_model = YOLO('/Users/vlad.sarm/Documents/sausage_rotation_estimation/weights/yolo11n_trained.pt')
    device = 'mps'
    model = SinCosResModel(feature_extract=False).to(device)
    checkpoint = torch.load("/Users/vlad.sarm/Documents/sausage_rotation_estimation/angle_recognition/sincos_/MAE-2.04_EPOCH-29.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Loaded model weights and set model to eval mode")

    if not USE_WEBCAM:
        imgs = os.listdir(DATA_FOLDER)
        for img in imgs:
            img_path = os.path.join(DATA_FOLDER, img)
            image = cv.imread(img_path)
            if image is None:
                print(f"Error: Could not read image {img_path}.")
                continue
            result, box = process_frame(image, model, yolo_model)
            image = cv.putText(image, f"{result:.2f}", (30, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4)
            w, h = image.shape[1], image.shape[0]
            image = cv.rectangle(image, (w//2 - IMG_SIZE//2, h//2 - IMG_SIZE//2), (w//2 + IMG_SIZE//2, h//2 + IMG_SIZE//2), (0, 0, 255), 2)
            if not box is None:
                x1, y1, x2, y2 = box
                x1 += w//2 - IMG_SIZE//2
                y1 += h//2 - IMG_SIZE//2
                x2 += w//2 - IMG_SIZE//2
                y2 += h//2 - IMG_SIZE//2
                image = cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv.imshow(f'{img}', image)

        cv.waitKey(0)
        exit()

    # Open a connection to the webcam
    cap_side = cv.VideoCapture(0)
    cap_top = cv.VideoCapture(1)

    cap_side.set(cv.CAP_PROP_FRAME_WIDTH, 4000)
    cap_side.set(cv.CAP_PROP_FRAME_HEIGHT, 2000)
    if not cap_side.isOpened() and not cap_top.isOpened():
        print("Error: Could not open webcam.")
        exit()
    # Read a frame from the webcam
    t_prev = time.time()
    while True:
        side_ang = camera(cap_side, "Side Camera")
        # top_ang = camera(cap_top, "Top Camera")
        top_ang = side_ang
        # top_ang -= 90
        # top_ang = top_ang % 360

        result = circular_average_deg(side_ang, top_ang)
        result_color = (0, 255, 0) if abs(result-180)%360 <= ERROR_TRESHOLD else (0, 0, 255)

        zeros = np.zeros((400,400,3), np.uint8)
        zeros = cv.putText(zeros, f"Rotation: {result:.1f} deg", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)
        zeros = cv.putText(zeros, f"Side: {side_ang:.1f} deg", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)
        # zeros = cv.putText(zeros, f"Top: {top_ang:.1f} deg", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)
        center = (150, 250)
        r = 100
        zeros = cv.circle(zeros, center, r, result_color, -1)
        result = np.deg2rad(result)
        pt2 = np.array([center[0] + np.cos(result)*r,
                center[1] + np.sin(result)*r])
        zeros = cv.line(zeros, center, pt2.astype(int), (255,255,255), 3)
        cv.imshow("Result", zeros)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap_side.release()
    cap_top.release()
    cv.destroyAllWindows()