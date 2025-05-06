
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from ultralytics import YOLO
import math
from torchvision import datasets, transforms
import os
import time

class CustomCNN(nn.Module):
    """
    A custom convolutional neural network that takes 64x128 images
    and outputs a 2-dimensional vector representing (cos, sin).
    Outputs are normalized to ensure they lie on the unit circle.
    """
    def __init__(self, input_channels: int = 3):
        super(CustomCNN, self).__init__()
        # Feature extractor
        self.features = nn.Sequential(
            # Conv block 1: 3 -> 16, output size: 64x128 -> 32x64
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 2: 16 -> 32, output size: 32x64 -> 16x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 3: 32 -> 64, output size: 16x32 -> 8x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 4: 64 -> 128, output size: 8x16 -> 4x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier head: flatten and output two values
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
            # nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_channels, 64, 128)
        x = self.features(x)
        x = self.classifier(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
img_tf = transforms.Compose([
    transforms.ToPILImage(),    
    transforms.Resize((128, 64)), # resize to 128x64
    # transforms.CenterCrop(IMG_SIZE),     # images are “vertical rectangles”; crop square centre
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)   # map pixel range to about (-1,1)
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
    cropped = image[y1:y2, x1:x2]

    return predict_angle_cv2(cropped, model), (x1, y1, x2, y2)

USE_WEBCAM = True
DATA_FOLDER = "test_dataset"

if __name__ == "__main__":
    yolo_model = YOLO('/Users/vlad.sarm/Documents/sausage_rotation_estimation/weights/yolo11n_trained.pt')
    device = 'mps'
    model = CustomCNN().to(device)
    checkpoint = torch.load("/Users/vlad.sarm/Documents/sausage_rotation_estimation/weights/angle_cnn_weights_2.pth", map_location=device)
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
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    # Read a frame from the webcam
    t_prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            cap.release()
            exit()

        # Convert the frame to RGB (if needed)
        # image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result, box = process_frame(frame, model, yolo_model)
        image = cv.putText(frame, f"Rotation: {result:.1f} deg", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        fps = 1 / (time.time() - t_prev)
        t_prev = time.time()
        image = cv.putText(image, f"FPS : {int(fps)}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        image = cv.putText(image, f"Inference : {int(1/fps*1000)} ms", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        w, h = frame.shape[1], frame.shape[0]
        image = cv.rectangle(image, (w//2 - IMG_SIZE//2, h//2 - IMG_SIZE//2), (w//2 + IMG_SIZE//2, h//2 + IMG_SIZE//2), (0, 255, 0), 2)
        if not box is None:
            x1, y1, x2, y2 = box
            x1 += w//2 - IMG_SIZE//2
            y1 += h//2 - IMG_SIZE//2
            x2 += w//2 - IMG_SIZE//2
            y2 += h//2 - IMG_SIZE//2
            image = cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.imshow('Webcam Feed', image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()