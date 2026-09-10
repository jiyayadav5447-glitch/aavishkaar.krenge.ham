import cv2
import torch
from ultralytics import YOLO
import pyttsx3

# Setup Voice
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Start Camera
cap = cv2.VideoCapture(0)

print("Downloading and Loading AI Models... This may take a minute!")
# Load YOLO (Finds objects)
yolo = YOLO('yolov8n.pt') 

# Load MiDaS (Calculates depth)
midas = torch.hub.load("isl-org/MiDaS", "MiDaS_small")
midas.eval()
transforms = torch.hub.load("isl-org/MiDaS", "transforms")
transform = transforms.small_transform

print("Models loaded! Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Ask YOLO to find objects
    results = yolo(frame, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_id = int(box.cls[0])
            object_name = yolo.names[class_id]
            cv2.putText(frame, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 2. Ask MiDaS to check depth
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Check the center of the screen
    center_y, center_x = depth_map.shape[0] // 2, depth_map.shape[1] // 2
    center_depth = depth_map[center_y, center_x]
    
    # Visual Warning
    if center_depth > 500: # Adjust this number later based on your room!
        cv2.putText(frame, "WARNING: CLOSE OBJECT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # speak("Warning!") # We leave this commented out for the first test so it doesn't talk forever

    # Show the video feed on screen
    cv2.imshow("Avishkar Prototype", frame)

    # Press 'q' on your keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()