"""
OmniSense AI - Indoor Navigation Assistant for the Visually Impaired
======================================================================
Pipeline:
  1. Capture video from a phone (DroidCam) over Wi-Fi
  2. YOLOv8 -> "WHAT" is in the scene (object + bounding box)
  3. MiDaS  -> "HOW FAR" is everything (per-pixel relative depth)
  4. Fusion -> combine box + depth to get "object X is Y away"
  5. pyttsx3 -> speak warnings, without freezing the video loop

Run:  python app.py
Quit: press 'q' in the video window
"""

import cv2
import torch
import numpy as np
import pyttsx3
import threading
import queue
import time
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 0. CONFIGURATION - change these to match your setup
# ----------------------------------------------------------------------
DROIDCAM_URL = "http://192.168.137.77:4747/video"   # <-- your phone's stream URL

# Objects that matter most for navigation get a louder / more urgent tone
PRIORITY_CLASSES = {"person", "chair", "door", "couch", "dining table", "stairs"}

# Distance zones, in *approximate* meters (see calibration note in explanation)
ZONE_DANGER = 1.0     # "STOP"
ZONE_WARNING = 2.5    # "Caution"
# anything farther -> not announced (reduces noise)

# How often to run the (expensive) MiDaS depth model.
# Every frame is overkill and slow on CPU; every 3rd-5th frame is plenty
# because depth doesn't change drastically frame-to-frame.
DEPTH_EVERY_N_FRAMES = 4

# Don't repeat the same warning for the same object every single frame.
# Cooldown in seconds, per (class, zone) pair.
ANNOUNCE_COOLDOWN = 3.0

# Resize incoming frames for speed. Smaller = faster, less accurate.
PROCESS_WIDTH = 480

# ----------------------------------------------------------------------
# 1. TEXT-TO-SPEECH: a background worker so speaking never blocks video
# ----------------------------------------------------------------------
class Speaker:
    """
    pyttsx3 is blocking (engine.say + runAndWait() freezes your program
    while it talks). We run it in its own thread with a queue, so the
    main video loop keeps grabbing frames smoothly while the voice
    catches up in the background.
    """
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)   # speaking speed
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()

    def say(self, text):
        # Drop the message if the queue is already backed up, so we don't
        # end up several warnings "behind" reality.
        if self.q.qsize() < 2:
            self.q.put(text)


# ----------------------------------------------------------------------
# 2. LOAD MODELS
# ----------------------------------------------------------------------
def load_models():
    print("[INFO] Loading YOLOv8n...")
    yolo_model = YOLO("yolov8n.pt")   # nano model -> fastest, good enough for CPU

    print("[INFO] Loading MiDaS_small...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print(f"[INFO] Running on device: {device}")
    return yolo_model, midas, transform, device


# ----------------------------------------------------------------------
# 3. DEPTH ESTIMATION
# ----------------------------------------------------------------------
def estimate_depth(frame_rgb, midas, transform, device):
    """
    Returns a 2D numpy array, same-ish size as the frame, where each
    pixel is a RELATIVE inverse-depth value (bigger number = CLOSER).
    This is NOT metric (not literally "meters") out of the box -
    see the calibration explanation below.
    """
    input_batch = transform(frame_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],   # resize depth map back to frame size
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map


def relative_depth_to_meters(depth_value, depth_min, depth_max):
    """
    MiDaS outputs RELATIVE inverse depth (higher = closer), not metric
    distance. To turn that into an approximate "meters" number for a
    demo, we normalize it against the min/max depth seen in the CURRENT
    frame, then map it onto a rough real-world range (0.3m to 6m),
    inverted because higher MiDaS value = closer.

    This is a heuristic good enough for a hackathon demo, not a
    laser rangefinder. Real metric depth needs stereo cameras, LiDAR,
    or a calibrated single-camera model.
    """
    if depth_max - depth_min < 1e-6:
        return 3.0  # fallback if depth map is flat (rare)

    normalized = (depth_value - depth_min) / (depth_max - depth_min)  # 0..1, 1 = closest
    meters = 6.0 - normalized * (6.0 - 0.3)   # invert: closest -> 0.3m, farthest -> 6m
    return round(meters, 2)


# ----------------------------------------------------------------------
# 4. FUSION: combine YOLO boxes with the MiDaS depth map
# ----------------------------------------------------------------------
def fuse_detections_with_depth(results, depth_map, yolo_model):
    """
    For each YOLO bounding box, we look at the MATCHING region of the
    depth map (same pixel coordinates) and pull out a representative
    depth value for that object - the MEDIAN depth inside the box.

    Why median, not average?
      - Average gets skewed by a few background pixels leaking into
        the box edges (e.g. a chair box that includes wall behind it).
      - Median is robust to those outlier pixels and represents the
        "typical" depth of the object itself.
    """
    detections = []

    depth_min = float(np.min(depth_map))
    depth_max = float(np.max(depth_map))

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = yolo_model.names[cls_id]

        # Clip box to frame bounds (safety, in case YOLO gives edge coords)
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Shrink the box by 20% inward before sampling depth.
        # This avoids sampling background pixels right at the object's edges.
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
        cx1, cx2 = x1 + pad_x, x2 - pad_x
        cy1, cy2 = y1 + pad_y, y2 - pad_y
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2  # fallback to full box

        object_depth_region = depth_map[cy1:cy2, cx1:cx2]
        median_depth_value = float(np.median(object_depth_region))

        distance_m = relative_depth_to_meters(median_depth_value, depth_min, depth_max)

        detections.append({
            "label": label,
            "conf": conf,
            "box": (x1, y1, x2, y2),
            "distance_m": distance_m,
        })

    return detections


# ----------------------------------------------------------------------
# 5. DECIDE WHAT TO SAY
# ----------------------------------------------------------------------
last_announced = {}  # {(label, zone): last_time_announced}

def decide_and_announce(detections, speaker):
    now = time.time()

    # Announce the CLOSEST dangerous object first (most urgent)
    detections_sorted = sorted(detections, key=lambda d: d["distance_m"])

    for det in detections_sorted:
        label = det["label"]
        dist = det["distance_m"]

        if dist <= ZONE_DANGER:
            zone = "danger"
            message = f"Stop. {label} very close, {dist} meters ahead."
        elif dist <= ZONE_WARNING:
            zone = "warning"
            message = f"Caution. {label} ahead, {dist} meters."
        else:
            continue  # too far to matter right now

        key = (label, zone)
        if key not in last_announced or (now - last_announced[key]) > ANNOUNCE_COOLDOWN:
            speaker.say(message)
            last_announced[key] = now
            break  # only announce ONE thing per cycle so speech doesn't overlap/queue up


# ----------------------------------------------------------------------
# 6. DRAWING (visual debug overlay - not needed by the blind user,
#    but essential for YOU to debug/demo the system)
# ----------------------------------------------------------------------
def draw_overlay(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        dist = det["distance_m"]

        if dist <= ZONE_DANGER:
            color = (0, 0, 255)      # red
        elif dist <= ZONE_WARNING:
            color = (0, 165, 255)    # orange
        else:
            color = (0, 255, 0)      # green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {dist}m"
        cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


# ----------------------------------------------------------------------
# 7. MAIN LOOP
# ----------------------------------------------------------------------
def main():
    speaker = Speaker()
    yolo_model, midas, transform, device = load_models()

    print(f"[INFO] Connecting to DroidCam stream: {DROIDCAM_URL}")
    cap = cv2.VideoCapture(DROIDCAM_URL)

    if not cap.isOpened():
        print("[ERROR] Could not open video stream. Check that:")
        print("  1. DroidCam app is running on your phone")
        print("  2. Phone and laptop are on the SAME Wi-Fi network")
        print("  3. The URL/IP in DROIDCAM_URL matches what DroidCam shows")
        return

    frame_count = 0
    cached_depth_map = None
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0

    print("[INFO] Starting main loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying...")
            time.sleep(0.5)
            continue

        # Resize for speed: smaller frame = faster YOLO + faster MiDaS
        h, w = frame.shape[:2]
        scale = PROCESS_WIDTH / w
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- YOLO runs every frame (it's relatively cheap on yolov8n) ---
        results = yolo_model(frame_rgb, verbose=False)[0]

        # --- MiDaS runs only every N frames (it's the expensive part) ---
        if frame_count % DEPTH_EVERY_N_FRAMES == 0 or cached_depth_map is None:
            cached_depth_map = estimate_depth(frame_rgb, midas, transform, device)
        depth_map = cached_depth_map

        # --- Fuse: attach a distance to every detected object ---
        detections = fuse_detections_with_depth(results, depth_map, yolo_model)

        # --- Decide if we need to speak a warning ---
        decide_and_announce(detections, speaker)

        # --- Draw debug overlay for the demo screen ---
        frame = draw_overlay(frame, detections)

        # FPS counter (useful to show judges it runs in real time)
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        cv2.putText(frame, f"FPS: {fps_display}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("OmniSense AI - Live Feed", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    speaker.q.put(None)


if __name__ == "__main__":
    main()