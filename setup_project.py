import os

files = {}

# ==========================================
# 1. config.py
# ==========================================
files["config.py"] = '''"""System configuration and hyperparameter constants."""
import numpy as np

# Camera Intrinsics (Default 640x480 webcam profile)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5

# Open3D Odometry & PointCloud Parameters
VOXEL_SIZE = 0.05          # 5 cm voxel downsampling
MAX_DEPTH_METERS = 5.0    # MiDaS clip range
MIN_DEPTH_METERS = 0.3

# Floor Occupancy Grid & Navigation
GRID_RESOLUTION = 0.1      # 10 cm per cell
GRID_SIZE_X = 10.0         # 10m x 10m space
GRID_SIZE_Z = 10.0
OBSTACLE_CLEARANCE = 0.3   # Safety margin in meters

# Voice & Speech Controls
SPEECH_COOLDOWN_SEC = 3.0
PROXIMITY_ALERT_DIST = 1.5 # Trigger warning if obstacle < 1.5m
'''

# ==========================================
# 2. models.py
# ==========================================
files["models.py"] = '''"""AI Models: MiDaS Monocular Depth & YOLOv8 Object Detector."""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import config

class PerceptionPipeline:
    def __init__(self):
        # Choose Apple Silicon MPS GPU if available, else CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load MiDaS Small
        print(f"[AI] Loading MiDaS depth estimator on {self.device}...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        # Load YOLOv8 Nano
        print("[AI] Loading YOLOv8n object detector...")
        self.yolo = YOLO("yolov8n.pt")

    @torch.no_grad()
    def estimate_depth(self, rgb_frame):
        img_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        prediction = self.midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_raw = prediction.cpu().numpy()
        # Invert & normalize inverse relative depth to metric approximation
        depth_norm = cv2.normalize(depth_raw, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        depth_metric = (1.0 - depth_norm) * (config.MAX_DEPTH_METERS - config.MIN_DEPTH_METERS) + config.MIN_DEPTH_METERS
        return depth_metric.astype(np.float32)

    def detect_objects(self, rgb_frame):
        results = self.yolo(rgb_frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            label = results.names[cls_id]
            conf = float(box.conf[0].item())
            detections.append({"bbox": (x1, y1, x2, y2), "label": label, "conf": conf})
        return detections
'''

# ==========================================
# 3. slam.py
# ==========================================
files["slam.py"] = '''"""Open3D Visual SLAM and 3D point cloud mapping."""
import open3d as o3d
import numpy as np
import config

class VisualSLAM:
    def __init__(self):
        self.pinhole_camera = o3d.camera.PinholeCameraIntrinsic(
            config.FRAME_WIDTH, config.FRAME_HEIGHT,
            config.FX, config.FY, config.CX, config.CY
        )
        self.current_pose = np.eye(4)
        self.global_map = o3d.geometry.PointCloud()
        self.prev_rgbd = None

    def create_rgbd_image(self, rgb_frame, depth_map):
        color = o3d.geometry.Image(rgb_frame)
        depth = o3d.geometry.Image((depth_map * 1000.0).astype(np.uint16))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1000.0,
            depth_trunc=config.MAX_DEPTH_METERS,
            convert_rgb_to_intensity=False
        )

    def process_frame(self, rgb_frame, depth_map):
        curr_rgbd = self.create_rgbd_image(rgb_frame, depth_map)
        curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            curr_rgbd, self.pinhole_camera
        )

        if self.prev_rgbd is not None:
            # Steinbrucker2011 RGBD odometry
            success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                self.prev_rgbd, curr_rgbd,
                self.pinhole_camera,
                np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                o3d.pipelines.odometry.OdometryOption()
            )
            if success:
                self.current_pose = self.current_pose @ trans

        self.prev_rgbd = curr_rgbd

        # Accumulate transformed pointcloud into global map
        pcd_transformed = curr_pcd.transform(self.current_pose)
        self.global_map += pcd_transformed
        
        # Bug fix: Open3D uses voxel_down_sample
        self.global_map = self.global_map.voxel_down_sample(voxel_size=config.VOXEL_SIZE)

        return self.current_pose, curr_pcd

    def save_map(self, filename="room_map.pcd"):
        o3d.io.write_point_cloud(filename, self.global_map)
        print(f"[SLAM] Map saved to {filename} ({len(self.global_map.points)} points)")
'''

# ==========================================
# 4. semantic_map.py
# ==========================================
files["semantic_map.py"] = '''"""Transforms 2D YOLO detections to 3D World space."""
import numpy as np
import config

class SemanticMap:
    def __init__(self):
        self.objects = []  # List of dicts: {label, world_pos, conf}

    def update(self, detections, depth_map, camera_pose):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            uc = (x1 + x2) // 2
            vc = (y1 + y2) // 2

            # Inner 60% region depth median
            dx = int((x2 - x1) * 0.2)
            dy = int((y2 - y1) * 0.2)
            crop_depth = depth_map[y1 + dy:y2 - dy, x1 + dx:x2 - dx]

            if crop_depth.size == 0:
                continue
            z = float(np.median(crop_depth))
            if z <= 0.2 or z > config.MAX_DEPTH_METERS:
                continue

            # Inverse pinhole camera model
            xc = (uc - config.CX) * z / config.FX
            yc = (vc - config.CY) * z / config.FY
            p_cam = np.array([xc, yc, z, 1.0])

            # Rigid transform to world frame
            p_world = camera_pose @ p_cam

            self.objects.append({
                "label": det["label"],
                "pos": p_world[:3],
                "conf": det["conf"]
            })
'''

# ==========================================
# 5. navigation.py
# ==========================================
files["navigation.py"] = '''"""Occupancy grid flattening, 8-connected A* search, and spatial vectors."""
import heapq
import numpy as np
import config

class Navigator:
    def __init__(self):
        self.res = config.GRID_RESOLUTION

    def pointcloud_to_grid(self, pcd, grid_span=10.0):
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        grid_cells = int(grid_span / self.res)
        grid = np.zeros((grid_cells, grid_cells), dtype=np.uint8)
        
        # Flatten onto X-Z plane
        for p in points:
            gx = int((p[0] + grid_span / 2) / self.res)
            gz = int((p[2] + grid_span / 2) / self.res)
            if 0 <= gx < grid_cells and 0 <= gz < grid_cells:
                grid[gx, gz] = 1
        return grid

    def a_star(self, grid, start, goal):
        rows, cols = grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def h(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            for dx, dz in neighbors:
                nbr = (current[0] + dx, current[1] + dz)
                if 0 <= nbr[0] < rows and 0 <= nbr[1] < cols:
                    if grid[nbr[0], nbr[1]] == 1:
                        continue  # Obstacle
                    tentative_g = g_score[current] + np.hypot(dx, dz)
                    if tentative_g < g_score.get(nbr, float("inf")):
                        came_from[nbr] = current
                        g_score[nbr] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(nbr, goal), nbr))
        return None

    def get_spatial_direction(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dz = target_pos[2] - current_pos[2]
        dist = np.sqrt(dx**2 + dz**2)
        angle = np.degrees(np.arctan2(dx, dz))

        if abs(angle) < 15:
            direction = "straight ahead"
        elif angle >= 15:
            direction = f"turn right by {int(angle)} degrees"
        else:
            direction = f"turn left by {int(abs(angle))} degrees"

        return dist, direction
'''

# ==========================================
# 6. voice.py
# ==========================================
files["voice.py"] = '''"""Non-blocking macOS text-to-speech speaker."""
import subprocess
import time
import config

class VoiceAssistant:
    def __init__(self):
        self.last_speech_time = 0

    def speak(self, text, force=False):
        now = time.time()
        if force or (now - self.last_speech_time > config.SPEECH_COOLDOWN_SEC):
            subprocess.Popen(["say", text])
            self.last_speech_time = now
'''

# ==========================================
# 7. main.py
# ==========================================
files["main.py"] = '''"""Main Orchestrator with OpenCV Webcam interface."""
import cv2
import time
import numpy as np
import config
from models import PerceptionPipeline
from slam import VisualSLAM
from semantic_map import SemanticMap
from navigation import Navigator
from voice import VoiceAssistant

def main():
    print("[System Ready] Controls: 's' = Scan Mode | 'g' = Guide Mode | 't' = Cycle Target | 'q' = Quit")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    ai = PerceptionPipeline()
    slam = VisualSLAM()
    s_map = SemanticMap()
    nav = Navigator()
    voice = VoiceAssistant()

    mode = "IDLE"
    targets = ["chair", "person", "bottle", "door", "cup"]
    target_idx = 0

    voice.speak("Navigation assistant ready.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = ai.estimate_depth(frame)
        detections = ai.detect_objects(frame)

        if mode == "SCAN":
            cam_pose, _ = slam.process_frame(frame, depth_map)
            s_map.update(detections, depth_map, cam_pose)
            cv2.putText(frame, "[SCANNING ROOM]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif mode == "GUIDE":
            curr_target = targets[target_idx]
            cam_pos = slam.current_pose[:3, 3]

            # Find latest matching target
            matches = [obj for obj in s_map.objects if obj["label"] == curr_target]
            if matches:
                target_pos = matches[-1]["pos"]
                dist, direction = nav.get_spatial_direction(cam_pos, target_pos)
                msg = f"{curr_target} is {dist:.1f} meters, {direction}"
                voice.speak(msg)
                cv2.putText(frame, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                voice.speak(f"Searching for {curr_target}")

            cv2.putText(frame, f"[GUIDING TO: {curr_target}]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw 2D detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, f"{det['label']} {det['conf']:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Depth heatmap visualization
        depth_vis = cv2.applyColorMap((depth_map / config.MAX_DEPTH_METERS * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        cv2.imshow("Avishkar AI Navigation - Color Feed", frame)
        cv2.imshow("Avishkar AI Navigation - Depth Map", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if mode == "SCAN":
                mode = "IDLE"
                slam.save_map("room_map.pcd")
                voice.speak("Scanning stopped. Map saved.")
            else:
                mode = "SCAN"
                voice.speak("Scanning room map started.")
        elif key == ord('g'):
            mode = "GUIDE" if mode != "GUIDE" else "IDLE"
            voice.speak("Guide mode activated." if mode == "GUIDE" else "Guide mode stopped.")
        elif key == ord('t'):
            target_idx = (target_idx + 1) % len(targets)
            voice.speak(f"Target set to {targets[target_idx]}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''

for filename, content in files.items():
    with open(filename, "w") as f:
        f.write(content.strip() + "\n")
    print(f"Created: {filename}")

print("\nAll 7 project files successfully written and updated.")
