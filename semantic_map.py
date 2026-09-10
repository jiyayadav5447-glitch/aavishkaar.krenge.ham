"""Transforms 2D YOLO detections to 3D World space."""
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
