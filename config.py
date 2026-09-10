"""System configuration and hyperparameter constants."""
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
