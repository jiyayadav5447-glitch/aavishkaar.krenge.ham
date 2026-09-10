"""
test_pipeline.py
================
Headless unit verification script for AI Models, Open3D RGBD processing,
A* Pathfinding algorithm, and macOS Speech synthesis.
"""

import numpy as np
import open3d as o3d
import torch
import config
from models import DepthEstimator, ObjectDetector
from slam import VisualSLAM
from semantic_map import SemanticMapManager
from navigation import OccupancyGridMap, AStarPlanner, NavigationGuide
from voice import VoiceAssistant


def test_system():
    print("[Test 1/5] Testing A* Pathfinding Algorithm...")
    grid_map = OccupancyGridMap(resolution=0.1)
    grid_map.width, grid_map.height = 50, 50
    grid_map.origin_x, grid_map.origin_z = 0.0, 0.0
    grid_map.grid = np.zeros((50, 50), dtype=np.uint8)
    
    # Add wall obstacle in middle
    grid_map.grid[20:30, 25] = 1

    planner = AStarPlanner(grid_map)
    start = (1.0, 1.0)
    goal = (4.0, 4.0)
    path = planner.plan(start, goal)
    assert len(path) > 0, "A* pathfinding failed!"
    print(f"  -> Path calculated successfully! Start: {start}, Goal: {goal}, Path steps: {len(path)}")

    print("[Test 2/5] Testing 2D-to-3D Projection Geometry...")
    mapper = SemanticMapManager()
    u, v, z = 320.0, 240.0, 2.5
    p_cam = mapper.project_2d_to_3d(u, v, z)
    expected_p = np.array([0.0, 0.0, 2.5])
    np.testing.assert_allclose(p_cam, expected_p, atol=1e-5)
    print(f"  -> Projection correct: (u={u}, v={v}, z={z}) -> 3D {p_cam}")

    print("[Test 3/5] Testing Spatial Guidance Text Formatting...")
    msg1 = NavigationGuide.format_voice_instruction("gate", 3.0, -25.0)
    msg2 = NavigationGuide.format_voice_instruction("person", 1.5, 0.0, is_warning=True)
    print(f"  -> Voice alert 1: '{msg1}'")
    print(f"  -> Voice alert 2: '{msg2}'")

    print("[Test 4/5] Testing macOS Asynchronous Voice Assistant...")
    voice = VoiceAssistant(cooldown=0.1)
    voice.speak("System test sequence verified successfully.")
    print("  -> Voice command dispatched asynchronously.")

    print("[Test 5/5] Testing AI Models & Visual SLAM Frame Processing...")
    depth_est = DepthEstimator()
    detector = ObjectDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = depth_est.estimate_depth(dummy_frame)
    dets = detector.detect(dummy_frame)
    print(f"  -> MiDaS depth output shape: {depth.shape}, YOLO detections: {len(dets)}")

    vslam = VisualSLAM()
    pose, pcd = vslam.process_frame(dummy_frame, depth)
    pose2, pcd2 = vslam.process_frame(dummy_frame, depth)
    print(f"  -> Open3D SLAM processed 2 frames successfully! Global map points: {len(vslam.global_map.points)}")

    print("\n==================================================")
    print("      ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("==================================================")



if __name__ == "__main__":
    test_system()
