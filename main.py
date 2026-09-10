"""Main Orchestrator with OpenCV Webcam interface."""
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
