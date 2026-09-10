"""Open3D Visual SLAM and 3D point cloud mapping."""
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
