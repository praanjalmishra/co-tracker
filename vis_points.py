import numpy as np
import open3d as o3d

# Load the point clouds
static_points = np.load("trajectory_analysis_prism/static_points_batch0.npy")
moving_points = np.load("trajectory_analysis_prism/moving_points_batch0.npy")

pcd_static = o3d.geometry.PointCloud()
pcd_static.points = o3d.utility.Vector3dVector(static_points)
pcd_static.paint_uniform_color([1, 0, 0])  # Red

pcd_moving = o3d.geometry.PointCloud()
pcd_moving.points = o3d.utility.Vector3dVector(moving_points)
pcd_moving.paint_uniform_color([0, 1, 0])  # Green

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_static)
vis.add_geometry(pcd_moving)

# Get render options and set point size
opt = vis.get_render_option()
opt.point_size = 10.0   # Increase point size (default is 1.0)

# Run
vis.run()
vis.destroy_window()