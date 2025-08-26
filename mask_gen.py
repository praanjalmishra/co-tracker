import torch
import numpy as np
import cv2
import os

def extract_change_masks(output_dir):
    """Extract 2D and 3D change masks from trajectory analysis results."""
    
    # Load the analysis results
    tracks_3d = torch.load(os.path.join(output_dir, "tracks_3d.pt"))
    valid_3d = torch.load(os.path.join(output_dir, "valid_3d.pt"))
    motion_labels = torch.load(os.path.join(output_dir, "motion_labels.pt"))
    
    B, T, N, _ = tracks_3d.shape
    
    # 1. Per-frame 2D masks (already in motion_masks/, but here's how to recreate)
    print("Creating 2D per-frame change masks...")
    masks_2d_dir = os.path.join(output_dir, "2d_change_masks")
    os.makedirs(masks_2d_dir, exist_ok=True)
    
    # You'd need original video dimensions - assuming from your code context
    # Replace with actual H, W from your video
    H, W = 480, 640  # Adjust based on your video resolution
    
    for t in range(T):
        mask_2d = np.zeros((H, W), dtype=np.uint8)
        
        for b in range(B):
            for n in range(N):
                if valid_3d[b, t, n] and motion_labels[b, n] > 0:  # Moving point
                    # Get 2D pixel coordinates (you'd need the original pred_tracks)
                    # This assumes you save pred_tracks separately or pass them to this function
                    pass
        
        # Save 2D mask for frame t
        cv2.imwrite(os.path.join(masks_2d_dir, f"change_mask_2d_{t:06d}.png"), mask_2d)
    
    # 2. 3D object masks - voxel grid or point-based
    print("Creating 3D change masks...")
    
    # Method A: Point-based 3D mask
    static_points_3d = []
    moving_points_3d = []
    
    for b in range(B):
        for n in range(N):
            if motion_labels[b, n] == 0:  # Static
                # Get all valid 3D positions for this static point
                valid_frames = valid_3d[b, :, n]
                if valid_frames.any():
                    first_valid = torch.where(valid_frames)[0][0]
                    static_points_3d.append(tracks_3d[b, first_valid, n].cpu().numpy())
            else:  # Moving
                # Get all valid 3D positions for this moving point
                valid_frames = valid_3d[b, :, n]
                for t in range(T):
                    if valid_frames[t]:
                        moving_points_3d.append(tracks_3d[b, t, n].cpu().numpy())

    # Save 3D point sets
    if static_points_3d:
        np.save(os.path.join(output_dir, "static_3d_mask.npy"), np.array(static_points_3d))
    if moving_points_3d:
        np.save(os.path.join(output_dir, "moving_3d_mask.npy"), np.array(moving_points_3d))
    
    # Method B: Voxel-based 3D mask
    print("Creating voxel-based 3D change mask...")
    create_voxel_change_mask(tracks_3d, valid_3d, motion_labels, output_dir)
    
    return masks_2d_dir

def create_voxel_change_mask(tracks_3d, valid_3d, motion_labels, output_dir, voxel_size=0.05):
    """Create a voxel grid showing static vs moving regions in 3D space."""
    
    B, T, N, _ = tracks_3d.shape
    
    # Find 3D bounds
    valid_points = tracks_3d[valid_3d]
    if len(valid_points) == 0:
        return
    
    min_bounds = valid_points.min(dim=0)[0]
    max_bounds = valid_points.max(dim=0)[0]
    
    # Create voxel grid
    grid_dims = ((max_bounds - min_bounds) / voxel_size).int() + 1
    static_voxels = torch.zeros(grid_dims.tolist(), dtype=torch.bool)
    moving_voxels = torch.zeros(grid_dims.tolist(), dtype=torch.bool)
    
    # Fill voxel grid
    for b in range(B):
        for n in range(N):
            label = motion_labels[b, n]
            valid_frames = valid_3d[b, :, n]
            
            if not valid_frames.any():
                continue
            
            if label == 0:  # Static
                # Use first valid position
                first_valid = torch.where(valid_frames)[0][0]
                pos = tracks_3d[b, first_valid, n]
                voxel_idx = ((pos - min_bounds) / voxel_size).int()
                if all(0 <= voxel_idx[i] < grid_dims[i] for i in range(3)):
                    static_voxels[voxel_idx[0], voxel_idx[1], voxel_idx[2]] = True
            else:  # Moving
                # Mark all valid positions
                for t in range(T):
                    if valid_frames[t]:
                        pos = tracks_3d[b, t, n]
                        voxel_idx = ((pos - min_bounds) / voxel_size).int()
                        if all(0 <= voxel_idx[i] < grid_dims[i] for i in range(3)):
                            moving_voxels[voxel_idx[0], voxel_idx[1], voxel_idx[2]] = True
    
    # Save voxel grids
    voxel_data = {
        'static_voxels': static_voxels,
        'moving_voxels': moving_voxels,
        'min_bounds': min_bounds,
        'max_bounds': max_bounds,
        'voxel_size': voxel_size,
        'grid_dims': grid_dims
    }
    torch.save(voxel_data, os.path.join(output_dir, "3d_voxel_change_mask.pt"))
    
    print(f"3D voxel grid: {grid_dims} voxels")
    print(f"Static voxels: {static_voxels.sum().item()}")
    print(f"Moving voxels: {moving_voxels.sum().item()}")

def visualize_3d_masks(output_dir):
    """Load and visualize 3D masks using matplotlib."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Load point clouds
    if os.path.exists(os.path.join(output_dir, "static_3d_mask.npy")):
        static_points = np.load(os.path.join(output_dir, "static_3d_mask.npy"))
        moving_points = np.load(os.path.join(output_dir, "moving_3d_mask.npy"))
        
        fig = plt.figure(figsize=(12, 5))
        
        # Static points
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(static_points[:, 0], static_points[:, 1], static_points[:, 2], 
                   c='blue', s=1, alpha=0.6)
        ax1.set_title('Static 3D Points')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Moving points
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(moving_points[:, 0], moving_points[:, 1], moving_points[:, 2], 
                   c='red', s=1, alpha=0.6)
        ax2.set_title('Moving 3D Points')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3d_change_visualization.png"), dpi=150)
        plt.show()

# Usage example:
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output", help="Directory to save output files")
    args = parser.parse_args()
    output_dir = args.output_dir
    extract_change_masks(output_dir)
    visualize_3d_masks(output_dir)