import os
import torch
import argparse
import numpy as np
from PIL import Image
import json
import cv2
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

import torch
import torch.nn.functional as F

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def load_depth_sequence(depth_dir, n_frames):
    """Load depth sequence from directory of depth images."""
    depth_sequence = []
    # Load every 2nd frame to match downsampled video
    for i in range(0, n_frames * 2, 2):  # Skip frames to match video downsampling
        depth_path = os.path.join(depth_dir, f"frame_{i:06d}.png")
        if os.path.exists(depth_path):
            # Load as 16-bit depth (millimeters)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # Convert back to meters
            depth_meters = depth_img.astype(np.float32) / 1000.0
            depth_sequence.append(depth_meters)
        else:
            print(f"Warning: Missing depth frame {depth_path}")
            
    return np.stack(depth_sequence, axis=0)  # Shape: (T, H, W)



def lift_tracks_to_3d(tracks_2d, visibility, depths, camera_intrinsics):
    """
    Vectorized version: Lift 2D tracks to 3D using depth maps.
    
    Args:
        tracks_2d: (B, T, N, 2) - 2D pixel coordinates
        visibility: (B, T, N) - track visibility
        depths: (T, H, W) - depth maps
        camera_intrinsics: (3, 3) - intrinsics
    Returns:
        tracks_3d: (B, T, N, 3)
        valid_3d: (B, T, N)
    """
    B, T, N, _ = tracks_2d.shape
    device = tracks_2d.device
    H, W = depths.shape[1], depths.shape[2]

    # Reshape for grid_sample: combine B and N dimensions
    tracks_flat = tracks_2d.view(B * T, N, 2)  # (B*T, N, 2)
    
    # normalize coords to [-1,1] for grid_sample
    u = tracks_flat[..., 0] / (W - 1) * 2 - 1
    v = tracks_flat[..., 1] / (H - 1) * 2 - 1
    grid = torch.stack([u, v], dim=-1)  # (B*T, N, 2)

    # Prepare depths for sampling: (B*T, 1, H, W)
    depths_expanded = depths.unsqueeze(1).repeat(B, 1, 1, 1)  # (B*T, 1, H, W)
    
    # Sample depth values - grid should be (B*T, N, 1, 2) for grid_sample
    grid_4d = grid.unsqueeze(2)  # (B*T, N, 1, 2)
    
    sampled_depths = F.grid_sample(
        depths_expanded,  # (B*T, 1, H, W)
        grid_4d,  # (B*T, N, 1, 2)
        align_corners=True,
        mode="bilinear"
    ).squeeze(1).squeeze(-1)  # (B*T, N)
    
    # Reshape back to original dimensions
    sampled_depths = sampled_depths.view(B, T, N)  # (B, T, N)

    # Intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Back-project
    z = sampled_depths
    x = (tracks_2d[..., 0] - cx) * z / fx
    y = (tracks_2d[..., 1] - cy) * z / fy
    tracks_3d = torch.stack([x, y, z], dim=-1)

    # Valid mask
    valid_3d = (z > 0) & (z < 100) & visibility.bool()

    return tracks_3d, valid_3d


def cluster_motion_trajectories(tracks_3d, valid_3d, motion_threshold=0.05):
    """
    Cluster 3D trajectories based on motion patterns.
    
    Args:
        tracks_3d: (B, T, N, 3) - 3D trajectories
        valid_3d: (B, T, N) - validity mask
        motion_threshold: float - threshold for static vs moving classification
        
    Returns:
        motion_labels: (B, N) - 0 for static, 1+ for different moving objects
        motion_vectors: list - average motion vector per cluster
    """
    B, T, N, _ = tracks_3d.shape
    device = tracks_3d.device  # Ensure we use the same device
    motion_labels = torch.zeros(B, N, dtype=torch.long, device=device)
    motion_vectors = []
    
    for b in range(B):
        # Calculate total displacement for each track
        displacements = []
        valid_tracks = []
        
        for n in range(N):
            valid_frames = valid_3d[b, :, n]
            if valid_frames.sum() < 2:
                continue
                
            # Get first and last valid 3D positions
            valid_indices = torch.where(valid_frames)[0]
            start_pos = tracks_3d[b, valid_indices[0], n]
            end_pos = tracks_3d[b, valid_indices[-1], n]
            
            displacement = torch.norm(end_pos - start_pos).item()
            displacements.append(displacement)
            valid_tracks.append(n)
        
        if len(displacements) == 0:
            continue
            
        # Classify as static or moving
        displacements = np.array(displacements)
        static_mask = displacements < motion_threshold
        
        # Label static points as 0
        for i, track_idx in enumerate(valid_tracks):
            if static_mask[i]:
                motion_labels[b, track_idx] = 0
            else:
                motion_labels[b, track_idx] = 1  # All moving points as same object for now
                
        # Calculate average motion vector for moving points (keep on device)
        moving_tracks = [valid_tracks[i] for i in range(len(valid_tracks)) if not static_mask[i]]
        if moving_tracks:
            motion_sum = torch.zeros(3, device=device)  # Keep on same device
            count = 0
            for track_idx in moving_tracks:
                valid_frames = valid_3d[b, :, track_idx]
                if valid_frames.sum() >= 2:
                    valid_indices = torch.where(valid_frames)[0]
                    start_pos = tracks_3d[b, valid_indices[0], track_idx]
                    end_pos = tracks_3d[b, valid_indices[-1], track_idx]
                    motion_sum += (end_pos - start_pos)
                    count += 1
            
            if count > 0:
                avg_motion = motion_sum / count
                motion_vectors.append(avg_motion)
            
    return motion_labels, motion_vectors

def save_3d_trajectories(tracks_3d, valid_3d, motion_labels, output_dir):
    """Save 3D trajectories and motion analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw 3D tracks
    torch.save(tracks_3d, os.path.join(output_dir, "tracks_3d.pt"))
    torch.save(valid_3d, os.path.join(output_dir, "valid_3d.pt"))
    torch.save(motion_labels, os.path.join(output_dir, "motion_labels.pt"))
    
    # Save as point clouds for visualization
    B, T, N, _ = tracks_3d.shape
    for b in range(B):
        # Static points
        static_points = []
        moving_points = []
        
        for n in range(N):
            if motion_labels[b, n] == 0:  # Static
                valid_frames = valid_3d[b, :, n]
                if valid_frames.any():
                    # Use first valid position for static points
                    first_valid = torch.where(valid_frames)[0][0]
                    static_points.append(tracks_3d[b, first_valid, n].cpu().numpy())
            else:  # Moving
                valid_frames = valid_3d[b, :, n]
                for t in range(T):
                    if valid_frames[t]:
                        moving_points.append(tracks_3d[b, t, n].cpu().numpy())
        
        # Save point clouds
        if static_points:
            static_pcd = np.array(static_points)
            np.save(os.path.join(output_dir, f"static_points_batch{b}.npy"), static_pcd)
            
        if moving_points:
            moving_pcd = np.array(moving_points)
            np.save(os.path.join(output_dir, f"moving_points_batch{b}.npy"), moving_pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4", 
        help="path to a video",
    )
    parser.add_argument(
        "--depth_dir",
        default="./depth_frames",
        help="directory containing depth frame sequence",
    )
    parser.add_argument(
        "--camera_metadata",
        default="./video_metadata.json",
        help="path to camera intrinsics and metadata",
    )
    parser.add_argument(
        "--output_dir", 
        default="./trajectory_output",
        help="output directory for 3D analysis results",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.05,
        help="3D motion threshold in meters for static/moving classification",
    )
    
    args = parser.parse_args()
    
    # Load video with memory optimization
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    T, H, W = video.shape[1], video.shape[3], video.shape[4]
    
    # Memory optimization: downsample video if too large
    if T > 30 or H > 480:
        print(f"Downsampling video from {T} frames to reduce memory usage...")
        # Temporal downsampling: take every 2nd frame
        video = video[:, ::2]  # Shape: (1, T//2, 3, H, W)
        T = video.shape[1]
        print(f"New video shape: {video.shape}")
    
    # Load corresponding depth sequence
    depths = load_depth_sequence(args.depth_dir, T)
    depths = torch.from_numpy(depths)  # Shape: (T, H, W)
    
    # Load camera intrinsics
    with open(args.camera_metadata, 'r') as f:
        metadata = json.load(f)
    camera_intrinsics = torch.tensor(metadata["camera_intrinsics"])
    
    print(f"Loaded video: {video.shape}")
    print(f"Loaded depths: {depths.shape}")
    print(f"Camera intrinsics: {camera_intrinsics}")
    
    # Load CoTracker3 model with memory management
    print("Loading CoTracker3 model...")
    torch.cuda.empty_cache()  # Clear any existing GPU memory
    
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(DEFAULT_DEVICE)
    
    # Move data to device with memory checks
    available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    print(f"Available GPU memory: {available_memory / 1e9:.2f} GB")
    
    video = video.to(DEFAULT_DEVICE)
    depths = depths.to(DEFAULT_DEVICE)
    camera_intrinsics = camera_intrinsics.to(DEFAULT_DEVICE)
    
    # Track 2D trajectories with gradient disabled to save memory
    print("Computing 2D trajectories...")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
        )
    
    print(f"2D tracks shape: {pred_tracks.shape}")
    print(f"Visibility shape: {pred_visibility.shape}")
    
    # Lift 2D tracks to 3D using depth
    print("Lifting tracks to 3D...")
    with torch.no_grad(): 
        tracks_3d, valid_3d = lift_tracks_to_3d(
            pred_tracks, pred_visibility, depths, camera_intrinsics
        )
    

    
    print(f"3D tracks shape: {tracks_3d.shape}")
    print(f"Valid 3D points: {valid_3d.sum().item()} / {valid_3d.numel()}")
    
    # Cluster trajectories by motion patterns
    print("Clustering motion trajectories...")
    motion_labels, motion_vectors = cluster_motion_trajectories(
        tracks_3d, valid_3d, motion_threshold=args.motion_threshold
    )
    
    # Analysis summary
    static_count = (motion_labels == 0).sum().item()
    moving_count = (motion_labels > 0).sum().item()
    print(f"Static points: {static_count}")
    print(f"Moving points: {moving_count}")
    
    if motion_vectors:
        for i, motion_vec in enumerate(motion_vectors):
            print(f"Motion cluster {i+1}: avg displacement = {motion_vec}")
    
    # Save results
    save_3d_trajectories(tracks_3d, valid_3d, motion_labels, args.output_dir)
    
    # Save 2D visualization
    seq_name = os.path.splitext(os.path.basename(args.video_path))[0]
    vis = Visualizer(save_dir=args.output_dir, pad_value=120, linewidth=3, tracks_leave_trace=-1)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=args.grid_query_frame,
        filename=f"{seq_name}_2d_tracks",
    )
    
    # Create 3D motion mask (binary static vs moving)
    B, T, N = tracks_3d.shape[:3]
    motion_mask_3d = torch.zeros(T, H, W, dtype=torch.bool)
    
    for b in range(B):
        for t in range(T):
            for n in range(N):
                if valid_3d[b, t, n] and motion_labels[b, n] > 0:  # Moving point
                    u, v = pred_tracks[b, t, n, 0], pred_tracks[b, t, n, 1]
                    u_int, v_int = int(u.round()), int(v.round())
                    if 0 <= u_int < W and 0 <= v_int < H:
                        motion_mask_3d[t, v_int, u_int] = True
    
    # Save motion masks
    motion_masks_dir = os.path.join(args.output_dir, "motion_masks")
    os.makedirs(motion_masks_dir, exist_ok=True)
    
    for t in range(T):
        mask_img = (motion_mask_3d[t].cpu().numpy() * 255).astype(np.uint8)
        mask_path = os.path.join(motion_masks_dir, f"motion_mask_{t:06d}.png")
        cv2.imwrite(mask_path, mask_img)
    
    print(f"Results saved to {args.output_dir}")
    print("3D trajectories: tracks_3d.pt")
    print("Motion labels: motion_labels.pt") 
    print("Motion masks: motion_masks/")
    print(f"2D visualization: {seq_name}_2d_tracks.mp4")