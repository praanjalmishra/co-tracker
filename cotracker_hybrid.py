from ast import arg
import os
import torch
import argparse
import numpy as np
from PIL import Image
import json
import cv2
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
from effsam_utils import effsam_batch_predict, effsam_refine_masks, expand_2D_bbox, effsam_embedding
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
# from image_utils import image_align

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
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

def image_diff(capture, render, dir, threshold=1e-2, kernel_ratio=0.03):
    """
    Image differencing for change detection (first vs last frame)

    Args:
        capture (1x3xHxW): Captured image (first frame)
        render (1x3xHxW): Rendered/last frame
        dir (str): Debug output directory
        threshold (float): Mask area ratio threshold
        kernel_ratio (float): Gaussian blur kernel fractional size

    Returns:
        masks (Mx1xHxW): Masks for the changed regions
        masks_all (Mx1xHxW): masks + small change regions
    """
    os.makedirs(dir, exist_ok=True)

    H, W = capture.shape[-2:]
    device = render.device

    # Convert to numpy for OpenCV / EffSAM
    capture_np = capture[0].permute(1, 2, 0).cpu().numpy()
    render_np  = render[0].permute(1, 2, 0).cpu().numpy()
    align_mask = np.ones((H, W), dtype=np.uint8)

    # --- Save raw input frames for debug
    cv2.imwrite(f"{dir}/capture.png", (capture_np[..., ::-1] * 255).astype(np.uint8))
    cv2.imwrite(f"{dir}/render.png",  (render_np[..., ::-1] * 255).astype(np.uint8))

    # Optional blur
    if kernel_ratio > 0:
        kernel_size = int(W * kernel_ratio)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        capture_np = cv2.GaussianBlur(capture_np, (kernel_size, kernel_size), 0)
        render_np  = cv2.GaussianBlur(render_np,  (kernel_size, kernel_size), 0)

    # --- EffSAM embeddings
    with torch.no_grad():
        emb1 = effsam_embedding(capture_np)
        emb2 = effsam_embedding(render_np)

    norm1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    norm2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

    similarity_map = torch.nn.functional.cosine_similarity(norm1, norm2, dim=1)
    similarity_map = similarity_map.squeeze().cpu().numpy()
    similarity_map = (similarity_map * 255).astype(np.uint8)

    cv2.imwrite(f"{dir}/similarity_map.png", similarity_map)

    # --- Thresholding
    thresh = cv2.threshold(
        similarity_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    # thresh = thresh * align_mask
    # Uncomment to debug
    # cv2.imwrite(f"{self.debug_dir}/thresh.png", thresh)
    # Find contours in the thresholded binary image
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Obtain masks for large enough contours
    masks, masks_all = [], []
    for contour in contours:
        mask = np.zeros((H, W))
        cv2.drawContours(
            mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
        )
        mask = torch.from_numpy(mask).unsqueeze(0)
        masks_all.append(mask)
        if cv2.contourArea(contour) >= threshold * H * W:
            masks.append(mask)
    masks = torch.stack(masks, dim=0).to(device)
    masks_all = torch.stack(masks_all, dim=0).to(device)
    # Uncomment to debug
    # for i, mask in enumerate(masks):
    #     cv2.imwrite(
    #         f"{self.debug_dir}/mask_{i}.png",
    #         mask.squeeze().cpu().numpy()
    #     )
    return masks, masks_all


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



def cluster_motion_trajectories(tracks_3d, valid_3d, motion_threshold=0.05, eps=0.05, min_samples=5):
    """
    Cluster 3D trajectories based on first-to-last displacement using DBSCAN.
    
    Args:
        tracks_3d: (B, T, N, 3) - 3D trajectories
        valid_3d: (B, T, N) - validity mask
        motion_threshold: float - threshold for static vs moving classification
        eps: float - DBSCAN neighborhood size (distance threshold)
        min_samples: int - minimum samples per cluster in DBSCAN
        
    Returns:
        motion_labels: (B, N) - 0 for static, 1..K for moving clusters
        motion_vectors: list of torch.Tensor - average motion vector per cluster
    """
    B, T, N, _ = tracks_3d.shape
    device = tracks_3d.device
    motion_labels = torch.zeros(B, N, dtype=torch.long, device=device)
    motion_vectors = []

    for b in range(B):
        displacements = []
        motion_vecs = []
        valid_tracks = []

        # Collect displacement vectors for all valid tracks
        for n in range(N):
            valid_frames = valid_3d[b, :, n]
            if valid_frames.sum() < 2:
                continue
            valid_indices = torch.where(valid_frames)[0]
            start_pos = tracks_3d[b, valid_indices[0], n]
            end_pos = tracks_3d[b, valid_indices[-1], n]
            disp_vec = end_pos - start_pos
            disp_mag = torch.norm(disp_vec).item()

            if disp_mag < motion_threshold:
                # static
                motion_labels[b, n] = 0
            else:
                displacements.append(disp_vec.cpu().numpy())
                motion_vecs.append(disp_vec)  # keep tensor version
                valid_tracks.append(n)

        if len(displacements) == 0:
            continue

        # Cluster moving points using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(displacements)
        labels = clustering.labels_  # -1 = noise, 0..K = clusters

        # Assign cluster labels (shifted by +1 to keep 0 for static)
        for i, track_idx in enumerate(valid_tracks):
            if labels[i] == -1:
                # noise → treat as static or assign its own label
                motion_labels[b, track_idx] = 0
            else:
                motion_labels[b, track_idx] = labels[i] + 1

        # Compute average motion vector per cluster
        unique_labels = set(labels) - {-1}
        for cluster_id in unique_labels:
            cluster_vecs = [motion_vecs[i] for i in range(len(valid_tracks)) if labels[i] == cluster_id]
            if cluster_vecs:
                avg_motion = torch.stack(cluster_vecs, dim=0).mean(dim=0).to(device)
                motion_vectors.append(avg_motion)

    return motion_labels, motion_vectors


def get_moving_points_2d(tracks_3d, valid_3d, motion_labels, camera_intrinsics, poses=None):
    """
    Project 3D moving points back into 2D per frame.
    Args:
        tracks_3d: (B, T, N, 3) 3D trajectories
        valid_3d: (B, T, N) mask of valid points
        motion_labels: (B, N) cluster labels (0=static, >0=moving cluster)
        camera_intrinsics: (3,3) intrinsic matrix
        poses: (T,4,4) optional extrinsics (for moving camera), None for static
    Returns:
        points_2d_per_frame: list of [ (K,2) numpy arrays per frame ]
    """
    B, T, N, _ = tracks_3d.shape
    assert B == 1, "Only single batch supported here"
    
    moving_mask = motion_labels[0] > 0
    points_2d_per_frame = []
    
    for t in range(T):
        valid = valid_3d[0, t] & moving_mask
        pts3d = tracks_3d[0, t, valid]
        
        if pts3d.numel() == 0:
            points_2d_per_frame.append(np.zeros((0,2)))
            continue
        
        # If camera static, just project with intrinsics
        if poses is None:
            fx, fy, cx, cy = camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0,2], camera_intrinsics[1,2]
            x, y, z = pts3d[:,0], pts3d[:,1], pts3d[:,2]
            u = (fx * x / z + cx).cpu().numpy()
            v = (fy * y / z + cy).cpu().numpy()
        else:
            pts_h = torch.cat([pts3d, torch.ones_like(pts3d[:,:1])], dim=-1).T  # 4xN
            cam_pts = (poses[t] @ pts_h).T  # N x 4
            x, y, z = cam_pts[:,0], cam_pts[:,1], cam_pts[:,2]
            fx, fy, cx, cy = camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0,2], camera_intrinsics[1,2]
            u = (fx * x / z + cx).cpu().numpy()
            v = (fy * y / z + cy).cpu().numpy()
        
        points_2d_per_frame.append(np.stack([u, v], axis=-1))
    
    return points_2d_per_frame


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

def save_moving_objects(tracks_3d, valid_3d, motion_labels, output_dir, camera_extrinsics):
    """
    Save moving clusters as object templates with centroid in world frame.

    Args:
        tracks_3d: (B, T, N, 3) - 3D trajectories (camera frame)
        valid_3d: (B, T, N) - validity mask
        motion_labels: (B, N) - cluster labels (0=static, >0=moving)
        output_dir: directory to save tensors
        camera_extrinsics: (4, 4) transform matrix (camera -> world)
    """
    os.makedirs(output_dir, exist_ok=True)
    B, T, N, _ = tracks_3d.shape

    def transform_to_world(points, extrinsics):
        ones = torch.ones((points.shape[0], 1), device=points.device)
        hom_points = torch.cat([points, ones], dim=1)  # (N, 4)
        world_points = (extrinsics.to(points.device) @ hom_points.T).T[:, :3]
        return world_points


    def opencv_to_nerf(points):
        R = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]], 
                        dtype=points.dtype, device=points.device)
        return points @ R.T



    for b in range(B):
        clusters = motion_labels[b].unique()
        cluster_sizes = {}

        for cluster_id in clusters:
            if cluster_id == 0:  # skip static
                continue

            # Collect all valid points for this cluster
            cluster_points = []
            for n in range(N):
                if motion_labels[b, n] == cluster_id:
                    valid_frames = valid_3d[b, :, n]
                    pts = tracks_3d[b, :, n][valid_frames]  # (F, 3)
                    if pts.numel() > 0:
                        cluster_points.append(pts)

            if cluster_points:
                cluster_points = torch.cat(cluster_points, dim=0)
                cluster_sizes[int(cluster_id.item())] = cluster_points.shape[0]
        
        if not cluster_sizes:
            print(f"⚠️ No valid points found for object cluster {cluster_id.item()} in batch {b}. Skipping...")
            continue

        # Pick largest cluster
        largest_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
        print(f"Largest cluster = {largest_cluster_id} with {cluster_sizes[largest_cluster_id]} points")

        # Re-collect its points
        cluster_points = []
        for n in range(N):
            if motion_labels[b, n] == largest_cluster_id:
                valid_frames = valid_3d[b, :, n]
                pts = tracks_3d[b, :, n][valid_frames]
                if pts.numel() > 0:
                    cluster_points.append(pts)

        cluster_points = torch.cat(cluster_points, dim=0)
        cluster_points = opencv_to_nerf(cluster_points)
        cluster_points_world = transform_to_world(cluster_points, camera_extrinsics)

        centroid_world = cluster_points_world.mean(dim=0)

        object_template = {
            "cluster_id": int(largest_cluster_id),
            "centroid_world": centroid_world.cpu().numpy(),
            "points_world": cluster_points_world.cpu().numpy(),
        }

        save_path = os.path.join(output_dir, f"obj_moved.pt")
        torch.save(object_template, save_path)
        print(f"✅ Saved largest object cluster {largest_cluster_id} with {cluster_points_world.shape[0]} points at {save_path}")



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

    # --- Intrinsics ---
    fx = metadata["fl_x"]
    fy = metadata["fl_y"]
    cx = metadata["cx"]
    cy = metadata["cy"]

    camera_intrinsics = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32)

    # --- Extrinsics (use first frame only since static) ---
    first_frame = metadata["frames"][0]
    camera_extrinsics = torch.tensor(first_frame["transform_matrix"], dtype=torch.float32)

    # --- Optionally get paths ---
    rgb_path = first_frame["file_path"]
    depth_path = first_frame.get("depth_file_path", None)

    print("Camera intrinsics:\n", camera_intrinsics)
    print("Camera extrinsics:\n", camera_extrinsics)
    print("RGB frame:", rgb_path)
    print("Depth frame:", depth_path)



    first_frame = video[0, 0]   # (3, H, W)
    last_frame  = video[0, -1]  # (3, H, W)

    # Convert to numpy for image_diff compatibility
    first_frame_np = first_frame.permute(1, 2, 0).cpu().numpy()
    last_frame_np  = last_frame.permute(1, 2, 0).cpu().numpy()

    coarse_masks, _ = image_diff(
        first_frame.unsqueeze(0), last_frame.unsqueeze(0), args.output_dir)
    change_mask = coarse_masks.sum(dim=0, keepdim=True).clamp(max=1)  # (1, H, W)

    print(f"Initial change mask shape: {change_mask.shape}")

    print("Loading CoTracker3 model...")
    torch.cuda.empty_cache() 
    
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(DEFAULT_DEVICE)
    
    available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    print(f"Available GPU memory: {available_memory / 1e9:.2f} GB")
    
    video = video.to(DEFAULT_DEVICE)
    depths = depths.to(DEFAULT_DEVICE)
    camera_intrinsics = camera_intrinsics.to(DEFAULT_DEVICE)

    # Track 2D trajectories with gradient disabled to save memory
    print("Computing 2D trajectories...")

    mask_np = change_mask.squeeze().cpu().numpy()

    # Compute bounding box
    ys, xs = np.where(mask_np > 0)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    bbox_mask = torch.zeros_like(change_mask)
    bbox_mask[:, :, ymin:ymax+1, xmin:xmax+1] = 1.0  # float32
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            segm_mask=bbox_mask.to(video.device).float()
        )

    print(f"2D tracks shape: {pred_tracks.shape}")
    print(f"Visibility shape: {pred_visibility.shape}")

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

    save_moving_objects(tracks_3d, valid_3d, motion_labels, args.output_dir, camera_extrinsics)
    
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
    

    # Create 3D motion mask (binary static vs moving)
    B, T, N = tracks_3d.shape[:3]
    motion_mask_3d = torch.zeros(T, H, W, dtype=torch.bool, device=tracks_3d.device)

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


