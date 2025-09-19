"""
Joint Model Implementations for 4D RANSAC Pipeline

This module defines the geometric models for different joint types (hinge, slider, etc.)
and provides methods for:
- Fitting joint parameters from trajectory samples
- Calculating how well trajectories match joint models
- Refining joint parameters using all inlier trajectories
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from data_structures import (
    Trajectory3D, Point3D, HingeParameters, SliderParameters,
    JointType, ModelFitResult
)


class JointModelBase(ABC):
    """
    Abstract base class for all joint models.
    
    Each joint type (hinge, slider, etc.) implements this interface
    to provide RANSAC with consistent fitting and error calculation methods.
    """
    
    @abstractmethod
    def get_joint_type(self) -> JointType:
        """Return the type of joint this model represents."""
        pass
    
    @abstractmethod
    def minimal_sample_size(self) -> int:
        """Return minimum number of trajectory points needed to fit this model."""
        pass
    
    @abstractmethod
    def fit_from_sample(self, trajectory_sample: List[Trajectory3D]) -> Optional[object]:
        """
        Fit joint parameters from a minimal sample of trajectories.
        
        Args:
            trajectory_sample: Minimal set of trajectories to fit model
            
        Returns:
            Joint parameters object (HingeParameters, SliderParameters, etc.)
            or None if fitting failed
        """
        pass
    
    @abstractmethod
    def calculate_trajectory_error(self, trajectory: Trajectory3D, joint_params: object) -> float:
        """
        Calculate how well a trajectory fits the joint model.
        
        Args:
            trajectory: 3D trajectory to evaluate
            joint_params: Joint parameters from fit_from_sample
            
        Returns:
            RMS error in meters (lower is better fit)
        """
        pass
    
    @abstractmethod
    def refine_parameters(self, 
                         inlier_trajectories: List[Trajectory3D], 
                         initial_params: object) -> object:
        """
        Refine joint parameters using all inlier trajectories.
        
        Args:
            inlier_trajectories: All trajectories that fit the model
            initial_params: Initial parameters from minimal sample
            
        Returns:
            Refined joint parameters
        """
        pass
    
    def validate_sample(self, trajectory_sample: List[Trajectory3D]) -> bool:
        """
        Check if trajectory sample is valid for fitting.
        
        Args:
            trajectory_sample: Trajectories to validate
            
        Returns:
            True if sample can be used for fitting
        """
        if len(trajectory_sample) < self.minimal_sample_size():
            return False
        
        # Check that trajectories have sufficient length and temporal overlap
        for traj in trajectory_sample:
            if len(traj.points) < 2:
                return False
        
        return True


class HingeJointModel(JointModelBase):
    """
    Model for hinge/revolute joints.
    
    A hinge joint rotates around a fixed axis. Points on the moving part
    trace circular arcs around this axis.
    """
    
    def get_joint_type(self) -> JointType:
        return JointType.HINGE
    
    def minimal_sample_size(self) -> int:
        # Need at least 2 trajectories with 2 time points each
        # This gives us 4 point correspondences to solve for rotation + translation
        return 2
    
    def fit_from_sample(self, trajectory_sample: List[Trajectory3D]) -> Optional[HingeParameters]:
        """
        Fit hinge parameters from trajectory sample.
        
        Algorithm:
        1. Extract point correspondences between two time steps
        2. Use Kabsch algorithm or similar to find rotation matrix R
        3. Extract rotation axis from R
        4. Find pivot point on the axis
        """
        if not self.validate_sample(trajectory_sample):
            return None
        
        try:
            # Get point correspondences from trajectories
            points_t1, points_t2 = self._extract_point_correspondences(trajectory_sample)
            
            if len(points_t1) < 3:  # Need at least 3 point pairs
                return None
            
            # Estimate rotation and translation using procrustes analysis
            R, t, pivot, axis = self._estimate_rigid_transform(points_t1, points_t2)
            
            if axis is None:
                return None
            
            return HingeParameters(axis=axis, pivot=pivot)
            
        except Exception as e:
            # Fitting failed - this is normal in RANSAC
            return None
    
    def _extract_point_correspondences(self, 
                                     trajectories: List[Trajectory3D]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract corresponding 3D points from different time steps."""
        points_t1 = []
        points_t2 = []
        
        for traj in trajectories:
            if len(traj.points) < 2:
                continue
            
            # Take first and last points for maximum motion
            p1 = traj.points[0]
            p2 = traj.points[-1]
            
            points_t1.append([p1.x, p1.y, p1.z])
            points_t2.append([p2.x, p2.y, p2.z])
        
        return np.array(points_t1), np.array(points_t2)
    
    def _estimate_rigid_transform(self, 
                                points_t1: np.ndarray, 
                                points_t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate rigid transformation between two point sets.
        
        Returns:
            R: Rotation matrix (3x3)
            t: Translation vector (3,)
            pivot: Pivot point on rotation axis (3,)
            axis: Rotation axis unit vector (3,)
        """
        # Center the point sets
        centroid_t1 = np.mean(points_t1, axis=0)
        centroid_t2 = np.mean(points_t2, axis=0)
        
        centered_t1 = points_t1 - centroid_t1
        centered_t2 = points_t2 - centroid_t2
        
        # Use Kabsch algorithm to find optimal rotation
        H = centered_t1.T @ centered_t2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = centroid_t2 - R @ centroid_t1
        
        # Extract rotation axis and angle
        axis, angle = self._rotation_matrix_to_axis_angle(R)
        
        # Find pivot point (point on axis closest to centroids)
        pivot = self._find_pivot_point(centroid_t1, centroid_t2, axis, R)
        
        return R, t, pivot, axis
    
    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert rotation matrix to axis-angle representation."""
        # Use scipy for robust conversion
        rot = Rotation.from_matrix(R)
        axis_angle = rot.as_rotvec()
        
        if np.linalg.norm(axis_angle) < 1e-6:
            # No significant rotation - return arbitrary axis
            return np.array([0, 0, 1]), 0.0
        
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle
        
        return axis, angle
    
    def _find_pivot_point(self, 
                         p1: np.ndarray, 
                         p2: np.ndarray, 
                         axis: np.ndarray, 
                         R: np.ndarray) -> np.ndarray:
        """
        Find pivot point on rotation axis - simplified approach.
        
        For hinge motion, the pivot is approximately the midpoint between
        the centroids of the two point sets, projected onto the rotation axis.
        """
        # Simple approach: use midpoint between centroids
        midpoint = (p1 + p2) / 2
        
        # Project midpoint onto rotation axis
        # Find point on axis closest to midpoint
        # If axis passes through origin: pivot = (midpoint · axis) * axis
        
        # More general: assume axis passes through the centroid of all points
        axis_origin = midpoint  # Simple assumption
        
        # The pivot is just the centroid for this simplified version
        return axis_origin
    
    def _rotate_point_around_axis(self, 
                                 point: np.ndarray, 
                                 pivot: np.ndarray, 
                                 axis: np.ndarray, 
                                 angle: float) -> np.ndarray:
        """Rotate a point around an axis by given angle."""
        # Translate so pivot is at origin
        translated = point - pivot
        
        # Create rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Rotate and translate back
        rotated = R @ translated + pivot
        return rotated
    
    def calculate_trajectory_error(self, trajectory: Trajectory3D, joint_params: HingeParameters) -> float:
        """
        Calculate RMS error for trajectory fitting hinge model.
        
        For each point in trajectory, find the rotation angle that best explains
        its position, then calculate error.
        """
        if len(trajectory.points) < 2:
            return float('inf')
        
        points = trajectory.get_all_positions()
        
        # Calculate distance from each point to the rotation axis
        distances_to_axis = []
        
        for point in points:
            # Distance from point to line (axis through pivot)
            # Formula: ||(point - pivot) - ((point - pivot) · axis) * axis||
            
            point_to_pivot = point - joint_params.pivot
            projection_on_axis = np.dot(point_to_pivot, joint_params.axis) * joint_params.axis
            perpendicular_component = point_to_pivot - projection_on_axis
            distance_to_axis = np.linalg.norm(perpendicular_component)
            
            distances_to_axis.append(distance_to_axis)
        
        # For a perfect hinge, all distances should be the same
        distances_to_axis = np.array(distances_to_axis)
        
        # Error is the standard deviation of distances (should be close to 0)
        if len(distances_to_axis) < 2:
            return float('inf')
        
        error = np.std(distances_to_axis)
        
        # Also add a small penalty for very small movements (static points)
        total_movement = np.linalg.norm(points[-1] - points[0])
        if total_movement < 0.01:  # Less than 1cm movement
            error += 0.1  # Add penalty
        
        return error
    
    def _estimate_rotation_angle(self, 
                               point1: np.ndarray, 
                               point2: np.ndarray, 
                               axis: np.ndarray, 
                               pivot: np.ndarray) -> float:
        """Estimate rotation angle between two points around an axis."""
        # Project points onto plane perpendicular to axis
        v1 = point1 - pivot
        v2 = point2 - pivot
        
        # Remove component along axis
        v1_proj = v1 - np.dot(v1, axis) * axis
        v2_proj = v2 - np.dot(v2, axis) * axis
        
        # Calculate angle between projected vectors
        cos_angle = np.dot(v1_proj, v2_proj) / (np.linalg.norm(v1_proj) * np.linalg.norm(v2_proj) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        angle = np.arccos(cos_angle)
        
        # Determine sign using cross product
        cross = np.cross(v1_proj, v2_proj)
        if np.dot(cross, axis) < 0:
            angle = -angle
        
        return angle
    
    def refine_parameters(self, 
                         inlier_trajectories: List[Trajectory3D], 
                         initial_params: HingeParameters) -> HingeParameters:
        """Refine hinge parameters using all inlier trajectories."""
        if not inlier_trajectories:
            return initial_params
        
        # Extract all point correspondences from inlier trajectories
        all_points_t1 = []
        all_points_t2 = []
        
        for traj in inlier_trajectories:
            if len(traj.points) >= 2:
                p1 = traj.points[0]
                p2 = traj.points[-1]
                all_points_t1.append([p1.x, p1.y, p1.z])
                all_points_t2.append([p2.x, p2.y, p2.z])
        
        if len(all_points_t1) < 3:
            return initial_params
        
        # Re-estimate parameters with more data
        points_t1 = np.array(all_points_t1)
        points_t2 = np.array(all_points_t2)
        
        try:
            R, t, pivot, axis = self._estimate_rigid_transform(points_t1, points_t2)
            return HingeParameters(axis=axis, pivot=pivot)
        except:
            return initial_params


class SliderJointModel(JointModelBase):
    """
    Model for slider/prismatic joints.
    
    A slider joint translates along a fixed direction. All points on the
    moving part move parallel to each other along this direction.
    """
    
    def get_joint_type(self) -> JointType:
        return JointType.SLIDER
    
    def minimal_sample_size(self) -> int:
        # Need at least 1 trajectory with 2 time points
        return 1
    
    def fit_from_sample(self, trajectory_sample: List[Trajectory3D]) -> Optional[SliderParameters]:
        if not self.validate_sample(trajectory_sample):
            return None

        displacements = []
        start_points = []

        for traj in trajectory_sample:
            pts = traj.get_all_positions()
            if len(pts) < 2:
                continue
            start_points.append(pts[0])
            displacements.extend(np.diff(pts, axis=0))  # use all steps

        if len(displacements) == 0:
            return None

        displacements = np.vstack(displacements)
        # PCA on displacements
        U, S, Vt = np.linalg.svd(displacements)
        direction = Vt[0] / np.linalg.norm(Vt[0])

        reference_point = np.mean(start_points, axis=0)

        return SliderParameters(direction=direction, reference_point=reference_point)

    
    def calculate_trajectory_error(self, trajectory: Trajectory3D, joint_params: SliderParameters) -> float:
        """
        Calculate error for trajectory fitting slider model.
        
        For slider motion, check that all displacements are parallel to slide direction
        and that the trajectory shows actual translation movement.
        """
        if len(trajectory.points) < 2:
            return float('inf')
        
        points = trajectory.get_all_positions()
        
        # Calculate frame-to-frame displacements
        displacements = np.diff(points, axis=0)
        
        if len(displacements) == 0:
            return float('inf')
        
        errors = []
        
        for displacement in displacements:
            displacement_magnitude = np.linalg.norm(displacement)
            
            # Skip very small displacements (noise)
            if displacement_magnitude < 1e-6:
                continue
            
            # Project displacement onto slide direction
            projected_length = np.dot(displacement, joint_params.direction)
            predicted_displacement = projected_length * joint_params.direction
            
            # Error is the perpendicular component (deviation from pure translation)
            error_vector = displacement - predicted_displacement
            error = np.linalg.norm(error_vector)
            errors.append(error)
        
        if len(errors) == 0:
            return float('inf')
        
        # Add penalty for trajectories that don't show significant translation
        total_displacement = np.linalg.norm(points[-1] - points[0])
        if total_displacement < 0.02:  # Less than 2cm total movement
            return float('inf')  # Not a good slider candidate
        
        return np.mean(errors)
    
    def refine_parameters(self, 
                         inlier_trajectories: List[Trajectory3D], 
                         initial_params: SliderParameters) -> SliderParameters:
        """Refine slider parameters using all inlier trajectories."""
        if not inlier_trajectories:
            return initial_params
        
        # Collect all displacement vectors
        all_displacements = []
        
        for traj in inlier_trajectories:
            if len(traj.points) >= 2:
                p1 = traj.points[0]
                p2 = traj.points[-1]
                
                displacement = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
                
                if np.linalg.norm(displacement) > 1e-6:
                    all_displacements.append(displacement)
        
        if len(all_displacements) == 0:
            return initial_params
        
        # Refine direction using all displacements
        avg_displacement = np.mean(all_displacements, axis=0)
        refined_direction = avg_displacement / np.linalg.norm(avg_displacement)
        
        return SliderParameters(
            direction=refined_direction,
            reference_point=initial_params.reference_point,
            translation_min=initial_params.translation_min,
            translation_max=initial_params.translation_max
        )


# Factory function for creating joint models
def create_joint_models() -> Dict[JointType, JointModelBase]:
    """Create instances of all available joint models."""
    return {
        JointType.HINGE: HingeJointModel(),
        JointType.SLIDER: SliderJointModel(),
    }


def get_joint_model(joint_type: JointType) -> Optional[JointModelBase]:
    """Get a specific joint model by type."""
    models = create_joint_models()
    return models.get(joint_type)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F

    # --- Helper to make synthetic 3D trajectories ---
    def make_hinge_trajectories(axis, pivot, angles, n_points=5, noise=0.01):
        """Generate synthetic hinge trajectories."""
        trajectories = []
        axis = axis / np.linalg.norm(axis)

        # random initial points not on axis
        for i in range(n_points):
            base_point = pivot + np.random.randn(3)  # random offset
            points = []
            for angle in angles:
                # Rodrigues rotation formula
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                pt = R @ (base_point - pivot) + pivot
                pt += noise * np.random.randn(3)
                points.append(Point3D(frame=len(points), x=pt[0], y=pt[1], z=pt[2]))
            trajectories.append(Trajectory3D(track_id=i, points=points))
        return trajectories

    def make_slider_trajectories(direction, start_point, steps, n_points=5, noise=0.01):
        """Generate synthetic slider trajectories."""
        trajectories = []
        direction = direction / np.linalg.norm(direction)
        for i in range(n_points):
            offset = np.random.randn(3) * 0.5 # random offset perpendicular
            base_point = start_point + offset
            points = []
            for step in steps:
                pt = base_point + step * direction
                pt += noise * np.random.randn(3)
                points.append(Point3D(frame=len(points), x=pt[0], y=pt[1], z=pt[2]))
            trajectories.append(Trajectory3D(track_id=i, points=points))
        return trajectories


    # --- Test Hinge Model ---
    hinge_model = HingeJointModel()
    true_axis = np.array([0, 0, 1])
    true_pivot = np.array([0, 0, 0])
    angles = np.linspace(0, np.pi/4, 10)
    hinge_trajs = make_hinge_trajectories(true_axis, true_pivot, angles)

    hinge_params = hinge_model.fit_from_sample(hinge_trajs)
    print("\n=== Hinge Joint Test ===")
    print("True axis:", true_axis, " | Estimated axis:", hinge_params.axis)
    print("True pivot:", true_pivot, " | Estimated pivot:", hinge_params.pivot)

    errors = [hinge_model.calculate_trajectory_error(t, hinge_params) for t in hinge_trajs]
    print("Mean trajectory error:", np.mean(errors))

    # --- Visualization for Hinge ---
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    for traj in hinge_trajs:
        pts = traj.get_all_positions()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', alpha=0.7)

    # Plot estimated axis as a line through pivot
    pivot = hinge_params.pivot
    axis = hinge_params.axis
    line_pts = np.array([pivot - 2 * axis, pivot + 2 * axis])
    ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 'r--', lw=2, label="Estimated axis")
    ax.set_title("Hinge trajectories & estimated axis")
    ax.legend()

    # --- Test Slider Model ---
    slider_model = SliderJointModel()
    true_dir = np.array([1, 0, 0])
    start = np.array([0, 0, 0])
    steps = np.linspace(0, 1, 10)
    slider_trajs = make_slider_trajectories(true_dir, start, steps)

    slider_params = slider_model.fit_from_sample(slider_trajs)
    print("\n=== Slider Joint Test ===")
    print("True direction:", true_dir, " | Estimated direction:", slider_params.direction)
    print("Reference point:", slider_params.reference_point)

    errors = [slider_model.calculate_trajectory_error(t, slider_params) for t in slider_trajs]
    print("Mean trajectory error:", np.mean(errors))

    # --- Visualization for Slider ---
    ax2 = fig.add_subplot(122, projection='3d')
    for traj in slider_trajs:
        pts = traj.get_all_positions()
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', alpha=0.7)

    # Plot estimated slide direction as a line from reference point
    ref_pt = slider_params.reference_point
    dir_vec = slider_params.direction
    line_pts = np.array([ref_pt - 2 * dir_vec, ref_pt + 2 * dir_vec])
    ax2.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 'g--', lw=2, label="Estimated direction")
    ax2.set_title("Slider trajectories & estimated direction")
    ax2.legend()

    plt.tight_layout()
    plt.show()