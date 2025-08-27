import numpy as np
from typing import Tuple
import sys
import os

# Add rm4d to path
sys.path.append('/home/rgong/Desktop/rm4d')

try:
    from rm4d import ReachabilityMap4D
except ImportError as e:
    print(f"Warning: Could not import rm4d: {e}")
    ReachabilityMap4D = None


class ReachabilityChecker:
    """
    A reachability checker that uses rm4d to determine if poses are reachable.
    """
    
    def __init__(self, map_path: str = None):
        """
        Initialize the reachability checker with a pre-computed reachability map.
        
        Args:
            map_path: Path to the .npy file containing the reachability map
        """
        if map_path is None:
            map_path = ("/home/rgong/Desktop/rm4d/experiment_scripts/data/"
                        "rm4d_franka_joint_42_0.025/20000000/rmap.npy")
        
        if ReachabilityMap4D is None:
            raise ImportError(
                "rm4d package not available. Please install it first.")
        
        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f"Reachability map not found at {map_path}")
        
        # Load the reachability map
        self.reachability_map = ReachabilityMap4D.from_file(map_path)
        print(f"Loaded reachability map from {map_path}")
        
        self.reachability_map.print_structure()
        
        # Store map bounds for compatibility
        self.map_bounds = {
            'xy_limits': self.reachability_map.xy_limits,
            'z_limits': self.reachability_map.z_limits,
            'theta_limits': self.reachability_map.theta_limits,
            'voxel_res': self.reachability_map.voxel_res,
            'n_bins_theta': self.reachability_map.n_bins_theta,
        }
    
    def is_pose_reachable(self, pose: np.ndarray) -> bool:
        """
        Check if a single pose is reachable.
        
        Args:
            pose: 4x4 transformation matrix representing the end-effector pose
            
        Returns:
            bool: True if the pose is reachable, False otherwise
        """
        try:
            # Use proper rm4d reachability checking
            indices = self.reachability_map.get_indices_for_ee_pose(pose)
            return self.reachability_map.is_reachable(indices)
        except (IndexError, ValueError):
            # Pose is outside the map bounds
            return False
    
    def are_poses_reachable(self, poses: np.ndarray) -> np.ndarray:
        """
        Check if multiple poses are reachable.
        
        Args:
            poses: Array of 4x4 transformation matrices (N, 4, 4)
            
        Returns:
            np.ndarray: Boolean array indicating which poses are reachable (N,)
        """
        if len(poses.shape) == 2:
            poses = poses[np.newaxis, ...]
        
        # Use CPU implementation for reachability checking
        results = np.zeros(len(poses), dtype=bool)
        for i, pose in enumerate(poses):
            try:
                indices = self.reachability_map.get_indices_for_ee_pose(pose)
                results[i] = self.reachability_map.is_reachable(indices)
            except (IndexError, ValueError):
                results[i] = False
        return results
    
    def get_reachability_scores(self, poses: np.ndarray) -> np.ndarray:
        """
        Get reachability scores for multiple poses.
        
        Args:
            poses: Array of 4x4 transformation matrices (N, 4, 4)
            
        Returns:
            np.ndarray: Array of reachability scores (N,)
        """
        if len(poses.shape) == 2:
            poses = poses[np.newaxis, ...]
        
        scores = np.zeros(len(poses), dtype=float)
        
        for i, pose in enumerate(poses):
            try:
                indices = self.reachability_map.get_indices_for_ee_pose(pose)
                scores[i] = float(self.reachability_map.is_reachable(indices))
            except (IndexError, ValueError):
                scores[i] = 0.0
        
        return scores
    
    def get_base_positions_for_pose(self, pose: np.ndarray,
                                   as_3d: bool = False) -> np.ndarray:
        """
        Get possible base positions for a given end-effector pose.
        
        Args:
            pose: 4x4 transformation matrix representing the end-effector pose
            as_3d: If True, return 3D positions with z=0, otherwise 2D
                   positions
            
        Returns:
            np.ndarray: Array of base positions with reachability scores
        """
        try:
            return self.reachability_map.get_base_positions(pose, as_3d=as_3d)
        except (IndexError, ValueError):
            # Pose is outside the map bounds
            return np.empty((0, 4 if as_3d else 3))
    
    def filter_reachable_poses(self, poses: np.ndarray) -> Tuple[np.ndarray,
                                                                np.ndarray]:
        """
        Filter poses to only include reachable ones.
        
        Args:
            poses: Array of 4x4 transformation matrices (N, 4, 4)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (reachable_poses, reachable_indices)
        """
        reachable_mask = self.are_poses_reachable(poses)
        reachable_indices = np.where(reachable_mask)[0]
        reachable_poses = poses[reachable_indices]
        
        return reachable_poses, reachable_indices
    
    def get_map_bounds(self) -> dict:
        """
        Get the bounds of the reachability map.
        
        Returns:
            dict: Dictionary containing xy_limits, z_limits, and theta_limits
        """
        return self.map_bounds
