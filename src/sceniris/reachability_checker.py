# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from typing import Tuple
import os

try:
    from rm4d import ReachabilityMap4D, ReachabilityMap4DGPU
    import rm4d
    rm4d_path = rm4d.__path__[0]
except ImportError as e:
    print(f"Warning: Could not import rm4d: {e}")
    ReachabilityMap4D = None
    ReachabilityMap4DGPU = None
    rm4d_path = None


class ReachabilityChecker:
    """
    A reachability checker that uses rm4d to determine if poses are reachable.
    """

    # TODO: deal with the case that robot could be within a bound (e.g. mobility robot)
    
    def __init__(self, map_path: str | None = None, use_gpu: bool = True):
        """
        Initialize the reachability checker with a pre-computed reachability map.
        
        Args:
            map_path: Path to the .npy file containing the reachability map
            use_gpu: Whether to use GPU acceleration for reachability
        """
        if map_path is None:
            if rm4d_path is None:
                raise ValueError("rm4d_path is not set")
            map_path = os.path.join(
                os.path.dirname(rm4d_path),
                "experiment_scripts/data/rm4d_franka_joint_42_0.05/20000000/rmap.npy"
            )
            print (f"Using default rm4d map path: {map_path}")

        if ReachabilityMap4D is None:
            raise ImportError(
                "rm4d package not available. Please install it first.")
        
        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f"Reachability map not found at {map_path}")
        
        # Load the reachability map with GPU support if requested
        self.use_gpu = (use_gpu and torch.cuda.is_available() and
                        ReachabilityMap4DGPU is not None)
        
        if self.use_gpu:
            self.reachability_map = ReachabilityMap4DGPU.from_file(
                map_path, use_gpu=True)
            print(f"Loaded GPU-accelerated reachability map from "
                  f"{map_path}")
        else:
            self.reachability_map = ReachabilityMap4D.from_file(map_path)
            print(f"Loaded CPU reachability map from {map_path}")
            if use_gpu and not torch.cuda.is_available():
                print("Warning: GPU requested but CUDA not available. "
                      "Using CPU.")
            elif use_gpu and ReachabilityMap4DGPU is None:
                print("Warning: GPU requested but ReachabilityMap4DGPU not "
                      "available. Using CPU.")
        
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
        
        # Use GPU implementation if available
        if self.use_gpu and hasattr(self.reachability_map, 'are_poses_reachable_gpu'):
            try:
                result = self.reachability_map.are_poses_reachable_gpu(poses)
                # Convert to numpy if it's a tensor
                if isinstance(result, torch.Tensor):
                    return result.cpu().numpy()
                return result
            except Exception as e:
                print(f"GPU reachability check failed, falling back to CPU: {e}")
        
        # Use CPU implementation
        results = np.zeros(len(poses), dtype=bool)
        for i, pose in enumerate(poses):
            try:
                indices = self.reachability_map.get_indices_for_ee_pose(pose)
                results[i] = self.reachability_map.is_reachable(indices)
            except (IndexError, ValueError):
                results[i] = False
        return results
    
    def are_positions_reachable_with_orientation_threshold(
            self, positions: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Check if positions are reachable with orientation threshold.
        Automatically uses GPU or CPU based on the reachability map type.
        
        Args:
            positions: Array of 3D positions (N, 3)
            threshold: Minimum fraction of orientations that must be reachable
            
        Returns:
            np.ndarray: Boolean array indicating which positions are reachable
        """
        try:
            # Try the reachability map's implementation first (GPU or CPU)
            return self.reachability_map.are_positions_reachable_with_orientation_threshold(
                positions, threshold=threshold)
        except Exception as e:
            print(f"Reachability check failed, using manual fallback: {e}")
            return self._positions_reachable_manual(positions, threshold)
    
    def are_positions_reachable_with_orientation_threshold_gpu(
            self, positions: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        GPU-accelerated batch checking if positions are reachable with 
        orientation threshold.
        
        Args:
            positions: Array of 3D positions (N, 3)
            threshold: Minimum fraction of orientations that must be reachable
            
        Returns:
            np.ndarray: Boolean array indicating which positions are reachable
        """

        
        # Always use the reachability map's implementation (GPU or CPU)
        try:
            return self.reachability_map.are_positions_reachable_with_orientation_threshold(
                positions, threshold=threshold)
        except Exception as e:
            print(f"Reachability check failed, using manual fallback: {e}")
            return self._positions_reachable_manual(positions, threshold)
        
    
    def _positions_reachable_manual(self, positions: np.ndarray, 
                                   threshold: float) -> np.ndarray:
        """
        Manual CPU implementation for position reachability with orientation threshold.
        """
        results = np.zeros(len(positions), dtype=bool)
        n_orientations = self.reachability_map.n_bins_theta
        
        for i, pos in enumerate(positions):
            reachable_count = 0
            for theta_idx in range(n_orientations):
                # Create a pose with this position and orientation
                theta = (theta_idx / n_orientations) * 2 * np.pi
                pose = np.eye(4)
                pose[:3, 3] = pos
                pose[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
                
                if self.is_pose_reachable(pose):
                    reachable_count += 1
            
            results[i] = (reachable_count / n_orientations) >= threshold
        
        return results
    
    def _positions_reachable_manual_gpu(self, positions: np.ndarray,
                                       threshold: float) -> np.ndarray:
        """
        Manual GPU implementation for position reachability with orientation threshold.
        """
        if not isinstance(positions, torch.Tensor):
            positions = torch.from_numpy(positions).cuda().float()
        
        n_positions = len(positions)
        n_orientations = self.reachability_map.n_bins_theta
        results = torch.zeros(n_positions, dtype=torch.bool, device=positions.device)
        
        # Create all poses for all positions and orientations
        theta_values = torch.linspace(0, 2 * np.pi, n_orientations, 
                                     device=positions.device, dtype=torch.float32)
        
        for i, pos in enumerate(positions):
            poses = torch.zeros((n_orientations, 4, 4), device=positions.device)
            poses[:, 3, 3] = 1.0  # homogeneous coordinate
            poses[:, :3, 3] = pos  # position
            
            # Set rotation matrices for all orientations
            cos_theta = torch.cos(theta_values)
            sin_theta = torch.sin(theta_values)
            poses[:, 0, 0] = cos_theta
            poses[:, 0, 1] = -sin_theta
            poses[:, 1, 0] = sin_theta
            poses[:, 1, 1] = cos_theta
            poses[:, 2, 2] = 1.0
            
            # Check reachability for all orientations
            if hasattr(self.reachability_map, 'are_poses_reachable_gpu'):
                reachable = self.reachability_map.are_poses_reachable_gpu(poses)
            else:
                # Fall back to CPU for this position
                poses_cpu = poses.cpu().numpy()
                reachable = self.are_poses_reachable(poses_cpu)
                reachable = torch.from_numpy(reachable).to(positions.device)
            
            # Check if threshold is met
            reachable_fraction = reachable.float().mean()
            results[i] = reachable_fraction >= threshold
        
        return results.cpu().numpy()
    
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
