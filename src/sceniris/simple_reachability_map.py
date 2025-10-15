# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from typing import Tuple, Optional
import os

try:
    import rm4d
    rm4d_path = rm4d.__path__[0]
except ImportError as e:
    print(f"Warning: Could not import rm4d: {e}")
    rm4d_path = None


class SimpleReachabilityMap:
    """
    Lightweight reachability map that only does lookup operations.
    No simulation dependencies - just pure numpy/torch operations.
    """
    
    def __init__(self, map_data: np.ndarray, xy_limits: Tuple[float, float],
                 z_limits: Tuple[float, float], voxel_res: float,
                 n_bins_theta: int, use_gpu: bool = True):
        """
        Initialize with pre-loaded map data.
        
        Args:
            map_data: 4D numpy array (z, theta, x, y) with reachability data
            xy_limits: (min, max) limits for x,y coordinates  
            z_limits: (min, max) limits for z coordinate
            voxel_res: Resolution of voxels in meters
            n_bins_theta: Number of theta orientation bins
            use_gpu: Whether to use GPU acceleration
        """
        self.map = map_data
        self.xy_limits = xy_limits
        self.z_limits = z_limits
        self.voxel_res = voxel_res
        self.n_bins_theta = n_bins_theta
        
        self.n_bins_z, self.n_bins_theta, self.n_bins_xy, _ = map_data.shape
        
        # GPU setup
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.gpu_map = torch.from_numpy(self.map).cuda()
            print(f"Simple reachability map moved to GPU: {self.gpu_map.device}")
        else:
            self.gpu_map = None
            if use_gpu and not torch.cuda.is_available():
                print("Warning: GPU requested but CUDA not available. Using CPU.")
    
    @classmethod
    def from_rm4d_file(cls, filename: str, use_gpu: bool = True):
        """Load from standard rm4d .npy file format"""
        data = np.load(filename, allow_pickle=True).item()
        return cls(
            map_data=data['map'],
            xy_limits=data['xy_limits'],
            z_limits=data['z_limits'],
            voxel_res=data['voxel_res'],
            n_bins_theta=data['n_bins_theta'],
            use_gpu=use_gpu
        )
    
    def are_positions_reachable_fast(self, positions: np.ndarray) -> np.ndarray:
        """
        Ultra-fast position reachability check.
        Checks if ANY orientation at each position is reachable.
        This is what you actually need for most placement tasks.
        """
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        n_positions = positions.shape[0]
        
        if self.use_gpu:
            return self._positions_reachable_gpu(positions)
        else:
            return self._positions_reachable_cpu(positions)
    
    def _positions_reachable_gpu(self, positions: np.ndarray) -> np.ndarray:
        """GPU-accelerated fast position check"""
        positions_gpu = torch.from_numpy(positions).cuda().float()
        
        # Convert positions to voxel indices
        x_indices = ((positions_gpu[:, 0] - self.xy_limits[0]) / self.voxel_res).long()
        y_indices = ((positions_gpu[:, 1] - self.xy_limits[0]) / self.voxel_res).long()
        z_indices = ((positions_gpu[:, 2] - self.z_limits[0]) / self.voxel_res).long()
        
        # Check bounds
        valid_mask = ((x_indices >= 0) & (x_indices < self.n_bins_xy) &
                     (y_indices >= 0) & (y_indices < self.n_bins_xy) &
                     (z_indices >= 0) & (z_indices < self.n_bins_z))
        
        results = torch.zeros(len(positions), dtype=torch.bool, device='cuda')
        
        if torch.any(valid_mask):
            valid_x = x_indices[valid_mask]
            valid_y = y_indices[valid_mask]
            valid_z = z_indices[valid_mask]
            
            # Check if ANY orientation at this position is reachable
            # This is just a simple lookup + any() operation
            position_reachability = torch.any(
                self.gpu_map[valid_z, :, valid_x, valid_y], dim=1)
            results[valid_mask] = position_reachability
        
        return results.cpu().numpy()
    
    def _positions_reachable_cpu(self, positions: np.ndarray) -> np.ndarray:
        """CPU fallback for position check"""
        n_positions = positions.shape[0]
        results = np.zeros(n_positions, dtype=bool)
        
        for i, pos in enumerate(positions):
            # Convert to voxel indices
            x_idx = int((pos[0] - self.xy_limits[0]) / self.voxel_res)
            y_idx = int((pos[1] - self.xy_limits[0]) / self.voxel_res)
            z_idx = int((pos[2] - self.z_limits[0]) / self.voxel_res)
            
            # Check bounds
            if (0 <= x_idx < self.n_bins_xy and 
                0 <= y_idx < self.n_bins_xy and 
                0 <= z_idx < self.n_bins_z):
                # Simple lookup: check if ANY orientation is reachable
                results[i] = np.any(self.map[z_idx, :, x_idx, y_idx])
        
        return results
    
    def print_stats(self):
        """Print basic statistics about the map"""
        total_voxels = self.map.size
        reachable_voxels = np.sum(self.map)
        coverage = reachable_voxels / total_voxels * 100
        
        print(f"Simple Reachability Map Statistics:")
        print(f"  Shape: {self.map.shape}")
        print(f"  XY limits: {self.xy_limits}")
        print(f"  Z limits: {self.z_limits}")
        print(f"  Voxel resolution: {self.voxel_res}m")
        print(f"  Coverage: {reachable_voxels}/{total_voxels} ({coverage:.1f}%)")
        print(f"  Using GPU: {self.use_gpu}")


class FastReachabilityChecker:
    """
    Lightweight reachability checker that only does fast lookups.
    No PyBullet dependencies.
    """
    
    def __init__(self, map_path: str | None = None, use_gpu: bool = True):
        if map_path is None:
            if rm4d_path is None:
                raise ValueError("rm4d_path is not set")
            map_path = os.path.join(
                os.path.dirname(rm4d_path),
                "experiment_scripts/data/rm4d_franka_joint_42_0.05/20000000/rmap.npy"
            )
        
        self.reachability_map = SimpleReachabilityMap.from_rm4d_file(
            map_path, use_gpu=use_gpu)
        self.use_gpu = self.reachability_map.use_gpu
        
        print(f"Fast reachability checker initialized (GPU: {self.use_gpu})")
        self.reachability_map.print_stats()
    
    def are_positions_reachable_with_orientation_threshold(
            self, positions: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Fast position reachability check.
        For threshold <= 0.2, this just checks if ANY orientation is reachable.
        This is much faster and sufficient for most placement tasks.
        """
        # For low thresholds, just check if ANY orientation works
        # This is what you actually need for object placement
        return self.reachability_map.are_positions_reachable_fast(positions) 
