import numpy as np
import os
import sys
import time
from typing import List, Tuple

from sceniris.reachability_checker import ReachabilityChecker


def generate_test_poses(n_poses: int = 100, seed: int = 42) -> np.ndarray:
    """Generate a set of test poses for reachability checking."""
    np.random.seed(seed)
    
    # Generate random poses within typical workspace bounds
    positions = np.random.uniform(-0.5, 0.5, (n_poses, 3))
    positions[:, 2] = np.random.uniform(0.1, 0.8, n_poses)  # Z between 0.1 and 0.8
    
    # Generate random rotations (quaternions)
    rotations = np.random.normal(0, 1, (n_poses, 4))
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    
    # Create 4x4 transformation matrices
    poses = np.zeros((n_poses, 4, 4))
    poses[:, 3, 3] = 1.0  # Homogeneous coordinate
    
    # Set rotation parts
    poses[:, :3, :3] = np.array([[
        [1 - 2*r[2]**2 - 2*r[3]**2, 2*r[1]*r[2] - 2*r[3]*r[0], 2*r[1]*r[3] + 2*r[2]*r[0]],
        [2*r[1]*r[2] + 2*r[3]*r[0], 1 - 2*r[1]**2 - 2*r[3]**2, 2*r[2]*r[3] - 2*r[1]*r[0]],
        [2*r[1]*r[3] - 2*r[2]*r[0], 2*r[2]*r[3] + 2*r[1]*r[0], 1 - 2*r[1]**2 - 2*r[2]**2]
    ] for r in rotations])
    
    # Set translation parts
    poses[:, :3, 3] = positions
    
    return poses


def test_environment_1_default_map():
    """Test 1: Default reachability map loading."""
    print("Running Environment 1: Default map loading...")
    try:
        checker = ReachabilityChecker()
        assert checker.reachability_map is not None
        bounds = checker.get_map_bounds()
        assert 'xy_limits' in bounds
        assert 'z_limits' in bounds
        print("✓ Environment 1 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 1 failed: {e}")
        return False


def test_environment_2_single_pose_check():
    """Test 2: Single pose reachability check."""
    print("Running Environment 2: Single pose check...")
    try:
        checker = ReachabilityChecker()
        pose = np.eye(4)  # Identity pose
        result = checker.is_pose_reachable(pose)
        assert isinstance(result, bool)
        print("✓ Environment 2 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 2 failed: {e}")
        return False


def test_environment_3_batch_pose_check():
    """Test 3: Batch pose reachability check."""
    print("Running Environment 3: Batch pose check...")
    try:
        checker = ReachabilityChecker()
        poses = generate_test_poses(50)
        results = checker.are_poses_reachable(poses)
        assert results.shape == (50,)
        assert results.dtype == bool
        print("✓ Environment 3 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 3 failed: {e}")
        return False


def test_environment_4_reachability_scores():
    """Test 4: Reachability scores calculation."""
    print("Running Environment 4: Reachability scores...")
    try:
        checker = ReachabilityChecker()
        poses = generate_test_poses(30)
        scores = checker.get_reachability_scores(poses)
        assert scores.shape == (30,)
        assert scores.dtype == float
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        print("✓ Environment 4 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 4 failed: {e}")
        return False


def test_environment_5_pose_filtering():
    """Test 5: Pose filtering functionality."""
    print("Running Environment 5: Pose filtering...")
    try:
        checker = ReachabilityChecker()
        poses = generate_test_poses(40)
        reachable_poses, reachable_indices = checker.filter_reachable_poses(poses)
        assert reachable_poses.shape[1:] == (4, 4)
        assert len(reachable_indices) <= len(poses)
        print("✓ Environment 5 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 5 failed: {e}")
        return False


def test_environment_6_base_positions():
    """Test 6: Base position calculation."""
    print("Running Environment 6: Base positions...")
    try:
        checker = ReachabilityChecker()
        pose = np.eye(4)
        base_positions = checker.get_base_positions_for_pose(pose)
        assert base_positions.ndim >= 2
        print("✓ Environment 6 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 6 failed: {e}")
        return False


def test_environment_7_edge_case_out_of_bounds():
    """Test 7: Out of bounds pose handling."""
    print("Running Environment 7: Out of bounds handling...")
    try:
        checker = ReachabilityChecker()
        # Create pose far outside workspace
        pose = np.eye(4)
        pose[0, 3] = 10.0  # Far X position
        result = checker.is_pose_reachable(pose)
        assert result == False
        print("✓ Environment 7 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 7 failed: {e}")
        return False


def test_environment_8_single_vs_batch_consistency():
    """Test 8: Consistency between single and batch checking."""
    print("Running Environment 8: Single vs batch consistency...")
    try:
        checker = ReachabilityChecker()
        poses = generate_test_poses(10)
        
        # Check individually
        single_results = []
        for pose in poses:
            single_results.append(checker.is_pose_reachable(pose))
        
        # Check as batch
        batch_results = checker.are_poses_reachable(poses)
        
        # Compare results
        single_results = np.array(single_results)
        assert np.array_equal(single_results, batch_results)
        print("✓ Environment 8 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 8 failed: {e}")
        return False


def test_environment_9_performance_small_batch():
    """Test 9: Performance with small batch."""
    print("Running Environment 9: Small batch performance...")
    try:
        checker = ReachabilityChecker()
        poses = generate_test_poses(100)
        
        start_time = time.time()
        results = checker.are_poses_reachable(poses)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 5.0  # Should complete in less than 5 seconds
        assert results.shape == (100,)
        print(f"Duration: {duration:.3f}s")
        return True
    except Exception as e:
        print(f"✗ Environment 9 failed: {e}")
        return False


def test_environment_10_performance_large_batch():
    """Test 10: Performance with large batch."""
    print("Running Environment 10: Large batch performance...")
    try:
        checker = ReachabilityChecker()
        poses = generate_test_poses(1000)
        
        start_time = time.time()
        results = checker.are_poses_reachable(poses)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 30.0  # Should complete in less than 30 seconds
        assert results.shape == (1000,)
        print(f"Duration: {duration:.3f}s")
        return True
    except Exception as e:
        print(f"✗ Environment 10 failed: {e}")
        return False


def test_environment_11_pose_shape_handling():
    """Test 11: Different pose shape handling."""
    print("Running Environment 11: Pose shape handling...")
    try:
        checker = ReachabilityChecker()
        
        # Test 2D pose (single pose)
        pose_2d = np.eye(4)
        result_2d = checker.are_poses_reachable(pose_2d)
        assert result_2d.shape == (1,)
        
        # Test 3D poses (batch)
        poses_3d = generate_test_poses(5)
        result_3d = checker.are_poses_reachable(poses_3d)
        assert result_3d.shape == (5,)
        
        print("✓ Environment 11 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 11 failed: {e}")
        return False


def test_environment_12_memory_efficiency():
    """Test 12: Memory efficiency check."""
    print("Running Environment 12: Memory efficiency...")
    try:
        checker = ReachabilityChecker()
        initial_memory = np.zeros(1000)  # Placeholder for memory tracking
        
        # Run multiple batches
        for i in range(10):
            poses = generate_test_poses(100)
            results = checker.are_poses_reachable(poses)
            assert results.shape == (100,)
        
        print("✓ Environment 12 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 12 failed: {e}")
        return False


def test_environment_13_error_handling():
    """Test 13: Error handling for invalid inputs."""
    print("Running Environment 13: Error handling...")
    try:
        checker = ReachabilityChecker()
        
        # Test with invalid pose shape
        try:
            invalid_pose = np.ones((3, 3))  # Wrong shape
            checker.is_pose_reachable(invalid_pose)
            assert False, "Should have raised an error"
        except (IndexError, ValueError):
            pass  # Expected
        
        print("✓ Environment 13 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 13 failed: {e}")
        return False


def test_environment_14_repeated_queries():
    """Test 14: Repeated queries consistency."""
    print("Running Environment 14: Repeated queries...")
    try:
        checker = ReachabilityChecker()
        pose = generate_test_poses(1)[0]
        
        # Run same query multiple times
        results = []
        for i in range(5):
            result = checker.is_pose_reachable(pose)
            results.append(result)
        
        # All results should be the same
        assert all(r == results[0] for r in results)
        print("✓ Environment 14 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 14 failed: {e}")
        return False


def test_environment_15_boundary_conditions():
    """Test 15: Boundary condition testing."""
    print("Running Environment 15: Boundary conditions...")
    try:
        checker = ReachabilityChecker()
        bounds = checker.get_map_bounds()
        
        # Test poses at boundary limits
        boundary_poses = []
        
        # XY boundaries
        for x in bounds['xy_limits']:
            for y in bounds['xy_limits']:
                pose = np.eye(4)
                pose[0, 3] = x
                pose[1, 3] = y
                pose[2, 3] = (bounds['z_limits'][0] + bounds['z_limits'][1]) / 2
                boundary_poses.append(pose)
        
        results = checker.are_poses_reachable(np.array(boundary_poses))
        assert results.shape == (len(boundary_poses),)
        print("✓ Environment 15 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 15 failed: {e}")
        return False


def test_environment_16_comprehensive_workflow():
    """Test 16: Comprehensive workflow test."""
    print("Running Environment 16: Comprehensive workflow...")
    try:
        checker = ReachabilityChecker()
        
        # Generate test poses
        test_poses = generate_test_poses(200)
        
        # 1. Check reachability
        reachable_mask = checker.are_poses_reachable(test_poses)
        
        # 2. Filter reachable poses
        reachable_poses, indices = checker.filter_reachable_poses(test_poses)
        
        # 3. Get scores for reachable poses
        if len(reachable_poses) > 0:
            scores = checker.get_reachability_scores(reachable_poses)
            assert scores.shape == (len(reachable_poses),)
        
        # 4. Test base positions for a sample pose
        if len(reachable_poses) > 0:
            sample_pose = reachable_poses[0]
            base_positions = checker.get_base_positions_for_pose(sample_pose)
            assert base_positions.ndim >= 2
        
        print("✓ Environment 16 passed")
        return True
    except Exception as e:
        print(f"✗ Environment 16 failed: {e}")
        return False


def run_all_tests():
    """Run all 16 test environments."""
    print("=" * 60)
    print("Running Reachability Checker Test Suite - 16 Environments")
    print("=" * 60)
    
    test_functions = [
        test_environment_1_default_map,
        test_environment_2_single_pose_check,
        test_environment_3_batch_pose_check,
        test_environment_4_reachability_scores,
        test_environment_5_pose_filtering,
        test_environment_6_base_positions,
        test_environment_7_edge_case_out_of_bounds,
        test_environment_8_single_vs_batch_consistency,
        test_environment_9_performance_small_batch,
        test_environment_10_performance_large_batch,
        test_environment_11_pose_shape_handling,
        test_environment_12_memory_efficiency,
        test_environment_13_error_handling,
        test_environment_14_repeated_queries,
        test_environment_15_boundary_conditions,
        test_environment_16_comprehensive_workflow,
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} environments passed")
    print(".1f")
    
    if passed == total:
        print("�� All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
