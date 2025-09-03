# Reachability Checker Test Suite - 16 Environments

This test suite provides comprehensive testing of the `ReachabilityChecker` class across 16 different environments and scenarios.

## Test Environments Overview

### Core Functionality Tests (1-6)
1. **Default Map Loading** - Tests basic initialization with default reachability map
2. **Single Pose Check** - Tests individual pose reachability checking
3. **Batch Pose Check** - Tests batch processing of multiple poses
4. **Reachability Scores** - Tests score calculation for pose reachability
5. **Pose Filtering** - Tests filtering of reachable vs unreachable poses
6. **Base Positions** - Tests base position calculation for given end-effector poses

### Edge Cases & Error Handling (7-8)
7. **Out of Bounds Handling** - Tests behavior with poses outside workspace
8. **Single vs Batch Consistency** - Ensures single and batch results match

### Performance Tests (9-10)
9. **Small Batch Performance** - Tests performance with 100 poses
10. **Large Batch Performance** - Tests performance with 1000 poses

### Robustness Tests (11-12)
11. **Pose Shape Handling** - Tests different input shape handling
12. **Memory Efficiency** - Tests memory usage with repeated operations

### Validation Tests (13-15)
13. **Error Handling** - Tests error handling for invalid inputs
14. **Repeated Queries** - Tests consistency of repeated identical queries
15. **Boundary Conditions** - Tests behavior at workspace boundaries

### Integration Test (16)
16. **Comprehensive Workflow** - Tests complete workflow from pose generation to base position calculation

## Running the Tests

### Prerequisites
- Python 3.7+
- NumPy
- Access to rm4d reachability maps
- sceniris package installed

### Execute Tests
```bash
cd /home/rgong/Desktop/sceniris
python tests/test_reachability_checker_16_environments.py
```

### Expected Output
```
============================================================
Running Reachability Checker Test Suite - 16 Environments
============================================================
Running Environment 1: Default map loading...
✓ Environment 1 passed

Running Environment 2: Single pose check...
✓ Environment 2 passed

[... test output ...]

============================================================
Test Results: 16/16 environments passed
Success Rate: 100.0%
🎉 All tests passed!
```

## Test Details

### Test Data Generation
The test suite uses a deterministic random seed (42) to generate test poses within typical workspace bounds:
- **Position bounds**: X, Y: [-0.5, 0.5], Z: [0.1, 0.8]
- **Rotation**: Random unit quaternions
- **Pose count**: Variable depending on test (10-1000 poses)

### Performance Benchmarks
- **Small batch (100 poses)**: Expected completion < 5 seconds
- **Large batch (1000 poses)**: Expected completion < 30 seconds

### Error Scenarios Tested
- Out-of-bounds poses
- Invalid input shapes
- Missing reachability maps
- Memory constraints

## Customization

### Modifying Test Poses
Edit the `generate_test_poses()` function to change:
- Number of test poses
- Workspace bounds
- Random seed for reproducibility

### Adding New Test Environments
1. Create a new test function following the pattern `test_environment_X_description()`
2. Add the function to the `test_functions` list in `run_all_tests()`
3. Update the test count and documentation

### Changing Performance Thresholds
Modify the time assertions in performance tests:
```python
assert duration < 5.0  # Small batch threshold
assert duration < 30.0  # Large batch threshold
```

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure rm4d is properly installed and accessible
- Check Python path includes both rm4d and sceniris

**Performance Issues**
- Large batch tests may take time on slower systems
- Consider reducing pose count for performance-limited environments

**Memory Errors**
- Reduce batch sizes in performance tests
- Ensure sufficient RAM for large pose arrays

### Debug Mode
Add debug prints to individual test functions:
```python
print(f"Debug: Processing {len(poses)} poses")
print(f"Debug: Results shape: {results.shape}")
```

## Test Coverage

This test suite covers:
- ✅ Basic functionality
- ✅ Edge cases
- ✅ Performance characteristics
- ✅ Error handling
- ✅ Memory efficiency
- ✅ Data consistency
- ✅ Integration workflows

Each test environment is designed to be independent and can run in isolation for debugging purposes.
