# Pose detection console guide

Run these from the repository root.

```powershell
# Live pose labels
python pose-detection/pose_cli.py --camera 0

# Make a calibration profile
python pose-detection/pose_cli.py --calibrate --camera 0

# Guided snapshot test
python pose-detection/pose_test.py --camera 0

# Inspect the pose backend
python pose-detection/ex-pc-detect.py --camera 0

# Score saved test results
python pose-detection/evaluate_head_pose.py <results-folder>
```

Use `--help` on any command for its options.
