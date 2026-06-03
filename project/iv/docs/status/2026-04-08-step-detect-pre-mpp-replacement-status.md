# Step Detect Pre-MPP Replacement Status

## Current Status

- Status: `completed`
- Last updated: `2026-04-08`

## Completed Work

- Replaced pre-MPP detection with the slope-median deviation implementation.
- Added configuration keys:
  - `slope_diff_min`
  - `min_region_ratio`
  - `min_interpolate_points`
- Extended `StepResult` and `analyze()` output with:
  - `step_deviation`
  - `current_span`
  - `voltage_span`
- Updated `diagnosis_config.yaml` and `docs/step_detect_pattern_detector.md`.
- Added/updated offline-safe tests for detector behavior and config selection.

## Verification

- `D:\software\miniforge\install\envs\python310\python.exe -m py_compile src\diagnosis\pattern_detectors\step_detect_pattern_detector.py`
- `D:\software\miniforge\install\envs\python310\python.exe -m py_compile tests\test_rule_predictor4_logic.py`
- `D:\software\miniforge\install\envs\python310\python.exe -m py_compile tests\test_predictor_config_selection.py`
- `D:\software\miniforge\install\envs\python310\python.exe -m unittest tests.test_rule_predictor4_logic tests.test_predictor_config_selection -v`

## Notes

- `pre_window_size` and `pre_slope_diff_ratio` remain in YAML for compatibility, but the new pre-MPP path now relies on `slope_diff_min`, `min_region_ratio`, and `min_interpolate_points`.
