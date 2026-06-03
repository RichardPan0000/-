# Step Detect Pre-MPP Replacement Plan

## Summary

- Replace the current pre-MPP step detection in `StepDetectPatternDetector` with the enhanced slope-median deviation algorithm.
- Keep the existing detector entrypoints and `RulePredictor4` integration unchanged.
- Extend step outputs so each detected step carries location, deviation, current span, and voltage span.

## Planned Changes

- Update `src/diagnosis/pattern_detectors/step_detect_pattern_detector.py`.
  - Add new pre-step configuration keys: `slope_diff_min`, `min_region_ratio`, `min_interpolate_points`.
  - Replace `_pre_mpp_detect(...)` with the interpolation-aware slope-median deviation implementation.
  - Extend `StepResult` to store `step_deviation`, `current_span`, and `voltage_span`.
  - Keep pre-step `step_degree` aligned with `current_span` so existing scoring logic remains compatible.
  - Preserve post-MPP logic, but enrich post step results with the same extra span/deviation fields.
  - Expand `analyze()` output under `steps` with pre/post and merged span/deviation arrays.

- Update `src/config/diagnosis_config.yaml`.
  - Add `slope_diff_min: 0.15`.
  - Add `min_region_ratio: 0.03`.
  - Add `min_interpolate_points: 200`.

- Update tests.
  - Add coverage for sparse-curve interpolation, region-length filtering, threshold filtering, and enriched step outputs.
  - Align config-selection assertions with the current YAML values and new keys.
  - Keep configuration-selection tests runnable offline by stubbing database/config loading dependencies.

- Update documentation.
  - Refresh `docs/step_detect_pattern_detector.md` so pre-MPP logic and output fields match the implementation.
  - Record execution status in `docs/status/2026-04-08-step-detect-pre-mpp-replacement-status.md`.

## Acceptance Checks

- `StepDetectPatternDetector` detects clear pre-MPP steps with the new algorithm.
- Sparse pre-MPP curves can still produce steps after interpolation.
- Steps below `pre_min_current_drop` / `pre_min_voltage_span` are filtered out.
- `steps` output includes `step_deviations`, `current_spans`, and `voltage_spans` in stable merged/pre/post forms.
- `tests.test_rule_predictor4_logic` and `tests.test_predictor_config_selection` pass.

