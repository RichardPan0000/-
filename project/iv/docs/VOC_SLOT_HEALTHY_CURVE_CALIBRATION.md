# Voc Slot Healthy Calibration

This document describes the current group-level `Voc_total` slot fitting used in the `healthy_curve + RulePredictor4` flow, and how it is applied to calibrate the healthy IV curve voltage axis.

## 1. Background

The healthy-curve flow first maps a datasheet / STC healthy IV curve to runtime conditions using estimated irradiance and temperature.

The voltage mapping is effectively a constant shift:

```text
V_new = V_stc - Rs * DeltaI + beta_abs * DeltaT
```

`DeltaI` and `DeltaT` are constants for the whole curve in this step, so the voltage axis is shifted by an approximately constant amount. If the temperature estimate is off, the generated healthy `Voc` can be shifted away from the normal strings in the same group.

Most strings in a group are usually healthy. That allows us to fit the observed group `Voc_total` distribution to an arithmetic slot model and use that model to calibrate the healthy reference once per group.

Implementation entry points:

- `src/diagnosis/voc_slot_calibration.py`
- `src/diagnosis/flows/healthy_curve_flow.py`
- `src/diagnosis/predictors/rule_predictor4.py`

## 2. Slot Model

The current model is:

```text
slot(k) = a + k * d
Voc_total = a + k * d
```

Where:

- `k`: integer slot index, usually aligned to `n_components`
- `d`: slot spacing, approximately the runtime single-module `Voc`
- `a`: slot phase / intercept

For a candidate `a, d`, each measured total Voc is projected to the nearest slot:

```text
k_i = round((Voc_i - a) / d)
ref_i = a + k_i * d
residual_i = abs(Voc_i - ref_i)
```

The best model is the one with the lowest clipped residual sum.

## 3. Fit Input Filtering

The helper fits on:

```text
VocSlotObservation(
    string_id,
    voc_total,
    isc,
    n_components
)
```

Strings with invalid `Voc_total` are ignored. Strings with too-low `Isc` are excluded from the fit so near-open-circuit or badly degraded strings do not drag the healthy slot model away from the normal population.

Default threshold:

```yaml
voc_ap_min_fit_isc_abs: 0.2
```

Excluded strings are still diagnosed later; they are only removed from slot fitting.

## 4. Search Space and Scoring

The `d` search range is centered around the original healthy single-module `Voc`:

```text
d_min = voc_ref_unit - voc_ap_ref_window
d_max = voc_ref_unit + voc_ap_ref_window
```

Current defaults:

```yaml
voc_ap_ref_window: 5.0
voc_ap_d_step: 0.1
voc_ap_a_step: 0.1
voc_ap_tol: 10.0
voc_ap_clip_factor: 2.0
voc_ap_anchor_values: [0.0, 0.001, 0.005, 0.03]
```

`a` is searched in `[0, d)`.

Anchor values are added during scoring so the fitted model stays aligned to a reasonable phase near the origin:

```text
score = sum(min(residual_i, voc_ap_tol * voc_ap_clip_factor))
```

The clipped score reduces the influence of extreme outliers.

## 5. Healthy Curve Calibration

Once the slot model is fitted, the helper computes the original healthy total Voc:

```text
voc_ref_total_raw = voc_ref_unit * n_components
```

With the default policy:

```yaml
healthy_curve_flow:
  voc_slot_calibration:
    enabled: true
    trust_n_components: true
    min_fit_points: 3
```

The target slot is:

```text
k_target = n_components
voc_ref_total_corrected = a + k_target * d
```

Then:

```text
DeltaV_total = voc_ref_total_corrected - voc_ref_total_raw
DeltaV_unit = DeltaV_total / n_components
```

The healthy voltage axis is shifted by `DeltaV_unit`, and the flow recomputes:

- `healthy_ref.voc`
- `healthy_ref.vmp`
- `healthy_ref.pmp`
- `healthy_ref.ff`
- `healthy_ref.voc_total`
- `healthy_ref.voc_unit_ref_corrected`

This calibrated healthy reference is then used by later comparison logic.

## 6. Curve Sanitization

After shifting the voltage axis, the helper sanitizes the healthy curve:

1. Remove non-finite points and negative-current points.
2. Sort by voltage.
3. Remove negative-voltage points.
4. Remove points above corrected `Voc`.
5. Insert `V = 0` if needed.
6. Insert corrected `Voc` endpoint if needed.
7. Force the endpoint current to `0`.
8. Deduplicate voltage points and keep non-negative current.

This keeps the calibrated healthy curve physically usable for later interpolation and feature extraction.

## 7. RulePredictor4 Reuse

After pre-calibration, `healthy_curve_flow` stores:

```python
group_stats["voc_slot_model"]
group_stats["voc_slot_calibration"]
```

`RulePredictor4._analyze_total_voc()` will use the precomputed slot model first. If present, it does not refit `a, d`; it directly computes:

- `nearest_ref`
- `residual`
- `signed_residual`
- `outlier`

If no precomputed model is available, RulePredictor4 keeps its fallback fit path with the same `a + k*d` semantics.

## 8. Configuration

Calibration enablement is controlled by:

```yaml
healthy_curve_flow:
  voc_slot_calibration:
    enabled: true
    trust_n_components: true
    min_fit_points: 3
```

Rule4 slot-fit parameters are:

```yaml
healthy_curve_flow:
  rule_predictor4:
    voc_ap_tol: 10.0
    voc_ap_ref_window: 5.0
    voc_ap_d_step: 0.1
    voc_ap_a_step: 0.1
    voc_ap_clip_factor: 2.0
    voc_ap_anchor_values: [0.0, 0.001, 0.005, 0.03]
    voc_ap_min_fit_isc_abs: 0.2
```

## 9. Debug Outputs

Calibration metadata is written into `healthy_ref["voc_slot_calibration"]` and propagated through `group_stats["voc_slot_calibration"]`.

Important fields:

- `applied`
- `target_slot_k`
- `target_slot_source`
- `raw_voc_total`
- `corrected_voc_total`
- `delta_v_total`
- `delta_v_unit`
- `slot_model`
- `observation_count`

Important slot-model fields:

- `best_a`
- `best_d`
- `fit_input_string_ids`
- `fit_excluded_low_isc_string_ids`
- `refs_by_string`
- `residuals_by_string`
- `outlier_by_string`

These fields are for calibration and low-Voc analysis only.

## 10. Temperature Output

Temperature output is not derived from the slot model.

Current CSV outputs only keep the original environment-based estimates:

- `backsheet_temperature`
- `estimated_cell_temperature`
- `estimated_ambient_temperature`

Those values come from the flow's environment estimation path, not from `best_d`.

## 11. Notes

- Default `trust_n_components: true` is intended to avoid absorbing true `low_voc` faults into a lower target slot.
- If `trust_n_components` is disabled, the helper can target the nearest slot to the group median measured `Voc_total`, but that is riskier when the group contains many real low-Voc strings.
- The `voc_ap_ref_window` stays at `5.0` by design. A larger window can pull `best_d` too far away from the physical single-module Voc and shift the calibrated healthy curve unrealistically.
