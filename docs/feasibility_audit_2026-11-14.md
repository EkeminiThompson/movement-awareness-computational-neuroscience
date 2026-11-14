# IACKD feasibility audit and experimental execution status

**Audit date:** 2026-11-14  
**Dataset:** IACKD, OpenNeuro `ds006840`, version `v1.0.0`  
**Repository baseline:** `main`, commit `87b5b596e1d7830bf51451665a6073fa37a581ea`  
**Audit type:** Documentary/source-level feasibility audit. This is not a file-level audit of downloaded participant recordings.

## Executive decision

**Documentary feasibility: GO, subject to raw-file verification.** The published dataset description documents both one-second pre-movement EEG segments and trial-aligned EEG/Leap Motion execution records. It also documents trial indices linking the two derivative families. This is stronger evidence than the protocol's initial description, which treated the one-second window as only a candidate.

**Empirical execution: BLOCKED.** The v1.0.0 data archive is approximately 6.4 GB. It was not present in the repository or current execution environment, and a direct network acquisition attempt from this environment failed at DNS resolution. Consequently, no participant files were inspected here, no trial-match table was computed, and no new predictive experiment was run. Do not describe this audit as a completed raw-data audit and do not report model scores as study results.

## 1. Source-supported dataset facts

The 2026 Scientific Data descriptor reports:

- 15 healthy participants, labelled s01-s15.
- 7,200 intended trials, 7,040 recorded trials, and 6,671 retained valid trials.
- 32-channel EEG recorded at 1,024 Hz.
- Pre-movement EEG derivatives sampled at 200 Hz, with fixed one-second `segs_1s` windows and variable-length `segs_var` windows.
- Movement-execution derivatives containing aligned EEG and 3-D hand positions at 100 Hz.
- Original Leap Motion hand streams, described as approximately 170 Hz in the data-file description; the methods section describes acquisition at 200 Hz. This raw-rate discrepancy must be resolved from per-file sidecars and timestamps.
- Trial metadata in pre-movement derivatives, including `keep_mask_1s`, `trial_csv`, `trial_info`, condition labels and mark timing.
- Aligned execution records containing `eeg_seg_index`, `im_trial`, `t_ms`, `EEG`, `x_mm`, `y_mm`, `z_mm`, `ball_color`, `move_direct`, `start_res_ms` and `cov`.

The descriptor reports 6,671 valid trials. The 6,720 count in the original project README should be treated as a historical/provenance discrepancy, not as an alternative verified count. The file-level audit must reproduce counts from the pinned release and state exactly which stage each count represents.

## 2. Feasibility matrix

| Requirement | Documentary evidence | Audit status |
|---|---|---|
| 15-participant coverage | Participant and session structure documented | Supported; inventory not run |
| Pre-movement EEG | `derivatives/premove/sub-XX/ihc/pre_eeg_*.mat`; `segs_1s`, `times_1s` | Supported; arrays not inspected |
| Execution kinematics | `derivatives/aligned/sub-XX/ihc/eeg_leap_*.mat`; 3-D positions | Supported; arrays not inspected |
| Trial-level linkage | `eeg_seg_index` and `im_trial` documented | Plausible; must verify uniqueness and referential integrity |
| Timing | EEG event 14 and Leap start/hit anchors; affine alignment | Supported by published methods; per-trial residuals not recomputed |
| Validity / exclusions | `keep_mask_1s`, `cov`, `start_res_ms` documented | Must recompute |
| Outcome availability | movement duration and 3-D trajectories documented | Supported; exact event/end-point fields must be verified |
| Participant-independent prediction | 15 participants permit outer LOPO | Methodologically feasible; no model run |
| Demographic generalisation | One small, young, right-handed cohort | Limited; not population-representative |

## 3. Important protocol corrections

1. **Use the actual derivative architecture.** The intended pairing is pre-movement EEG from the `premove` derivatives and kinematic targets from the corresponding `aligned` execution derivatives. Do not pair by filename similarity alone.
2. **Preserve participant, hand, run and trial identity.** The linkage must be verified through `eeg_seg_index`, `im_trial`, trial metadata and event anchors. A one-to-one match assertion is required before modelling.
3. **Resolve the sampling-rate descriptions.** The descriptor reports EEG at 1,024 Hz, pre-movement derivatives at 200 Hz, aligned derivatives at 100 Hz, and describes raw Leap streams at approximately 170 Hz in one section but 200 Hz in the methods. The 100-Hz aligned derivative is the common time base for execution outcomes.
4. **Audit temporal filtering for leakage.** The published pre-movement preprocessing describes zero-phase FIR filtering on continuous EEG before epoch extraction. Zero-phase filtering can allow post-onset samples to influence samples immediately before onset. The confirmatory predictor should therefore be regenerated from raw continuous EEG using a causal/minimum-phase filter or a pre-onset guard interval, with the exact implementation documented. The precomputed `segs_1s` can be used for replication/sensitivity analysis, not assumed leakage-safe without verifying filtering boundaries.
5. **Do not fit ICA on the held-out participant.** All learned artefact policies, channel selection, ICA fitting, scaling and model selection must be confined to training participants. The current LOPO preprocessing implementation is a starting point, not proof that the entire analysis is leakage-safe.
6. **Predefine a continuous primary outcome.** Movement completion time is directly described by the dataset paper. Path length and peak 3-D speed are reasonable secondary candidates, but their definitions must be fixed from the actual arrays and validity rules before inspecting model performance.
7. **Keep condition labels distinct from perturbations.** Red/yellow are instructed congruent/incongruent mappings, not unexpected perturbations. No consciousness, awareness or agency measure is present.

## 4. Proposed confirmatory experiment

### Primary estimand

Participant-held-out prediction error for movement completion time using only EEG from the verified one-second interval before event 14 / movement onset, evaluated on valid EEG–kinematic matched trials.

### Outcomes

- **Primary:** movement completion time, from verified movement onset to the corresponding verified hit/trajectory termination.
- **Secondary:** 3-D path length and peak 3-D speed, only where valid, sufficiently sampled x/y/z trajectories and endpoint timing are available.

For positions (p_t=(x_t,y_t,z_t)), path length is `sum_t ||p_t - p_(t-1)||_2`. Speed is calculated from the time derivative of the 3-D position on the documented 100-Hz aligned time base; peak speed is the maximum valid speed during the execution interval. No interpolation across long missing segments and no extrapolation beyond recorded coverage.

### Predictors

Prespecified pre-movement EEG summaries: channel-wise log band power (theta 4–8 Hz, alpha/mu 8–13 Hz, beta 13–30 Hz) and, as a separate feature family, readiness-potential summaries over motor channels C3/C4/Cz. Frequency bands and channel mapping must be frozen after the raw-file/channel audit. No movement-execution EEG may enter the predictors.

### Models and validation

1. Training-participant mean baseline.
2. Ridge regression with scaling and regularisation selected only inside grouped training folds.
3. Optional nonlinear baseline only after the simple models and trial counts are verified.

Outer validation: 15-fold leave-one-participant-out. All trials from the held-out participant remain excluded from preprocessing fitting, feature selection, hyperparameter tuning and calibration. Report MAE and RMSE per held-out participant, pooled out-of-fold predictions, and participant-level uncertainty. Do not treat thousands of trials as thousands of independent participants.

### Required controls

- Participant-disjoint split assertions.
- Trial-key uniqueness and one-to-one EEG/kinematics match assertions.
- No post-onset samples in predictor arrays.
- Training-only fitting of every learned transform.
- Target-only execution arrays used solely to calculate labels.
- Sensitivity to pre-movement window/filter choice, artefact thresholds and participant influence.
- A label-permutation or within-training-fold negative-control analysis, designed to preserve the grouped validation boundary.

## 5. Published technical validation (not results from this repository)

The dataset paper reports the following source-author validation:

- More than 99.375% of retained cross-modal trial pairs had synchronization residuals within ±30 ms; the reported residual distribution was centred near 7.94 ms, with mean signed error below 1.47 ms.
- Reported mean completion times were 303.84 ± 116.74 ms for left-hand and 299.98 ± 121.17 ms for right-hand movements.
- Congruent trials averaged 294.53 ± 119.52 ms and incongruent trials 309.23 ± 117.03 ms.

These are published dataset-characterisation results. They are not participant-held-out EEG-to-kinematics model results, are not reproduced by this project, and must not be presented as such.

## 6. Repository implementation status

The merged `main` commit `87b5b596` contains LOPO-safe EEG preprocessing and ICA labelling code. The existing tests cover LOPO fold generation, but the repository does not yet contain a verified IACKD .mat loader, a participant/run/trial linkage audit, a kinematic outcome extraction pipeline, or an executed regression experiment with held-out predictions.

The current preprocessor's defaults (0.5–40 Hz, 250 Hz resampling) differ from the dataset paper's described pre-movement processing (0.1–45 Hz, 200 Hz). These should not be silently changed or assumed equivalent. A raw-data reprocessing configuration must be justified and versioned, and a faithful derivative-replication analysis should be kept distinct from the leakage-controlled confirmatory analysis.

## 7. Acquisition and exact next execution gate

The pinned NEMAR copy provides the v1.0.0 archive and recommends:

```bash
nemar dataset download on006840
```

The full archive is approximately 6.4 GB. A subset can be requested by participant, but a full 15-participant LOPO study requires coverage of all participants. Keep the raw dataset outside Git; commit only manifests, audit summaries, code, model predictions without identifying metadata, and figures.

The first executable gate after acquisition is:

1. Inventory all `s01`–`s15` premovement and aligned MAT files.
2. Load the MAT structs and record array shapes, field presence, trial counts and sampling/time axes.
3. Reconcile the `segs_1s` trial mask with aligned `eeg_seg_index` and `im_trial` references.
4. Recompute per-participant and total counts, exclusion reasons, coverage and timing residuals.
5. Freeze the matched-trial manifest and primary outcome definition.
6. Only then run the 15 outer LOPO folds.

## 8. Current conclusion

The research question is **documentarily feasible and worth pursuing**. The data descriptor supplies the required pre-movement EEG and aligned movement records, and reports strong technical cross-modal synchronization. However, the project has **not yet passed the raw-file feasibility gate** because the v1.0.0 data files have not been downloaded and inspected in this execution environment. There are therefore **no new experimental model results** to report. The next milestone is actual file-level audit and LOPO execution, not manuscript polishing or stronger claims.

## References

1. Cai, M., Fu, R., Wang, Y., Lu, B., Guo, S., & Xu, F. (2026). *Intention-Action Conflict EEG-Hand Kinematics Dataset for Unimanual Control under Congruent and Incongruent Conditions*. Scientific Data, 13, 933. https://doi.org/10.1038/s41597-026-07146-x
2. OpenNeuro ds006840, v1.0.0. https://doi.org/10.18112/openneuro.ds006840.v1.0.0
3. NEMAR on006840, v1.0.0. https://nemar.org/dataset/on006840
