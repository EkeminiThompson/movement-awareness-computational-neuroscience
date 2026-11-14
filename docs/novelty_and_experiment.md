# Novelty positioning and confirmatory experiment

## Why the broad claim is not novel

Pre-movement EEG decoding, prediction of movement speed/force, continuous hand-trajectory decoding, and participant-independent evaluation have all appeared in prior work. Therefore, this project will not claim to be the first to predict movement from pre-movement EEG, the first to decode hand kinematics, or the first to use leave-one-participant-out validation.

The dataset itself (IACKD / OpenNeuro ds006840) has a 2026 data descriptor and includes instructed congruent/incongruent visuomotor mappings. The dataset release is not a contribution of this study.

## Defensible contribution

The proposed contribution is a dataset-specific, condition-sensitive, participant-held-out evaluation of continuous movement execution outcomes from strictly pre-movement EEG in the IACKD visuomotor mapping paradigm, with an explicit temporal-leakage audit and a fully nested participant-level model-selection boundary.

The study asks:

1. Does strictly pre-movement EEG predict continuous execution kinematics for a participant entirely absent from model fitting?
2. Does prediction error differ between the instructed congruent and incongruent mapping conditions?
3. Are any observed relationships robust to temporal filtering choices, EEG feature-family ablations, participant influence, and a training-mean baseline?

The condition contrast is a testable hypothesis, not an assumed effect. Congruent/incongruent mappings must not be described as unexpected perturbations.

## Related-work boundary

The introduction must discuss at least:
- Gatti et al. (2021), pre-movement EEG and hand-movement speed/force prediction.
- Jain and Kumar (2023), pre-movement EEG and 3-D hand-trajectory prediction with subject-independent evaluation.
- Cai et al. (2026), IACKD dataset description.

Before submission, verify the complete bibliographic metadata, exact task differences, outcome definitions, validation design, and any overlap with the present protocol from the primary papers. Do not claim priority without a systematic search.

## Preregisterable design

- **Primary outcome:** movement completion time, with its onset and termination definitions fixed from the verified aligned kinematic arrays.
- **Secondary outcomes:** 3-D path length and peak 3-D speed, only if arrays and coverage pass the audit.
- **Predictors:** EEG-only features computed from a verified pre-movement interval. Every feature column passed to the regression script must use the `eeg_` prefix.
- **Outer validation:** leave one participant out; all trials of the held-out participant remain outside fitting and tuning.
- **Inner validation:** GroupKFold by participant on training participants for Ridge regularisation selection.
- **Baseline:** training-participant mean.
- **Condition analysis:** score the same held-out model separately in each condition; report per-participant MAE and RMSE, and a participant-level paired error contrast where both conditions are represented.
- **Uncertainty:** bootstrap participant-level fold metrics, not individual trials.
- **Leakage checks:** no post-onset EEG; no execution kinematics among predictors; training-only scaling and tuning; verified trial key uniqueness; raw-data filtering audit.
- **Sensitivity:** pre-movement guard/filter configuration, frequency-feature families, artifact policy, valid-trial exclusions, and participant influence.

## Claim discipline

A successful software test is not an empirical finding. Running the model on synthetic fixtures tests code only. A result can be described as an IACKD experiment only after the input table is traced to the pinned raw/derivative release and its trial matching, timing, and feature-generation audit is passed. Never report the dataset paper's technical-validation statistics as results of this project.
