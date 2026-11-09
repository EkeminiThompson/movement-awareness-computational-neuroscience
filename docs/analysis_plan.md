# Provisional Analysis Plan

**Status:** Pre-analysis plan; not preregistered. Conditional on the data audit.

## Question and estimand

Can pre-movement EEG features predict prespecified continuous kinematic characteristics of the subsequent hand movement in a participant excluded entirely from model fitting?

The primary outcome(s) must be selected before inspecting model performance. Candidate outcomes may include movement duration, path length, peak velocity, endpoint error, or trajectory summaries, but only those supported by the actual motion files and task definitions may be used.

The estimand is participant-held-out predictive performance on valid matched trials, not a causal effect of EEG on movement.

## Unit of analysis

A matched trial nested within participant. Repeated trials from a held-out participant must not appear in training or tuning data.

## Validation

- Outer loop: leave-one-participant-out.
- Inner loop: participant-grouped validation among training participants, where sample size permits.
- Fit imputation, scaling, feature selection, dimensionality reduction, calibration, and hyperparameter selection only within training folds.
- Keep all trials from the held-out participant out of every model-selection stage.
- Report fold-level and participant-level metrics; do not treat trials as independent participants.

## Predictors

Use only EEG data from a verified pre-movement interval. Any filtering, epoching, artefact rejection, and feature extraction must be documented. Ensure zero-phase filtering does not leak post-onset information into the pre-movement epoch; use causal filtering or an adequately guarded segment and sensitivity analysis.

Candidate representations, subject to data feasibility:
- channel-wise time-domain summaries;
- prespecified band-limited power and temporal summaries;
- regularised multivariate representations;
- temporal models only after simple baselines and sample-size checks.

## Baselines

1. Training-fold mean predictor.
2. Regularised linear regression.
3. A simple nonlinear baseline if justified by sample size.

Complex models are secondary and must demonstrate added value under the same participant-held-out splits.

## Outcomes and metrics

For continuous outcomes, report MAE and RMSE; report participant-level correlation only as a supplementary descriptive measure. Compare against the training-mean baseline. Include uncertainty intervals that respect participant clustering (participant-level bootstrap where the number of participants permits; otherwise show all participant-fold results and clearly state limitations).

## Secondary questions

Condition decoding or condition-stratified performance may be explored only if task labels are verified and sample sizes support it. The instructed congruent/incongruent mapping must not be relabelled as unexpected perturbation.

## Robustness and sensitivity

- Alternative valid trial-exclusion thresholds.
- Artefact handling choices.
- Pre-movement epoch definition.
- Feature-family ablations.
- Participant influence analysis.
- Missingness and imbalance by participant and condition.

## Leakage and integrity tests

Automated tests should verify:
- participant disjointness between train and test;
- no trial appears in more than one split;
- all learned preprocessing is fitted within training data;
- no execution-period EEG is used as a predictor;
- labels and trial matches are not inferred from outcome files in a way that leaks targets;
- random seeds, package versions, and data version are recorded.

## Interpretation

Predictive performance does not establish causality, conscious access, movement awareness, or sense of agency. The dataset contains no direct subjective measure of these constructs as currently documented. Interpret findings as associations between pre-movement neural signals and subsequent movement behaviour.

## Analysis freeze

Before confirmatory modelling, commit:
- data-audit report and checksum manifest;
- exact inclusion/exclusion rules;
- primary outcome(s);
- model specifications and metrics;
- validation code and leakage tests.

Changes after this freeze must be logged as deviations.
