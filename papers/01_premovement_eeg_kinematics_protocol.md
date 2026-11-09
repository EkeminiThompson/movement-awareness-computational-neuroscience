# Pre-movement EEG and Subsequent Action Kinematics: A Participant-Independent Modelling Protocol

**Author:** Ekemini Thompson  
**Status:** Working protocol manuscript; no empirical model results reported  
**Dataset:** IACKD, OpenNeuro ds006840

## Abstract

Movement preparation and execution are linked processes, but whether neural activity immediately preceding an action predicts its subsequent kinematic expression in unseen individuals remains an empirical question. This protocol specifies a participant-independent analysis of the IACKD dataset, which combines EEG and three-dimensional hand kinematics. The candidate primary analysis predicts prespecified continuous movement-execution outcomes from EEG recorded strictly before movement onset. The planned evaluation uses leave-one-participant-out validation, with preprocessing, feature selection and model tuning confined to training participants. A feasibility-first design specifies checks for non-empty pre-movement EEG, event timing, trial-level EEG–kinematic correspondence and participant coverage. The published descriptor and original project README report different valid-trial counts; the discrepancy is explicitly identified for file-level reconciliation. The study does not treat instructed congruent/incongruent mapping conditions as unexpected perturbations and does not claim to measure consciousness or sense of agency. This manuscript reports a protocol and makes no predictive performance claim.

**Keywords:** EEG; hand kinematics; movement preparation; participant-independent validation; multimodal modelling; reproducibility.

## 1. Background

Multimodal recordings that combine neural activity and movement trajectories provide an opportunity to examine relationships between pre-action brain signals and subsequent movement behaviour. Predictive analyses must account for repeated observations within individuals. If trials from one participant appear in both training and test partitions, performance may reflect participant-specific information rather than generalisation to a new person.

This protocol therefore places participant-independent evaluation at the centre of the proposed analysis. It is based on IACKD (OpenNeuro ds006840), a dataset combining EEG and hand-motion measurements during visuomotor tasks.

## 2. Aim and research question

The primary research question is whether features extracted from EEG recorded strictly before movement onset can predict prespecified continuous kinematic properties of the subsequent action in a participant excluded entirely from model fitting.

The estimand is participant-held-out predictive performance on valid, matched trials. It is not a causal effect of neural activity on movement.

## 3. Dataset and provenance

The dataset is available as OpenNeuro accession ds006840 (versioned DOI: https://doi.org/10.18112/openneuro.ds006840.v1.0.0). Its data descriptor reports 15 participants, 32-channel EEG sampled at 1,024 Hz, three-dimensional hand kinematics sampled at 170 Hz, 7,040 recorded trials and 6,671 valid trials. The original project README has reported 6,720 valid trials. The difference is recorded here as a provenance and reproducibility issue to be resolved against the actual released files.

The congruent and incongruent conditions are instructed visuomotor mapping conditions. They are not described as unexpected perturbations. The dataset does not directly provide subjective reports of consciousness, conscious access or sense of agency.

## 4. Feasibility and data audit

Before confirmatory modelling, the analysis requires a version-pinned inventory of participants, sessions, EEG recordings, event tables and motion files. The audit protocol records file sizes, empty files, event-table row counts and optional checksums. It also requires a dataset-specific trial-matching table documenting participant, run, event identifier, condition, EEG interval, movement-onset definition, kinematic interval and exclusion reason.

A one-second pre-movement window is a candidate interval, subject to verification against the released recordings and task timing. The audit must establish that EEG is present throughout the interval, that event markers identify the same movement trial, and that filtering does not introduce post-onset information into the pre-onset features. Filename similarity alone is not accepted as evidence of trial correspondence.

## 5. Planned modelling and validation

The primary candidate task is regression from pre-movement EEG features to continuous kinematic outcomes selected before model comparison. Candidate outcomes include movement duration, path length, peak velocity, endpoint error or trajectory summaries, but only outcomes supported by the verified motion files and task definitions will be used.

The outer evaluation loop is leave-one-participant-out. All trials from the held-out participant remain outside preprocessing fitting, feature selection, dimensionality reduction, hyperparameter selection and calibration. Where sample size permits, inner model selection will also use participant-grouped partitions.

A training-fold mean predictor and regularised linear regression form the initial baselines. More complex nonlinear or temporal models are secondary and must demonstrate added value under the same participant-held-out splits.

For continuous outcomes, the planned metrics are mean absolute error and root mean squared error. Participant-level fold results will be reported. Uncertainty methods will respect participant clustering; trial-level observations will not be treated as independent participants.

## 6. Leakage control and sensitivity analyses

Predictors will use only the verified pre-movement interval. Filtering, artefact rejection, feature extraction, imputation, scaling and feature selection will be documented. Zero-phase filtering will not be used in a way that allows post-onset samples to influence pre-onset features; causal filtering or a guarded interval will be considered.

Sensitivity analyses are specified for epoch definitions, artefact handling, valid-trial thresholds, feature families and participant influence. The pipeline will record data version, package versions, random seeds, split assignments and analysis deviations.

## 7. Interpretation

Predictive performance will not be interpreted as causality, conscious access, movement awareness or sense of agency. The dataset's currently documented task conditions do not directly measure those constructs. The sample of 15 participants also constrains the precision and scope of participant-level generalisation.

## 8. Reproducibility assets completed

The repository contains a data-audit protocol, provisional analysis plan, this protocol manuscript, a BIDS inventory utility, unit tests, a reference file and a continuous-integration workflow. The utility inventories files and event tables without loading raw EEG or motion arrays. Its unit tests cover TSV event-row counting and empty event tables. The CI workflow compiles the source and tests and runs the unit tests.

These software checks validate the utility's basic behaviour; they do not constitute an audit of the actual OpenNeuro recordings or evidence of model performance.

## Conclusion

This protocol specifies a leakage-conscious, participant-independent framework for investigating whether pre-movement EEG predicts subsequent hand kinematics. Its completed contribution is the documented research design and reproducibility infrastructure. No empirical prediction results are reported in this manuscript.

## Declarations

- **Data:** IACKD, OpenNeuro ds006840, https://doi.org/10.18112/openneuro.ds006840.v1.0.0.
- **Data descriptor:** Cai M, Fu R, Wang Y, Lu B, Guo S, Xu F. Intention-Action Conflict EEG-Hand Kinematics Dataset for Unimanual Control under Congruent and Incongruent Conditions. Scientific Data. 2026;13:933. https://doi.org/10.1038/s41597-026-07146-x.
- **Code:** https://github.com/EkeminiThompson/movement-awareness-computational-neuroscience.
- **Funding:** To be completed by the author.
- **Competing interests:** To be completed by the author.
- **Ethics:** Confirm applicable dataset governance and institutional requirements before analysis or submission.