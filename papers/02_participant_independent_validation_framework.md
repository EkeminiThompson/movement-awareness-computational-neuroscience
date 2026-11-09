# Participant-Independent Validation for Multimodal EEG–Kinematic Prediction: A Reproducible Leakage-Control Framework

**Author:** Ekemini Thompson  
**Status:** Methods paper draft; no empirical performance results reported  
**Project:** IACKD / OpenNeuro ds006840

## Abstract

Multimodal EEG–kinematic datasets contain repeated trials nested within participants, creating a risk that trial-wise random splitting overestimates performance on unseen individuals. This methods paper documents a participant-independent validation framework developed for a proposed pre-movement EEG to subsequent movement-kinematics prediction study. The framework defines the participant as the outer generalisation unit, requires all learned preprocessing and model selection to occur within training participants, and establishes explicit checks for temporal leakage, trial matching and participant overlap. A lightweight BIDS inventory utility, unit tests and CI workflow are included in the project repository. The utility inventories file metadata and event-table rows but does not inspect raw signal validity or establish EEG–kinematic trial correspondence. The contribution is a reproducible validation specification and supporting software checks; no model results or claims about movement awareness are made.

**Keywords:** cross-validation; EEG; kinematics; data leakage; grouped validation; BIDS; reproducibility.

## 1. Introduction

Repeated trials from the same participant are statistically dependent and may contain participant-specific signatures. When trials are randomly divided between training and test sets, a model can exploit those signatures and appear to generalise better than it does to a new individual. This issue is especially important in multimodal neuroscience datasets, where subject-specific signal characteristics, recording conditions and movement patterns can be learned by flexible models.

This paper describes the participant-independent validation design implemented for the IACKD EEG–kinematic research project. It focuses on validation architecture and reproducibility controls rather than reporting empirical model performance.

## 2. Validation target and unit of generalisation

The target is predictive performance for a participant whose data have not been used in any model-development step. The unit of outer generalisation is therefore the participant, not the trial.

The proposed outer loop is leave-one-participant-out validation. In each fold, all trials from one participant are held out. The remaining participants form the training data. The held-out participant is not used for feature selection, imputation, scaling, dimensionality reduction, hyperparameter tuning, calibration or other learned transformations.

Where sample size permits, inner model selection uses participant-grouped validation within the training participants. If the available number of participants does not support reliable nested tuning, the analysis will use prespecified low-complexity models and report the limitation rather than reverting to trial-wise random splitting.

## 3. Preprocessing boundaries

Every learned preprocessing operation is fitted only on the training participants in each outer fold. This includes imputation, standardisation, feature selection, dimensionality reduction and hyperparameter selection. The fitted transformations are then applied unchanged to the held-out participant.

Predictors are restricted to a verified pre-movement EEG interval. Filtering and epoch construction must prevent post-onset samples from influencing pre-onset features. In particular, zero-phase filtering across a boundary that includes movement execution can introduce temporal leakage. The project protocol therefore requires causal filtering or a guarded interval, with the choice documented and evaluated through sensitivity analysis.

## 4. Trial correspondence and outcome integrity

A valid modelling row requires a documented link between a participant, task event, EEG interval and kinematic record. Filename similarity is insufficient. The data audit specification requires explicit event identifiers, onset definitions, sample/time bounds, condition labels and exclusion reasons.

Kinematic outcomes must be defined from the motion data before model comparison. Outcome extraction must not use information from the held-out participant to tune predictors or select a favourable target definition. Any trial without a verifiable EEG–kinematic correspondence is excluded under a documented rule rather than matched heuristically.

## 5. Baselines and evaluation

The initial baselines are a training-fold mean predictor and regularised linear regression. More complex models are secondary and are evaluated using identical participant-held-out folds.

For continuous kinematic outcomes, the prespecified primary metrics are mean absolute error and root mean squared error. Participant-level fold performance is reported to expose heterogeneity across held-out individuals. Uncertainty summaries must respect the small number of participants; trial counts are not treated as independent sample size for participant-level inference.

Sensitivity analyses are specified for pre-movement epoch definitions, artefact handling, feature families, trial exclusions and participant influence. Split integrity tests should verify that no participant or trial appears in both training and test partitions.

## 6. Software and validation assets

The project repository includes the following completed assets:

- A standard-library BIDS inventory utility that inventories files, participant/session labels, empty files, suffixes and event-table row counts.
- Unit tests for event TSV row counting and header-only event tables.
- A data-audit protocol specifying provenance, inventory and trial-matching checks.
- An analysis plan defining the estimand, participant-held-out validation, baselines, metrics and leakage controls.
- A CI workflow compiling source and tests and executing the unit tests.
- A protocol manuscript and bibliography.

The CI workflow passed its source compilation and unit-test jobs. These checks establish basic software behaviour for the inventory utility; they do not validate raw EEG signal quality, actual trial matching or predictive performance.

## 7. Limitations and interpretation

The framework is designed for a dataset with a limited number of participants. Leave-one-participant-out evaluation provides a direct test of participant transfer, but the number of outer folds is small and uncertainty estimates will be limited by participant count. Nested participant-grouped model selection may also be constrained by the small number of training participants.

The proposed dataset includes instructed congruent and incongruent mapping conditions. These must not be relabelled as unexpected perturbations. The dataset does not directly measure subjective consciousness, conscious access or sense of agency. A model predicting kinematic outcomes from EEG would establish a predictive association under the specified design, not a causal explanation or measurement of movement awareness.

## Conclusion

The completed methods contribution is a participant-independent validation specification supported by an auditable repository structure, inventory utility, unit tests and CI workflow. It makes the participant—not the individual trial—the unit of outer generalisation and explicitly restricts learned preprocessing to training participants. No empirical model results are reported.

## Declarations

- **Dataset:** IACKD, OpenNeuro ds006840, https://doi.org/10.18112/openneuro.ds006840.v1.0.0.
- **Data descriptor:** Cai M, Fu R, Wang Y, Lu B, Guo S, Xu F. Scientific Data. 2026;13:933. https://doi.org/10.1038/s41597-026-07146-x.
- **Code:** https://github.com/EkeminiThompson/movement-awareness-computational-neuroscience.
- **Funding:** To be completed by the author.
- **Competing interests:** To be completed by the author.

**Manuscript note:** This methods paper is a companion to the project's protocol manuscript. Its focus is participant-independent validation and leakage control. The shared dataset and protocol relationship should be disclosed to editors, and overlapping text should be minimised in any eventual submissions.