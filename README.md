# Human Movement Perception and Computational Neuroscience

**Researcher:** Ekemini Thompson  
**Status:** Research protocol and data-feasibility stage. No analyses or results are claimed.  
**Target opportunity:** University of Copenhagen, Postdoc in Cognitive Computational Neuroscience and Human Movement Perception; project *Mapping Experience of Action: Redefining Theories of Consciousness*.

## Research question

Can neural activity recorded before a voluntary action predict subsequent movement execution in participants not used to train the model?

The provisional study will investigate pre-movement EEG features and subsequent hand kinematics in the IACKD dataset (OpenNeuro `ds006840`). The primary question, outcome definitions, and estimands remain conditional on a reproducible audit of the released BIDS data, event timing, actual pre-movement EEG availability, trial matching, and participant-level completeness.

**Provisional title:** *Pre-movement EEG and Subsequent Action Kinematics: Participant-Independent Modelling of Human Visuomotor Behaviour.*

This is a study of movement preparation and execution. The experiment's congruent/incongruent visuomotor mappings were instructed conditions, not unexpected perturbations. The released dataset does not directly measure subjective consciousness, conscious access, or sense of agency. No claim to measure those constructs will be made.

## Dataset provenance

- OpenNeuro: [ds006840](https://doi.org/10.18112/openneuro.ds006840.v1.0.0)
- Data descriptor: [Scientific Data article](https://doi.org/10.1038/s41597-026-07146-x)
- Original project repository: [Boketto1/IACKD](https://github.com/Boketto1/IACKD)
- Reported modalities: 32-channel EEG (1,024 Hz) and 3-D hand kinematics (170 Hz).
- The published data descriptor reports 7,040 recorded trials and 6,671 valid trials from 15 participants. The original project README has reported 6,720 valid trials. This discrepancy must be reconciled against released BIDS events and trial files before analysis.
- Demographic fields are not assumed to be available; the participant table has been reported to contain unavailable values.

Verify the current OpenNeuro version, file checksums, licence, and citation before analysis or redistribution. Do not commit raw participant data to Git.

## Feasibility gate — required before fixing the primary analysis

1. Verify accession, version, provenance, licence, and complete file inventory.
2. Inventory subjects, sessions, EEG recordings, channels, events, motion files, and derivatives.
3. Confirm that pre-movement EEG exists as actual non-empty data, not placeholders, with usable event markers.
4. Establish reproducible trial matching between EEG and kinematics.
5. Reconcile the 6,671 versus 6,720 valid-trial counts.
6. Establish movement-onset timing and a pre-movement window that cannot include post-onset information.
7. Quantify missingness and exclusions by participant and condition.
8. Freeze a versioned data manifest before model development.

If the pre-movement EEG or trial matching is unavailable, revise the question before modelling. Do not silently substitute execution EEG or infer missing labels.

## Provisional analysis plan

- **Primary candidate:** participant-independent prediction of prespecified continuous execution-kinematic outcomes from strictly pre-movement EEG.
- **Validation:** leave-one-participant-out evaluation. All preprocessing, feature selection, hyperparameter tuning, and calibration must be fitted within training participants.
- **Baselines:** participant-independent mean and regularised linear models before complex temporal or deep-learning models.
- **Secondary analyses:** condition decoding and EEG feature ablations, only where event structure and sample sizes justify them.
- **Reporting:** participant-level performance distributions, uncertainty, missing-data handling, exclusion sensitivity, and model calibration where appropriate.
- Trial-wise random splits across participants must not be presented as the principal generalisation result.
- Predictive association must not be described as causal evidence.

## Planned repository structure

- `docs/data_audit_protocol.md` — provenance and feasibility checklist.
- `docs/analysis_plan.md` — estimands, validation, baselines, leakage controls.
- `src/` — reproducible inventory and analysis code after feasibility checks.
- `tests/` — event, trial-matching, split-integrity, and leakage tests.
- `manuscript/` — protocol and manuscript drafts; results only after reproducible execution.
- `reports/` — generated audit summaries and analysis outputs, excluding raw participant data.

## Scientific integrity

- No results, effect sizes, or model performance are claimed at this stage.
- No participant-level raw EEG or kinematic data should be committed to Git.
- No direct consciousness, awareness, or agency measurement is claimed.
- No causal inference is made from predictive associations.
- The research question will be revised if the data audit invalidates its assumptions.
- Published counts and independently reproduced counts will be reported separately.

## Alignment with the Copenhagen project

The repository is a computational work sample, not a substitute for the host group's original proposal. The vacancy requires applicants to request the original project proposal from Associate Professor Mark Schram Christensen (`markc@psy.ku.dk`) before applying. The eventual project description must be aligned with that proposal; this repository implies no affiliation, collaboration, or endorsement.

## Citation

Cite the OpenNeuro dataset and its data descriptor in any resulting work. Confirm the final metadata and citation format before manuscript submission.
