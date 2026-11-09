# Pre-movement EEG and Subsequent Action Kinematics: A Participant-Independent Modelling Protocol

**Working protocol — no results reported**

## Abstract

Movement preparation and movement execution are linked processes, but the extent to which neural activity immediately preceding an action predicts its subsequent kinematic expression across individuals remains an empirical question. This protocol proposes a participant-independent computational analysis of the IACKD dataset (OpenNeuro ds006840), which combines electroencephalography (EEG) and three-dimensional hand kinematics. The candidate primary analysis will predict prespecified continuous movement-execution outcomes from EEG recorded strictly before movement onset, evaluating generalisation with leave-one-participant-out validation. All learned preprocessing, feature selection, and model tuning will be confined to training participants. A feasibility audit will first verify the availability of non-empty pre-movement EEG, event timing, trial-level EEG–kinematic matching, and participant coverage. The published data descriptor and original project README report different valid-trial counts, which will be reconciled from the released files. The study will not treat instructed congruent/incongruent mapping conditions as unexpected perturbations and will not claim to measure consciousness or sense of agency, which are not directly assessed by the dataset. No results are available at protocol stage.

## 1. Background and rationale

Human action is preceded by neural and behavioural processes that may contain information about the ensuing movement. Multimodal datasets combining EEG with continuous kinematic recordings offer an opportunity to examine these relationships at the level of individual trials. However, predictive studies can overestimate generalisation when trials from the same participant occur in both training and test sets. A participant-held-out design is therefore central to the proposed analysis.

The IACKD dataset is a candidate resource because it combines EEG and hand-motion measurements during visuomotor tasks. Its suitability for the specific pre-movement prediction question cannot be assumed from the data descriptor alone; the released files and event structure must first be audited.

## 2. Aim and hypotheses

The primary aim is to determine whether verified pre-movement EEG contains predictive information about prespecified continuous kinematic properties of the subsequent action in unseen participants.

The primary hypothesis is that a model using pre-movement EEG will outperform a training-fold mean baseline under leave-one-participant-out evaluation. This is a testable hypothesis, not an established finding. The primary kinematic outcome will be selected only after the data dictionary, motion files, and trial structure have been verified and before model comparison.

## 3. Dataset and feasibility

The study will use OpenNeuro ds006840 and cite its data descriptor. The descriptor reports 15 participants, 32-channel EEG sampled at 1,024 Hz, three-dimensional hand kinematics sampled at 170 Hz, 7,040 recorded trials, and 6,671 valid trials. The original project README has reported 6,720 valid trials. The discrepancy will be resolved by an explicit file-level and event-level audit.

The feasibility audit will verify non-empty pre-movement EEG, event timing, movement-onset definitions, trial matching, missingness, and exclusions. A one-second pre-movement interval is a candidate window, subject to confirmation against the released recordings and task timing. If the required signals or matching cannot be established, the primary question will be revised or abandoned.

## 4. Modelling and validation

The primary evaluation will use leave-one-participant-out validation. The held-out participant will be excluded from preprocessing fit, feature selection, hyperparameter tuning, and calibration. Candidate predictors include prespecified time-domain and frequency-domain summaries extracted solely from the verified pre-movement interval. Regularised linear models will be compared with a training-mean baseline before more complex models are considered.

Continuous outcomes will be assessed using mean absolute error and root mean squared error. Participant-level fold results will be reported, with uncertainty methods chosen to reflect the small number of participants. Sensitivity analyses will examine artefact handling, epoch definitions, trial exclusions, and feature-family ablations.

## 5. Interpretation and limitations

The dataset's congruent and incongruent mapping conditions are instructed task conditions and will not be described as unexpected perturbations. Predictive associations will not be interpreted causally. The dataset does not directly provide subjective reports of conscious experience, conscious access, or sense of agency; accordingly, this analysis cannot by itself establish or measure those constructs. The sample of 15 participants also limits the precision and scope of participant-level generalisation claims.

## 6. Reproducibility

The project will version the data manifest, environment, event-matching rules, exclusions, analysis code, random seeds, and validation splits. Raw participant data will not be committed to the repository. No numerical results will be added until the data audit and analysis pipeline have been executed and independently checked.

## Data and code availability

Data are available from OpenNeuro under accession ds006840, subject to the dataset's current metadata and licence. Analysis code and non-sensitive audit outputs will be maintained in the accompanying GitHub repository. No raw EEG or kinematic recordings will be redistributed through GitHub.
