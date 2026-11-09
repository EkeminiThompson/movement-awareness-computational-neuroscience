# IACKD / OpenNeuro ds006840: Data Audit Protocol

## Purpose

Determine whether the released dataset supports participant-independent prediction of movement-execution kinematics from EEG recorded strictly before movement onset. This audit precedes confirmatory modelling.

## Provenance checks

- Record OpenNeuro accession, version, retrieval date, DOI, licence, and checksums.
- Preserve the original dataset citation and data-descriptor citation.
- Record the exact repository commit/tag used for any companion code.
- Do not upload raw participant data, identifiable metadata, or restricted derivatives to this repository.

## Inventory

Create a machine-readable inventory of:
- participants and sessions;
- EEG files, channel counts, sampling rates, units, and annotations;
- events.tsv files and event codes;
- motion/kinematic files, coordinate definitions, units, and sampling rates;
- trial identifiers and any mapping files;
- missing, empty, corrupted, or placeholder files.

## Trial-matching audit

Construct an explicit table with one row per candidate trial:
- participant;
- session/run;
- event/trial identifier;
- condition;
- EEG recording and sample/time bounds;
- movement onset definition;
- kinematic recording and sample/time bounds;
- match status and exclusion reason.

Do not assume filename similarity establishes trial identity. Verify event timing against the data descriptor and original task documentation.

## Pre-movement window

The candidate window is the one-second interval immediately preceding independently verified movement onset. Verify that:
1. EEG exists throughout the interval;
2. the event marker corresponds to the same movement trial;
3. filtering does not introduce post-onset leakage into the pre-onset interval;
4. baseline correction and normalisation use only information allowed by the evaluation design;
5. no execution-period EEG enters the predictor.

If these conditions cannot be established, stop and revise the research question.

## Count reconciliation

The published data descriptor reports 6,671 valid trials, while the original project README has reported 6,720. Reproduce counts from the actual released events and files. Report:
- recorded trial count;
- candidate matched trial count;
- valid trial count under each explicit exclusion rule;
- per-participant and per-condition counts;
- exact reason for every excluded trial.

Do not choose one count merely because it appears in a paper or README.

## Deliverables

- `reports/data_inventory.json`
- `reports/trial_match_audit.csv`
- `reports/exclusion_summary.csv`
- `reports/data_audit.md`
- checksums and environment metadata.

The reports must not include raw EEG or kinematic samples.

## Stop/go decision

**GO** only if actual pre-movement EEG, reliable event timing, reproducible EEG–kinematic trial matching, and sufficient participant coverage are demonstrated.

**REVISE** if a related but different question is supported by the released data.

**STOP** if the required pre-movement signals or trial associations cannot be verified. No predictive result should be reported before this gate passes.
