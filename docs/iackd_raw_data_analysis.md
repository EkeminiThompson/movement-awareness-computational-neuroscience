# IACKD raw-data acquisition and confirmatory analysis

## Dataset acquisition

The target is NEMAR IACKD accession `on006840`, corresponding to OpenNeuro `ds006840`, release v1.0.0. NEMAR lists 15 participants, approximately 7.42 GB and a CC0 license. The full download must include both `sourcedata/` and `derivatives/`; the default NEMAR download skips large `derivatives/` and `stimuli/` trees.

The NEMAR data plane exposes a version-pinned manifest and stable file URLs. The repository downloader fetches `https://data.nemar.org/on006840/v1.0.0/manifest.json`, downloads every manifest entry (including raw EEG, sourcedata and derivatives), resumes interrupted transfers, and verifies each file using the manifest's SHA-256, MD5 or Git-blob checksum.

Run:

```bash
bash scripts/download_iackd.sh data/on006840
```

No NEMAR account, CLI, DataLad or git-annex installation is required for this public release. Keep the downloaded data outside Git. Never commit raw EEG or participant recordings to this repository.

## Why the supplied pre-movement EEG derivative is not the predictor

The data descriptor reports that the pre-movement EEG was downsampled to 200 Hz and filtered 0.1–45 Hz using a zero-phase FIR filter. A zero-phase filter can allow samples after movement onset to influence samples before onset. The supplied `segs_1s` is therefore not assumed to be leakage-safe.

The extractor reads the raw BrainVision EEG, applies a one-pass causal 0.1–45 Hz Butterworth filter to the continuous recording, average-references instantaneously, and extracts only the premove interval ending at the event-14 / execution-onset boundary. The supplied premove MATLAB file is used only for trial metadata/index reconciliation, not as an EEG predictor. This avoids post-onset signal leakage into the predictors. No ICA model is fitted on the held-out participant.

## Trial reconciliation

For each participant, hand and run, the script requires:
- BIDS `*_events.tsv` with premove, execution and hit rows;
- raw BrainVision `*_eeg.vhdr` plus its data file;
- `derivatives/premove/.../pre_eeg_L*.mat` or `pre_eeg_R*.mat`;
- `derivatives/aligned/.../eeg_leap_L*.mat` or `eeg_leap_R*.mat`.

The MATLAB `OUT` record's EEG segment index is resolved against `trial_info`, then checked against the BIDS event-14 time. Premove end must agree with execution onset, and execution end must agree with hit onset within 50 ms. A trial cannot be silently matched by filename or row order alone. Duplicate event matches fail the run.

Kinematic outcomes are calculated from the aligned MATLAB `OUT` arrays: movement duration, 3-D path length and peak 3-D speed using `x_mm`, `y_mm` and `z_mm`. The dataset-provided aligned coordinates are 4-Hz zero-phase-smoothed; this affects the kinematic target definition, not the pre-movement EEG predictor. A raw Leap-stream sensitivity analysis should be added before final journal submission. Alignment QC uses the predeclared coverage threshold (0.75) and absolute start residual threshold (30 ms). All excluded trials remain represented in `all_matched_trials_pre_qc.csv` and the run audit.

## Run locally

```bash
python -m pip install -r requirements-iackd-analysis.txt
python scripts/audit_and_extract_iackd.py --dataset data/on006840 --output-dir reports/iackd
python scripts/run_lopo_experiment.py --input-csv reports/iackd/audited_eeg_kinematics_features.csv --target movement_duration_ms --output-dir reports/iackd/lopo_duration
python scripts/run_lopo_experiment.py --input-csv reports/iackd/audited_eeg_kinematics_features.csv --target path_length_mm --output-dir reports/iackd/lopo_path_length
python scripts/run_lopo_experiment.py --input-csv reports/iackd/audited_eeg_kinematics_features.csv --target peak_speed_mm_s --output-dir reports/iackd/lopo_peak_speed
```

The GitHub Actions workflow uses the same version-pinned manifest downloader and uploads the reports as an artifact. It can take several hours and may hit runner disk or download limits. The first CLI-based attempt failed during git-annex remote initialisation; the current workflow uses the documented direct data-plane manifest instead.

## Interpretation and reporting boundary

The workflow must stop if any of the 15 participants is missing, event matching is ambiguous, the raw EEG interval cannot be verified, or either condition is absent. A successful run produces three LOPO experiments, each with 15 outer folds. Scaling and Ridge tuning remain inside grouped training-participant folds. Condition-specific scores and participant-level bootstrap intervals are reported.

The 6,671 valid-trial figure in the 2026 data descriptor and the 6,720 figure in the original project README are reference counts, not acceptance targets to force. The file-level audit determines the observed counts. Never claim that the discrepancy is resolved until the inventory and exclusions explain it.


## EthicsApprovals: source article approval and participant consent

The original IACKD data descriptor explicitly reports that all experimental procedures complied with the Declaration of Helsinki and were approved by the **Ethics Committee of the First Hospital of Qinhuangdao (Approval No. 2024G006)**. The article further states that the committee approved both the experimental protocol and anonymized public sharing of the dataset. **Written informed consent was obtained from every participant before data collection**, and the consent form explicitly covered anonymized data storage, sharing, and open public release for scientific research. The article also states that the released data were de-identified before publication.

This is the ethics approval and consent reported by the original dataset investigators; it is not an approval granted to the present secondary-analysis project. The current project must follow the applicable institutional requirements for secondary analysis and must not imply that the original approval automatically covers a new research protocol.

Source: Cai et al. (2026), *Intention-Action Conflict EEG-Hand Kinematics Dataset for Unimanual Control under Congruent and Incongruent Conditions*, Scientific Data 13, Article 933. https://doi.org/10.1038/s41597-026-07146-x
