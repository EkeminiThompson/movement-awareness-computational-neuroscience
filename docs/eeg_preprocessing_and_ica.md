# EEG preprocessing and ICA for pre-movement analysis

## Pipeline order

1. Load FIF, EDF, BDF, or BrainVision recordings with MNE.
2. Apply the configured 50 Hz notch and 0.5–40 Hz band-pass filters using causal minimum-phase FIR filtering, then resample to 250 Hz.
3. Detect noisy/flat EEG channels, set the reference, and construct epochs ending at movement onset. The default epoch is **−1.0 to 0.0 seconds** with a baseline of −1.0 to −0.8 seconds. Positive epoch end times are rejected.
4. Fit ICA, label components with ICLabel when available, and apply the configured artifact exclusion thresholds (0.8 by default for eye, muscle, line-noise, and channel-noise classes).
5. Extract per-epoch log band-power features for delta, theta, alpha, and beta bands.

The default ICA method is Picard, with Infomax as a fallback if Picard cannot be used. If `mne-icalabel` is unavailable or classification fails, the pipeline reports that it used a heuristic fallback. The fallback is a triage aid, not a clinically validated artifact classifier.

## LOPO leakage rules

- The outer split is by participant. Every trial from the held-out participant stays out of training.
- In `preprocess_lopo_subject`, bad-channel detection is run on concatenated training recordings only.
- ICA is fitted on concatenated pre-movement epochs from training participants only.
- The resulting channel policy, ICA decomposition, and component exclusions are applied unchanged to the held-out participant.
- Do not fit ICA, choose bad channels, tune thresholds, select features, or estimate normalization statistics using the held-out participant.
- If channel layouts differ, resolve the montage/channel policy using training data and documented acquisition metadata; do not silently interpolate or learn a policy from the held-out participant.

The pipeline uses a pre-movement epoch boundary at or before time zero. This does not by itself prove that filtering is free of temporal leakage: zero-phase filtering can carry post-onset information backwards. For confirmatory work, evaluate causal filtering or a guarded pre-onset interval and document the choice.

## Installation

Use a supported Python environment (Python 3.10–3.12 is recommended for current scientific Python compatibility):

```bash
python -m pip install -r requirements-eeg.txt
```

MNE is a soft dependency for configuration-only imports. Actual loading, filtering, epoching, ICA, and feature extraction require MNE.

## Minimal API

```python
from src.eeg_preprocess import PreprocessConfig
from src.lopo_preprocess import iter_lopo_folds, preprocess_lopo_subject, save_report

config = PreprocessConfig()
for train_ids, held_out_id in iter_lopo_folds(subject_ids):
    result, report = preprocess_lopo_subject(
        train_ids + [held_out_id], held_out_id, recordings, config
    )
    save_report(report, f"reports/{held_out_id}.json")
```

Each recording is a mapping with `raw`, `events`, and optional `event_id`, or a tuple `(raw, events[, event_id])`. The LOPO result contains participant-indexed training features and held-out test features. It does not perform model fitting or calculate predictive metrics.

## Scope and limitations

This implementation does **not** download OpenNeuro ds006840, reconcile the reported valid-trial counts (6,671 versus 6,720), verify EEG–kinematic trial matching, define validated movement outcomes, or report predictive LOPO results. Those tasks remain upstream. No dataset-level results are fabricated or implied by passing unit tests.

Component labels from the heuristic fallback are approximate. Inspect component maps and spectra and document decisions before scientific use. Validate MNE and mne-icalabel versions against the target dataset and acquisition montage.
