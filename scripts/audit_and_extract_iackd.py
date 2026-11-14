#!/usr/bin/env python3
"""Audit IACKD MATLAB derivatives and build leakage-controlled EEG/kinematic trials.

The pre-movement EEG predictors are recomputed from raw BrainVision recordings
with a one-pass causal IIR filter. Supplied zero-phase pre-movement EEG
derivatives are used only for trial-index metadata, never as predictors.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.integrate import trapezoid
from scipy.signal import butter, periodogram, sosfilt


def load_mat(path: Path) -> dict[str, Any]:
    try:
        import mat73
        return mat73.loadmat(str(path))
    except Exception as first_error:
        try:
            return loadmat(str(path), squeeze_me=True, struct_as_record=False)
        except Exception as second_error:
            raise RuntimeError(
                f"Cannot read MATLAB file {path}. mat73 error={first_error}; scipy error={second_error}"
            ) from second_error


def field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    return default


def scalar(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(value)
        except Exception:
            return default


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    arr = np.asarray(value).squeeze()
    if arr.size == 1:
        value = arr.item()
    elif arr.dtype.kind in {'U', 'S'}:
        value = ''.join(str(x) for x in arr.reshape(-1))
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return str(value).strip().strip(chr(34)).strip(chr(39))


def records(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        lengths = []
        for v in value.values():
            if isinstance(v, (str, bytes)):
                continue
            try:
                if np.asarray(v).ndim > 0:
                    lengths.append(len(v))
            except Exception:
                pass
        n = max(lengths, default=1)
        out = []
        for i in range(n):
            row = {}
            for k, v in value.items():
                try:
                    arr = np.asarray(v)
                    row[k] = arr[i] if arr.ndim > 0 and len(arr) == n else v
                except Exception:
                    row[k] = v
            out.append(row)
        return out
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [arr.item()]
    return list(arr.reshape(-1))


def vector(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float).squeeze().reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def normalise_label(value: Any) -> str:
    return text_value(value).lower().replace(" ", "").replace("_", "")


def read_out_records(path: Path) -> list[Any]:
    mat = load_mat(path)
    if "OUT" not in mat:
        raise KeyError(f"{path} has no OUT variable")
    return records(mat["OUT"])


def read_trial_info(path: Path) -> tuple[list[Any], float]:
    mat = load_mat(path)
    info = records(mat.get("trial_info"))
    fs = scalar(mat.get("RESAMPLE_HZ"), default=np.nan)
    if not np.isfinite(fs) or fs <= 0:
        fs = scalar(mat.get("srate"), default=200.0)
    if not info:
        raise ValueError(f"No trial_info records found in {path}")
    return info, fs


def trial_info_time_seconds(info: Any, fs: float) -> float:
    sample = scalar(field(info, "t14_samp", "t14_sample", "mark14_samp", "event14_sample"))
    if not np.isfinite(sample):
        return np.nan
    return sample / fs


def event_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"onset", "duration", "trial_type", "hand", "congruency", "ball_color"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing event columns: {missing}")
    return frame


def event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Match premove -> execution by event-14 time and trial labels, not row position."""
    pre = frame.loc[frame.trial_type.astype(str).str.lower() == "premove"].sort_values("onset")
    execution = frame.loc[frame.trial_type.astype(str).str.lower() == "execution"].sort_values("onset")
    hit = frame.loc[frame.trial_type.astype(str).str.lower() == "hit"].sort_values("onset")
    used_exec: set[int] = set()
    used_hit: set[int] = set()
    matched = []
    unmatched_premove = 0
    inconsistent_hit_timing = 0

    def labels(row: pd.Series) -> tuple[str, ...]:
        return tuple(normalise_label(row.get(k, "")) for k in
                     ("hand", "congruency", "direction", "ball_color"))

    exec_records = list(execution.iterrows())
    hit_records = list(hit.iterrows())
    for _, p in pre.iterrows():
        event14 = float(p.onset) + float(p.duration)
        candidates = [
            (abs(float(e.onset) - event14), pos, e)
            for pos, (_, e) in enumerate(exec_records)
            if pos not in used_exec and labels(e) == labels(p)
            and abs(float(e.onset) - event14) <= 0.05
        ]
        if not candidates:
            unmatched_premove += 1
            continue
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 1e-9:
            raise ValueError(f"Ambiguous execution event for premove at {p.onset}s")
        _, exec_pos, e = candidates[0]
        used_exec.add(exec_pos)

        hit_candidates = [
            (float(h.onset) - float(e.onset), pos, h)
            for pos, (_, h) in enumerate(hit_records)
            if pos not in used_hit and labels(h) == labels(p)
            and float(h.onset) >= float(e.onset) - 0.05
        ]
        hit_onset = np.nan
        hit_delay_ms = np.nan
        hit_found = False
        if hit_candidates:
            hit_candidates.sort(key=lambda x: x[0])
            delay, hit_pos, h = hit_candidates[0]
            if delay <= 7.0:
                used_hit.add(hit_pos)
                hit_onset = float(h.onset)
                hit_delay_ms = delay * 1000.0
                hit_found = True

        duration_field_ms = float(e.duration) * 1000.0
        duration_consistent = None
        if hit_found and duration_field_ms > 0:
            duration_consistent = abs(duration_field_ms - hit_delay_ms) <= 50.0
            if not duration_consistent:
                inconsistent_hit_timing += 1
        if duration_field_ms > 0:
            duration_ms = duration_field_ms
            duration_source = "events_tsv_execution_duration"
        elif hit_found:
            duration_ms = hit_delay_ms
            duration_source = "hit_onset_minus_execution_onset_fallback"
        else:
            duration_ms = np.nan
            duration_source = "unavailable"

        row = p.to_dict()
        row.update({
            "event14_s": event14,
            "event_execution_duration_ms": duration_ms,
            "event_execution_duration_field_ms": duration_field_ms,
            "event_duration_source": duration_source,
            "hit_onset_s": hit_onset,
            "event_hit_delay_ms": hit_delay_ms,
            "hit_event_found": hit_found,
            "event_hit_timing_consistent": duration_consistent,
            "trial_order_in_run": len(matched),
        })
        matched.append(row)

    unmatched_execution = len(exec_records) - len(used_exec)
    unmatched_hit = len(hit_records) - len(used_hit)
    if not matched:
        raise ValueError("No premove events could be matched to execution events by time and labels.")
    result = pd.DataFrame(matched).sort_values("event14_s").reset_index(drop=True)
    result.attrs["event_audit"] = {
        "n_premove_total": int(len(pre)),
        "n_execution_total": int(len(execution)),
        "n_hit_total": int(len(hit)),
        "n_premove_matched_to_execution": int(len(matched)),
        "n_premove_without_execution": int(unmatched_premove),
        "n_execution_unmatched": int(unmatched_execution),
        "n_hit_unmatched": int(unmatched_hit),
        "n_hit_timing_inconsistent_when_duration_present": int(inconsistent_hit_timing),
    }
    return result

def map_out_index_to_trial_info(out_record: Any, info: list[Any], fs: float,
                                events: pd.DataFrame) -> tuple[int, float, float]:
    """Resolve MATLAB's one-based EEG segment index against trial_info and event time."""
    raw_idx = scalar(field(out_record, "eeg_seg_index", "eeg_seg_idx", "eeg_seg_index"))
    if not np.isfinite(raw_idx):
        raise ValueError("OUT record has no eeg_seg_index/eeg_seg_idx")
    idx = int(raw_idx)
    valid_idx = [
        i for i, item in enumerate(info)
        if bool(scalar(field(item, "valid"), default=0))
    ]
    candidates = set()
    for j in (idx - 1, idx):
        if 0 <= j < len(info):
            candidates.add(j)
        if 0 <= j < len(valid_idx):
            candidates.add(valid_idx[j])
    scored = []
    event_times = events.event14_s.to_numpy(dtype=float)
    for info_i in candidates:
        tsec = trial_info_time_seconds(info[info_i], fs)
        if not np.isfinite(tsec):
            continue
        event_i = int(np.argmin(np.abs(event_times - tsec)))
        error = abs(float(event_times[event_i]) - tsec)
        scored.append((error, info_i, event_i, tsec))
    if not scored:
        raise ValueError(f"Cannot resolve eeg segment index {idx} to trial_info/event time")
    scored.sort()
    error, info_i, event_i, tsec = scored[0]
    if error > 0.05:
        raise ValueError(
            f"Trial identity mismatch for eeg segment {idx}: closest events.tsv time error={error:.4f}s"
        )
    if len(scored) > 1 and abs(scored[1][0] - error) < 1e-6 and scored[1][2] != event_i:
        raise ValueError(f"Ambiguous event match for EEG segment index {idx}")
    return info_i, event_i, error


def extract_kinematics(out_record: Any) -> dict[str, float]:
    t = vector(field(out_record, "t_ms"))
    x = vector(field(out_record, "x_mm"))
    y = vector(field(out_record, "y_mm"))
    z = vector(field(out_record, "z_mm"))
    if t.size == 0 or x.size != t.size or y.size != t.size or z.size != t.size:
        raise ValueError("OUT record has incompatible t_ms/x_mm/y_mm/z_mm arrays")
    good = (
        np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (t >= 0)
    )
    if good.sum() < 5:
        raise ValueError("Fewer than five valid post-onset 3-D hand-position samples")
    ids = np.flatnonzero(good)
    # Keep the longest contiguous valid run; do not bridge NaN gaps.
    breaks = np.where(np.diff(ids) > 1)[0]
    groups = np.split(ids, breaks + 1)
    ids = max(groups, key=len)
    if len(ids) < 5:
        raise ValueError("No sufficiently long contiguous 3-D kinematic segment")
    tt = t[ids]
    xyz = np.column_stack([x[ids], y[ids], z[ids]])
    dt = np.diff(tt) / 1000.0
    if np.any(dt <= 0):
        raise ValueError("Kinematic time axis is not strictly increasing")
    displacement = np.diff(xyz, axis=0)
    step = np.linalg.norm(displacement, axis=1)
    speed = step / dt
    return {
        "movement_duration_ms": float(tt[-1] - max(0.0, tt[0])),
        "path_length_mm": float(step.sum()),
        "peak_speed_mm_s": float(np.max(speed)),
        "kinematic_n_samples": int(len(ids)),
        "kinematic_dimensions": 3,
    }

def causal_filter(data_uv: np.ndarray, fs: float) -> np.ndarray:
    """One-pass causal Butterworth band-pass; no post-window samples influence the past."""
    sos = butter(4, [0.1, min(45.0, fs * 0.45)], btype="bandpass", fs=fs, output="sos")
    filtered = sosfilt(sos, data_uv, axis=-1)
    # Average reference is instantaneous and does not mix future samples.
    filtered = filtered - np.mean(filtered, axis=0, keepdims=True)
    return filtered


def eeg_features(epoch_uv: np.ndarray, fs: float, channel_names: list[str]) -> dict[str, float]:
    if epoch_uv.ndim != 2 or epoch_uv.shape[0] != len(channel_names):
        raise ValueError("EEG epoch has unexpected channel dimensions")
    if epoch_uv.shape[1] < int(0.90 * fs):
        raise ValueError(f"EEG pre-movement epoch too short: {epoch_uv.shape[1]} samples")
    if not np.isfinite(epoch_uv).all():
        raise ValueError("EEG epoch contains non-finite samples")
    freq, power = periodogram(epoch_uv, fs=fs, axis=-1, window="hann",
                              nfft=max(2048, 2 ** int(np.ceil(np.log2(epoch_uv.shape[-1])))),
                              scaling="density")
    bands = {
        "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
    }
    out: dict[str, float] = {}
    for ch_i, ch in enumerate(channel_names):
        safe_ch = re.sub(r"[^A-Za-z0-9]+", "_", ch).strip("_").lower()
        sig = epoch_uv[ch_i]
        out[f"eeg_{safe_ch}_std_uv"] = float(np.std(sig))
        out[f"eeg_{safe_ch}_rms_uv"] = float(np.sqrt(np.mean(sig ** 2)))
        out[f"eeg_{safe_ch}_ptp_uv"] = float(np.ptp(sig))
        out[f"eeg_{safe_ch}_line_length"] = float(np.mean(np.abs(np.diff(sig))))
        for band, (lo, hi) in bands.items():
            mask = (freq >= lo) & (freq < hi)
            bp = float(trapezoid(power[ch_i, mask], freq[mask])) if mask.sum() > 1 else np.nan
            out[f"eeg_{safe_ch}_{band}_logpower"] = float(np.log(max(bp, 1e-20)))
    return out


def process_run(dataset: Path, participant: str, hand: str, run: int,
                min_cov: float, max_start_res_ms: float) -> tuple[list[dict], dict]:
    subdir = dataset / participant
    acq = "left" if hand == "L" else "right"
    run_id = f"{run:02d}"
    prefix = f"{participant}_task-ihc_acq-{acq}_run-{run_id}"
    event_path = subdir / "eeg" / f"{prefix}_events.tsv"
    vhdr_path = subdir / "eeg" / f"{prefix}_eeg.vhdr"
    pre_path = subdir / "derivatives" / "premove" / participant / "ihc" / f"pre_eeg_{hand}{run}.mat"
    aligned_path = subdir / "derivatives" / "aligned" / participant / "ihc" / f"eeg_leap_{hand}{run}.mat"
    missing = [str(p) for p in (event_path, vhdr_path, pre_path, aligned_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required run files: " + "; ".join(missing))

    events = event_rows(event_table(event_path))
    info, info_fs = read_trial_info(pre_path)
    out_records = read_out_records(aligned_path)

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) < 8:
        raise ValueError(f"Only {len(picks)} EEG channels found in {vhdr_path}")
    channel_names = [raw.ch_names[i] for i in picks]
    raw_uv = raw.get_data(picks=picks) * 1e6
    filtered_uv = causal_filter(raw_uv, fs)

    rows = []
    matched_event_indices = set()
    for out_i, out_rec in enumerate(out_records):
        info_i, event_i, time_error = map_out_index_to_trial_info(
            out_rec, info, info_fs, events
        )
        if event_i in matched_event_indices:
            raise ValueError(f"Multiple OUT records map to the same events.tsv trial {event_i}")
        matched_event_indices.add(event_i)
        ev = events.iloc[event_i]
        start_s = float(ev.onset)
        end_s = float(ev.event14_s)
        start_sample = int(round(start_s * fs))
        end_sample = int(round(end_s * fs))
        if end_sample <= start_sample or end_sample - start_sample < int(0.90 * fs):
            raise ValueError(f"Invalid pre-movement interval for {participant} {hand}{run} trial {event_i}")
        if start_sample < 0 or end_sample > filtered_uv.shape[1]:
            raise ValueError(f"Pre-movement interval outside raw EEG recording for trial {event_i}")
        epoch = filtered_uv[:, start_sample:end_sample]
        feat = eeg_features(epoch, fs, channel_names)

        kin = extract_kinematics(out_rec)
        peak_abs_uv = float(np.max(np.abs(epoch)))
        eeg_qc_pass = bool(peak_abs_uv <= 300.0)
        cov = scalar(field(out_rec, "cov"), default=np.nan)
        residual = scalar(field(out_rec, "start_res_ms"), default=np.nan)
        eeg_idx = scalar(field(out_rec, "eeg_seg_index", "eeg_seg_idx"))
        lm_idx = scalar(field(out_rec, "im_trial", "lm_trial"), default=np.nan)
        color = text_value(field(out_rec, "ball_color")).lower()
        condition = str(ev.congruency).lower()
        if color and color not in {"unknown", "nan"}:
            expected = "congruent" if "red" in color else "incongruent" if "yellow" in color else ""
            if expected and expected != condition:
                raise ValueError(
                    f"Condition mismatch for {participant} {hand}{run} trial {event_i}: "
                    f"OUT ball_color={color}, events.tsv={condition}"
                )
        alignment_qc_pass = bool(
            np.isfinite(cov) and cov >= min_cov and
            np.isfinite(residual) and abs(residual) <= max_start_res_ms
        )
        passed = bool(alignment_qc_pass and eeg_qc_pass)
        row = {
            "participant_id": participant,
            "trial_id": f"{participant}_{hand}{run:02d}_{event_i:04d}",
            "run": run,
            "hand": hand,
            "condition": condition,
            "ball_color": str(ev.ball_color).lower(),
            "direction": str(ev.direction).lower(),
            "event14_s": end_s,
            "trial_info_event_time_error_s": time_error,
            "eeg_segment_index": eeg_idx,
            "leap_trial_index": lm_idx,
            "start_res_ms": residual,
            "coverage": cov,
            "passed_alignment_qc": alignment_qc_pass,
            "passed_eeg_qc": eeg_qc_pass,
            "passed_all_qc": passed,
            "qc_peak_abs_eeg_uv": peak_abs_uv,
            "raw_eeg_sampling_hz": fs,
            "causal_filter": "4th-order Butterworth 0.1-45 Hz, one-pass SOS; average reference",
            "kinematic_source": "aligned MATLAB OUT 3-D coordinates; dataset-provided 4-Hz zero-phase-smoothed trajectories",
            **kin,
            **feat,
        }
        rows.append(row)

    report = {
        "participant_id": participant, "hand": hand, "run": run,
        "n_events_premove": int(events.attrs.get("event_audit", {}).get("n_premove_total", len(events))),
        "n_premove_matched_to_execution": int(len(events)),
        "n_premove_without_execution": int(events.attrs.get("event_audit", {}).get("n_premove_without_execution", 0)),
        "n_execution_unmatched": int(events.attrs.get("event_audit", {}).get("n_execution_unmatched", 0)),
        "n_hit_unmatched": int(events.attrs.get("event_audit", {}).get("n_hit_unmatched", 0)),
        "n_hit_timing_inconsistent": int(events.attrs.get("event_audit", {}).get("n_hit_timing_inconsistent_when_duration_present", 0)),
        "n_trial_info": int(len(info)),
        "n_aligned_OUT": int(len(out_records)), "n_matched": int(len(matched_event_indices)),
        "n_alignment_qc_pass": int(sum(r["passed_alignment_qc"] for r in rows)),
        "n_alignment_qc_fail": int(sum(not r["passed_alignment_qc"] for r in rows)),
        "n_combined_qc_pass": int(sum(r["passed_all_qc"] for r in rows)),
        "n_combined_qc_fail": int(sum(not r["passed_all_qc"] for r in rows)),
        "n_eeg_amplitude_qc_pass": int(sum(r["passed_eeg_qc"] for r in rows)),
        "n_eeg_amplitude_qc_fail": int(sum(not r["passed_eeg_qc"] for r in rows)),
        "n_event_rows_unmatched": int(len(events) - len(matched_event_indices)),
        "raw_eeg_fs_hz": fs, "n_eeg_channels": len(channel_names),
        "event_file": str(event_path.relative_to(dataset)),
        "raw_eeg_file": str(vhdr_path.relative_to(dataset)),
        "premove_mat": str(pre_path.relative_to(dataset)),
        "aligned_mat": str(aligned_path.relative_to(dataset)),
    }
    return rows, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("reports/iackd"))
    ap.add_argument("--min-coverage", type=float, default=0.75)
    ap.add_argument("--max-start-residual-ms", type=float, default=30.0)
    ap.add_argument("--participants", nargs="*", default=[f"sub-{i:02d}" for i in range(1, 16)])
    args = ap.parse_args()

    dataset = args.dataset.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    description = json.loads((dataset / "dataset_description.json").read_text(encoding="utf-8"))
    participants_file = dataset / "participants.tsv"
    if not participants_file.exists():
        raise FileNotFoundError("participants.tsv missing from dataset root")
    expected = {f"sub-{i:02d}" for i in range(1, 16)}
    if not expected.issubset(set(pd.read_csv(participants_file, sep="\t").participant_id)):
        raise ValueError("Dataset participants.tsv does not list all 15 expected participants")

    all_rows: list[dict] = []
    run_reports: list[dict] = []
    errors: list[dict] = []
    for participant in args.participants:
        for hand in ("L", "R"):
            for run in range(1, 5):
                try:
                    rows, report = process_run(
                        dataset, participant, hand, run,
                        args.min_coverage, args.max_start_residual_ms
                    )
                    all_rows.extend(rows)
                    run_reports.append(report)
                except Exception as exc:
                    errors.append({
                        "participant_id": participant, "hand": hand, "run": run,
                        "error_type": type(exc).__name__, "error": str(exc),
                    })

    if not all_rows:
        raise RuntimeError(
            "No trial rows were extracted. Inspect audit_errors.json and verify the release layout."
        )
    table = pd.DataFrame(all_rows)
    table.to_csv(output / "all_matched_trials_pre_qc.csv", index=False)
    pd.DataFrame(run_reports).to_csv(output / "run_audit.csv", index=False)
    (output / "audit_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    qc = table.loc[table.passed_all_qc].copy()
    # Require both conditions and all 15 participants before the confirmatory model.
    participant_counts = qc.groupby("participant_id").size().to_dict()
    conditions = sorted(qc.condition.dropna().unique().tolist())
    summary = {
        "dataset_name": description.get("Name"),
        "dataset_version": description.get("DatasetVersion"),
        "dataset_doi": description.get("DatasetDOI"),
        "participants_expected": 15,
        "participants_with_qc_trials": len(participant_counts),
        "participant_trial_counts": participant_counts,
        "n_run_attempts": len(args.participants) * 8,
        "n_runs_successfully_audited": len(run_reports),
        "n_run_errors": len(errors),
        "n_out_records_extracted": int(len(table)),
        "n_qc_pass": int(len(qc)),
        "n_qc_fail": int(len(table) - len(qc)),
        "conditions": conditions,
        "n_congruent": int((qc.condition == "congruent").sum()),
        "n_incongruent": int((qc.condition == "incongruent").sum()),
        "trial_count_descriptor_reference": 6671,
        "trial_count_original_readme_reference": 6720,
        "count_discrepancy_resolved_by_this_audit": False,
        "raw_EEG_reprocessing": "Causal one-pass 0.1-45 Hz filtering; no supplied zero-phase pre-movement EEG used as predictor.",
        "kinematic_targets": ["movement_duration_ms", "path_length_mm", "peak_speed_mm_s"],
        "software": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "mne": mne.__version__,
        },
        "errors": errors,
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if errors or len(run_reports) != len(args.participants) * 8:
        raise RuntimeError(
            f"Only {len(run_reports)}/{len(args.participants) * 8} expected participant-hand-runs passed the audit; "
            "inspect audit_errors.json and resolve missing/failed runs before modelling."
        )
    if len(participant_counts) != 15:
        raise RuntimeError(
            f"Only {len(participant_counts)}/15 participants have QC-passing trials; "
            "do not run confirmatory LOPO."
        )
    if not {"congruent", "incongruent"}.issubset(set(conditions)):
        raise RuntimeError("Both congruent and incongruent conditions are required.")
    # The runner only receives EEG-prefixed predictors; identifiers and kinematic outcomes are labels.
    qc.to_csv(output / "audited_eeg_kinematics_features.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
