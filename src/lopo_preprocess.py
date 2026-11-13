"""Leave-one-participant-out preprocessing with training-only artifact fitting."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .eeg_preprocess import (PreprocessConfig, _require_mne, apply_ica,
        detect_bad_channels, epoch_premovement, extract_band_power_features,
        filter_raw, fit_ica, label_ica_components, select_exclude_indices,
        set_reference)
except ImportError:
    from eeg_preprocess import (PreprocessConfig, _require_mne, apply_ica,
        detect_bad_channels, epoch_premovement, extract_band_power_features,
        filter_raw, fit_ica, label_ica_components, select_exclude_indices,
        set_reference)


@dataclass
class PreprocessReport:
    held_out_subject: str
    training_subjects: list[str]
    bad_channels: list[str]
    excluded_components: list[int]
    ica_method: str
    train_epoch_counts: dict[str, int] = field(default_factory=dict)
    test_epoch_count: int = 0
    feature_names: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_lopo_folds(subjects: Sequence[str]) -> Iterable[tuple[list[str], str]]:
    """Yield (training subject IDs, held-out subject ID) in input order."""
    ordered = list(dict.fromkeys(str(s) for s in subjects))
    if len(ordered) < 2:
        raise ValueError("LOPO requires at least two distinct participants.")
    for held_out in ordered:
        yield [subject for subject in ordered if subject != held_out], held_out


def _unpack_recording(record: Any) -> tuple[Any, np.ndarray, Any]:
    if isinstance(record, Mapping):
        return record["raw"], np.asarray(record["events"]), record.get("event_id")
    if isinstance(record, (tuple, list)) and len(record) >= 2:
        return record[0], np.asarray(record[1]), record[2] if len(record) > 2 else None
    raise TypeError("Each subject recording must be {raw, events, event_id?} or (raw, events[, event_id]).")


def _prepare_raw(raw: Any, config: PreprocessConfig, bads: Sequence[str] | None = None) -> Any:
    raw = raw.copy().load_data()
    filter_raw(raw, config)
    raw.info["bads"] = list(bads) if bads is not None else list(raw.info.get("bads", []))
    set_reference(raw, config)
    return raw


def preprocess_lopo_subject(subjects: Sequence[str], held_out_subject: str,
                            recordings: Mapping[str, Any],
                            config: PreprocessConfig | None = None) -> tuple[dict[str, Any], PreprocessReport]:
    """Fit bad-channel policy and ICA on training subjects, then transform held-out data.

    recordings maps each subject ID to {raw, events, event_id?} or
    (raw, events[, event_id]). No held-out samples enter artifact fitting.
    """
    lib = _require_mne()
    config = config or PreprocessConfig()
    if config.tmax > 0:
        raise ValueError("LOPO preprocessing is restricted to pre-movement epochs (tmax <= 0).")
    train_subjects = [str(s) for s in subjects if str(s) != str(held_out_subject)]
    if not train_subjects or held_out_subject not in recordings:
        raise ValueError("Provide training subjects and a recording for the held-out subject.")
    missing = [s for s in train_subjects if s not in recordings]
    if missing:
        raise KeyError(f"Missing training recordings: {missing}")

    # Detect channel outliers using only training participant recordings.
    train_raws = [_prepare_raw(_unpack_recording(recordings[s])[0], config) for s in train_subjects]
    combined_raw = lib.concatenate_raws([r.copy() for r in train_raws], preload=True, verbose="ERROR")
    bads = detect_bad_channels(combined_raw)
    # Retain only channels actually present in every training recording and test recording.
    test_raw, test_events, test_event_id = _unpack_recording(recordings[held_out_subject])
    common = set(train_raws[0].ch_names) & set(test_raw.ch_names)
    for r in train_raws[1:]:
        common &= set(r.ch_names)
    common_order = [ch for ch in train_raws[0].ch_names if ch in common]
    if not set(bads).issubset(common):
        bads = [ch for ch in bads if ch in common]
    if not common:
        raise ValueError("Training and held-out recordings have no common channels.")

    train_epochs = []
    train_counts: dict[str, int] = {}
    for subject, raw in zip(train_subjects, train_raws):
        raw.pick_channels(common_order, ordered=True)
        raw.info["bads"] = list(bads)
        events = _unpack_recording(recordings[subject])[1]
        event_id = _unpack_recording(recordings[subject])[2]
        epochs = epoch_premovement(raw, events, event_id, config)
        if len(epochs) == 0:
            raise ValueError(f"No valid pre-movement epochs for training participant {subject}.")
        train_counts[subject] = len(epochs)
        train_epochs.append(epochs)

    # The fitted ICA sees training epochs only; test epochs are never concatenated here.
    combined_epochs = lib.concatenate_epochs(train_epochs, add_offset=True, verbose="ERROR")
    ica = fit_ica(combined_epochs, config)
    labels = label_ica_components(ica, combined_epochs)
    exclude = select_exclude_indices(labels, config)
    ica.exclude = list(exclude)

    train_features: dict[str, np.ndarray] = {}
    feature_names: list[str] = []
    for subject, epochs in zip(train_subjects, train_epochs):
        cleaned = apply_ica(ica, epochs, exclude=exclude)
        features, names = extract_band_power_features(cleaned)
        train_features[subject] = features
        feature_names = names

    # Apply the exact training-derived channel list, reference policy and ICA to held-out data.
    test_raw = _prepare_raw(test_raw, config, bads=bads)
    test_raw.pick_channels(common_order, ordered=True)
    test_epochs = epoch_premovement(test_raw, test_events, test_event_id, config)
    cleaned_test = apply_ica(ica, test_epochs, exclude=exclude)
    test_features, test_names = extract_band_power_features(cleaned_test)
    if feature_names != test_names:
        raise RuntimeError("Feature schema differs between training and held-out participants.")

    report = PreprocessReport(
        held_out_subject=str(held_out_subject), training_subjects=train_subjects,
        bad_channels=list(bads), excluded_components=list(exclude),
        ica_method=str(getattr(ica, "_fit_method_used", "unknown")),
        train_epoch_counts=train_counts, test_epoch_count=len(cleaned_test),
        feature_names=feature_names, config=config.to_dict(),
        notes=["Bad-channel detection and ICA fitting used training participants only.",
               "Held-out participant was transformed with training-fitted ICA; no held-out refitting performed."])
    result = {"train_features": train_features, "test_features": test_features,
              "test_epochs": cleaned_test, "ica": ica, "labels": labels,
              "exclude": list(exclude), "bads": list(bads),
              "feature_names": feature_names}
    return result, report


def save_report(report: PreprocessReport | Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON preprocessing report and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if isinstance(report, PreprocessReport) else dict(report)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path
