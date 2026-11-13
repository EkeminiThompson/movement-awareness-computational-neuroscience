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
    ordered = list(dict.fromkeys(str(subject) for subject in subjects))
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


def _prepare_raw(raw: Any, events: np.ndarray, config: PreprocessConfig,
                 bads: Sequence[str] | None = None) -> tuple[Any, np.ndarray]:
    raw = raw.copy().load_data()
    raw, events = filter_raw(raw, config, events=events)
    raw.info["bads"] = list(bads) if bads is not None else list(raw.info.get("bads", []))
    return raw, events


def preprocess_lopo_subject(subjects: Sequence[str], held_out_subject: str,
                            recordings: Mapping[str, Any],
                            config: PreprocessConfig | None = None) -> tuple[dict[str, Any], PreprocessReport]:
    """Fit channel policy and ICA on training participants; transform held-out data unchanged.

    Channel selection is derived from training participants only. Every training and
    held-out event array is resampled together with its EEG recording before epoching.
    """
    lib = _require_mne()
    config = config or PreprocessConfig()
    if config.tmax > 0:
        raise ValueError("LOPO preprocessing is restricted to pre-movement epochs (tmax <= 0).")
    held_out_subject = str(held_out_subject)
    train_subjects = list(dict.fromkeys(str(s) for s in subjects if str(s) != held_out_subject))
    if not train_subjects or held_out_subject not in recordings:
        raise ValueError("Provide training subjects and a recording for the held-out subject.")
    missing = [s for s in train_subjects if s not in recordings]
    if missing:
        raise KeyError(f"Missing training recordings: {missing}")

    # Resample EEG and event sample indices together for each training participant.
    train_raws: dict[str, Any] = {}
    train_events: dict[str, np.ndarray] = {}
    train_event_ids: dict[str, Any] = {}
    for subject in train_subjects:
        raw, events, event_id = _unpack_recording(recordings[subject])
        prepared, resampled_events = _prepare_raw(raw, events, config)
        train_raws[subject] = prepared
        train_events[subject] = resampled_events
        train_event_ids[subject] = event_id

    # Channel selection is based on training layouts only; held-out data must contain
    # that exact schema rather than influencing the selection policy.
    common_train = set(train_raws[train_subjects[0]].ch_names)
    for subject in train_subjects[1:]:
        common_train.intersection_update(train_raws[subject].ch_names)
    common_order = [name for name in train_raws[train_subjects[0]].ch_names if name in common_train]
    if not common_order:
        raise ValueError("Training participants have no common channels.")

    reference_types = train_raws[train_subjects[0]].get_channel_types(picks=common_order)
    for subject in train_subjects[1:]:
        candidate_types = train_raws[subject].get_channel_types(picks=common_order)
        if candidate_types != reference_types:
            raise ValueError(f"Channel types/order differ for training participant {subject}.")
    test_raw_original, test_events_original, test_event_id = _unpack_recording(recordings[held_out_subject])
    missing_test_channels = [name for name in common_order if name not in test_raw_original.ch_names]
    if missing_test_channels:
        raise ValueError(
            "Held-out participant is missing training-defined channels: "
            f"{missing_test_channels}. Resolve montage compatibility before LOPO."
        )
    test_types = test_raw_original.get_channel_types(picks=common_order)
    if test_types != reference_types:
        raise ValueError("Held-out channel types/order do not match the training-defined schema.")
    for subject, raw in train_raws.items():
        raw.pick_channels(common_order, ordered=True)

    # Aggregate manual training bad-channel annotations, then detect additional
    # outliers on concatenated training recordings only.
    training_bads = sorted({name for raw in train_raws.values() for name in raw.info.get("bads", [])
                            if name in common_order})
    for raw in train_raws.values():
        raw.info["bads"] = list(training_bads)
    combined_raw = lib.concatenate_raws(
        [train_raws[s].copy() for s in train_subjects], preload=True, verbose="ERROR"
    )
    bads = detect_bad_channels(combined_raw)

    train_epochs: dict[str, Any] = {}
    train_counts: dict[str, int] = {}
    for subject in train_subjects:
        raw = train_raws[subject]
        raw.info["bads"] = list(bads)
        set_reference(raw, config)
        epochs = epoch_premovement(raw, train_events[subject],
                                   train_event_ids[subject], config)
        if len(epochs) == 0:
            raise ValueError(f"No valid pre-movement epochs for training participant {subject}.")
        train_counts[subject] = len(epochs)
        train_epochs[subject] = epochs

    # ICA sees training epochs only. Held-out EEG and events are not passed to fit_ica.
    combined_epochs = lib.concatenate_epochs(
        [train_epochs[s] for s in train_subjects], add_offset=True, verbose="ERROR"
    )
    ica = fit_ica(combined_epochs, config)
    labels = label_ica_components(ica, combined_epochs)
    exclude = select_exclude_indices(labels, config)
    ica.exclude = list(exclude)

    train_features: dict[str, np.ndarray] = {}
    feature_names: list[str] = []
    for subject in train_subjects:
        cleaned = apply_ica(ica, train_epochs[subject], exclude=exclude)
        features, names = extract_band_power_features(cleaned)
        if feature_names and names != feature_names:
            raise RuntimeError(f"Training feature schema differs for participant {subject}.")
        train_features[subject] = features
        feature_names = names

    # Transform the held-out recording with the exact training-derived channel,
    # reference, ICA and exclusion policy. Its event indices are resampled in lockstep.
    test_raw, test_events = _prepare_raw(test_raw_original, test_events_original,
                                         config, bads=bads)
    test_raw.pick_channels(common_order, ordered=True)
    set_reference(test_raw, config)
    cleaned_test_epochs = epoch_premovement(test_raw, test_events, test_event_id, config)
    if len(cleaned_test_epochs) == 0:
        raise ValueError(f"No valid pre-movement epochs for held-out participant {held_out_subject}.")
    cleaned_test = apply_ica(ica, cleaned_test_epochs, exclude=exclude)
    test_features, test_names = extract_band_power_features(cleaned_test)
    if feature_names != test_names:
        raise RuntimeError("Feature schema differs between training and held-out participants.")

    report = PreprocessReport(
        held_out_subject=held_out_subject, training_subjects=train_subjects,
        bad_channels=list(bads), excluded_components=list(exclude),
        ica_method=str(getattr(ica, "_fit_method_used", "unknown")),
        train_epoch_counts=train_counts, test_epoch_count=len(cleaned_test),
        feature_names=feature_names, config=config.to_dict(),
        notes=[
            "Bad-channel detection and ICA fitting used training participants only.",
            "Channel selection was derived from training participants; held-out montage was checked, not used to select channels.",
            "EEG and event sample indices were resampled together before epoching.",
            "Held-out participant was transformed with training-fitted ICA; no held-out refitting performed.",
        ])
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
