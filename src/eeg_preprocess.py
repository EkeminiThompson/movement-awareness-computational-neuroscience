"""Leakage-conscious EEG preprocessing for strictly pre-movement analysis.

MNE is an optional runtime dependency: importing configuration and helpers does not
require MNE to be installed. NumPy remains a lightweight required dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

try:
    import mne  # type: ignore
except ImportError:
    mne = None


@dataclass
class PreprocessConfig:
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freq: float = 50.0
    resample_sfreq: float = 250.0
    reference: str = "average"
    tmin: float = -1.0
    tmax: float = 0.0
    baseline: tuple[float, float] | None = (-1.0, -0.8)
    ica_method: str = "picard"
    eye_threshold: float = 0.8
    muscle_threshold: float = 0.8
    line_noise_threshold: float = 0.8
    channel_noise_threshold: float = 0.8
    ica_n_components: float | int = 0.99
    random_state: int = 97
    filter_method: str = "fir"
    filter_phase: str = "minimum"
    reject: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.l_freq <= 0 or self.h_freq <= self.l_freq:
            raise ValueError("Require 0 < l_freq < h_freq.")
        if self.tmax > 0:
            raise ValueError("tmax must be <= 0 for pre-movement epochs.")
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be earlier than tmax.")
        if self.resample_sfreq <= 0:
            raise ValueError("resample_sfreq must be positive.")
        if self.baseline is not None:
            if self.baseline[0] < self.tmin or self.baseline[1] > self.tmax:
                raise ValueError("Baseline must lie inside the configured epoch.")
            if self.baseline[0] > self.baseline[1]:
                raise ValueError("Baseline start must not exceed baseline end.")
        for name in ("eye_threshold", "muscle_threshold", "line_noise_threshold", "channel_noise_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_mne() -> Any:
    if mne is None:
        raise ImportError("EEG processing requires MNE. Install requirements-eeg.txt.")
    return mne


def load_raw(path: str | Path, preload: bool = True, **kwargs: Any) -> Any:
    """Load FIF, EDF, BDF, or BrainVision recordings with MNE."""
    lib = _require_mne()
    path = Path(path)
    suffix = path.suffix.lower()
    readers = {
        ".fif": lib.io.read_raw_fif,
        ".edf": lib.io.read_raw_edf,
        ".bdf": lib.io.read_raw_bdf,
        ".vhdr": lib.io.read_raw_brainvision,
    }
    if suffix not in readers:
        raise ValueError(f"Unsupported EEG format: {suffix}")
    return readers[suffix](str(path), preload=preload, **kwargs)


def filter_raw(raw: Any, config: PreprocessConfig, *,
               events: np.ndarray | None = None) -> Any:
    """Filter/resample Raw; if events are supplied, return their resampled indices too.

    MNE's Raw.resample(events=...) is essential: resampling EEG without resampling
    event sample indices shifts epoch onsets and can invalidate every trial.
    """
    _require_mne()
    raw_sfreq = float(raw.info["sfreq"])
    nyquist = raw_sfreq / 2.0
    if config.h_freq >= nyquist:
        raise ValueError(f"h_freq={config.h_freq} must be below input Nyquist ({nyquist}).")
    if config.notch_freq >= nyquist:
        raise ValueError(f"notch_freq={config.notch_freq} must be below input Nyquist ({nyquist}).")
    raw.notch_filter(freqs=[config.notch_freq], phase=config.filter_phase, verbose="ERROR")
    raw.filter(l_freq=config.l_freq, h_freq=config.h_freq,
               method=config.filter_method, phase=config.filter_phase, verbose="ERROR")
    if config.resample_sfreq and not np.isclose(raw_sfreq, config.resample_sfreq):
        if events is None:
            raw.resample(config.resample_sfreq, npad="auto", verbose="ERROR")
        else:
            raw, events = raw.resample(config.resample_sfreq, events=events,
                                       npad="auto", verbose="ERROR")
    return (raw, events) if events is not None else raw


def detect_bad_channels(raw: Any, *, return_scores: bool = False) -> Any:
    """Detect noisy/flat EEG channels; mutates raw.info['bads']."""
    lib = _require_mne()
    picks = lib.pick_types(raw.info, eeg=True, exclude=[])
    eeg_names = [raw.ch_names[i] for i in picks]
    if not eeg_names:
        raise ValueError("No EEG channels found.")
    bads: list[str] = []
    scores: dict[str, Any] = {}
    try:
        from mne.preprocessing import find_bad_channels_lof
        detected = find_bad_channels_lof(raw, picks="eeg", return_scores=True, verbose="ERROR")
        if isinstance(detected, tuple):
            bads, raw_scores = detected
            scores = dict(zip(eeg_names, np.asarray(raw_scores).tolist()))
        else:
            bads = list(detected)
    except (ImportError, TypeError, AttributeError, RuntimeError, ValueError):
        data = raw.get_data(picks=eeg_names)
        variances = np.nanvar(data, axis=1)
        median = np.nanmedian(variances)
        mad = np.nanmedian(np.abs(variances - median)) + np.finfo(float).eps
        z_scores = 0.6745 * (variances - median) / mad
        bads = [name for name, variance, z in zip(eeg_names, variances, z_scores)
                if abs(z) > 5 or variance < 1e-15]
        scores = dict(zip(eeg_names, z_scores.tolist()))
    raw.info["bads"] = sorted(set(raw.info.get("bads", [])) | set(bads))
    return (list(raw.info["bads"]), scores) if return_scores else list(raw.info["bads"])


def set_reference(raw: Any, config: PreprocessConfig) -> Any:
    _require_mne()
    if config.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif config.reference not in ("none", None):
        raw.set_eeg_reference(ref_channels=config.reference, projection=False, verbose="ERROR")
    return raw


def epoch_premovement(raw: Any, events: np.ndarray, event_id: Any = None,
                      config: PreprocessConfig | None = None, **kwargs: Any) -> Any:
    """Create epochs ending at movement onset (t=0); post-onset epochs are rejected."""
    config = config or PreprocessConfig()
    tmin = float(kwargs.pop("tmin", config.tmin))
    tmax = float(kwargs.pop("tmax", config.tmax))
    if tmax > 0:
        raise ValueError("tmax must be <= 0 for pre-movement epochs.")
    if tmin >= tmax:
        raise ValueError("tmin must be earlier than tmax.")
    baseline = kwargs.pop("baseline", config.baseline)
    if baseline is not None and (baseline[0] < tmin or baseline[1] > tmax):
        raise ValueError("Baseline must lie inside the requested pre-movement epoch.")
    if "events" in kwargs or "event_id" in kwargs:
        raise TypeError("Pass events and event_id through their named function arguments.")
    lib = _require_mne()
    return lib.Epochs(raw, events, event_id=event_id, tmin=tmin,
                      tmax=tmax, baseline=baseline, preload=True,
                      reject_by_annotation=True, picks="eeg", verbose="ERROR", **kwargs)


def fit_ica(epochs: Any, config: PreprocessConfig | None = None) -> Any:
    """Fit ICA on the supplied training epochs only."""
    _require_mne()
    config = config or PreprocessConfig()
    if config.tmax > 0 or epochs.times[-1] > 1e-9:
        raise ValueError("ICA input must contain pre-movement epochs only.")
    from mne.preprocessing import ICA
    methods = [config.ica_method]
    if config.ica_method == "picard":
        methods.append("infomax")
    last_error: Exception | None = None
    for method in methods:
        try:
            ica = ICA(n_components=config.ica_n_components, method=method,
                      random_state=config.random_state, max_iter="auto")
            ica.fit(epochs, picks="eeg", verbose="ERROR")
            ica._fit_method_used = method
            return ica
        except (ImportError, RuntimeError, ValueError) as exc:
            last_error = exc
            if method != methods[-1]:
                warnings.warn(f"ICA method {method!r} unavailable; trying infomax.", RuntimeWarning)
    raise RuntimeError(f"Could not fit ICA using {methods}: {last_error}") from last_error


def label_ica_components(ica: Any, epochs: Any = None) -> Any:
    try:
        from .ica_labeling import label_components
    except ImportError:
        from ica_labeling import label_components
    return label_components(ica, epochs=epochs)


def select_exclude_indices(labels: Any, config: PreprocessConfig | None = None) -> list[int]:
    try:
        from .ica_labeling import exclude_from_labels
    except ImportError:
        from ica_labeling import exclude_from_labels
    config = config or PreprocessConfig()
    return exclude_from_labels(labels, eye_threshold=config.eye_threshold,
        muscle_threshold=config.muscle_threshold,
        line_noise_threshold=config.line_noise_threshold,
        channel_noise_threshold=config.channel_noise_threshold)


def apply_ica(ica: Any, instance: Any, exclude: Sequence[int] | None = None,
              copy: bool = True) -> Any:
    target = instance.copy() if copy else instance
    fitted_channels = set(getattr(ica, "ch_names", []) or [])
    available_channels = set(getattr(target, "ch_names", []) or [])
    missing = sorted(fitted_channels - available_channels)
    if missing:
        raise ValueError(f"Recording is missing ICA training channels: {missing}")
    ica.apply(target, exclude=list(exclude if exclude is not None else ica.exclude), verbose="ERROR")
    return target


def preprocess_recording(raw: Any, events: np.ndarray, event_id: Any = None,
                         config: PreprocessConfig | None = None, *,
                         fit_artifacts: bool = True, fitted: Mapping[str, Any] | None = None,
                         **epoch_kwargs: Any) -> dict[str, Any]:
    """Preprocess one recording; held-out data must reuse training-fitted artifacts."""
    config = config or PreprocessConfig()
    raw = raw.copy().load_data()
    raw, events = filter_raw(raw, config, events=events)
    if fit_artifacts:
        bads = detect_bad_channels(raw)
        raw.info["bads"] = bads
    else:
        if fitted is None or fitted.get("ica") is None:
            raise ValueError("Held-out preprocessing requires training-fitted ICA artifacts.")
        raw.info["bads"] = list(fitted.get("bads", []))
    set_reference(raw, config)
    epochs = epoch_premovement(raw, events, event_id, config, **epoch_kwargs)
    if fit_artifacts:
        ica = fit_ica(epochs, config)
        labels = label_ica_components(ica, epochs)
        exclude = select_exclude_indices(labels, config)
        ica.exclude = exclude
    else:
        ica = fitted["ica"]
        labels = fitted.get("labels")
        exclude = list(fitted.get("exclude", ica.exclude))
    cleaned = apply_ica(ica, epochs, exclude=exclude)
    report = {
        "bads": list(raw.info["bads"]), "ica_method": getattr(ica, "_fit_method_used", "pre-fitted"),
        "n_components": int(ica.n_components_), "excluded_components": list(exclude),
        "n_epochs": len(cleaned), "config": config.to_dict(),
        "fit_artifacts": fit_artifacts,
    }
    return {"raw": raw, "epochs": cleaned, "ica": ica, "labels": labels,
            "exclude": list(exclude), "bads": list(raw.info["bads"]), "report": report}


def extract_band_power_features(epochs: Any,
                                bands: Mapping[str, tuple[float, float]] | None = None,
                                epsilon: float = 1e-20) -> tuple[np.ndarray, list[str]]:
    """Return per-epoch, per-channel log mean spectral power for delta/theta/alpha/beta."""
    bands = bands or {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    data = epochs.get_data(picks="eeg")
    if data.ndim != 3 or data.shape[-1] < 2:
        raise ValueError("Expected EEG epochs with shape (epochs, channels, time).")
    sfreq = float(epochs.info["sfreq"])
    freqs = np.fft.rfftfreq(data.shape[-1], 1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(data, axis=-1)) ** 2 / data.shape[-1]
    channel_names = list(epochs.copy().pick("eeg").ch_names)
    feature_blocks: list[np.ndarray] = []
    names: list[str] = []
    for band, (low, high) in bands.items():
        if low <= 0 or high <= low or high > sfreq / 2:
            raise ValueError(f"Invalid band {band}: {(low, high)} for sfreq={sfreq}")
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            raise ValueError(f"No FFT bins fall inside band {band}: {(low, high)}.")
        power = np.mean(spectrum[..., mask], axis=-1)
        feature_blocks.append(np.log(power + epsilon))
        names.extend([f"{band}_{channel}" for channel in channel_names])
    return np.concatenate(feature_blocks, axis=1), names
