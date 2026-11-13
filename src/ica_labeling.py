"""ICLabel integration and conservative rule-based fallback for ICA components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

ICLABEL_ORDER = ("brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other")


@dataclass
class LabelResult:
    labels: list[str]
    probabilities: np.ndarray
    source: str = "unknown"
    component_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        if self.probabilities.ndim != 2 or self.probabilities.shape[1] != len(ICLABEL_ORDER):
            raise ValueError(f"probabilities must have shape (n_components, {len(ICLABEL_ORDER)})")
        if len(self.labels) != self.probabilities.shape[0]:
            raise ValueError("labels and probabilities must have the same component count")
        if not self.component_indices:
            self.component_indices = list(range(len(self.labels)))

    def summary(self) -> dict[str, Any]:
        counts = {label: self.labels.count(label) for label in ICLABEL_ORDER}
        return {"source": self.source, "n_components": len(self.labels),
                "counts": counts,
                "components": [{"index": i, "label": label,
                    "probabilities": {name: float(p) for name, p in zip(ICLABEL_ORDER, row)}}
                    for i, (label, row) in enumerate(zip(self.labels, self.probabilities))]}


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, sums, out=np.full_like(values, 1 / values.shape[1]), where=sums > 0)


def _fallback_labels(ica: Any, epochs: Any = None) -> LabelResult:
    """Heuristic labels: useful for triage, not a substitute for validated ICLabel."""
    maps = np.asarray(ica.get_components(), dtype=float)
    n = maps.shape[1]
    probs = np.zeros((n, len(ICLABEL_ORDER)), dtype=float)
    probs[:, -1] = 0.55
    # Focal scalp maps are more likely ocular/muscle/noisy than broadly distributed maps.
    abs_maps = np.abs(maps)
    focality = abs_maps.max(axis=0) / (abs_maps.mean(axis=0) + 1e-12)
    spread = abs_maps.std(axis=0) / (abs_maps.mean(axis=0) + 1e-12)
    if epochs is not None:
        try:
            sources = ica.get_sources(epochs).get_data()
            sfreq = float(epochs.info["sfreq"])
            freqs = np.fft.rfftfreq(sources.shape[-1], 1 / sfreq)
            power = np.mean(np.abs(np.fft.rfft(sources, axis=-1)) ** 2, axis=(0, 1))
            high = power[(freqs >= 20) & (freqs <= min(40, sfreq / 2 - 1))].mean(axis=0) if power.ndim > 1 else None
            # Use per-component spectra when available.
            spec = np.mean(np.abs(np.fft.rfft(sources, axis=-1)) ** 2, axis=0)
            low = np.mean(spec[:, (freqs >= 1) & (freqs < 4)], axis=1)
            beta = np.mean(spec[:, (freqs >= 20) & (freqs < min(40, sfreq / 2))], axis=1)
            line_mask = np.abs(freqs - 50).argmin()
            line = spec[:, line_mask]
            total = np.mean(spec, axis=1) + 1e-12
            for i in range(n):
                if i < len(low):
                    probs[i, 1] = min(0.8, float(beta[i] / (total[i] + 1e-12)) * 2.5)
                    probs[i, 4] = min(0.8, float(line[i] / (total[i] + 1e-12)) * 10)
        except Exception:
            pass
    for i in range(n):
        if focality[i] > 5.0:
            probs[i, 2] = 0.65
        elif spread[i] > 1.0:
            probs[i, 1] = max(probs[i, 1], 0.45)
        probs[i, 0] = max(0.05, 0.35 - probs[i, 1] - probs[i, 2] - probs[i, 4])
    probs = _normalise_rows(probs)
    labels = [ICLABEL_ORDER[int(np.argmax(row))] for row in probs]
    return LabelResult(labels, probs, source="heuristic_fallback")


def label_components(ica: Any, epochs: Any = None) -> LabelResult:
    """Use mne-icalabel where available; otherwise return transparent heuristic labels."""
    try:
        from mne_icalabel import label_components as mne_label_components
        if epochs is None:
            raise ValueError("ICLabel requires an instance (epochs or raw) for classification.")
        result = mne_label_components(epochs, ica, method="iclabel")
        raw_labels = list(result["labels"])
        raw_probs = np.asarray(result["y_pred_proba"], dtype=float)
        # ICLabel class order is the documented ICLABEL_ORDER.
        labels = [str(label).lower().replace(" ", "_") for label in raw_labels]
        return LabelResult(labels, raw_probs, source="mne-icalabel")
    except (ImportError, ValueError, RuntimeError, KeyError, TypeError):
        return _fallback_labels(ica, epochs)


def exclude_from_labels(labels: LabelResult | Mapping[str, Any] | Sequence[str], *,
                        eye_threshold: float = 0.8, muscle_threshold: float = 0.8,
                        line_noise_threshold: float = 0.8,
                        channel_noise_threshold: float = 0.8) -> list[int]:
    """Return component indices exceeding configured non-brain class thresholds."""
    thresholds = {"eye": eye_threshold, "muscle": muscle_threshold,
                  "line_noise": line_noise_threshold,
                  "channel_noise": channel_noise_threshold}
    if isinstance(labels, LabelResult):
        probs = labels.probabilities
        return [i for i, row in enumerate(probs)
                if any(row[ICLABEL_ORDER.index(name)] >= threshold
                       for name, threshold in thresholds.items())]
    if isinstance(labels, Mapping):
        class_names = labels.get("classes", ICLABEL_ORDER)
        probabilities = np.asarray(labels.get("probabilities", labels.get("y_pred_proba")), dtype=float)
        if probabilities.ndim != 2:
            raise ValueError("Mapping must include a 2-D probabilities/y_pred_proba array.")
        return [i for i, row in enumerate(probabilities)
                if any(name in class_names and row[list(class_names).index(name)] >= threshold
                       for name, threshold in thresholds.items())]
    # Label-only fallback: conservatively exclude explicit artifact labels.
    return [i for i, label in enumerate(labels) if str(label).lower().replace(" ", "_") in thresholds]
