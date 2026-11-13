import numpy as np

from src.ica_labeling import ICLABEL_ORDER, LabelResult, exclude_from_labels


def test_high_eye_probability_excludes_component():
    probabilities = np.zeros((2, 7), dtype=float)
    probabilities[0, ICLABEL_ORDER.index("eye")] = 0.91
    probabilities[0, ICLABEL_ORDER.index("brain")] = 0.09
    probabilities[1, ICLABEL_ORDER.index("brain")] = 0.95
    probabilities[1, ICLABEL_ORDER.index("other")] = 0.05
    result = LabelResult(["eye", "brain"], probabilities, source="test")
    assert exclude_from_labels(result) == [0]


def test_iclabel_order_has_seven_classes():
    assert len(ICLABEL_ORDER) == 7
