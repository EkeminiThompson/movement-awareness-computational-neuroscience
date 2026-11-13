import unittest

import numpy as np

from src.eeg_preprocess import PreprocessConfig, select_exclude_indices
from src.ica_labeling import ICLABEL_ORDER, LabelResult, exclude_from_labels


class TestICAlabeling(unittest.TestCase):
    def test_high_eye_probability_excludes_component(self):
        probabilities = np.zeros((2, 7), dtype=float)
        probabilities[0, ICLABEL_ORDER.index("eye")] = 0.91
        probabilities[0, ICLABEL_ORDER.index("brain")] = 0.09
        probabilities[1, ICLABEL_ORDER.index("brain")] = 0.95
        probabilities[1, ICLABEL_ORDER.index("other")] = 0.05
        result = LabelResult(["eye", "brain"], probabilities, source="test")
        self.assertEqual(exclude_from_labels(result), [0])

    def test_iclabel_order_has_seven_classes(self):
        self.assertEqual(len(ICLABEL_ORDER), 7)

    def test_package_level_exclusion_helper_uses_config_thresholds(self):
        probabilities = np.zeros((1, 7), dtype=float)
        probabilities[0, ICLABEL_ORDER.index("muscle")] = 0.81
        probabilities[0, ICLABEL_ORDER.index("brain")] = 0.19
        result = LabelResult(["muscle"], probabilities)
        self.assertEqual(select_exclude_indices(result, PreprocessConfig()), [0])


if __name__ == "__main__":
    unittest.main()
