import unittest
from unittest.mock import patch

import numpy as np

from src.eeg_preprocess import PreprocessConfig, epoch_premovement, filter_raw


class TestPreprocessConfig(unittest.TestCase):
    def test_preprocess_defaults_are_valid_and_pre_movement_only(self):
        config = PreprocessConfig()
        self.assertLess(config.l_freq, config.h_freq)
        self.assertLessEqual(config.tmax, 0)
        self.assertEqual(config.tmin, -1.0)
        self.assertEqual(config.tmax, 0.0)

    def test_positive_tmax_is_rejected_by_config(self):
        with self.assertRaisesRegex(ValueError, "tmax"):
            PreprocessConfig(tmax=0.01)

    def test_epoch_helper_rejects_positive_tmax_without_requiring_mne(self):
        with self.assertRaisesRegex(ValueError, "tmax"):
            epoch_premovement(None, events=[], tmax=0.01)

    def test_baseline_must_be_inside_epoch(self):
        with self.assertRaisesRegex(ValueError, "Baseline"):
            PreprocessConfig(tmin=-0.5)

    def test_positive_epoch_end_is_never_allowed(self):
        with self.assertRaisesRegex(ValueError, "tmax"):
            epoch_premovement(None, events=[], config=PreprocessConfig(), tmax=0.001)

    def test_filter_resamples_event_sample_indices_with_raw(self):
        class FakeRaw:
            def __init__(self):
                self.info = {"sfreq": 200.0}
                self.ch_names = ["Cz"]

            def notch_filter(self, **kwargs):
                return self

            def filter(self, **kwargs):
                return self

            def resample(self, sfreq, events=None, **kwargs):
                old_sfreq = self.info["sfreq"]
                self.info["sfreq"] = sfreq
                if events is None:
                    return self
                adjusted = events.copy()
                adjusted[:, 0] = np.rint(adjusted[:, 0] * sfreq / old_sfreq).astype(int)
                return self, adjusted

        raw = FakeRaw()
        events = np.array([[2000, 0, 1], [4000, 0, 2]])
        with patch("src.eeg_preprocess.mne", new=object()):
            filtered, resampled_events = filter_raw(raw, PreprocessConfig(), events=events)
        self.assertEqual(filtered.info["sfreq"], 250.0)
        self.assertEqual(resampled_events[:, 0].tolist(), [2500, 5000])
        self.assertEqual(events[:, 0].tolist(), [2000, 4000])


if __name__ == "__main__":
    unittest.main()
