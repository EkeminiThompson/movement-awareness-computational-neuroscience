from src.eeg_preprocess import PreprocessConfig


def test_preprocess_defaults_are_valid_and_pre_movement_only():
    config = PreprocessConfig()
    assert config.l_freq < config.h_freq
    assert config.tmax <= 0
    assert config.tmin == -1.0
    assert config.tmax == 0.0


def test_positive_tmax_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="tmax"):
        PreprocessConfig(tmax=0.01)
