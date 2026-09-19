import numpy as np
import pytest

from scripts.rknn_pipeline import MODEL_SPECS, build_calibration_samples, validate_input_array


def test_model_specs_match_encoder_and_decoder_contracts():
    assert MODEL_SPECS["encoder"].input_name == "obs_dict"
    assert MODEL_SPECS["encoder"].input_shape == (1, 1247)
    assert MODEL_SPECS["encoder"].output_name == "encoded_tokens"
    assert MODEL_SPECS["encoder"].output_shape == (1, 64)
    assert MODEL_SPECS["decoder"].input_name == "obs_dict"
    assert MODEL_SPECS["decoder"].input_shape == (1, 994)
    assert MODEL_SPECS["decoder"].output_name == "action"
    assert MODEL_SPECS["decoder"].output_shape == (1, 29)


def test_calibration_samples_are_float32_and_have_fixed_batch_shape():
    samples = build_calibration_samples("decoder", count=7, seed=3)
    assert samples.dtype == np.float32
    assert samples.shape == (7, 1, 994)
    assert np.isfinite(samples).all()


def test_validate_input_array_rejects_wrong_shape_and_dtype():
    with pytest.raises(ValueError, match="shape"):
        validate_input_array("encoder", np.zeros((1, 994), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        validate_input_array("encoder", np.zeros((1, 1247), dtype=np.float64))
