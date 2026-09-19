import numpy as np

from scripts.validate_rknn import summarize_error


def test_summarize_error_reports_exact_match_as_zero():
    values = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
    summary = summarize_error(values, values.copy())

    assert summary["max_abs"] == 0.0
    assert summary["rmse"] == 0.0
    assert summary["max_relative"] == 0.0


def test_summarize_error_uses_stable_relative_denominator():
    reference = np.array([[0.0, 2.0]], dtype=np.float32)
    actual = np.array([[1.0, 3.0]], dtype=np.float32)
    summary = summarize_error(reference, actual)

    assert summary["max_abs"] == 1.0
    assert np.isclose(summary["rmse"], 1.0)
    assert summary["max_relative"] == 0.5
