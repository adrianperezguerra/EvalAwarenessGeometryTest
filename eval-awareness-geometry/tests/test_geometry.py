import json
import numpy as np
import pytest

from eval_awareness_prelim import (
    benchmark_direction, compare_datasets, cosine_sim, load_pairs_json,
    participation_ratio, project_out, residual_energy_fraction,
    run_mock, specificity_ratio, within_set_cosine_reference,
)


def test_direction_is_unit_and_points_toward_mean():
    x = np.array([[2., 0.], [4., 0.]])
    d = benchmark_direction(x)
    assert np.allclose(d, [1., 0.])
    assert np.isclose(np.linalg.norm(d), 1.)


def test_projection_removes_parallel_component():
    target = np.array([3., 4.])
    control = np.array([1., 0.])
    assert np.allclose(project_out(target, control), [0., 4.])
    assert np.isclose(specificity_ratio(target, control), 0.8)
    assert np.isclose(residual_energy_fraction(target, control), 0.64)


def test_cosine_checks_dimensions_and_zero_vectors():
    assert np.isclose(cosine_sim(np.array([1., 0.]), np.array([2., 0.])), 1.)
    with pytest.raises(ValueError):
        cosine_sim(np.array([1.]), np.array([1., 2.]))
    with pytest.raises(ValueError):
        cosine_sim(np.zeros(2), np.ones(2))


def test_participation_ratio_known_cases():
    x = np.array([[1., 0.], [2., 0.], [3., 0.]])
    assert np.isclose(participation_ratio(x), 1.)
    assert participation_ratio(np.zeros((3, 2))) == 0.


def test_centered_and_uncentered_are_distinct_diagnostics():
    x = np.array([[10., 0.], [10., 1.], [10., -1.]])
    assert not np.isclose(participation_ratio(x), participation_ratio(x, center=True))


def test_within_set_reference_skips_zero_rows():
    sims = within_set_cosine_reference(np.array([[1., 0.], [0., 0.], [0., 1.]]))
    assert sims.size == 1
    assert np.isclose(sims[0], 0.)


def test_compare_reports_consistent_energy_fraction():
    result = compare_datasets(np.array([[1., 0.], [2., 0.]]), np.array([[1., 1.], [2., 2.]]))
    assert np.isclose(result["cosine_eval_control"], 1 / np.sqrt(2))
    assert np.isclose(result["residual_energy_fraction"], 0.5)
    assert result["n_eval_pairs"] == 2


def test_validation_rejects_bad_arrays():
    with pytest.raises(ValueError):
        benchmark_direction(np.array([]))
    with pytest.raises(ValueError):
        participation_ratio(np.array([[np.nan, 1.]]))


def test_mock_mode_is_deterministic_and_synthetic():
    a, b = run_mock(seed=4), run_mock(seed=4)
    assert a == b
    assert a["mode"] == "mock_synthetic_only"


def test_pair_json_validation(tmp_path):
    path = tmp_path / "pairs.json"
    payload = {"eval_deploy": [{"a": "a", "b": "b"}], "control": [{"a": "c", "b": "d"}]}
    path.write_text(json.dumps(payload))
    a, b = load_pairs_json(path)
    assert a[0]["a"] == "a" and b[0]["b"] == "d"
    path.write_text('{"eval_deploy": [], "control": []}')
    with pytest.raises(ValueError):
        load_pairs_json(path)
