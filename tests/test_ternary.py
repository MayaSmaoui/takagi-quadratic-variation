import numpy as np
import pytest

from takagi_qv import (
    exact_ternary_quadratic_sum,
    multiplicative_order_two_mod_power_three,
    partition_data,
    takagi_partial_sum,
    ternary_partial_sum_values,
)


@pytest.mark.parametrize("p", range(1, 8))
def test_multiplicative_order(p: int) -> None:
    modulus = 3**p
    order = multiplicative_order_two_mod_power_three(p)
    assert pow(2, order, modulus) == 1
    assert pow(2, order // 2, modulus) != 1
    if p > 1:
        assert pow(2, order // 3, modulus) != 1


def test_residue_evaluation_matches_direct_evaluation() -> None:
    p, n = 4, 9
    nodes = np.arange(3**p + 1) / 3**p
    direct = takagi_partial_sum(nodes, n)
    residue_based = ternary_partial_sum_values(p, n)
    np.testing.assert_allclose(residue_based, direct, rtol=0.0, atol=2e-13)


@pytest.mark.parametrize("p", range(1, 7))
def test_exact_formula_matches_controlled_truncation(p: int) -> None:
    data = partition_data(p, n_plot=12, n_full=70)
    difference = abs(exact_ternary_quadratic_sum(p) - data.full_approximation)
    assert difference <= max(5.0 * data.truncation_error_bound, 5e-12)


def test_first_reported_ternary_values() -> None:
    expected = {
        1: 2.5904120554,
        2: 1.3488251111,
        3: 0.8200694953,
    }
    for p, value in expected.items():
        assert exact_ternary_quadratic_sum(p) == pytest.approx(value, abs=6e-10)

