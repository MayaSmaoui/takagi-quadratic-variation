import numpy as np
import pytest

from takagi_qv import (
    distance_to_integers,
    quadratic_sum,
    regular_partition,
    takagi_partial_sum,
)


def test_distance_to_integers() -> None:
    values = np.array([-0.25, 0.0, 0.25, 0.5, 0.75, 1.25])
    expected = np.array([0.25, 0.0, 0.25, 0.5, 0.25, 0.25])
    np.testing.assert_allclose(distance_to_integers(values), expected)


@pytest.mark.parametrize("p", range(1, 11))
def test_exact_dyadic_identity(p: int) -> None:
    partition = regular_partition(2, p)
    values = takagi_partial_sum(partition, p)
    assert quadratic_sum(values) == pytest.approx(1.0 - 2.0 ** (-p), abs=2e-12)


def test_endpoints_vanish() -> None:
    values = takagi_partial_sum(np.array([0.0, 1.0]), 20)
    np.testing.assert_array_equal(values, np.zeros(2))

