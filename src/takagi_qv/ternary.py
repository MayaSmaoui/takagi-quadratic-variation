"""Exact and controlled computations on regular ternary partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .core import quadratic_sum


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def tent_values_from_residues(
    residues: NDArray[np.integer], denominator: int
) -> NDArray[np.float64]:
    """Evaluate ``dist(residues / denominator, Z)`` using integers."""
    distances = np.minimum(residues, denominator - residues)
    return distances.astype(float) / float(denominator)


def ternary_partial_sum_values(p: int, n: int) -> NDArray[np.float64]:
    """Evaluate ``x_n`` at every node of the level-``p`` ternary partition.

    Residues are updated modulo ``3**p``.  This remains stable for large
    truncation levels, unlike direct formation of ``2**m * t`` in floating
    point arithmetic.
    """
    p = _positive_integer(p, "p")
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer.")

    denominator = 3**p
    residues = np.arange(denominator + 1, dtype=np.int64) % denominator
    values = np.zeros(denominator + 1, dtype=float)

    for m in range(int(n)):
        values += 2.0 ** (-m / 2.0) * tent_values_from_residues(
            residues, denominator
        )
        residues = (2 * residues) % denominator

    return values


def multiplicative_order_two_mod_power_three(p: int) -> int:
    r"""Return ``ord_{3**p}(2) = 2 * 3**(p-1)``."""
    p = _positive_integer(p, "p")
    return 2 * 3 ** (p - 1)


def effective_tent_period(p: int) -> int:
    r"""Return the period ``3**(p-1)`` after the symmetry ``phi(u)=phi(-u)``."""
    return multiplicative_order_two_mod_power_three(p) // 2


def exact_ternary_node_values(p: int) -> NDArray[np.float64]:
    """Evaluate the limiting function exactly up to floating-point arithmetic.

    The orbit of ``dist(2**m j / 3**p, Z)`` has period ``3**(p-1)``.  The
    infinite sum can therefore be grouped into finitely many geometric series.
    """
    p = _positive_integer(p, "p")
    denominator = 3**p
    period = effective_tent_period(p)

    residues = np.arange(denominator + 1, dtype=np.int64) % denominator
    one_period_values = np.zeros(denominator + 1, dtype=float)
    inverse_square_root_two = 2.0 ** (-0.5)
    weight = 1.0

    for _ in range(period):
        one_period_values += weight * tent_values_from_residues(
            residues, denominator
        )
        residues = (2 * residues) % denominator
        weight *= inverse_square_root_two

    geometric_factor = 1.0 / (1.0 - 2.0 ** (-period / 2.0))
    return geometric_factor * one_period_values


def exact_ternary_quadratic_sum(p: int) -> float:
    """Return the finite-period evaluation of ``Q_{T_p}(x)``."""
    return quadratic_sum(exact_ternary_node_values(p))


def uniform_tail_bound(n: int) -> float:
    """Bound ``||x - x_n||_infinity`` by its geometric majorant."""
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer.")
    return 0.5 * 2.0 ** (-int(n) / 2.0) / (1.0 - 2.0 ** (-0.5))


def quadratic_sum_truncation_bound(values: NDArray[np.float64], n: int) -> float:
    """Bound ``|Q(x)-Q(x_n)|`` on the partition underlying ``values``."""
    increments = np.diff(values)
    increment_error = 2.0 * uniform_tail_bound(n)
    number_of_intervals = increments.size
    return float(
        2.0 * increment_error * np.sum(np.abs(increments))
        + number_of_intervals * increment_error**2
    )


@dataclass(frozen=True)
class PartitionData:
    """Numerical data for one ternary partition level."""

    q_curve: NDArray[np.float64]
    full_approximation: float
    truncation_error_bound: float


def partition_data(p: int, n_plot: int = 35, n_full: int = 80) -> PartitionData:
    """Compute ``Q_{T_p}(x_n)`` and a controlled approximation of ``Q_{T_p}(x)``."""
    p = _positive_integer(p, "p")
    n_plot = _positive_integer(n_plot, "n_plot")
    n_full = _positive_integer(n_full, "n_full")
    if n_full < n_plot:
        raise ValueError("n_full must be at least n_plot.")

    denominator = 3**p
    residues = np.arange(denominator + 1, dtype=np.int64) % denominator
    values = np.zeros(denominator + 1, dtype=float)
    q_curve = np.empty(n_plot, dtype=float)

    for m in range(n_full):
        values += 2.0 ** (-m / 2.0) * tent_values_from_residues(
            residues, denominator
        )
        if m < n_plot:
            q_curve[m] = quadratic_sum(values)
        residues = (2 * residues) % denominator

    approximation = quadratic_sum(values)
    error_bound = quadratic_sum_truncation_bound(values, n_full)

    return PartitionData(q_curve, approximation, error_bound)

