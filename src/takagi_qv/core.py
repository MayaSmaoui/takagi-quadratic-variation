"""Core definitions for the normalised constant-coefficient Takagi function."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def distance_to_integers(values: ArrayLike) -> NDArray[np.float64]:
    """Return ``dist(values, Z)`` elementwise.

    Computing from the fractional part avoids the tie-handling behaviour of
    rounding functions and gives the periodic tent function used throughout
    the report.
    """
    array = np.asarray(values, dtype=float)
    fractional_part = array - np.floor(array)
    return np.minimum(fractional_part, 1.0 - fractional_part)


def takagi_partial_sum(t: ArrayLike, n: int) -> NDArray[np.float64]:
    r"""Evaluate the first ``n`` levels of the function.

    .. math::

       x_n(t)=\sum_{m=0}^{n-1}2^{-m/2}\,\operatorname{dist}(2^m t,\mathbb Z).
    """
    n = _nonnegative_integer(n, "n")
    points = np.asarray(t, dtype=float)
    values = np.zeros_like(points, dtype=float)

    for m in range(n):
        values += 2.0 ** (-m / 2.0) * distance_to_integers((2**m) * points)

    return values


def quadratic_sum(values: ArrayLike) -> float:
    """Return the sum of squared consecutive increments."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("values must be a one-dimensional array of length at least two.")
    increments = np.diff(array)
    return float(np.dot(increments, increments))


def regular_partition(base: int, level: int) -> NDArray[np.float64]:
    """Return ``{j / base**level : 0 <= j <= base**level}``."""
    if not isinstance(base, (int, np.integer)) or base < 2:
        raise ValueError("base must be an integer at least two.")
    level = _nonnegative_integer(level, "level")
    number_of_intervals = int(base) ** level
    return np.linspace(0.0, 1.0, number_of_intervals + 1)

