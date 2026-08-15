"""Computations for the constant-coefficient normalised Takagi function."""

from .core import (
    distance_to_integers,
    quadratic_sum,
    regular_partition,
    takagi_partial_sum,
)
from .ternary import (
    exact_ternary_node_values,
    exact_ternary_quadratic_sum,
    multiplicative_order_two_mod_power_three,
    partition_data,
    ternary_partial_sum_values,
)

__all__ = [
    "distance_to_integers",
    "exact_ternary_node_values",
    "exact_ternary_quadratic_sum",
    "multiplicative_order_two_mod_power_three",
    "partition_data",
    "quadratic_sum",
    "regular_partition",
    "takagi_partial_sum",
    "ternary_partial_sum_values",
]

