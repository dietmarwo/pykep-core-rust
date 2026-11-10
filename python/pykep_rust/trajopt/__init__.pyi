from collections.abc import Sequence

from . import gym as gym
from .gym import Problem as Problem, launcher_mass as launcher_mass

class SimsFlanaganChain:
    """Fixed-duration multiple shooting with a sparse analytic Jacobian."""
    def __init__(
        self, departure: Sequence[float], arrival: Sequence[float],
        initial_mass: float, arcs: int, segments_per_arc: int,
        time_of_flight: float, maximum_thrust: float, exhaust_velocity: float,
        mu: float, cut: float = 0.5,
    ) -> None: ...
    @property
    def dimension(self) -> int: ...
    @property
    def jacobian_indptr(self) -> list[int]: ...
    @property
    def jacobian_indices(self) -> list[int]: ...
    @property
    def elements(self) -> list[list[int]]: ...
    def evaluate(self, x: Sequence[float]) -> tuple[float, list[float]]: ...
    def jacobian_values(self, x: Sequence[float]) -> list[float]: ...
    def lift(
        self, throttle: Sequence[Sequence[float]], arrival_mass: float,
        departure_excess: Sequence[float],
    ) -> list[float]: ...
