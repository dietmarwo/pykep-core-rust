from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

Backend: TypeAlias = Literal["taylor", "dop853"]
TimeEncoding: TypeAlias = Literal["uniform", "softmax"]
Decisions: TypeAlias = Sequence[float] | npt.NDArray[np.float64]

class Problem:
    def __init__(
        self, id: str, *, nseg: int | None = None, cut: float = 0.5,
        time_encoding: TimeEncoding = "uniform", inequalities_for_tc: bool | None = None,
        backend: Backend = "taylor", tolerance: float = 1e-13,
        derivative_tolerance: float = 1e-13, maximum_steps: int | None = None,
    ) -> None: ...
    def get_name(self) -> str: ...
    def get_extra_info(self) -> str: ...
    def get_bounds(self) -> tuple[list[float], list[float]]: ...
    def get_nobj(self) -> int: ...
    def get_nec(self) -> int: ...
    def get_nic(self) -> int: ...
    def has_gradient(self) -> bool: ...
    def metadata(self) -> dict[str, Any]: ...
    def fitness(self, x: Decisions) -> list[float]: ...
    def gradient(self, x: Decisions) -> list[float]: ...
    def evaluate(self, x: Decisions) -> dict[str, Any]: ...
    def fitness_batch(
        self, x: npt.NDArray[np.float64], workers: int = 1,
    ) -> npt.NDArray[np.float64]: ...
    def decode(self, x: Decisions, samples_per_segment: int = 0) -> dict[str, Any]: ...
    def forward_residual(self, x: Decisions) -> list[float]: ...

class _Factory(Protocol):
    def __call__(
        self, prob_name: str, cut: float = 0.5, nseg: int | None = None,
        time_encoding: TimeEncoding = "uniform", inequalities_for_tc: bool | None = None,
        *, backend: Backend = "taylor", tolerance: float = 1e-13,
        derivative_tolerance: float = 1e-13, maximum_steps: int | None = None,
    ) -> Problem: ...

tops_twobody: _Factory
tops_twobody_mb: _Factory
tops_mee: _Factory
tops_mee_mb: _Factory
tops_cr3bp: _Factory
tops_cr3bp_mb: _Factory
tops_ss: _Factory
tops_ss_mb: _Factory

cassini1: Problem
cassini1_a: Problem
cassini1_n: Problem
cassini2: Problem
eve_mga1dsm: Problem
eve_mga1dsm_a: Problem
eve_mga1dsm_n: Problem
rosetta: Problem
messenger: Problem
em3imp: Problem
em5imp: Problem
em7imp: Problem
juice: Problem
juice_mo: Problem

def ids() -> list[str]: ...
def launcher_mass(name: str, speed_kms: float, declination_deg: float = 0.0) -> float: ...
