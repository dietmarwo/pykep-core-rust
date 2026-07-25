"""Type declarations for the native pykep-rust extension."""

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

PI: Final[float]
HALF_PI: Final[float]
ASTRONOMICAL_UNIT: Final[float]
CAVENDISH_CONSTANT: Final[float]
MU_SUN: Final[float]
MU_EARTH: Final[float]
MU_MOON: Final[float]
EARTH_ORBITAL_VELOCITY: Final[float]
EARTH_J2: Final[float]
EARTH_RADIUS: Final[float]
DEGREES_TO_RADIANS: Final[float]
RADIANS_TO_DEGREES: Final[float]
DAY_TO_SECONDS: Final[float]
SECONDS_TO_DAY: Final[float]
JULIAN_YEAR_DAYS: Final[float]
DAYS_TO_JULIAN_YEAR: Final[float]
STANDARD_GRAVITY: Final[float]
CR3BP_MU_EARTH_MOON: Final[float]
BCP_MU_EARTH_MOON: Final[float]
BCP_MU_SUN: Final[float]
BCP_SUN_DISTANCE: Final[float]
BCP_SUN_ANGULAR_VELOCITY: Final[float]

class PykepError(RuntimeError):
    """Base exception for pykep-rust numerical failures."""

class ConvergenceError(PykepError):
    """An iterative numerical algorithm did not converge."""

class SingularGeometryError(PykepError):
    """The supplied geometry is singular for the requested operation."""

class UnsupportedCapabilityError(PykepError):
    """A provider does not implement a requested capability."""

class IntegrationError(PykepError):
    """A numerical integration could not be completed."""

class Optimality:
    """Indirect low-thrust optimality criterion."""

    Mass: Final[Optimality]
    Time: Final[Optimality]

class SimsFlanaganLeg:
    """Fixed-duration Sims–Flanagan low-thrust leg."""

    def __init__(
        self,
        departure_state: Sequence[float],
        departure_mass: float,
        throttles: Sequence[Sequence[float]],
        arrival_state: Sequence[float],
        arrival_mass: float,
        time_of_flight: float,
        maximum_thrust: float,
        exhaust_velocity: float,
        mu: float,
        cut: float = 0.5,
    ) -> None: ...
    def mismatch_constraints(self) -> list[float]: ...
    def throttle_constraints(self) -> list[float]: ...
    def mismatch_jacobian(
        self,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]: ...
    def throttle_jacobian(self) -> list[list[float]]: ...
    @property
    def segment_count(self) -> int: ...
    @property
    def forward_segment_count(self) -> int: ...
    @property
    def backward_segment_count(self) -> int: ...
    @property
    def cut(self) -> float: ...
    @property
    def time_of_flight(self) -> float: ...

class SimsFlanaganAlphaLeg:
    """Variable-duration Sims–Flanagan low-thrust leg."""

    def __init__(
        self,
        departure_state: Sequence[float],
        departure_mass: float,
        throttles: Sequence[Sequence[float]],
        segment_durations: Sequence[float],
        arrival_state: Sequence[float],
        arrival_mass: float,
        time_of_flight: float,
        maximum_thrust: float,
        exhaust_velocity: float,
        mu: float,
        cut: float = 0.5,
    ) -> None: ...
    @staticmethod
    def from_time_weights(
        departure_state: Sequence[float],
        departure_mass: float,
        throttles: Sequence[Sequence[float]],
        time_weights: Sequence[float],
        arrival_state: Sequence[float],
        arrival_mass: float,
        time_of_flight: float,
        maximum_thrust: float,
        exhaust_velocity: float,
        mu: float,
        cut: float = 0.5,
    ) -> SimsFlanaganAlphaLeg: ...
    def mismatch_constraints(self) -> list[float]: ...
    def throttle_constraints(self) -> list[float]: ...
    def throttle_jacobian(self) -> list[list[float]]: ...
    @property
    def segment_durations(self) -> list[float]: ...
    @property
    def segment_count(self) -> int: ...
    @property
    def forward_segment_count(self) -> int: ...
    @property
    def backward_segment_count(self) -> int: ...
    @property
    def cut(self) -> float: ...
    @property
    def time_of_flight(self) -> float: ...

class ZohModel:
    """Built-in dynamics for a generic zero-order-hold leg."""

    Kepler: Final[ZohModel]
    Cr3bp: Final[ZohModel]
    Equinoctial: Final[ZohModel]
    SolarSail: Final[ZohModel]

class ZohLeg:
    """Generic zero-order-hold low-thrust leg."""

    def __init__(
        self,
        model: ZohModel,
        initial_state: Sequence[float],
        controls: Sequence[Sequence[float]],
        final_state: Sequence[float],
        time_grid: Sequence[float],
        constants: Sequence[float],
        cut: float = 0.5,
        relative_tolerance: float = 1e-12,
        absolute_tolerance: float = 1e-12,
        maximum_step: float | None = None,
        maximum_steps: int = 100_000,
    ) -> None: ...
    def mismatch_constraints(self) -> list[float]: ...
    def mismatch_jacobian(
        self,
    ) -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[float]],
        list[list[float]],
    ]: ...
    def state_history(
        self, samples_per_segment: int = 2
    ) -> tuple[list[list[list[float]]], list[list[list[float]]]]: ...
    @staticmethod
    def mismatch_constraints_batch(legs: Sequence[ZohLeg]) -> list[list[float]]: ...
    @property
    def model(self) -> ZohModel: ...
    @property
    def state_dimension(self) -> int: ...
    @property
    def control_dimension(self) -> int: ...
    @property
    def segment_count(self) -> int: ...
    @property
    def forward_segment_count(self) -> int: ...
    @property
    def backward_segment_count(self) -> int: ...

class Epoch:
    """A microsecond-resolution proleptic-Gregorian epoch."""

    def __init__(self, value: float = 0.0, scale: str = "mjd2000") -> None:
        """Construct from an MJD2000, MJD, or JD day count."""

    @staticmethod
    def from_iso(text: str) -> Epoch:
        """Parse a cropped ISO calendar string."""

    @staticmethod
    def from_calendar(
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        millisecond: int = 0,
        microsecond: int = 0,
    ) -> Epoch:
        """Construct from validated calendar components."""

    @staticmethod
    def now() -> Epoch:
        """Return an epoch sampled from the current system clock."""

    @property
    def mjd2000(self) -> float:
        """Modified Julian Date 2000, in days."""

    @property
    def mjd(self) -> float:
        """Modified Julian Date, in days."""

    @property
    def jd(self) -> float:
        """Julian Date, in days."""

    @property
    def microseconds_since_mjd2000(self) -> int:
        """Signed internal microseconds from the MJD2000 origin."""

    def to_iso(self) -> str:
        """Return the canonical six-fractional-digit ISO representation."""

    def add_days(self, days: float) -> Epoch:
        """Return a new epoch offset by a finite number of days."""

    def sub_days(self, days: float) -> Epoch:
        """Return a new epoch offset backwards by a finite number of days."""

    def add_seconds(self, seconds: float) -> Epoch:
        """Return a new epoch offset by a finite number of seconds."""

    def seconds_since(self, other: Epoch) -> float:
        """Return self minus other in seconds."""

    def __add__(self, days: float) -> Epoch: ...
    def __sub__(self, days: float) -> Epoch: ...
    def __lt__(self, other: Epoch) -> bool: ...
    def __le__(self, other: Epoch) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __gt__(self, other: Epoch) -> bool: ...
    def __ge__(self, other: Epoch) -> bool: ...

class LambertSolution:
    """One branch in a deterministic Lambert solution family."""

    @property
    def departure_velocity(self) -> list[float]: ...
    @property
    def arrival_velocity(self) -> list[float]: ...
    @property
    def x(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def revolutions(self) -> int: ...
    @property
    def path(self) -> str: ...

class LambertProblem:
    """Solved single- or multi-revolution Lambert problem."""

    def __init__(
        self,
        initial_position: Sequence[float],
        final_position: Sequence[float],
        time: float,
        mu: float,
        clockwise: bool = False,
        maximum_revolutions: int = 1,
    ) -> None: ...
    @property
    def solutions(self) -> list[LambertSolution]: ...
    @property
    def maximum_revolutions(self) -> int: ...
    @property
    def initial_position(self) -> list[float]: ...
    @property
    def final_position(self) -> list[float]: ...
    @property
    def time(self) -> float: ...
    @property
    def mu(self) -> float: ...
    @property
    def clockwise(self) -> bool: ...

class Planet:
    """Thread-safe owner of an ephemeris provider."""

    @staticmethod
    def vsop2013(name: str, threshold: float = 1e-5) -> Planet: ...
    @staticmethod
    def vsop2013_available() -> bool: ...
    @staticmethod
    def vsop2013_minimum_threshold() -> float: ...
    @staticmethod
    def vsop2013_supported_bodies() -> list[str]: ...
    @staticmethod
    def jpl_low_precision(
        name: str, safe_radius: float | None = None
    ) -> Planet: ...
    @staticmethod
    def jpl_supported_bodies() -> list[str]: ...
    @staticmethod
    def keplerian_from_state(
        reference_epoch_mjd2000: float,
        state: Sequence[float],
        central_mu: float,
        name: str = "Unknown",
        body_mu: float | None = None,
        radius: float | None = None,
        safe_radius: float | None = None,
    ) -> Planet: ...
    @staticmethod
    def keplerian_from_classical(
        reference_epoch_mjd2000: float,
        elements: Sequence[float],
        central_mu: float,
        name: str = "Unknown",
        body_mu: float | None = None,
        radius: float | None = None,
        safe_radius: float | None = None,
    ) -> Planet: ...
    def state(self, epoch_mjd2000: float) -> list[float]: ...
    def states(
        self, epochs_mjd2000: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def acceleration(self, epoch_mjd2000: float) -> list[float]: ...
    def elements(
        self, epoch_mjd2000: float, representation: str = "classical_true"
    ) -> list[float]: ...
    def period(self, epoch_mjd2000: float) -> float | None: ...
    @property
    def name(self) -> str: ...
    @property
    def central_mu(self) -> float | None: ...
    @property
    def body_mu(self) -> float | None: ...
    @property
    def radius(self) -> float | None: ...
    @property
    def safe_radius(self) -> float | None: ...
    def has_acceleration(self) -> bool: ...

def port_status() -> str:
    """Return the current implementation status of the native core."""

def kepler_rhs(state: Sequence[float], mu: float) -> list[float]:
    """Evaluate two-body Cartesian dynamics."""

def cr3bp_rhs(state: Sequence[float], mu: float) -> list[float]:
    """Evaluate CR3BP dynamics in the synodic frame."""

def bcp_rhs(
    time: float,
    state: Sequence[float],
    mu: float,
    mu_sun: float,
    rho_sun: float,
    omega_sun: float,
) -> list[float]:
    """Evaluate time-dependent bicircular dynamics."""

def cr3bp_effective_potential(state: Sequence[float], mu: float) -> float:
    """Evaluate the positive CR3BP effective potential."""

def cr3bp_jacobi_constant(state: Sequence[float], mu: float) -> float:
    """Evaluate the CR3BP Jacobi constant."""

def pontryagin_cartesian_rhs(
    state: Sequence[float],
    optimality: Optimality,
    parameters: Sequence[float],
) -> list[float]:
    """Evaluate Cartesian Pontryagin state/costate dynamics."""

def pontryagin_equinoctial_rhs(
    state: Sequence[float],
    optimality: Optimality,
    parameters: Sequence[float],
) -> list[float]:
    """Evaluate modified-equinoctial Pontryagin state/costate dynamics."""

def pontryagin_cartesian_control(
    state: Sequence[float],
    optimality: Optimality,
    parameters: Sequence[float],
) -> tuple[float, list[float], float]:
    """Return throttle, Cartesian direction, and switching function."""

def pontryagin_equinoctial_control(
    state: Sequence[float],
    optimality: Optimality,
    parameters: Sequence[float],
) -> tuple[float, list[float], float]:
    """Return throttle, RTN direction, and switching function."""

def pontryagin_cartesian_hamiltonian(
    state: Sequence[float],
    optimality: Optimality,
    parameters: Sequence[float],
) -> float:
    """Evaluate the minimized Cartesian Pontryagin Hamiltonian."""

def pontryagin_equinoctial_hamiltonian(
    state: Sequence[float],
    optimality: Optimality,
    parameters: Sequence[float],
) -> float:
    """Evaluate the minimized modified-equinoctial Pontryagin Hamiltonian."""

def propagate_pontryagin_cartesian(
    state: Sequence[float],
    final_time: float,
    optimality: Optimality,
    parameters: Sequence[float],
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Propagate Cartesian Pontryagin state/costate dynamics."""

def propagate_pontryagin_equinoctial(
    state: Sequence[float],
    final_time: float,
    optimality: Optimality,
    parameters: Sequence[float],
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Propagate modified-equinoctial Pontryagin state/costate dynamics."""

def zoh_kepler_rhs(
    state: Sequence[float],
    thrust: float,
    direction: Sequence[float],
    mass_flow_coefficient: float,
) -> list[float]:
    """Evaluate normalized seven-state low-thrust Kepler dynamics."""

def zoh_cr3bp_rhs(
    state: Sequence[float],
    thrust: float,
    direction: Sequence[float],
    mass_flow_coefficient: float,
    mu: float,
) -> list[float]:
    """Evaluate seven-state low-thrust CR3BP dynamics."""

def zoh_equinoctial_rhs(
    state: Sequence[float],
    thrust: float,
    rtn_direction: Sequence[float],
    mass_flow_coefficient: float,
) -> list[float]:
    """Evaluate normalized modified-equinoctial low-thrust dynamics."""

def zoh_solar_sail_rhs(
    state: Sequence[float],
    alpha: float,
    beta: float,
    lightness: float,
) -> list[float]:
    """Evaluate normalized ideal solar-sail dynamics."""

def propagate_zoh_kepler(
    state: Sequence[float],
    boundaries: Sequence[float],
    controls: Sequence[Sequence[float]],
    mass_flow_coefficient: float,
    backward: bool = False,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Propagate a piecewise-constant low-thrust Kepler schedule."""

def propagate_zoh_cr3bp(
    state: Sequence[float],
    boundaries: Sequence[float],
    controls: Sequence[Sequence[float]],
    mass_flow_coefficient: float,
    mu: float,
    backward: bool = False,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Propagate a piecewise-constant low-thrust CR3BP schedule."""

def propagate_zoh_equinoctial(
    state: Sequence[float],
    boundaries: Sequence[float],
    controls: Sequence[Sequence[float]],
    mass_flow_coefficient: float,
    backward: bool = False,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Propagate a piecewise-constant modified-equinoctial schedule."""

def propagate_zoh_solar_sail(
    state: Sequence[float],
    boundaries: Sequence[float],
    controls: Sequence[Sequence[float]],
    lightness: float,
    backward: bool = False,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Propagate a piecewise-constant ideal solar-sail schedule."""

def jd_to_mjd(value: float) -> float:
    """Convert Julian date to modified Julian date, in days."""

def jd_to_mjd2000(value: float) -> float:
    """Convert Julian date to MJD2000, in days."""

def mjd_to_jd(value: float) -> float:
    """Convert modified Julian date to Julian date, in days."""

def mjd_to_mjd2000(value: float) -> float:
    """Convert modified Julian date to MJD2000, in days."""

def mjd2000_to_jd(value: float) -> float:
    """Convert MJD2000 to Julian date, in days."""

def mjd2000_to_mjd(value: float) -> float:
    """Convert MJD2000 to modified Julian date, in days."""

def stumpff_c(value: float) -> float:
    """Evaluate the dimensionless Stumpff C function."""

def stumpff_s(value: float) -> float:
    """Evaluate the dimensionless Stumpff S function."""

def stumpff_c_batch(values: Sequence[float]) -> list[float]:
    """Evaluate Stumpff C for a sequence in input order."""

def stumpff_s_batch(values: Sequence[float]) -> list[float]:
    """Evaluate Stumpff S for a sequence in input order."""

def mean_to_eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    """Convert elliptic mean anomaly to principal eccentric anomaly."""

def eccentric_to_mean_anomaly(
    eccentric_anomaly: float, eccentricity: float
) -> float:
    """Convert eccentric anomaly to elliptic mean anomaly."""

def eccentric_to_true_anomaly(
    eccentric_anomaly: float, eccentricity: float
) -> float:
    """Convert eccentric anomaly to principal true anomaly."""

def true_to_eccentric_anomaly(true_anomaly: float, eccentricity: float) -> float:
    """Convert true anomaly to principal eccentric anomaly."""

def mean_to_true_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    """Convert elliptic mean anomaly to principal true anomaly."""

def true_to_mean_anomaly(true_anomaly: float, eccentricity: float) -> float:
    """Convert true anomaly to elliptic mean anomaly."""

def gudermannian_to_true_anomaly(
    gudermannian_anomaly: float, eccentricity: float
) -> float:
    """Convert Gudermannian anomaly to hyperbolic true anomaly."""

def true_to_gudermannian_anomaly(
    true_anomaly: float, eccentricity: float
) -> float:
    """Convert hyperbolic true anomaly to Gudermannian anomaly."""

def hyperbolic_mean_to_anomaly(
    mean_anomaly: float, eccentricity: float
) -> float:
    """Convert hyperbolic mean anomaly to hyperbolic anomaly."""

def hyperbolic_anomaly_to_mean(
    hyperbolic_anomaly: float, eccentricity: float
) -> float:
    """Convert hyperbolic anomaly to hyperbolic mean anomaly."""

def hyperbolic_anomaly_to_true(
    hyperbolic_anomaly: float, eccentricity: float
) -> float:
    """Convert hyperbolic anomaly to principal true anomaly."""

def true_to_hyperbolic_anomaly(
    true_anomaly: float, eccentricity: float
) -> float:
    """Convert true anomaly to hyperbolic anomaly."""

def hyperbolic_mean_to_true(mean_anomaly: float, eccentricity: float) -> float:
    """Convert hyperbolic mean anomaly to principal true anomaly."""

def true_to_hyperbolic_mean(true_anomaly: float, eccentricity: float) -> float:
    """Convert true anomaly to hyperbolic mean anomaly."""

def mean_to_eccentric_anomaly_batch(
    mean_anomalies: Sequence[float], eccentricity: float
) -> list[float]:
    """Convert elliptic mean anomalies in input order."""

def hyperbolic_mean_to_anomaly_batch(
    mean_anomalies: Sequence[float], eccentricity: float
) -> list[float]:
    """Convert hyperbolic mean anomalies in input order."""

def elliptic_kepler_residual(
    eccentric_anomaly: float, mean_anomaly: float, eccentricity: float
) -> float:
    """Evaluate the elliptic Kepler residual in radians."""

def elliptic_kepler_derivative(
    eccentric_anomaly: float, eccentricity: float
) -> float:
    """Evaluate the first derivative of the elliptic Kepler residual."""

def elliptic_kepler_second_derivative(
    eccentric_anomaly: float, eccentricity: float
) -> float:
    """Evaluate the second derivative of the elliptic Kepler residual."""

def hyperbolic_kepler_residual(
    hyperbolic_anomaly: float, mean_anomaly: float, eccentricity: float
) -> float:
    """Evaluate the hyperbolic Kepler residual."""

def hyperbolic_kepler_derivative(
    hyperbolic_anomaly: float, eccentricity: float
) -> float:
    """Evaluate the first derivative of the hyperbolic Kepler residual."""

def hyperbolic_kepler_second_derivative(
    hyperbolic_anomaly: float, eccentricity: float
) -> float:
    """Evaluate the second derivative of the hyperbolic Kepler residual."""

def elliptic_difference_residual(
    delta_eccentric_anomaly: float,
    delta_mean_anomaly: float,
    sigma0: float,
    sqrt_semi_major_axis: float,
    semi_major_axis: float,
    initial_radius: float,
) -> float:
    """Evaluate Kepler's equation in elliptic anomaly difference."""

def elliptic_difference_derivative(
    delta_eccentric_anomaly: float,
    sigma0: float,
    sqrt_semi_major_axis: float,
    semi_major_axis: float,
    initial_radius: float,
) -> float:
    """Evaluate the first derivative of the elliptic difference residual."""

def elliptic_difference_second_derivative(
    delta_eccentric_anomaly: float,
    sigma0: float,
    sqrt_semi_major_axis: float,
    semi_major_axis: float,
    initial_radius: float,
) -> float:
    """Evaluate the second derivative of the elliptic difference residual."""

def hyperbolic_difference_residual(
    delta_hyperbolic_anomaly: float,
    delta_mean_anomaly: float,
    sigma0: float,
    sqrt_abs_semi_major_axis: float,
    semi_major_axis: float,
    initial_radius: float,
) -> float:
    """Evaluate Kepler's equation in hyperbolic anomaly difference."""

def hyperbolic_difference_derivative(
    delta_hyperbolic_anomaly: float,
    sigma0: float,
    sqrt_abs_semi_major_axis: float,
    semi_major_axis: float,
    initial_radius: float,
) -> float:
    """Evaluate the first derivative of the hyperbolic difference residual."""

def hyperbolic_difference_second_derivative(
    delta_hyperbolic_anomaly: float,
    sigma0: float,
    sqrt_abs_semi_major_axis: float,
    semi_major_axis: float,
    initial_radius: float,
) -> float:
    """Evaluate the second derivative of the hyperbolic difference residual."""

def universal_kepler_residual(
    delta_s: float,
    delta_time: float,
    initial_radius: float,
    initial_radial_velocity: float,
    alpha: float,
    mu: float,
) -> float:
    """Evaluate universal-variable Kepler's equation."""

def universal_kepler_derivative(
    delta_s: float,
    initial_radius: float,
    initial_radial_velocity: float,
    alpha: float,
    mu: float,
) -> float:
    """Evaluate the first derivative of universal-variable Kepler's equation."""

def universal_kepler_second_derivative(
    delta_s: float,
    initial_radius: float,
    initial_radial_velocity: float,
    alpha: float,
    mu: float,
) -> float:
    """Evaluate the second derivative of universal-variable Kepler's equation."""

def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute the Euclidean dot product of two three-vectors."""

def norm(vector: Sequence[float]) -> float:
    """Compute the Euclidean norm of a three-vector."""

def normalize(vector: Sequence[float]) -> list[float]:
    """Return a normalized three-vector."""

def cross(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    """Compute the right-handed cross product of two three-vectors."""

def skew(vector: Sequence[float]) -> list[list[float]]:
    """Return a row-major 3 by 3 skew-symmetric matrix."""

def cartesian_to_classical(state: Sequence[float], mu: float) -> list[float]:
    """Convert Cartesian state to classical [a,e,i,Omega,omega,nu] elements."""

def classical_to_cartesian(elements: Sequence[float], mu: float) -> list[float]:
    """Convert classical elements to Cartesian [x,y,z,vx,vy,vz] state."""

def classical_to_modified_equinoctial(
    elements: Sequence[float], retrograde: bool = False
) -> list[float]:
    """Convert classical elements to modified equinoctial [p,f,g,h,k,L]."""

def modified_equinoctial_to_classical(
    elements: Sequence[float], retrograde: bool = False
) -> list[float]:
    """Convert modified equinoctial elements to classical elements."""

def cartesian_to_modified_equinoctial(
    state: Sequence[float], mu: float, retrograde: bool = False
) -> list[float]:
    """Convert Cartesian state directly to modified equinoctial elements."""

def modified_equinoctial_to_cartesian(
    elements: Sequence[float], mu: float, retrograde: bool = False
) -> list[float]:
    """Convert modified equinoctial elements directly to Cartesian state."""

def cartesian_to_modified_equinoctial_jacobian(
    state: Sequence[float], mu: float, retrograde: bool = False
) -> list[list[float]]:
    """Return the row-major Cartesian-to-equinoctial analytic Jacobian."""

def modified_equinoctial_to_cartesian_jacobian(
    elements: Sequence[float], mu: float, retrograde: bool = False
) -> list[list[float]]:
    """Return the row-major equinoctial-to-Cartesian analytic Jacobian."""

def cartesian_to_classical_batch(
    states: npt.NDArray[np.float64], mu: float
) -> npt.NDArray[np.float64]:
    """Batch-convert an N by 6 array of Cartesian states."""

def classical_to_cartesian_batch(
    elements: npt.NDArray[np.float64], mu: float
) -> npt.NDArray[np.float64]:
    """Batch-convert an N by 6 array of classical elements."""

def classical_to_modified_equinoctial_batch(
    elements: npt.NDArray[np.float64], retrograde: bool = False
) -> npt.NDArray[np.float64]:
    """Batch-convert an N by 6 array of classical elements."""

def modified_equinoctial_to_classical_batch(
    elements: npt.NDArray[np.float64], retrograde: bool = False
) -> npt.NDArray[np.float64]:
    """Batch-convert an N by 6 array of modified equinoctial elements."""

def cartesian_to_modified_equinoctial_batch(
    states: npt.NDArray[np.float64],
    mu: float,
    retrograde: bool = False,
) -> npt.NDArray[np.float64]:
    """Batch-convert an N by 6 array of Cartesian states."""

def modified_equinoctial_to_cartesian_batch(
    elements: npt.NDArray[np.float64],
    mu: float,
    retrograde: bool = False,
) -> npt.NDArray[np.float64]:
    """Batch-convert an N by 6 array of modified equinoctial elements."""

def propagate_lagrangian(
    state: Sequence[float], time: float, mu: float
) -> list[float]:
    """Propagate a Cartesian state with Lagrange coefficients."""

def propagate_universal(
    state: Sequence[float], time: float, mu: float
) -> list[float]:
    """Propagate a Cartesian state with universal variables."""

def propagate_keplerian(
    state: Sequence[float], time: float, mu: float
) -> list[float]:
    """Propagate a Cartesian state by advancing its mean anomaly."""

def propagate_kepler_dynamics(
    state: Sequence[float],
    final_time: float,
    mu: float,
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Adaptively propagate evaluated two-body dynamics."""

def propagate_cr3bp(
    state: Sequence[float],
    final_time: float,
    mu: float,
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Adaptively propagate CR3BP dynamics."""

def propagate_bcp(
    state: Sequence[float],
    final_time: float,
    mu: float,
    mu_sun: float,
    rho_sun: float,
    omega_sun: float,
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> list[float]:
    """Adaptively propagate time-dependent bicircular dynamics."""

def propagate_kepler_dynamics_with_stm(
    state: Sequence[float],
    final_time: float,
    mu: float,
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Propagate evaluated two-body dynamics with a 6 by 6 STM."""

def propagate_cr3bp_with_stm(
    state: Sequence[float],
    final_time: float,
    mu: float,
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Propagate CR3BP dynamics with a 6 by 6 STM."""

def propagate_bcp_with_stm(
    state: Sequence[float],
    final_time: float,
    mu: float,
    mu_sun: float,
    rho_sun: float,
    omega_sun: float,
    initial_time: float = 0.0,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    maximum_step: float | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Propagate bicircular dynamics with a 6 by 6 STM."""

def propagate_lagrangian_with_stm(
    state: Sequence[float], time: float, mu: float
) -> tuple[list[float], list[list[float]]]:
    """Propagate and return the Lagrangian state-transition matrix."""

def state_transition_matrix_lagrangian(
    state: Sequence[float], time: float, mu: float
) -> list[list[float]]:
    """Return the row-major Lagrangian state-transition matrix."""

def state_transition_matrix_reynolds(
    initial_state: Sequence[float],
    final_state: Sequence[float],
    time: float,
    mu: float,
) -> list[list[float]]:
    """Return the row-major Reynolds state-transition matrix."""

def propagate_lagrangian_batch(
    states: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mu: float,
) -> npt.NDArray[np.float64]:
    """Propagate N states for N durations while releasing the GIL."""

def propagate_universal_batch(
    states: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    mu: float,
) -> npt.NDArray[np.float64]:
    """Universally propagate N states for N durations while releasing the GIL."""

def propagate_lagrangian_grid(
    state: Sequence[float],
    time_grid: npt.NDArray[np.float64],
    mu: float,
) -> npt.NDArray[np.float64]:
    """Propagate one state over a time grid relative to its first entry."""

def hohmann(r1: float, r2: float, mu: float) -> tuple[float, float, list[float]]:
    """Return total delta-v, duration, and impulses for a Hohmann transfer."""

def bielliptic(
    r1: float, r2: float, rb: float, mu: float
) -> tuple[float, float, list[float]]:
    """Return total delta-v, duration, and impulses for a bi-elliptic transfer."""

def alpha_to_direct(alphas: Sequence[float], total_time: float) -> list[float]:
    """Decode alpha decision variables into direct durations."""

def direct_to_alpha(times: Sequence[float]) -> tuple[list[float], float]:
    """Encode direct durations and return alphas plus total time."""

def eta_to_direct(etas: Sequence[float], maximum_time: float) -> list[float]:
    """Decode eta variables into direct durations."""

def direct_to_eta(times: Sequence[float], maximum_time: float) -> list[float]:
    """Encode direct durations into eta variables."""

def flyby_constraints(
    incoming: Sequence[float],
    outgoing: Sequence[float],
    mu: float,
    safe_radius: float,
) -> list[float]:
    """Return flyby equality and inequality constraints."""

def flyby_constraints_jacobian(
    incoming: Sequence[float],
    outgoing: Sequence[float],
    mu: float,
    safe_radius: float,
) -> list[list[float]]:
    """Return the row-major two-by-six flyby constraint Jacobian."""

def flyby_delta_v(
    incoming: Sequence[float],
    outgoing: Sequence[float],
    mu: float,
    safe_radius: float,
) -> float:
    """Return minimum powered-flyby delta-v."""

def flyby_outgoing_velocity(
    incoming: Sequence[float],
    planet_velocity: Sequence[float],
    periapsis_radius: float,
    beta: float,
    mu: float,
) -> list[float]:
    """Map an incoming velocity through an unpowered flyby."""

def mima(
    departure_delta_v: Sequence[float],
    arrival_delta_v: Sequence[float],
    time: float,
    maximum_thrust: float,
    effective_exhaust_velocity: float,
) -> tuple[float, float]:
    """Return maximum initial mass and characteristic acceleration."""

def mima2(
    initial_state: Sequence[float],
    departure_delta_v: Sequence[float],
    arrival_delta_v: Sequence[float],
    time: float,
    maximum_thrust: float,
    effective_exhaust_velocity: float,
    mu: float,
) -> tuple[float, float]:
    """Return the STM-based maximum mass and acceleration estimate."""
