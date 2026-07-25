"""Type declarations for the native pykep-rust extension."""

from collections.abc import Sequence
from typing import Final

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

def port_status() -> str:
    """Return the current implementation status of the native core."""

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
