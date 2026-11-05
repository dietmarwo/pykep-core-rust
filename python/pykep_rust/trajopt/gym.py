"""Lazy Rust trajectory gym with the PyGMO problem protocol."""

from functools import partial
import json

from .. import _pykep_rust as _native


def _entry(name):
    try:
        return getattr(_native, name)
    except AttributeError:
        raise ImportError("this pykep-rust build requires the trajopt feature") from None


class Problem:
    """A named gym problem with raw minimized objectives and constraints."""

    def __init__(
        self, id, *, nseg=None, cut=0.5, time_encoding="uniform",
        inequalities_for_tc=None, backend="taylor", tolerance=1e-13,
        derivative_tolerance=1e-13,
    ):
        self._configuration = {
            "id": id, "nseg": nseg, "cut": cut, "time_encoding": time_encoding,
            "inequalities_for_tc": inequalities_for_tc, "backend": backend,
            "tolerance": tolerance, "derivative_tolerance": derivative_tolerance,
        }
        self._problem = _entry("_gym_problem")(
            id, nseg, cut, time_encoding, inequalities_for_tc, backend,
            tolerance, derivative_tolerance,
        )
        self._spec = json.loads(self._problem.spec_json())

    def __getstate__(self):
        return self._configuration.copy()

    def __setstate__(self, config):
        self.__init__(**config)

    def get_name(self):
        return self._spec["id"]

    def get_extra_info(self):
        return self._spec["name"]

    def get_bounds(self):
        return tuple([p[i] for p in self._spec["bounds"]] for i in (0, 1))

    def get_nobj(self):
        return len(self._spec["objectives"])

    def get_nec(self):
        return len(self._spec["equalities"])

    def get_nic(self):
        return len(self._spec["inequalities"])

    def has_gradient(self):
        return self._spec["has_gradient"]

    def metadata(self):
        return json.loads(self._problem.spec_json())

    def fitness(self, x):
        return self._problem.fitness(x)

    def gradient(self, x):
        """Return complete derivatives in fitness-row major order."""
        return self._problem.gradient(x)

    def evaluate(self, x):
        return json.loads(self._problem.evaluate_json(x))

    def fitness_batch(self, x, workers=1):
        """Evaluate a two-dimensional float64 array in input order."""
        return self._problem.fitness_batch(x, workers)

    def decode(self, x, samples_per_segment=0):
        return json.loads(self._problem.decode_json(x, samples_per_segment))

    def forward_residual(self, x):
        """Independently check TOPS terminal state by tighter forward DOP853."""
        return self._problem.forward_residual(x)


def ids():
    """Return the stable catalog identifiers."""
    return _entry("_gym_ids")()


def launcher_mass(name, speed_kms, declination_deg=0.0):
    """Evaluate an embedded launcher table in kg."""
    return _entry("_launcher_mass")(name, speed_kms, declination_deg)


def _tops(family, prob_name, cut=0.5, nseg=None, time_encoding="uniform",
          inequalities_for_tc=None, *, moving=False, **settings):
    suffix = ".moving" if moving else ""
    return Problem(f"tops.{family}.{prob_name}{suffix}", nseg=nseg, cut=cut,
                   time_encoding=time_encoding, inequalities_for_tc=inequalities_for_tc, **settings)


tops_twobody = partial(_tops, "twobody")
tops_mee = partial(_tops, "mee")
tops_cr3bp = partial(_tops, "cr3bp")
tops_ss = partial(_tops, "sail")
tops_twobody_mb = partial(_tops, "twobody", moving=True)
tops_mee_mb = partial(_tops, "mee", moving=True)
tops_cr3bp_mb = partial(_tops, "cr3bp", moving=True)
tops_ss_mb = partial(_tops, "sail", moving=True)

_CLASSICAL = frozenset((
    "cassini1", "cassini1_a", "cassini1_n", "cassini2", "eve_mga1dsm", "eve_mga1dsm_a",
    "eve_mga1dsm_n", "rosetta", "messenger", "em3imp", "em5imp", "em7imp", "juice", "juice_mo",
))


def __getattr__(name):
    if name in _CLASSICAL:
        problem = Problem(f"gym.{name}")
        globals()[name] = problem
        return problem
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | _CLASSICAL)
