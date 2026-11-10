"""Native trajectory optimization problems and launcher tables."""

from . import gym
from .gym import Problem, _entry, launcher_mass

__all__ = ["Problem", "SimsFlanaganChain", "gym", "launcher_mass"]


def __getattr__(name):
    if name == "SimsFlanaganChain":
        return _entry(name)
    raise AttributeError(name)
