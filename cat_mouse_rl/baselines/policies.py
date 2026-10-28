from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4


class BasePolicy:
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        raise NotImplementedError


@dataclass
class RandomPolicy(BasePolicy):
    action_space_n: int = 5
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        return int(self.rng.integers(0, self.action_space_n))


class _ScriptedPolicy(BasePolicy):
    @staticmethod
    def _relative_vector(observation: np.ndarray) -> tuple[float, float]:
        if observation.shape[0] < 6:
            return 0.0, 0.0
        dx, dy = float(observation[4]), float(observation[5])
        return dx, dy

    @staticmethod
    def _to_action(dx: float, dy: float, invert: bool = False) -> int:
        if invert:
            dx, dy = -dx, -dy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return ACTION_STAY
        if abs(dx) >= abs(dy):
            return ACTION_RIGHT if dx > 0 else ACTION_LEFT
        return ACTION_UP if dy > 0 else ACTION_DOWN


class ScriptedTaggerPolicy(_ScriptedPolicy):
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        dx, dy = self._relative_vector(observation)
        return self._to_action(dx, dy)


class ScriptedEvaderPolicy(_ScriptedPolicy):
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        dx, dy = self._relative_vector(observation)
        return self._to_action(dx, dy, invert=True)
