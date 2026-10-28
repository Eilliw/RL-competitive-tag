from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from cat_mouse_rl.baselines.policies import BasePolicy, RandomPolicy

TAGGER_NAME = "adversary_0"
EVADER_NAME = "agent_0"


@dataclass
class SimpleTagConfig:
    seed: int | None = None
    max_cycles: int = 200
    continuous_actions: bool = False
    num_adversaries: int = 1
    num_good: int = 1
    num_obstacles: int = 0
    step_penalty_tagger: float = -0.01
    step_bonus_evader: float = 0.01
    terminate_on_capture: bool = True
    evader_speed_scale: float = 1.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def create_parallel_env(config: SimpleTagConfig):
    from mpe2 import simple_tag_v3

    kwargs = {
        "num_good": config.num_good,
        "num_adversaries": config.num_adversaries,
        "num_obstacles": config.num_obstacles,
        "max_cycles": config.max_cycles,
        "continuous_actions": config.continuous_actions,
        **config.extra_kwargs,
    }
    try:
        env = simple_tag_v3.parallel_env(render_mode=None, **kwargs)
    except TypeError:
        kwargs.pop("render_mode", None)
        env = simple_tag_v3.parallel_env(**kwargs)
    _apply_evader_speed(env, config.evader_speed_scale)
    return env


def _apply_evader_speed(env: Any, scale: float) -> None:
    if abs(scale - 1.0) < 1e-9:
        return
    world = None
    for attr in ("aec_env", "env"):
        candidate = getattr(env, attr, None)
        if candidate is not None and hasattr(candidate, "world"):
            world = candidate.world
            break
    if world is None:
        return
    for agent in getattr(world, "agents", []):
        if not getattr(agent, "adversary", False):
            if not hasattr(agent, "_base_max_speed"):
                agent._base_max_speed = getattr(agent, "max_speed", None)
            if agent._base_max_speed is not None:
                agent.max_speed = agent._base_max_speed * scale


class SingleAgentSimpleTagEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        focal_agent: str,
        opponent_policy: BasePolicy | None = None,
        config: SimpleTagConfig | None = None,
    ):
        super().__init__()
        self.config = config or SimpleTagConfig()
        self.focal_agent = focal_agent
        self.opponent_agent = EVADER_NAME if focal_agent == TAGGER_NAME else TAGGER_NAME
        self.opponent_policy = opponent_policy or RandomPolicy()
        self.env = create_parallel_env(self.config)
        self._last_obs: dict[str, np.ndarray] = {}
        self.capture_reward = 10.0 if focal_agent == TAGGER_NAME else -10.0

        self.env.reset(seed=self.config.seed)
        self.observation_space = self.env.observation_space(self.focal_agent)
        self.action_space = self.env.action_space(self.focal_agent)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, _ = self.env.reset(seed=seed or self.config.seed, options=options)
        self._last_obs = obs
        return obs[self.focal_agent], {}

    def step(self, action: int):
        if self.focal_agent not in self._last_obs:
            raise RuntimeError("Environment must be reset before stepping")

        opponent_obs = self._last_obs[self.opponent_agent]
        opponent_action = self.opponent_policy.act(np.asarray(opponent_obs), deterministic=True)
        actions = {self.focal_agent: action, self.opponent_agent: opponent_action}
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        done = bool(terminations.get(self.focal_agent, False))
        truncated = bool(truncations.get(self.focal_agent, False))
        capture = False

        reward = float(rewards[self.focal_agent])
        if abs(reward) >= abs(self.capture_reward):
            capture = True
            if self.config.terminate_on_capture:
                done = True

        if not done and not truncated:
            if self.focal_agent == TAGGER_NAME:
                reward += self.config.step_penalty_tagger
            else:
                reward += self.config.step_bonus_evader

        if done or truncated:
            self._last_obs = {}
            next_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        else:
            self._last_obs = obs
            next_obs = obs[self.focal_agent]

        info = dict(infos.get(self.focal_agent, {}))
        info["opponent_action"] = int(opponent_action)
        info["capture"] = capture
        return next_obs, reward, done, truncated, info

    def close(self):
        self.env.close()
