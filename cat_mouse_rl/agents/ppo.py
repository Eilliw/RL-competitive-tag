from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO


class PPOPolicy:
    def __init__(self, env, device: str = "cpu", **ppo_kwargs: Any):
        policy_kwargs = ppo_kwargs.pop("policy_kwargs", {"net_arch": [64, 64]})
        self.model = PPO("MlpPolicy", env, device=device, policy_kwargs=policy_kwargs, verbose=0, **ppo_kwargs)

    def train(self, total_timesteps: int) -> None:
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation, deterministic: bool = True) -> int:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env, device: str = "cpu") -> "PPOPolicy":
        instance = cls(env=env, device=device)
        instance.model = PPO.load(str(path), env=env, device=device)
        return instance
