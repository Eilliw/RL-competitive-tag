from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from cat_mouse_rl.baselines import BasePolicy
from cat_mouse_rl.envs.simple_tag import SingleAgentSimpleTagEnv


@dataclass
class EvaluationSummary:
    role: str
    episodes: int
    capture_rate: float
    avg_capture_steps: float
    avg_survival_steps: float
    avg_episode_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_policy(
    env: SingleAgentSimpleTagEnv,
    policy: BasePolicy,
    role: str,
    episodes: int = 20,
    deterministic: bool = True,
) -> EvaluationSummary:
    rewards: list[float] = []
    capture_steps: list[int] = []
    survival_steps: list[int] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        ep_reward = 0.0

        while not done and not truncated:
            action = policy.act(np.asarray(obs), deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1

        rewards.append(ep_reward)
        survival_steps.append(step_count)
        if info.get("capture", False):
            capture_steps.append(step_count)

    return EvaluationSummary(
        role=role,
        episodes=episodes,
        capture_rate=len(capture_steps) / episodes,
        avg_capture_steps=float(np.mean(capture_steps)) if capture_steps else float("nan"),
        avg_survival_steps=float(np.mean(survival_steps)),
        avg_episode_reward=float(np.mean(rewards)),
    )


def save_results(summary: EvaluationSummary, path: str) -> None:
    frame = pd.DataFrame([summary.to_dict()])
    if path.endswith(".csv"):
        frame.to_csv(path, index=False)
    elif path.endswith(".json"):
        frame.to_json(path, orient="records", indent=2)
    else:
        raise ValueError("Output path must end with .csv or .json")
