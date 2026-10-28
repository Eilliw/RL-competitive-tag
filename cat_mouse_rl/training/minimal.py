from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cat_mouse_rl.agents import PPOPolicy
from cat_mouse_rl.baselines import ScriptedEvaderPolicy, ScriptedTaggerPolicy
from cat_mouse_rl.envs.simple_tag import EVADER_NAME, TAGGER_NAME, SimpleTagConfig, SingleAgentSimpleTagEnv


@dataclass
class TrainResult:
    role: str
    checkpoint_path: Path


def train_role(
    role: str,
    output_path: str | Path,
    timesteps: int,
    config: SimpleTagConfig | None = None,
    device: str = "cpu",
) -> TrainResult:
    cfg = config or SimpleTagConfig()
    if role == "tagger":
        env = SingleAgentSimpleTagEnv(TAGGER_NAME, opponent_policy=ScriptedEvaderPolicy(), config=cfg)
    elif role == "evader":
        env = SingleAgentSimpleTagEnv(EVADER_NAME, opponent_policy=ScriptedTaggerPolicy(), config=cfg)
    else:
        raise ValueError("role must be either 'tagger' or 'evader'")

    agent = PPOPolicy(env=env, device=device)
    agent.train(total_timesteps=timesteps)
    checkpoint = Path(output_path)
    agent.save(checkpoint)
    env.close()
    return TrainResult(role=role, checkpoint_path=checkpoint)
