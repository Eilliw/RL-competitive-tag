from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cat_mouse_rl.agents import PPOPolicy
from cat_mouse_rl.baselines import RandomPolicy, ScriptedEvaderPolicy, ScriptedTaggerPolicy
from cat_mouse_rl.envs.simple_tag import EVADER_NAME, TAGGER_NAME, SimpleTagConfig, SingleAgentSimpleTagEnv
from cat_mouse_rl.evaluation.metrics import evaluate_policy, save_results
from cat_mouse_rl.training.minimal import train_role


class SB3AdapterPolicy:
    def __init__(self, policy: PPOPolicy):
        self.policy = policy

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        return self.policy.predict(observation, deterministic=deterministic)


def _config_from_args(args: argparse.Namespace) -> SimpleTagConfig:
    return SimpleTagConfig(seed=args.seed, max_cycles=args.max_cycles, evader_speed_scale=args.evader_speed_scale)


def cmd_env_smoke(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    env = SingleAgentSimpleTagEnv(TAGGER_NAME, opponent_policy=ScriptedEvaderPolicy(), config=cfg)
    policy = RandomPolicy(action_space_n=env.action_space.n, seed=args.seed)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        while not done and not truncated:
            obs, _, done, truncated, _ = env.step(policy.act(obs))
            steps += 1
        print(f"episode={ep} steps={steps} done={done} truncated={truncated}")
    env.close()


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    result = train_role(args.role, args.output, args.timesteps, config=cfg, device=args.device)
    print(f"saved {result.role} checkpoint to {result.checkpoint_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    if args.role == "tagger":
        env = SingleAgentSimpleTagEnv(TAGGER_NAME, opponent_policy=ScriptedEvaderPolicy(), config=cfg)
    else:
        env = SingleAgentSimpleTagEnv(EVADER_NAME, opponent_policy=ScriptedTaggerPolicy(), config=cfg)
    model = PPOPolicy.load(args.model, env=env, device=args.device)
    summary = evaluate_policy(env, SB3AdapterPolicy(model), args.role, episodes=args.episodes, deterministic=args.deterministic)
    print(summary)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_results(summary, args.output)
        print(f"wrote results to {args.output}")
    env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cat and Mouse RL command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=7)
    common.add_argument("--max-cycles", type=int, default=200)
    common.add_argument("--evader-speed-scale", type=float, default=1.0)

    smoke = subparsers.add_parser("env-smoke", help="Run environment smoke test", parents=[common])
    smoke.add_argument("--episodes", type=int, default=3)
    smoke.set_defaults(func=cmd_env_smoke)

    train = subparsers.add_parser("train", help="Train a role-specific PPO policy", parents=[common])
    train.add_argument("--role", choices=["tagger", "evader"], required=True)
    train.add_argument("--timesteps", type=int, default=2000)
    train.add_argument("--output", type=str, required=True)
    train.add_argument("--device", type=str, default="cpu")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved PPO policy", parents=[common])
    evaluate.add_argument("--role", choices=["tagger", "evader"], required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--deterministic", action="store_true")
    evaluate.add_argument("--output", type=str)
    evaluate.add_argument("--device", type=str, default="cpu")
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
