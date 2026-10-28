# Cat and Mouse — Competitive Multi-Agent RL

This repository contains the initial milestone for a CSCI 4922 deep learning research project on co-evolving pursuit and evasion strategies in the Farama MPE2 `simple_tag_v3` environment.

## What works now

- Installable Python package (`pyproject.toml`, Python 3.11+)
- MPE2 1v1 Simple Tag environment abstraction
- Baseline opponents:
  - Random policy
  - Scripted tagger (pursuit)
  - Scripted evader (flee)
- Role-specific PPO training loops (`tagger`, `evader`)
- Model checkpoint save/load
- Basic evaluation metrics and CSV/JSON export
- CLI entry points for smoke tests, training, and evaluation
- CI workflow running tests

## Repository structure

- `/cat_mouse_rl/envs`: MPE2 setup and single-agent adapter
- `/cat_mouse_rl/baselines`: random + scripted baseline policies
- `/cat_mouse_rl/agents`: PPO wrapper
- `/cat_mouse_rl/training`: minimal role-specific training
- `/cat_mouse_rl/evaluation`: metrics and result export
- `/cat_mouse_rl/self_play`: checkpoint league foundation
- `/configs`: example debug/short-run configs
- `/docs`: methodology and implementation notes
- `/tests`: focused smoke/unit tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
```

## CLI usage

Environment smoke test:

```bash
cat-mouse env-smoke --episodes 2 --seed 7
```

Train tagger against scripted evader (short run):

```bash
cat-mouse train --role tagger --timesteps 2000 --output checkpoints/tagger_smoke.zip --device cpu
```

Evaluate saved tagger checkpoint:

```bash
cat-mouse evaluate --role tagger --model checkpoints/tagger_smoke.zip --episodes 20 --deterministic --output results/tagger_eval.csv
```

## Reproducibility and constraints

- Seeds are configurable via CLI (`--seed`)
- CPU-only training is supported; use `--device cuda` if available
- Reward shaping is applied per non-terminal step (`-0.01` tagger / `+0.01` evader)
- Short smoke runs are for validation only and not final experimental quality

## Known limitations / next milestones

- Expand alternating self-play into full train-freeze-train loops
- Improve checkpoint league sampling/evaluation (round-robin tournaments, heatmaps)
- Add richer experiment tracking and plotting pipelines
- Extend curriculum schedule controls when/if full MPE2 curriculum hooks are exposed

## AI-use disclosure

This repository is being developed with AI coding assistance. Contributors should log:

- tool name/model,
- files changed,
- type of assistance (scaffolding, debugging, test writing, docs),
- and manual verification performed.

See `docs/ai-use-log-template.md`.
