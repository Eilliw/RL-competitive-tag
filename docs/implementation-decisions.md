# Implementation decisions

- `SingleAgentSimpleTagEnv` adapts 2-agent parallel env to Gymnasium single-agent training API.
- Opponent behavior is injected as a policy object for scripted/random/frozen-model use.
- PPO wrapper centralizes save/load/train behavior for both roles.
- `CheckpointLeague` provides a minimal latest/historical sampling foundation.
