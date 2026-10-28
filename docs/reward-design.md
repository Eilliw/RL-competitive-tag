# Reward design

The environment uses native MPE2 rewards/capture signals. The wrapper adds per-step shaping only on non-terminal timesteps:

- tagger: `-0.01`
- evader: `+0.01`

This avoids double-counting collision/capture rewards in terminal transitions.
