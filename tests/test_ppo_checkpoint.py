import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cat_mouse_rl.agents import PPOPolicy


class OneStepEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(4, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = True
        info = {}
        return obs, reward, terminated, truncated, info


def test_ppo_save_and_load_roundtrip(tmp_path):
    env = OneStepEnv()
    policy = PPOPolicy(env=env, device="cpu", n_steps=8, batch_size=4)
    policy.train(total_timesteps=8)

    path = tmp_path / "ppo_model.zip"
    policy.save(path)
    loaded = PPOPolicy.load(path, env=env, device="cpu")

    action = loaded.predict(np.zeros(4, dtype=np.float32), deterministic=True)
    assert 0 <= action < env.action_space.n
