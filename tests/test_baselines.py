import numpy as np

from cat_mouse_rl.baselines.policies import ACTION_LEFT, ACTION_RIGHT, RandomPolicy, ScriptedEvaderPolicy, ScriptedTaggerPolicy


def test_random_policy_action_bounds():
    policy = RandomPolicy(action_space_n=5, seed=1)
    actions = [policy.act(np.zeros(8)) for _ in range(50)]
    assert all(0 <= a < 5 for a in actions)


def test_scripted_tagger_moves_toward_positive_x():
    obs = np.array([0, 0, 0, 0, 1.0, 0.2, 0, 0], dtype=np.float32)
    assert ScriptedTaggerPolicy().act(obs) == ACTION_RIGHT


def test_scripted_evader_moves_away_from_positive_x():
    obs = np.array([0, 0, 0, 0, 1.0, 0.0, 0, 0], dtype=np.float32)
    assert ScriptedEvaderPolicy().act(obs) == ACTION_LEFT
