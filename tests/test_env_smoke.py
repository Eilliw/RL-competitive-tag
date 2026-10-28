import pytest

from cat_mouse_rl.baselines import RandomPolicy
from cat_mouse_rl.envs import SimpleTagConfig, SingleAgentSimpleTagEnv
from cat_mouse_rl.envs.simple_tag import TAGGER_NAME


@pytest.mark.parametrize("seed", [0, 1])
def test_simple_tag_env_resets_and_steps(seed):
    pytest.importorskip("mpe2")
    env = SingleAgentSimpleTagEnv(TAGGER_NAME, opponent_policy=RandomPolicy(seed=seed), config=SimpleTagConfig(seed=seed))
    obs, info = env.reset()
    assert obs is not None
    assert info == {}

    action = env.action_space.sample()
    next_obs, reward, done, truncated, step_info = env.step(action)
    assert next_obs is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert "opponent_action" in step_info
    env.close()
