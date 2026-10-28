from pathlib import Path

from cat_mouse_rl.self_play import CheckpointLeague


def test_league_latest_and_historical_sampling():
    league = CheckpointLeague(seed=0)
    league.add("tagger", 1, Path("a.zip"))
    latest = league.add("tagger", 2, Path("b.zip"))

    assert league.sample("tagger", strategy="latest") == latest
    sample = league.sample("tagger", strategy="historical")
    assert sample.role == "tagger"
