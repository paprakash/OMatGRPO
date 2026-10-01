import pytest


class _NoUMA:
    """Stands in for the UMA potential so the reward constructor runs on CPU without the gated
    model. Tests that use it never call the potential."""

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def no_uma(monkeypatch):
    import omg.grpo.reward as reward_mod

    monkeypatch.setattr(reward_mod, "FairChemModel", _NoUMA)
    monkeypatch.setattr(reward_mod, "_DetachedFairChem", _NoUMA)
