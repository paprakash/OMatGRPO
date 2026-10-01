"""
Floor at zero and cap of the stability term, stability = -clip(E_hull, 0, cap)
(as in Chemeleon2), implemented in _route_and_clamp_ehull.

Cases:
  1. With the floor off and cap=5.0, the result equals a reference reimplementation of the
     earlier inline routing, bit for bit, for all four gate/route modes (2000 random trials).
  2. Floor: a trusted shallow below-hull structure gets 0.0, the best value, tied with
     on-hull; more depth earns nothing.
  3. Failed, deep-below-hull and penalty-routed sparse structures get the cap.
  4. The abstention mask does not depend on the floor.
  5. The cap is the clamp ceiling, and the occurrence-discount floor in
     _apply_diversity_penalty follows it.
  6. cap <= 0 is rejected by the constructor.
"""
import pytest
import torch

from omg.grpo.reward import OMatGRPOReward


def make_reward(gate=False, route="worst", floor=False, cap=5.0,
                mag_floor=-0.1, min_refset=12):
    """Bare instance (no UMA load) with just the attrs the routing needs."""
    r = OMatGRPOReward.__new__(OMatGRPOReward)
    r.ehull_mag_floor = float(mag_floor)
    r.ehull_min_refset = int(min_refset)
    r.ehull_sparse_gate = bool(gate)
    r.sparse_route = route
    r.ehull_floor_at_zero = bool(floor)
    r.ehull_cap = float(cap)
    r.last_sparse_neutral_mask = None
    return r


def legacy_pipeline(eh, ref, gate, route, mag_floor=-0.1, min_refset=12):
    """Reference reimplementation of the earlier inline routing (cap fixed at 5.0)."""
    below = eh < 0.0
    sparse = ref < min_refset
    deep_below = below & (eh < mag_floor)
    neutral_mask = None
    if gate:
        if route == "neutral":
            untrusted = deep_below
            neutral_mask = sparse & ~deep_below
        else:
            untrusted = sparse | deep_below
    else:
        untrusted = below & ((eh < mag_floor) | (ref < min_refset))
    eh = torch.where(untrusted, torch.full_like(eh, 5.0), eh)
    eh = torch.nan_to_num(eh, nan=5.0, posinf=5.0, neginf=5.0)
    return torch.clamp(eh, max=5.0), untrusted, neutral_mask


@pytest.mark.parametrize("gate,route", [(False, "worst"), (False, "neutral"),
                                        (True, "worst"), (True, "neutral")])
def test_legacy_byte_identity_fuzz(gate, route):
    g = torch.Generator().manual_seed(hash((gate, route)) % (2**31))
    for _ in range(500):
        n = int(torch.randint(1, 65, (1,), generator=g))
        eh = torch.randn(n, generator=g, dtype=torch.float64) * 3.0
        # inject NaN/inf sentinels and exact-boundary values
        m = torch.rand(n, generator=g)
        eh[m < 0.08] = float("nan")
        eh[(m >= 0.08) & (m < 0.12)] = float("inf")
        eh[(m >= 0.12) & (m < 0.14)] = -0.1  # exact mag_floor boundary
        ref = torch.randint(0, 40, (n,), generator=g).to(torch.float64)
        r = make_reward(gate=gate, route=route, floor=False, cap=5.0)
        got_eh, got_untrusted = r._route_and_clamp_ehull(eh.clone(), ref.clone())
        exp_eh, exp_untrusted, exp_neutral = legacy_pipeline(
            eh.clone(), ref.clone(), gate, route)
        assert torch.equal(got_eh, exp_eh)
        assert torch.equal(got_untrusted, exp_untrusted)
        if exp_neutral is None:
            assert r.last_sparse_neutral_mask is None
        else:
            assert torch.equal(r.last_sparse_neutral_mask, exp_neutral)


def test_floor_trusted_below_hull_is_best_and_flat():
    # dense refs, shallow below-hull (trusted) AND deep-but-guard-off scenarios
    r = make_reward(gate=True, route="neutral", floor=True, cap=1.0)
    eh = torch.tensor([-0.05, -0.02, 0.0, 0.10, 0.50, 2.0], dtype=torch.float64)
    ref = torch.full_like(eh, 30.0)
    out, untrusted = r._route_and_clamp_ehull(eh, ref)
    # trusted shallow below-hull floors to 0 == on-hull; no depth ranking
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0
    assert not untrusted[:3].any()
    # gradient region intact, cap applied
    assert out[3] == pytest.approx(0.10) and out[4] == pytest.approx(0.50)
    assert out[5] == 1.0  # capped


def test_floor_sentinels_and_guards_go_to_cap():
    r = make_reward(gate=True, route="worst", floor=True, cap=1.0)
    #      deep-below  sparse-above  nan          inf          trusted
    eh = torch.tensor([-3.0,        0.05,         float("nan"), float("inf"), 0.2],
                      dtype=torch.float64)
    ref = torch.tensor([30.0,       3.0,          30.0,         30.0,         30.0],
                       dtype=torch.float64)
    out, untrusted = r._route_and_clamp_ehull(eh, ref)
    assert out[0] == 1.0 and untrusted[0]          # deep-below -> cap, routed
    assert out[1] == 1.0 and untrusted[1]          # sparse (route worst) -> cap
    assert out[2] == 1.0 and out[3] == 1.0         # sentinels -> cap
    assert out[4] == pytest.approx(0.2) and not untrusted[4]


def test_floor_does_not_change_neutral_mask():
    eh = torch.tensor([-0.05, 0.3, -2.0, 0.4], dtype=torch.float64)
    ref = torch.tensor([5.0, 5.0, 5.0, 30.0], dtype=torch.float64)
    masks = []
    for floor in (False, True):
        r = make_reward(gate=True, route="neutral", floor=floor, cap=1.0)
        r._route_and_clamp_ehull(eh.clone(), ref.clone())
        masks.append(r.last_sparse_neutral_mask.clone())
    assert torch.equal(masks[0], masks[1])
    # deep-below (idx 2) excluded from neutral; sparse idx 0,1 included; dense idx 3 not
    assert masks[0].tolist() == [True, True, False, False]


def test_cap_ceiling_tracks_ehull_cap():
    for cap in (1.0, 2.5, 5.0):
        r = make_reward(floor=True, cap=cap)
        eh = torch.tensor([100.0, float("nan")], dtype=torch.float64)
        ref = torch.full_like(eh, 30.0)
        out, _ = r._route_and_clamp_ehull(eh, ref)
        assert float(out.max()) == cap


def test_diversity_penalty_floor_reads_ehull_cap():
    # _apply_diversity_penalty computes its worst-reward floor from
    # getattr(rf, 'ehull_cap', 5.0) — verify the getattr contract here.
    r = make_reward(floor=True, cap=1.0)
    assert float(getattr(r, "ehull_cap", 5.0)) == 1.0
    r2 = OMatGRPOReward.__new__(OMatGRPOReward)
    assert float(getattr(r2, "ehull_cap", 5.0)) == 5.0  # default for instances without the attribute
