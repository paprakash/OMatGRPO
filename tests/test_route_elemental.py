"""
Single-element guard (--route_elemental).

Single-element compositions get the cap (penalty) through the same mechanism as
deep-below-hull structures, before clamping; abstention for sparse compounds is unchanged.
Style follows tests/test_ehull_floor.py (bare instances, no UMA load).

Cases:
  1. With the guard off (or on an instance without the attribute), the result equals the
     routing without the guard, bit for bit, for all gate/route modes and both floor
     settings, even when a single-element mask is passed.
  2. elemental=None with the guard on changes nothing (the caller supplies the mask).
  3. Guard on: single-element structures get the cap and are untrusted whatever their
     E_hull, under both sparse routings and both floor settings.
  4. Compounds are unaffected: guard on with an all-False mask equals guard off.
  5. A sparse single-element structure cannot abstain: it is penalized and removed from
     last_sparse_neutral_mask; sparse compounds keep abstaining.
  6. The constructor rejects values other than 'off' and 'worst' (the no_uma fixture
     replaces the UMA potential, so this runs the real __init__).
"""
import pytest
import torch

from omg.grpo.reward import OMatGRPOReward


def make_reward(gate=False, route="worst", floor=False, cap=5.0,
                mag_floor=-0.1, min_refset=12, route_elemental="off",
                legacy_no_attr=False):
    """Bare instance (no UMA load) with just the attrs the routing needs.

    legacy_no_attr=True builds an instance without the route_elemental attribute
    (getattr default path).
    """
    r = OMatGRPOReward.__new__(OMatGRPOReward)
    r.ehull_mag_floor = float(mag_floor)
    r.ehull_min_refset = int(min_refset)
    r.ehull_sparse_gate = bool(gate)
    r.sparse_route = route
    r.ehull_floor_at_zero = bool(floor)
    r.ehull_cap = float(cap)
    r.last_sparse_neutral_mask = None
    if not legacy_no_attr:
        r.route_elemental = route_elemental
    return r


def fuzz_inputs(g, n):
    """Random eh (with NaN/inf/boundary injections), ref, elemental mask."""
    eh = torch.randn(n, generator=g, dtype=torch.float64) * 3.0
    m = torch.rand(n, generator=g)
    eh[m < 0.08] = float("nan")
    eh[(m >= 0.08) & (m < 0.12)] = float("inf")
    eh[(m >= 0.12) & (m < 0.14)] = -0.1  # exact mag_floor boundary
    ref = torch.randint(0, 40, (n,), generator=g).to(torch.float64)
    elemental = torch.rand(n, generator=g) < 0.3
    return eh, ref, elemental


@pytest.mark.parametrize("gate,route", [(False, "worst"), (False, "neutral"),
                                        (True, "worst"), (True, "neutral")])
@pytest.mark.parametrize("floor,cap", [(False, 5.0), (True, 1.0)])
def test_flag_off_byte_identity_fuzz(gate, route, floor, cap):
    """route_elemental='off' AND legacy no-attr instances match the reference
    (flagless call on a no-attr instance) exactly, mask passed or not."""
    g = torch.Generator().manual_seed(hash((gate, route, floor)) % (2**31))
    for _ in range(500):
        n = int(torch.randint(1, 65, (1,), generator=g))
        eh, ref, elemental = fuzz_inputs(g, n)

        ref_r = make_reward(gate=gate, route=route, floor=floor, cap=cap,
                            legacy_no_attr=True)
        exp_eh, exp_untrusted = ref_r._route_and_clamp_ehull(eh.clone(), ref.clone())
        exp_neutral = ref_r.last_sparse_neutral_mask

        for kwargs in (dict(route_elemental="off"), dict(legacy_no_attr=True)):
            for pass_mask in (False, True):
                r = make_reward(gate=gate, route=route, floor=floor, cap=cap, **kwargs)
                got_eh, got_untrusted = r._route_and_clamp_ehull(
                    eh.clone(), ref.clone(),
                    elemental=elemental.clone() if pass_mask else None)
                assert torch.equal(got_eh, exp_eh)
                assert torch.equal(got_untrusted, exp_untrusted)
                if exp_neutral is None:
                    assert r.last_sparse_neutral_mask is None
                else:
                    assert torch.equal(r.last_sparse_neutral_mask, exp_neutral)


def test_flag_on_no_mask_is_identity():
    """Flag ON but elemental=None → guard is inert (mask is the call site's job)."""
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        n = int(torch.randint(1, 65, (1,), generator=g))
        eh, ref, _ = fuzz_inputs(g, n)
        base = make_reward(gate=True, route="neutral", floor=True, cap=1.0)
        exp_eh, exp_untrusted = base._route_and_clamp_ehull(eh.clone(), ref.clone())
        r = make_reward(gate=True, route="neutral", floor=True, cap=1.0,
                        route_elemental="worst")
        got_eh, got_untrusted = r._route_and_clamp_ehull(eh.clone(), ref.clone(),
                                                         elemental=None)
        assert torch.equal(got_eh, exp_eh)
        assert torch.equal(got_untrusted, exp_untrusted)
        assert torch.equal(r.last_sparse_neutral_mask, base.last_sparse_neutral_mask)


@pytest.mark.parametrize("route", ["worst", "neutral"])
@pytest.mark.parametrize("floor,cap", [(False, 5.0), (True, 1.0)])
def test_elemental_routes_to_cap(route, floor, cap):
    """Arity-1 → cap + untrusted regardless of E_hull, incl. trusted dense
    on-hull/below-hull entries that every other guard would pass."""
    r = make_reward(gate=True, route=route, floor=floor, cap=cap,
                    route_elemental="worst")
    #    trusted-below trusted-on  trusted-above  sparse-above
    eh = torch.tensor([-0.05,      0.0,           0.2,          0.05],
                      dtype=torch.float64)
    ref = torch.tensor([30.0,      30.0,          30.0,         3.0],
                       dtype=torch.float64)
    elemental = torch.tensor([True, True, True, True])
    out, untrusted = r._route_and_clamp_ehull(eh, ref, elemental=elemental)
    assert untrusted.all()
    assert (out == cap).all()


def test_compounds_unaffected_fuzz():
    """Flag ON + all-False mask == flag OFF, on the canonical config."""
    g = torch.Generator().manual_seed(42)
    for _ in range(500):
        n = int(torch.randint(1, 65, (1,), generator=g))
        eh, ref, _ = fuzz_inputs(g, n)
        off = make_reward(gate=True, route="neutral", floor=True, cap=1.0,
                          route_elemental="off")
        exp_eh, exp_untrusted = off._route_and_clamp_ehull(eh.clone(), ref.clone())
        on = make_reward(gate=True, route="neutral", floor=True, cap=1.0,
                         route_elemental="worst")
        got_eh, got_untrusted = on._route_and_clamp_ehull(
            eh.clone(), ref.clone(), elemental=torch.zeros(n, dtype=torch.bool))
        assert torch.equal(got_eh, exp_eh)
        assert torch.equal(got_untrusted, exp_untrusted)
        assert torch.equal(on.last_sparse_neutral_mask, off.last_sparse_neutral_mask)


def test_mixed_batch_compound_slots_untouched():
    """In a mixed batch, non-elemental slots get bit-identical values to a
    flag-off pass; elemental slots go to cap."""
    eh = torch.tensor([-0.05, 0.3, -2.0, 0.4, 0.15], dtype=torch.float64)
    ref = torch.tensor([5.0, 5.0, 5.0, 30.0, 30.0], dtype=torch.float64)
    elemental = torch.tensor([False, True, False, True, False])
    off = make_reward(gate=True, route="neutral", floor=True, cap=1.0)
    off_eh, off_untrusted = off._route_and_clamp_ehull(eh.clone(), ref.clone())
    on = make_reward(gate=True, route="neutral", floor=True, cap=1.0,
                     route_elemental="worst")
    on_eh, on_untrusted = on._route_and_clamp_ehull(eh.clone(), ref.clone(),
                                                    elemental=elemental)
    comp = ~elemental
    assert torch.equal(on_eh[comp], off_eh[comp])
    assert torch.equal(on_untrusted[comp], off_untrusted[comp])
    assert (on_eh[elemental] == 1.0).all() and on_untrusted[elemental].all()


def test_neutral_mask_excludes_elementals():
    """A sparse elemental cannot be rescued by sparse→neutral: it is routed
    worst AND dropped from the neutral mask. Sparse compounds keep theirs."""
    #     sparse-elem  sparse-comp  sparse-deep  dense-comp
    eh = torch.tensor([0.05,        0.3,         -2.0,       0.4],
                      dtype=torch.float64)
    ref = torch.tensor([2.0,        5.0,          5.0,       30.0],
                       dtype=torch.float64)
    elemental = torch.tensor([True, False, False, False])
    r = make_reward(gate=True, route="neutral", floor=True, cap=1.0,
                    route_elemental="worst")
    out, untrusted = r._route_and_clamp_ehull(eh, ref, elemental=elemental)
    # elemental: worst + NOT neutral (flag-off would have neutral=True here)
    assert out[0] == 1.0 and untrusted[0]
    assert r.last_sparse_neutral_mask.tolist() == [False, True, False, False]
    # compound slots exactly as before: sparse-comp neutral+honest, deep worst
    assert out[1] == pytest.approx(0.3) and not untrusted[1]
    assert out[2] == 1.0 and untrusted[2]
    assert out[3] == pytest.approx(0.4) and not untrusted[3]


def test_constructor_validates_route_elemental(no_uma):
    """Real __init__ path (UMA replaced by no_uma): bad value → ValueError,
    valid values accepted and stored."""
    with pytest.raises(ValueError, match="route_elemental"):
        OMatGRPOReward(device="cpu", route_elemental="bogus")
    for v in ("off", "worst"):
        r = OMatGRPOReward(device="cpu", route_elemental=v)
        assert r.route_elemental == v
        assert r.last_elemental_fraction == 0.0
        assert r.last_elemental_routed_fraction == 0.0
