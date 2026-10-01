"""Upstream OMatG tests that fail at the vendored commit (and at upstream HEAD) for reasons that
do not concern OMatGRPO. They are expected failures (strict: a test that starts passing is reported).

- test_coupled_integrator: the test builds positions for 1000 atoms but declares 3000 (n_atoms=3 per
  structure), so it fails before reaching the integrator.
- VP score-based interpolants in test_integrators_ODE/SDE: upstream's SingleStochasticInterpolant
  rejects them on purpose ("requires antithetic sampling"); the tests predate that check.
"""
import pytest

STALE_UPSTREAM_TESTS = {
    "omg/tests/test_coupled_integrator.py::test_coupled_integrator",
    "omg/tests/test_integrators_ODE.py::test_ode_integrator[gamma=LatentGammaEncoderDecoder, interpolant=PeriodicScoreBasedDiffusionModelInterpolantVP]",
    "omg/tests/test_integrators_ODE.py::test_ode_integrator[gamma=LatentGammaEncoderDecoder, interpolant=ScoreBasedDiffusionModelInterpolantVP]",
    "omg/tests/test_integrators_ODE.py::test_ode_integrator[gamma=LatentGammaSqrt, interpolant=PeriodicScoreBasedDiffusionModelInterpolantVP]",
    "omg/tests/test_integrators_ODE.py::test_ode_integrator[gamma=LatentGammaSqrt, interpolant=ScoreBasedDiffusionModelInterpolantVP]",
    "omg/tests/test_integrators_ODE.py::test_ode_integrator[gamma=None, interpolant=PeriodicScoreBasedDiffusionModelInterpolantVP]",
    "omg/tests/test_integrators_ODE.py::test_ode_integrator[gamma=None, interpolant=ScoreBasedDiffusionModelInterpolantVP]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaEncoderDecoder, interpolant=PeriodicScoreBasedDiffusionModelInterpolantVP, epsilon=ConstantEpsilon]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaEncoderDecoder, interpolant=PeriodicScoreBasedDiffusionModelInterpolantVP, epsilon=VanishingEpsilon]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaEncoderDecoder, interpolant=ScoreBasedDiffusionModelInterpolantVP, epsilon=ConstantEpsilon]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaEncoderDecoder, interpolant=ScoreBasedDiffusionModelInterpolantVP, epsilon=VanishingEpsilon]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaSqrt, interpolant=PeriodicScoreBasedDiffusionModelInterpolantVP, epsilon=ConstantEpsilon]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaSqrt, interpolant=PeriodicScoreBasedDiffusionModelInterpolantVP, epsilon=VanishingEpsilon]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaSqrt, interpolant=ScoreBasedDiffusionModelInterpolantVP, epsilon=ConstantEpsilon]",
    "omg/tests/test_integrators_SDE.py::test_sde_integrator[gamma=LatentGammaSqrt, interpolant=ScoreBasedDiffusionModelInterpolantVP, epsilon=VanishingEpsilon]",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in STALE_UPSTREAM_TESTS:
            item.add_marker(pytest.mark.xfail(reason="stale upstream OMatG test", strict=True))
