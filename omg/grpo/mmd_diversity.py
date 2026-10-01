"""Compositional MMD leave-one-out (LOO) diversity reward.

Reimplementation of Chemeleon2's *compositional* diversity term
(`chemeleon2/src/rl_module/components.py::mmd_reward` +
`CompositionDiversityReward`, paper pp.34-35), with ONE substitution:

  Chemeleon2's compositional embedding (featurizer.py:67-70) is the mean over a
  structure's atoms of a *learned* VAE atom-type embedding:
      comp_feat = mean_atoms( atom_type_embedder(Z) )            # [emb_dim]
  We replace the learned atom-type table with the IDENTITY (one-hot) table, so
  embed(Z) = one_hot(Z) in R^119 and the mean over atoms collapses to the
  *fractional-element vector*:
      e_i[Z] = count(Z in structure i) / n_atoms_i               # [119]
  Same MMD-LOO machinery, no learned embedding (we have no VAE).

KEY NUMERICAL FACT — used by the trainer to skip the expensive reference
self-kernel: the per-sample leave-one-out credit `r_indiv` is INDEPENDENT of
`R_term` (the reference-only term), because R_term is the same in `mmd_full`
and `mmd_drop` and cancels in `r_indiv = mmd_drop - mmd_full`. So the per-sample
bonus needs only k_gg (BK x BK) and k_gr (BK x N) — never the N x N k_rr. Pass
`r_term=0.0` for the bonus; `mmd_reward(...)['r']` (the scalar MMD) still needs
the real R_term and computes it when `r_term is None`.

Sign / direction:
  r_div(X)   = -MMD^2(X_gen, X_ref)        (lower MMD to reference => higher reward)
  r_indiv(z) = r_div(X) - r_div(X minus {z})   (marginal utility, as in Chemeleon2)
             = mmd_drop - mmd_full
  A sample that IMPROVES coverage of the reference manifold lowers the full MMD,
  so removing it RAISES the MMD (mmd_drop > mmd_full) => r_indiv > 0 (rewarded).
  A redundant / over-covered sample => removing it lowers MMD => r_indiv < 0.
"""

import torch

COMP_DIM = 119  # atomic numbers 0..118 (Z=0 = mask/dummy, never populated)


def frac_comp_vector(Z: torch.Tensor, dim: int = COMP_DIM) -> torch.Tensor:
    """Fractional-element vector for ONE structure.

    Z : 1D long tensor of REAL atomic numbers (1..118) for the structure.
    Returns [dim] float64 with v[z] = count(z)/n_atoms. Asserts no Z=0 (mask).
    """
    assert Z.numel() > 0, "empty structure"
    assert int(Z.min()) > 0, "Z=0 (DFM mask token) survived into MMD comp vector"
    v = torch.bincount(Z, minlength=dim).to(torch.float64)[:dim]
    return v / float(Z.numel())


def build_comp_matrix(species_flat: torch.Tensor, n_atoms: torch.Tensor,
                      dim: int = COMP_DIM) -> torch.Tensor:
    """Stack per-structure fractional-element vectors into [N, dim] (float64).

    species_flat : flat long tensor of REAL Z, concatenated over structures
                   (== gen.species; sliced by n_atoms, identical to the
                   _apply_diversity_penalty / process_data slicing).
    n_atoms      : [N] long, atoms per structure.
    """
    splits = torch.split(species_flat, n_atoms.tolist())
    rows = [frac_comp_vector(Z, dim) for Z in splits]
    return torch.stack(rows, dim=0)


def minmax_scale(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Min-max scaling to [0,1] across the batch (Chemeleon2's `normalize`).

    Same FORMULA as chemeleon2/src/rl_module/components.py::normalize, but eps is a
    pure divide-by-zero guard (1e-12), not Chemeleon2's 1e-4. Their 1e-4 is calibrated for
    learned-VAE-embedding r_indiv; ours is fractional-composition r_indiv ~O(1e-5),
    which is below 1e-4, so their guard would zero the entire signal. Min-max is fully
    scale-invariant once eps << range, so lowering eps recovers a proper [0,1] span.
    Truly-constant batch (range < 1e-12) -> all zeros (no signal). Outlier-sensitive
    (one extreme value sets the range) — use zscore_scale if that bites.
    """
    rng = float(x.max() - x.min())
    if rng < eps:
        return torch.zeros_like(x)
    x = (x - x.min()) / (x.max() - x.min())
    return x.clamp(0.0, 1.0)


def zscore_scale(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-batch z-score (chemeleon2 `standardize`): mean 0, std ~1, outlier-robust.

    eps lowered to 1e-12 (pure guard) for the same reason as minmax_scale — our
    r_indiv std ~O(1e-5) is below Chemeleon2's 1e-4 guard. z-score is scale-invariant for
    eps << std, so the output is mean0/std1 regardless of the raw kernel magnitude.
    """
    sd = float(x.std())
    if sd < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / x.std()


def scale_per_batch(x: torch.Tensor, mode: str = "minmax") -> torch.Tensor:
    """Dispatch per-batch scaling of the per-sample MMD-LOO credit.

    'minmax' (default, as in Chemeleon2) -> [0,1]; 'zscore' -> mean0/std1; 'none' -> raw.
    Stabilizes the bonus magnitude so a small fixed w_mmd (~O(1)) is meaningful,
    instead of the raw r_indiv ~O(1e-5) that needs w~1e3-1e4.
    """
    if mode == "none":
        return x
    if mode == "minmax":
        return minmax_scale(x)
    if mode == "zscore":
        return zscore_scale(x)
    raise ValueError(f"unknown mmd_norm={mode!r} (use 'minmax', 'zscore', or 'none')")


def _kernel(z: torch.Tensor, y: torch.Tensor, kernel: str = "poly",
            c: float = 1.0, d: int = 3) -> torch.Tensor:
    """Kernel Gram matrix. Dimension-normalized dot product (Chemeleon2's `/d`).

    poly   : (z.y/dim + c)^d           (Chemeleon2's poly_k, default c=1 d=3)
    linear : z.y/dim                   (composition overlap on the simplex;
             MMD reduces to ||mean_gen - mean_ref||^2 — first moment only)
    """
    dim = z.size(-1)
    g = z @ y.T / dim
    if kernel == "linear":
        return g
    if kernel == "poly":
        return (g + c) ** d
    raise ValueError(f"unknown mmd_kernel={kernel!r} (use 'poly' or 'linear')")


def mmd_reward(z_gen: torch.Tensor, z_ref: torch.Tensor, kernel: str = "poly",
               c: float = 1.0, d: int = 3, r_term=None) -> dict:
    """Polynomial-kernel MMD^2 with per-sample leave-one-out marginal credit.

    Faithful to chemeleon2 components.py::mmd_reward (Eq. 11 + LOO), generalized
    with a kernel switch and exposed (c, d).

    z_gen : [M, dim] generated batch (M = B*K).
    z_ref : [N, dim] reference manifold (MP-20 compositions).
    r_term: optional precomputed reference-only scalar; pass 0.0 to skip k_rr
            when only `r_indiv` is needed (it is R_term-independent — see module
            docstring). When None, k_rr is computed (needed for the 'r' scalar).

    Returns {"r": -mmd_full (scalar), "r_indiv": [M] LOO marginal credit}.
    """
    M, N = z_gen.shape[0], z_ref.shape[0]
    assert M >= 2, f"MMD-LOO needs M>=2 generated samples, got M={M}"

    k_gg = _kernel(z_gen, z_gen, kernel, c, d)
    k_gr = _kernel(z_gen, z_ref, kernel, c, d)

    if r_term is None:
        k_rr = _kernel(z_ref, z_ref, kernel, c, d)
        R_term = (k_rr.sum() - k_rr.trace()) / (N * (N - 1))
    else:
        R_term = r_term

    G = k_gg.sum() - k_gg.trace()                       # sum_{i!=j} k(g_i,g_j)
    C = k_gr.sum()                                      # sum_{i,m} k(g_i,r_m)
    mmd_full = G / (M * (M - 1)) + R_term - 2 * C / (M * N)   # Eq. (11)

    # Per-sample drop (leave structure m out of the generated set).
    S = k_gg.sum(dim=1) - k_gg.diagonal()              # S_m = sum_{j!=m} k(g_m,g_j)
    T = k_gr.sum(dim=1)                                # T_m = sum_r k(g_m,r)
    Mp = M - 1
    Ap = Mp * (Mp - 1)
    mmd_drop = (G - 2 * S) / Ap + R_term - 2 * (C - T) / (Mp * N)

    r_indiv = mmd_drop - mmd_full                       # R_term cancels here
    return {"r": -mmd_full, "r_indiv": r_indiv}
