# omg/grpo/reward.py
# Reward for OMatGRPO. Structures are relaxed and scored with the UMA potential
# (uma-s-1p2, task omat) through TorchSim, which runs batched FIRE relaxations on the GPU.
# The stability term uses the energy above the LeMat-Bulk UMA convex hull.
import torch
import torch_sim as ts
from torch_sim.models.fairchem import FairChemModel
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import gc
from pymatgen.core.periodic_table import Element
from pymatgen.core import Lattice, Structure
import signal
from contextlib import contextmanager

from omg.grpo.guards import cell_geometry, cell_ok, classify_cell

@contextmanager
def timeout(seconds):
    """Timeout context manager"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Relaxation timeout after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _relaxed_cell_ok(cell_np):
    """Post-relaxation cell guard (used with relax_cell_dof), with the same thresholds as the
    pre-relaxation guard in calculate_batch_energy_reward. A relaxation that includes the cell
    can blow a cell up (huge, oblate or collapsed); scoring such a cell would feed UMA a
    pathological periodic graph, so the structure is treated as failed instead.
    Unit-tested in tests/test_relax_cell_dof.py."""
    return cell_ok(cell_np)


class _DetachedFairChem(FairChemModel):
    """FairChem's stress output is autograd-tracked; torch-sim's cell filters
    propagate it into the optimizer state and the next forward crashes on
    .numpy(). The reward is a detached scalar, so all outputs are detached. Used only
    when relax_cell_dof=True."""
    def forward(self, state):
        out = super().forward(state)
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}


class CreativityReward:
    """Creativity term (unique and novel), following Chemeleon2.

    Reference implementation: chemeleon2 src/rl_module/components.py:90-129 +
    src/utils/metrics.py:68-94. Per generated structure:
      unique AND novel   -> 1.0
      neither            -> 0.0
      mixed (u xor v)    -> min AMD distance (average-minimum-distance, k=100)
                            to same-reduced-formula structures (reference + the
                            rest of the rollout batch), CLAMPED to [0, 1] so the
                            fallback can never exceed the unique∧novel score
                            (deviation from chemeleon2, which leaves it unclamped;
                            clamp hits are counted in last_amd_clamp_count).
    Uniqueness = first-occurrence within the rollout batch (pairwise
    StructureMatcher fit against earlier same-formula batch members — equivalent
    to chemeleon2's group_structures first-of-group convention modulo fit
    non-transitivity). Novelty = no StructureMatcher fit against any reference
    structure with the same reduced formula (MatterGen convention, pymatgen defaults; the
    same matcher and reference as the MP-20 novelty of the internal evaluation, so this term
    and that novelty metric share one definition).

    Reference = MP-20 train bucketed by reduced formula (dict[str, list[Structure]]), by default
    <data_dir>/references/mp20_train_ref.json.gz (scripts/build_references.py; see
    omg/grpo/references.py).

    Per-structure TIMEOUT (sm_timeout, default 3 s) wraps that structure's
    StructureMatcher work; on timeout the structure scores 0.0 and is counted
    in last_timeout_count. AMD failures also score 0.0 (last_amd_fail_count).

    CPU-only, no UMA dependency — directly unit-testable
    (tests/test_creativity_reward.py).
    """

    def __init__(self, reference_path=None, sm_timeout=3.0, amd_k=100):
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from omg.grpo.references import default_creativity_reference
        self.sm_timeout = max(1, int(round(float(sm_timeout))))  # signal.alarm wants int >= 1
        self.amd_k = int(amd_k)
        self.matcher = StructureMatcher()  # pymatgen defaults = MatterGen conventions
        if reference_path is None:
            reference_path = default_creativity_reference()
        self.reference_path = str(reference_path)
        self._ref_by_formula = self._load_reference(reference_path)
        n_structs = sum(len(v) for v in self._ref_by_formula.values())
        print(f"[creativity] reference ready: {n_structs} structures, "
              f"{len(self._ref_by_formula)} reduced formulas ({reference_path}); "
              f"sm_timeout={self.sm_timeout}s amd_k={self.amd_k}")
        # Per-call stats (read by OMatGRPOReward / grpo_lightning logging).
        self.last_scores = []
        self.last_unique = []
        self.last_novel = []
        self.last_timeout_count = 0
        self.last_amd_fail_count = 0
        self.last_amd_clamp_count = 0
        self.last_amd_count = 0
        self.last_elapsed_sec = 0.0

    @staticmethod
    def _load_reference(path):
        """dict[reduced_formula -> list[Structure]] of MP-20 train (.json.gz or legacy .pkl)."""
        import os
        from omg.grpo.references import load_creativity_reference
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Creativity reference not found at {path}. Build it with "
                f"`python scripts/build_references.py creativity`.")
        return load_creativity_reference(path)

    def _min_amd_distance(self, structure, candidates):
        """Min positive AMD distance from `structure` to same-formula candidates
        (chemeleon2 fallback, k=100). Returns 0.0 when all distances are 0
        (identical fingerprints) — chemeleon2 would crash on the empty min."""
        import amd as _amd
        psets = [_amd.periodicset_from_pymatgen_structure(s)
                 for s in [structure] + list(candidates)]
        amds = np.array([_amd.AMD(p, self.amd_k) for p in psets])
        dists = _amd.AMD_cdist(amds, amds)[0]
        pos = dists[dists > 0]
        return float(pos.min()) if pos.size else 0.0

    def compute(self, structures):
        """Score a rollout batch. Returns torch.FloatTensor [N] in [0, 1]."""
        import time as _time
        t0 = _time.time()
        n = len(structures)
        formulas = []
        for s in structures:
            try:
                formulas.append(s.composition.reduced_formula)
            except Exception:
                formulas.append(None)   # unparsable composition -> scores 0.0

        scores = [0.0] * n
        unique_flags = [False] * n
        novel_flags = [False] * n
        self.last_timeout_count = 0
        self.last_amd_fail_count = 0
        self.last_amd_clamp_count = 0
        self.last_amd_count = 0

        # Batch member indices by formula for the uniqueness prefilter + AMD pool.
        from collections import defaultdict
        batch_by_formula = defaultdict(list)
        for i, f in enumerate(formulas):
            if f is not None:
                batch_by_formula[f].append(i)

        for i, s in enumerate(structures):
            f = formulas[i]
            if f is None:
                continue                 # score stays 0.0
            refs = self._ref_by_formula.get(f, [])
            try:
                with timeout(self.sm_timeout):
                    # Uniqueness: first occurrence within the rollout batch.
                    u = True
                    for j in batch_by_formula[f]:
                        if j >= i:
                            break
                        if self.matcher.fit(structures[j], s):
                            u = False
                            break
                    # Novelty: no fit against same-formula reference structures.
                    v = True
                    for ref_st in refs:
                        if self.matcher.fit(ref_st, s):
                            v = False
                            break
            except TimeoutError:
                self.last_timeout_count += 1
                continue                 # score stays 0.0, per spec
            unique_flags[i] = u
            novel_flags[i] = v
            if u and v:
                scores[i] = 1.0
            elif not u and not v:
                scores[i] = 0.0
            else:
                # Mixed: AMD-distance fallback vs same-formula refs + the REST of
                # the batch (chemeleon2 builds its by-formula map over ref + gen).
                candidates = list(refs) + [structures[j]
                                           for j in batch_by_formula[f] if j != i]
                self.last_amd_count += 1
                try:
                    d = self._min_amd_distance(s, candidates)
                except Exception as e:
                    self.last_amd_fail_count += 1
                    if self.last_amd_fail_count == 1:
                        print(f"[creativity] AMD fallback FIRST failure "
                              f"(structure {i}, {f}): {e}")
                    continue             # score stays 0.0
                if d > 1.0:
                    self.last_amd_clamp_count += 1
                    d = 1.0
                scores[i] = d

        self.last_scores = scores
        self.last_unique = unique_flags
        self.last_novel = novel_flags
        self.last_elapsed_sec = _time.time() - t0
        return torch.tensor(scores, dtype=torch.float)


def _apply_mask_guard(specie_array):
    """DFM mid-evolution mask guard for a single structure's species array.

    Any atom carrying mask token Z=0 means the structure is partially masked:
    pymatgen.Element.from_Z(0) raises, and even if bypassed UMA would score a
    placeholder-H cell whose energy is meaningless. Returns a (safe, has_mask)
    pair so the caller can (a) survive downstream pymatgen / ASE construction
    with a benign Z=1 placeholder and (b) route the structure to the +inf
    failed-structure path alongside small/oblate/huge degenerate cells.

    Extracted for direct pytest coverage without UMA model load.
    """
    if specie_array.size == 0:
        return specie_array, False
    if (specie_array == 0).any():
        return np.ones_like(specie_array), True
    return specie_array, False


class OMatGRPOReward:
    def __init__(self, device='cuda', relax_before_reward=False, relax_max_steps=100, w_ehull=1.0, ehull_mag_floor=-0.1, ehull_min_refset=12, ehull_sparse_gate=False, sparse_route='worst', ehull_floor_at_zero=False, ehull_cap=5.0, relax_cell_dof=False, route_elemental='off', w_creat=0.0, creativity_reference=None, creativity_sm_timeout=3.0, creativity_amd_k=100, creat_on_relaxed=False):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # UMA model via TorchSim — loaded once, reused every step (no state accumulation).
        # A relaxation that includes the cell (Frechet filter) needs the stress from the model
        # (compute_stress=True) and detached outputs (see _DetachedFairChem).
        self.relax_cell_dof = bool(relax_cell_dof)
        if self.relax_cell_dof:
            self._torchsim_model = _DetachedFairChem(
                model="uma-s-1p2",
                device=self.device,
                task_name="omat",
                compute_stress=True,
            )
            print("[reward] relax_cell_dof=True — FIRE relax includes CELL DOF "
                  "(Frechet filter) + post-relax cell re-guard.")
        else:
            self._torchsim_model = FairChemModel(
                model="uma-s-1p2",
                device=self.device,
                task_name="omat",
            )

        # gen will be set when __call__ is invoked
        self.gen = None
        self.atoms = []
        self.structures = []
        self.group = []
        self.score_details = {}  # Will be populated in calculate_rewards
        # Bulk-crystal per-element reference energies, ported from LeMat-GenBench's
        # reference_energies.py (omg/grpo/lemat_hull.py, element_chem_pot.json). Isolated-atom
        # references must not be used: they give cohesive-energy-like values about 7 eV/atom
        # too negative. Checked against a reference structure in tests/test_formation_energy_refs.py.
        self._form_refs_available = False
        try:
            import os
            from omg.grpo.lemat_hull import (
                get_formation_energy_per_atom_from_composition_energy,  # noqa: F401
            )
            chem_pot_path = os.path.join(
                os.path.dirname(__file__), "element_chem_pot.json"
            )
            if os.path.exists(chem_pot_path):
                self._form_refs_available = True
                print(f"[reward] LeMat-GenBench bulk-crystal refs ready: {chem_pot_path}")
            else:
                print(f"[reward] WARNING: element_chem_pot.json missing at {chem_pot_path} — formation-energy logging disabled")
        except Exception as e:
            print(f"[reward] Could not import lemat_hull formation-energy fn: {e}")
            self._form_refs_available = False

        # Stability term: -E_hull, routed and clamped (see _route_and_clamp_ehull), against the
        # LeMat-Bulk UMA hull (omg.grpo.lemat_hull.get_energy_above_hull). Lower E_hull = closer
        # to the hull = higher reward. The formation and absolute energy rewards of the paper's
        # reward-hacking appendix are on the reward-hacking branch.

        # Relaxation before scoring: FIRE, force_tol 0.05, at most
        # relax_max_steps steps (a full relaxation of B*K structures every rollout is too
        # costly), positions only or with the cell (relax_cell_dof). Scoring the relaxed
        # structure rewards the basin the generator lands in rather than the raw sample.
        self.relax_before_reward = bool(relax_before_reward)
        self.relax_max_steps = int(relax_max_steps)
        self.last_relax_rmsd = []
        self.last_relax_delta_e = []
        if self.relax_before_reward:
            print(f"[reward] relax_before_reward=True (relax_max_steps={self.relax_max_steps}, "
                  f"FIRE force_tol=0.05, cell {'included' if self.relax_cell_dof else 'fixed'})")

        # Weight of the stability (E_hull) term. 0 turns the term off, and with it every guard
        # penalty, because all penalties live inside this term.
        self.w_ehull = float(w_ehull)
        # Deep-below-hull guard and sparse-hull threshold. A structure far below the hull
        # (E_hull < ehull_mag_floor, beyond UMA's error) is not a trustworthy discovery; the
        # policy can reach it by exploiting errors of the potential or a poorly sampled hull.
        # A hull with fewer than ehull_min_refset reference entries in the chemical system is
        # too sparse to trust in either direction.
        self.ehull_mag_floor = float(ehull_mag_floor)
        self.ehull_min_refset = int(ehull_min_refset)
        # Sparse-hull gate. When on, every structure on a sparse hull is treated as untrusted
        # whatever the sign of its E_hull; without it, sparse above-hull structures stay
        # rewarded and the policy can drift into poorly sampled chemical systems. The reference
        # count belongs to the chemical system, not the structure, so the gate also affects
        # legitimate structures in under-explored systems (logged as reward/sparse_untrusted_frac).
        # When off, only the older guard acts: below-hull structures that are deep or sparse
        # are penalized.
        self.ehull_sparse_gate = bool(ehull_sparse_gate)
        # How the sparse clause is handled (only when the gate is on); deep-below-hull
        # structures are always penalized.
        #   'worst'   -> penalty: sparse structures get the cap value (worst stability term).
        #   'neutral' -> abstention: sparse structures keep their E_hull value, are flagged in
        #                last_sparse_neutral_mask, and get zero advantage in the trainer
        #                (no gradient) instead of a penalty. Structures that failed a guard or the
        #                hull lookup have no reference entries and therefore abstain too.
        # See _compute_group_advantages in grpo_lightning.py.
        self.sparse_route = str(sparse_route)
        if self.sparse_route not in ('worst', 'neutral'):
            raise ValueError(f"sparse_route must be 'worst' or 'neutral'; got {self.sparse_route!r}")
        # Single-element guard. Single-element compositions sit on sparse hulls, so abstention
        # never penalizes them, and the compositional MMD term barely separates them. Without
        # this guard the policy drifts toward single-element structures (reward-hacking appendix). When
        # 'worst', single-element structures get the cap value like deep-below-hull ones;
        # compounds and the abstention clause are unchanged.
        self.route_elemental = str(route_elemental)
        if self.route_elemental not in ('off', 'worst'):
            raise ValueError(f"route_elemental must be 'off' or 'worst'; got {self.route_elemental!r}")
        # Logged: share of single-element structures (always) and share penalized by the guard
        # (0.0 unless route_elemental='worst').
        self.last_elemental_fraction = 0.0
        self.last_elemental_routed_fraction = 0.0
        # Floor at zero and cap (stability term -clip(E_hull, 0, cap), as in Chemeleon2). With
        # the floor, a structure below the hull earns the same best value as one on the hull,
        # so there is no payout for depth below the hull (without it the policy chases
        # ever-lower E_hull, reward-hacking appendix). ehull_cap is also the value given to penalized and
        # failed structures. A cap of 1 keeps the gradient in the range that matters for
        # metastability; a cap of 5 dilutes it.
        self.ehull_floor_at_zero = bool(ehull_floor_at_zero)
        self.ehull_cap = float(ehull_cap)
        if self.ehull_cap <= 0.0:
            raise ValueError(f"ehull_cap must be > 0; got {self.ehull_cap}")
        # Per-structure bool [N]: abstaining structures whose advantage the trainer sets to zero
        # (set only when sparse_route=='neutral' and the gate is on).
        self.last_sparse_neutral_mask = None
        self.last_e_hull_ref_count = []
        self.last_ehull_untrusted_fraction = 0.0
        self.last_sparse_untrusted_fraction = 0.0
        self.last_sparse_neutral_fraction = 0.0
        # Per-structure clamped term magnitudes for within-group-std logging (set per call).
        self.last_ehull_clamped_term = []
        print(f"[reward] e_hull below-hull guard: trust E_hull<0 only if "
              f"E_hull>={self.ehull_mag_floor:g} AND ref_set>={self.ehull_min_refset} "
              f"(else → worst reward).")
        print(f"[reward] e_hull sparse-hull gate: {self.ehull_sparse_gate} "
              f"(when True, ref_set<{self.ehull_min_refset} is untrusted for both E_hull signs).")
        print(f"[reward] sparse_route={self.sparse_route} "
              f"({'sparse hull -> abstain (zero advantage); deep below hull -> penalty' if self.sparse_route=='neutral' else 'sparse hull -> penalty'}).")
        print(f"[reward] route_elemental={self.route_elemental} "
              f"({'single-element -> penalty' if self.route_elemental=='worst' else 'single-element guard off'}).")
        print(f"[reward] ehull_floor_at_zero={self.ehull_floor_at_zero} ehull_cap={self.ehull_cap:g} "
              f"(term = clamp(E_hull, {'0' if self.ehull_floor_at_zero else '-inf'}, {self.ehull_cap:g}); "
              f"penalized and failed structures -> cap).")
        if self.w_ehull == 0.0:
            print("[reward] w_ehull=0.0 → E_hull term zeroed (the relaxation and hull lookup still run).")

        # Creativity term (following Chemeleon2), additive: combined += w_creat * creativity,
        # creativity in [0, 1] per structure (see CreativityReward). With w_creat=0 the
        # CreativityReward is never built.
        self.w_creat = float(w_creat)
        self._creativity = None
        self.last_creativity = []
        # creat_on_relaxed: score creativity on the relaxed geometries kept by
        # calculate_batch_energy_reward (the generated structure is used, and counted, for slots
        # without a relaxed structure). Scored on unrelaxed samples the term stayed near its
        # maximum; the relaxed geometry is also what the evaluation scores.
        self.creat_on_relaxed = bool(creat_on_relaxed)
        self.last_relaxed_structures = []
        self.last_creat_relax_fallback = 0
        if self.creat_on_relaxed and not self.relax_before_reward:
            raise ValueError(
                "creat_on_relaxed=True requires relax_before_reward=True — the term must "
                "never silently score unrelaxed geometry when the flag is on."
            )
        if self.w_creat > 0.0:
            self._creativity = CreativityReward(
                reference_path=creativity_reference,
                sm_timeout=creativity_sm_timeout,
                amd_k=creativity_amd_k,
            )
            print(f"[reward] creativity term ON: w_creat={self.w_creat:g} "
                  f"(unique∧novel=1, neither=0, mixed=min-AMD clamp[0,1]; "
                  f"per-structure SM timeout {self._creativity.sm_timeout}s → 0; "
                  f"creat_on_relaxed={self.creat_on_relaxed})")

    def _route_and_clamp_ehull(self, eh: torch.Tensor, ref: torch.Tensor,
                               elemental: torch.Tensor = None):
        """Guard routing and clamping of the E_hull term.

        A separate method so the routing can be unit-tested without a UMA model
        (tests/test_ehull_floor.py). Inputs: eh [N] per-atom E_hull (NaN = failed guard or hull
        lookup), ref [N] hull reference-entry counts (0 for failed structures), elemental
        optional bool [N] (True = exactly one distinct element). Returns (eh_clamped, untrusted)
        and sets self.last_sparse_neutral_mask.

        Routing ("penalty" = the cap value, the worst stability term):
          deep below hull (eh < ehull_mag_floor)            -> penalty, always
          sparse (ref < ehull_min_refset), gate on, 'worst'   -> penalty
          sparse, gate on, 'neutral'                          -> keeps eh, abstains (neutral mask)
          gate off: below hull and (deeper than floor or sparse) -> penalty
        Single-element guard: with route_elemental='worst' and `elemental` given, single-element
        structures get the penalty and are removed from the neutral mask, so abstention can
        never shield them.
        Then NaN/inf -> cap, and clamp to [0 if ehull_floor_at_zero else -inf, cap]. A failed
        structure (NaN, ref 0) therefore counts as sparse: it abstains under 'neutral' and is
        penalized under 'worst' or with the gate off.
        """
        below = eh < 0.0
        sparse = ref < self.ehull_min_refset
        deep_below = below & (eh < self.ehull_mag_floor)   # always penalized
        self.last_sparse_neutral_mask = None               # reset each call
        if self.ehull_sparse_gate:
            if self.sparse_route == 'neutral':
                # Abstention: sparse structures keep their E_hull and get zero advantage in the
                # trainer. A structure that is both sparse and deep below the hull is penalized
                # and left out of the neutral mask, so the two routes never overlap.
                untrusted = deep_below
                self.last_sparse_neutral_mask = (sparse & ~deep_below)
            else:
                # 'worst': penalty for sparse structures of either E_hull sign.
                untrusted = sparse | deep_below
        else:
            # Gate off: only below-hull structures that are deep or sparse are penalized.
            untrusted = below & ((eh < self.ehull_mag_floor) | (ref < self.ehull_min_refset))
        if getattr(self, 'route_elemental', 'off') == 'worst' and elemental is not None:
            untrusted = untrusted | elemental
            if self.last_sparse_neutral_mask is not None:
                self.last_sparse_neutral_mask = self.last_sparse_neutral_mask & ~elemental
        cap = self.ehull_cap
        eh = torch.where(untrusted, torch.full_like(eh, cap), eh)
        eh = torch.nan_to_num(eh, nan=cap, posinf=cap, neginf=cap)
        if self.ehull_floor_at_zero:
            # Floor at zero: below-hull ties with on-hull at the best value (0), with no
            # gradient below zero, so depth below the hull earns nothing.
            eh_clamped = torch.clamp(eh, min=0.0, max=cap)
        else:
            eh_clamped = torch.clamp(eh, max=cap)
        return eh_clamped, untrusted

    def calculate_batch_energy_reward(self, group):
        """
        Energy-based reward for a batch of structures.

        1. Cell and masked-species guards: pathological cells and structures that still carry a
           mask token are not scored (their energy stays +inf, E_hull NaN).
        2. Optional FIRE relaxation (relax_before_reward), with the cell when relax_cell_dof;
           with the cell included, a post-relaxation cell guard follows.
        3. UMA energy per atom and formation energy (both logged only), and the energy above
           the LeMat-Bulk UMA hull, routed and clamped by _route_and_clamp_ehull.
        Returns the per-structure stability term, -w_ehull * routed E_hull.
        """
        unrelaxed_atoms = [AseAtomsAdaptor.get_atoms(s) for s in group]

        # ── Pathological-cell guard ──────────────────────────────────────────
        # A stochastic lattice channel can produce cells that UMA cannot score:
        #   (1) small/collapsed: volume near 0 or one vector near 0. The periodic replica
        #       count blows up (integer overflow in the radius graph).
        #   (2) huge/exploded: volume or a vector far beyond MP-20. Physically meaningless,
        #       and the UMA call can run out of GPU memory (an allocation of about 194 GiB
        #       was observed).
        #   (3) oblate/skewed: one perpendicular lattice-plane distance
        #       d_perp_i = V / |a_j x a_k| is tiny. UMA's replica count scales with this
        #       distance, not with |a_i|, so such a cell can pass (1) and (2) and still run
        #       out of memory.
        # These structures skip UMA: energy +inf, E_hull NaN, reference count 0. They are then
        # routed like any failed structure (see _route_and_clamp_ehull).
        # Priority: small > oblate > huge > ok. Thresholds live in omg/grpo/guards.py.

        N = len(unrelaxed_atoms)
        ok_idx, small_idx, oblate_idx, huge_idx, masked_idx = [], [], [], [], []
        masked_flags = list(getattr(self, 'last_has_mask', [False] * N))
        for i, atoms in enumerate(unrelaxed_atoms):
            # Structures with a mask token left are set aside before the geometry checks, so the
            # small/oblate/huge counts reflect genuine cell problems only.
            if masked_flags[i]:
                masked_idx.append(i)
                continue
            cell_np = np.asarray(atoms.get_cell())
            vol, min_len, max_len, min_perp = cell_geometry(cell_np)
            bucket = classify_cell(cell_np)
            if bucket == 'small':
                small_idx.append(i)
            elif bucket == 'oblate':
                oblate_idx.append(i)
            elif bucket == 'huge':
                huge_idx.append(i)
            else:
                bucket = None
                ok_idx.append(i)

            # First-fire print per bucket per run (for OOM post-mortem)
            if bucket is not None:
                flag_attr = f'_cell_guard_first_{bucket}'
                if not getattr(self, flag_attr, False):
                    print(f'cell guard FIRST {bucket.upper()}: '
                          f'vol={vol:.3f} min_len={min_len:.3f} '
                          f'max_len={max_len:.3f} min_perp={min_perp:.3f}')
                    setattr(self, flag_attr, True)

        degen_idx = small_idx + oblate_idx + huge_idx + masked_idx  # not scored (energy +inf)
        self.last_small_cell_fraction  = (len(small_idx)  / N) if N else 0.0
        self.last_oblate_cell_fraction = (len(oblate_idx) / N) if N else 0.0
        self.last_huge_cell_fraction   = (len(huge_idx)   / N) if N else 0.0
        self.last_masked_fraction      = (len(masked_idx) / N) if N else 0.0
        self.last_degenerate_cell_fraction = (len(degen_idx) / N) if N else 0.0
        print(f'cell guard: small={len(small_idx)}/{N} '
              f'oblate={len(oblate_idx)}/{N} huge={len(huge_idx)}/{N} '
              f'masked={len(masked_idx)}/{N} '
              f'ok={len(ok_idx)}/{N}')

        # Energy evaluation of the cells that passed the guard. With relax_before_reward, the structures are FIRE-relaxed first (at most relax_max_steps
        # steps, with the cell if relax_cell_dof), so energy and E_hull describe the relaxed
        # structure. Cells that failed the guard are not in ok_idx and keep energy +inf.
        self.last_relax_rmsd = [float('nan')] * N
        self.last_relax_delta_e = [float('nan')] * N
        # Relaxed geometries are kept for the creativity term (creat_on_relaxed). Slots that
        # failed a guard or the relaxation stay None; creativity then uses the generated
        # structure for them (counted as relax_fallback).
        self.last_relaxed_structures = [None] * N
        do_relax = bool(getattr(self, 'relax_before_reward', False))
        if ok_idx:
            ok_atoms = [unrelaxed_atoms[i] for i in ok_idx]
            if do_relax:
                # Unrelaxed energies and geometry, for the relaxation Delta-E / RMSD logs.
                gen_static = ts.static(system=ok_atoms, model=self._torchsim_model)
                e_gen_ok = [r['potential_energy'].detach().item() for r in gen_static]
                gen_pos = [a.get_positions().copy() for a in ok_atoms]
                gen_cell = [np.array(a.get_cell()) for a in ok_atoms]
                relaxed_atoms = None
                # relax_cell_dof: include the cell through the Frechet cell filter.
                _relax_kwargs = {}
                if getattr(self, 'relax_cell_dof', False):
                    from torch_sim.optimizers.cell_filters import CellFilter
                    _relax_kwargs['init_kwargs'] = {'cell_filter': CellFilter.frechet}
                try:
                    # Autocast must be off here: under bf16 training, autocast-eligible ops in the
                    # cell filter (torch.bmm in the fractional-coordinate transform) return bf16
                    # tensors, and torch.linalg.solve then fails against the fp32 deformation
                    # gradient. The reward is a detached scalar, so full precision costs nothing.
                    with timeout(900), torch.autocast('cuda', enabled=False):
                        relaxed_state = ts.optimize(
                            system=ok_atoms,
                            model=self._torchsim_model,
                            optimizer=ts.Optimizer.fire,
                            convergence_fn=ts.generate_force_convergence_fn(force_tol=0.05),
                            max_steps=int(getattr(self, 'relax_max_steps', 100)),
                            **_relax_kwargs,
                        )
                    relaxed_atoms = relaxed_state.to_atoms()
                    if not isinstance(relaxed_atoms, list):
                        relaxed_atoms = [relaxed_atoms]
                    self._relax_consec_failures = 0
                except (RuntimeError, TimeoutError) as e:
                    print(f"[e_hull relax] ts.optimize failed ({e}); using unrelaxed energies")
                    relaxed_atoms = None
                    # A single failure falls back to unrelaxed energies. Repeated failures mean
                    # the relaxation is broken, and a run that silently trains on unrelaxed
                    # energies would look healthy while being invalid, so stop after 5 in a row.
                    self._relax_consec_failures = getattr(self, '_relax_consec_failures', 0) + 1
                    if self._relax_consec_failures >= 5:
                        raise RuntimeError(
                            f"[e_hull relax] ts.optimize failed {self._relax_consec_failures} "
                            f"consecutive times (last: {e}). The reward depends on the "
                            f"relaxation (relax_before_reward), so the run stops instead of "
                            f"continuing on unrelaxed energies."
                        ) from e
                if relaxed_atoms is not None:
                    # Post-relaxation cell guard (only with relax_cell_dof): a cell that blew up
                    # during relaxation is not scored; its slot keeps energy +inf.
                    post_bad = set()
                    if getattr(self, 'relax_cell_dof', False):
                        for j, a in enumerate(relaxed_atoms):
                            if not _relaxed_cell_ok(np.array(a.get_cell())):
                                post_bad.add(j)
                        self.last_postrelax_guard_fraction = (
                            len(post_bad) / len(relaxed_atoms)) if relaxed_atoms else 0.0
                        if post_bad:
                            print(f"[e_hull relax] post-relax cell guard: "
                                  f"{len(post_bad)}/{len(relaxed_atoms)} blown up → worst")
                    if post_bad:
                        good_j = [j for j in range(len(relaxed_atoms)) if j not in post_bad]
                        static_good = (ts.static(system=[relaxed_atoms[j] for j in good_j],
                                                 model=self._torchsim_model)
                                       if good_j else [])
                        static_result_ok = [None] * len(ok_idx)
                        for jj, j in enumerate(good_j):
                            static_result_ok[j] = static_good[jj]
                    else:
                        static_result_ok = ts.static(system=relaxed_atoms, model=self._torchsim_model)
                    from ase.geometry import find_mic
                    for j, slot in enumerate(ok_idx):
                        if static_result_ok[j] is None:   # post-guard-failed slot
                            self.last_relax_delta_e[slot] = float('nan')
                            self.last_relax_rmsd[slot] = float('nan')
                            continue
                        e_rel = static_result_ok[j]['potential_energy'].detach().item()
                        self.last_relax_delta_e[slot] = float(e_gen_ok[j] - e_rel)
                        # keep this slot's relaxed geometry for the creativity term
                        try:
                            self.last_relaxed_structures[slot] = \
                                AseAtomsAdaptor.get_structure(relaxed_atoms[j])
                        except Exception:
                            self.last_relaxed_structures[slot] = None
                        try:
                            disp = relaxed_atoms[j].get_positions() - gen_pos[j]
                            mic, _ = find_mic(disp, gen_cell[j], pbc=True)
                            self.last_relax_rmsd[slot] = float(np.sqrt((mic ** 2).sum(axis=1).mean()))
                        except Exception:
                            self.last_relax_rmsd[slot] = float('nan')
                    del relaxed_state
                else:
                    static_result_ok = gen_static  # relaxation failed -> gen energies
                gc.collect()
                torch.cuda.empty_cache()
            else:
                static_result_ok = ts.static(
                    system=ok_atoms,
                    model=self._torchsim_model,
                )
        else:
            static_result_ok = []
        # ts.static returns list[dict] with 'potential_energy' (grad_fn tensor)

        # Assemble full-batch energies; degenerate slots get +inf (E_hull is then not looked up).
        energies_list = [float('inf')] * N
        for slot, r in zip(ok_idx, static_result_ok):
            if r is None:      # post-relax-guard-failed slot stays +inf → worst
                continue
            energies_list[slot] = r['potential_energy'].detach().item()
        energies = torch.tensor(energies_list)
        n_atoms = torch.tensor(
            [len(a) for a in unrelaxed_atoms], dtype=torch.float
        )
        energy_per_atom = energies / n_atoms

        # Clamp outliers (pathological structures can hit ~+50 eV/atom)
        energy_per_atom = torch.nan_to_num(energy_per_atom, nan=20.0, posinf=20.0, neginf=20.0)
        energy_per_atom_clamped = torch.clamp(energy_per_atom, min=-20.0, max=20.0)

        # Store absolute energies for logging (always)
        self.last_energy_per_atom = energy_per_atom.tolist()
        self.last_energy_per_atom_clamped = energy_per_atom_clamped.tolist()

        # Formation energy (logged only).
        # Bulk-crystal PBE refs via lemat_hull port (LeMat-GenBench convention) —
        # NOT iso-atom-in-vacuum refs. See __init__ for the rationale.
        if self._form_refs_available:
            from omg.grpo.lemat_hull import (
                get_formation_energy_per_atom_from_composition_energy,
            )
            from pymatgen.core import Composition
            from collections import Counter
            form_energies = []
            e_list = energies.tolist()
            for i, atoms in enumerate(unrelaxed_atoms):
                e_total = e_list[i]
                try:
                    comp = Composition(Counter(atoms.get_chemical_symbols()))
                    form_energies.append(
                        get_formation_energy_per_atom_from_composition_energy(
                            e_total, comp, functional="pbe"
                        )
                    )
                except Exception:
                    form_energies.append(float('nan'))
            self.last_formation_energy_per_atom = form_energies
        else:
            self.last_formation_energy_per_atom = [float('nan')] * len(group)

        # E_hull. Failures (cells that failed a guard, masked structures, a failed hull lookup)
        # get E_hull NaN and reference count 0; _route_and_clamp_ehull decides what that means
        # for the reward.
        self.last_e_hull_per_atom, self.last_e_hull_ref_count = self._lookup_e_hull(
            unrelaxed_atoms, ok_idx, energies.tolist())

        # Stability term: -clamp(routed E_hull, floor, cap). The cap bounds the bad tail and
        # is the value given to penalized and failed structures (nan_to_num maps NaN/inf to
        # the cap before clamping). Without the floor at zero, below-hull structures earn
        # more than on-hull ones; with it (ehull_floor_at_zero) they tie at the best value.
        # A below-hull value is trusted only when it is shallower than ehull_mag_floor
        # (deeper than UMA's error) and the hull has at least ehull_min_refset reference
        # entries; otherwise it is penalized, not clamped to the best value, which would
        # still pay the exploit. With the sparse-hull gate on, a sparse hull is untrusted for
        # both E_hull signs. See _route_and_clamp_ehull and the paper's reward-hacking appendix.
        eh = torch.tensor(self.last_e_hull_per_atom)
        ref = torch.tensor(self.last_e_hull_ref_count, dtype=eh.dtype)
        # Single-element mask: always computed (logged as elemental_frac); used for routing
        # only when route_elemental='worst'.
        elemental_mask = torch.tensor(
            [len(set(a.get_chemical_symbols())) == 1 for a in unrelaxed_atoms])
        _Ne = elemental_mask.numel()
        self.last_elemental_fraction = (
            elemental_mask.sum().item() / _Ne) if _Ne else 0.0
        if getattr(self, 'route_elemental', 'off') == 'worst':
            eh_clamped, untrusted = self._route_and_clamp_ehull(
                eh, ref, elemental=elemental_mask)
            self.last_elemental_routed_fraction = self.last_elemental_fraction
        else:
            eh_clamped, untrusted = self._route_and_clamp_ehull(eh, ref)
            self.last_elemental_routed_fraction = 0.0
        _N = eh.numel()
        self.last_ehull_untrusted_fraction = (untrusted.sum().item() / _N) if _N else 0.0
        if self.last_sparse_neutral_mask is not None:
            self.last_sparse_neutral_fraction = (
                self.last_sparse_neutral_mask.sum().item() / _N) if _N else 0.0
        else:
            self.last_sparse_neutral_fraction = 0.0
        # Share of structures on a sparse hull, logged with the gate on or off. Failed
        # structures (reference count 0) count as sparse here too.
        self.last_sparse_untrusted_fraction = (
            (ref < self.ehull_min_refset).sum().item() / _N) if _N else 0.0
        print(f'ehull_untrusted (penalized): '
              f'{int(untrusted.sum().item())}/{_N} = {self.last_ehull_untrusted_fraction:.3f} '
              f'| sparse_frac(ref<{self.ehull_min_refset})={self.last_sparse_untrusted_fraction:.3f} '
              f'| sparse_neutral_frac={self.last_sparse_neutral_fraction:.3f} '
              f'| elemental_frac={self.last_elemental_fraction:.3f} '
              f'(routed={self.last_elemental_routed_fraction:.3f}) '
              f'gate={self.ehull_sparse_gate} sparse_route={self.sparse_route} '
              f'route_elemental={getattr(self, "route_elemental", "off")}')
        # w_ehull scales the stability term; w_ehull=0 turns it (and every guard penalty) off.
        energy_rewards = -eh_clamped * self.w_ehull   # E_hull term weight = w_ehull only
        self.last_ehull_clamped_term = eh_clamped.tolist()
        print(f'E_hull/atom (reward): {[round(e, 4) for e in eh_clamped.tolist()]}')
        _n = eh_clamped.numel()
        # clamp hits = saturation at the cap (penalized/failed structures and the bad tail); below-hull
        # (E_hull<0) is genuine signal, NOT a clamp hit.
        _hits = (eh_clamped >= self.ehull_cap - 1e-6).sum().item()
        self.last_e_hull_clamp_hit_fraction = (_hits / _n) if _n else 0.0
        print(f'e_hull_clamp_hit_fraction: {_hits}/{_n} = {self.last_e_hull_clamp_hit_fraction:.3f}')

        print(f'Energy/atom (post-relax in relax path): {[round(e, 4) for e in self.last_energy_per_atom]}')
        if self._form_refs_available:
            print(f'Formation E/atom (log): {[round(e, 4) if np.isfinite(e) else float("nan") for e in self.last_formation_energy_per_atom]}')
        print(f'E_hull/atom (log): {[round(e, 4) if np.isfinite(e) else float("nan") for e in self.last_e_hull_per_atom]}')

        del unrelaxed_atoms, static_result_ok
        gc.collect()
        torch.cuda.empty_cache()

        return energy_rewards

    def _lookup_e_hull(self, atoms_list, ok_idx, e_list):
        """E_hull (eV/atom) and hull reference-entry count per structure.

        Structures outside ok_idx (failed guard, masked species) and structures with a
        non-finite energy are not looked up. A lookup that raises gives NaN and count 0, like a
        failed structure. Lookup failures are counted (last_hull_lookup_failures). If every
        attempted lookup in the batch fails, the hull reference is unavailable (for example no
        Hugging Face access), and training on it would silently turn the stability term into
        abstention or penalty for every structure, so this raises instead.
        """
        from collections import Counter
        from pymatgen.core import Composition
        from omg.grpo.lemat_hull import get_energy_above_hull
        e_hull_list, ref_count_list = [], []
        ok_idx_set = set(ok_idx)
        n_attempted = n_failed = 0
        first_error = None
        for i, atoms in enumerate(atoms_list):
            if i not in ok_idx_set or not np.isfinite(e_list[i]):
                # cell failed a guard, species still masked, or no finite energy
                e_hull_list.append(float('nan'))
                ref_count_list.append(0)
                continue
            n_attempted += 1
            try:
                comp = Composition(Counter(atoms.get_chemical_symbols()))
                eh, ref_count = get_energy_above_hull(
                    e_list[i], comp, hull_type='uma', threshold=0.001,
                    return_ref_count=True,
                )
                e_hull_list.append(float(eh))
                ref_count_list.append(int(ref_count))
            except Exception as e:
                n_failed += 1
                if first_error is None:
                    first_error = e
                e_hull_list.append(float('nan'))
                ref_count_list.append(0)
        self.last_hull_lookup_attempted = n_attempted
        self.last_hull_lookup_failures = n_failed
        if n_failed:
            print(f'hull lookup failed for {n_failed}/{n_attempted} structures '
                  f'(first error: {first_error})')
        if n_attempted > 0 and n_failed == n_attempted:
            raise RuntimeError(
                f"The hull lookup failed for all {n_attempted} structures of this batch "
                f"(first error: {first_error}). The LeMat-Bulk-MLIP-Hull reference is probably "
                f"unavailable: check Hugging Face access and the data directory."
            ) from first_error
        return e_hull_list, ref_count_list

    def _creativity_input_structures(self):
        """Structures the creativity term scores in this call.

        creat_on_relaxed=False -> the gen structures (self.group), 0 fallbacks.
        creat_on_relaxed=True  -> per slot the retained relaxed Structure, or
        the gen structure when that slot is None (degenerate/masked cell,
        post-relax guard fail, relax failure); returns (structs, n_fallback).
        A separate method so it can be tested without a UMA model.
        """
        if not self.creat_on_relaxed:
            return list(self.group), 0
        n = len(self.group)
        rel = list(getattr(self, 'last_relaxed_structures', []) or [])
        if len(rel) != n:            # relax path never ran for this batch size
            rel = [None] * n
        out, n_fallback = [], 0
        for i in range(n):
            if rel[i] is not None:
                out.append(rel[i])
            else:
                out.append(self.group[i])
                n_fallback += 1
        return out, n_fallback

    def calculate_rewards(self):
        """
        Computes the per-structure reward: stability term (calculate_batch_energy_reward)
        + w_creat * creativity.

        Returns
        -------
        tuple
            - score_details (dict): Per-structure reward breakdown.
            - total_reward (torch.Tensor): Sum across the group.
            - gathered_rewards (torch.Tensor): Per-structure combined rewards.
        """
        self.device = self.gen.cell.device
        n = len(self.group)

        # --- Stability term ---
        energy_rewards = self.calculate_batch_energy_reward(self.group)

        # --- Creativity term (only if w_creat > 0) ---
        # creat_on_relaxed=False scores the generated structures (Chemeleon2 convention);
        # creat_on_relaxed=True scores the relaxed geometries kept by
        # calculate_batch_energy_reward, with the generated structure for failed slots
        # (counted as relax_fallback). Runs on CPU either way.
        if self.w_creat > 0.0 and self._creativity is not None:
            creat_input, n_fallback = self._creativity_input_structures()
            self.last_creat_relax_fallback = n_fallback
            creat_scores = self._creativity.compute(creat_input)
            self.last_creativity = creat_scores.tolist()
            c = self._creativity
            fb = (f'relax_fallback={n_fallback} ' if self.creat_on_relaxed else '')
            print(f'[creativity] mean={creat_scores.mean().item():.4f} '
                  f'unique={sum(c.last_unique)}/{n} novel={sum(c.last_novel)}/{n} '
                  f'amd_fallback={c.last_amd_count} (fail={c.last_amd_fail_count}, '
                  f'clamped={c.last_amd_clamp_count}) timeouts={c.last_timeout_count} '
                  f'{fb}({c.last_elapsed_sec:.1f}s)')
        else:
            creat_scores = torch.zeros(n)
            self.last_creativity = []

        # --- Combine ---
        combined = energy_rewards + self.w_creat * creat_scores

        self.score_details = {idx + 1: {} for idx in range(n)}
        for idx in range(n):
            self.score_details[idx + 1]['energy_reward'] = energy_rewards[idx]
            self.score_details[idx + 1]['creativity_reward'] = self.w_creat * creat_scores[idx]
            self.score_details[idx + 1]['energy_per_atom'] = self.last_energy_per_atom[idx]
            self.score_details[idx + 1]['formation_energy_per_atom'] = self.last_formation_energy_per_atom[idx]
            self.score_details[idx + 1]['reward'] = combined[idx]

        gathered_rewards = torch.tensor(
            [self.score_details[idx + 1]['reward'] for idx in range(n)],
            device=self.device,
        )
        total_reward = torch.sum(gathered_rewards)

        print(
            f"reward breakdown: "
            f"energy_reward={[float(r) for r in energy_rewards]}"
        )

        # Clear group data to prevent RAM accumulation
        self.group = []
        self.structures = []
        self.atoms = []

        return self.score_details, total_reward, gathered_rewards

    def process_data(self, gen):
        def to_numpy(x):
            """Convert torch tensors (CPU or CUDA) to numpy, otherwise np.array."""
            if isinstance(x, torch.Tensor):
                # Convert BF16/FP16 to FP32 for compatibility with external libraries (pymatgen, ASE)
                if x.dtype in (torch.bfloat16, torch.float16):
                    x = x.float()
                return x.detach().cpu().numpy()
            return np.array(x)

        split_struct_pos = torch.split(gen.pos, gen.n_atoms.tolist())
        split_struct_species = torch.split(gen.species, gen.n_atoms.tolist())
        split_struct_cell = gen.cell

        structures = []
        has_mask_list = []
        n = split_struct_cell.shape[0]
        for i in range(n):
            lattice = to_numpy(split_struct_cell[i])
            pos = to_numpy(split_struct_pos[i])
            specie = to_numpy(split_struct_species[i])
            # Masked-species guard: a structure that still carries mask tokens (Z=0) gets
            # Z=1 placeholders so Element.from_Z works, and is flagged; it is not scored
            # (energy +inf in calculate_batch_energy_reward).
            specie, has_mask = _apply_mask_guard(specie)
            has_mask_list.append(has_mask)
            species = [Element.from_Z(Z).symbol for Z in specie]
            lattice_parameters = Lattice(lattice)

            structure = Structure(lattice = lattice_parameters,
                    species = species,
                    coords = pos)
            structures.append(structure)

        self.last_has_mask = has_mask_list
        adaptor = AseAtomsAdaptor()
        atoms = [adaptor.get_atoms(structure) for structure in structures]
        return atoms, structures
    
    def __call__(self, gen, aux=None, step=0):
        self.gen = gen

        self.atoms, self.structures = self.process_data(self.gen)
        self.group = self.structures
        r = self.calculate_rewards()
        r = list(r)
        gathered_rewards = r[2]

        # Build W&B metrics dict — aggregate stats across the full batch
        metrics = {}
        if self.last_energy_per_atom:
            epa = torch.tensor(self.last_energy_per_atom)
            finite = torch.isfinite(epa)
            if finite.any():
                epa_f = epa[finite]
                # raw UMA total energy per atom: depends on composition, so it tracks chemistry
                # drift and is not a stability metric (E_hull and formation energy are).
                metrics["energy/raw_total_per_atom_mean"] = epa_f.mean().item()
                metrics["energy/raw_total_per_atom_std"] = (epa_f.std().item()
                                                if epa_f.numel() > 1 else 0.0)
                metrics["energy/raw_total_per_atom_min"] = epa_f.min().item()
                metrics["energy/raw_total_per_atom_max"] = epa_f.max().item()
                metrics["energy/raw_total_per_atom_median"] = epa_f.median().item()
                metrics["energy/raw_total_clamp_hit_frac"] = (
                    ((epa < -20.0) | (epa > 20.0)).float().mean().item()
                )
                metrics["energy/raw_total_nonfinite_count"] = float((~finite).sum().item())

            if self._form_refs_available and self.last_formation_energy_per_atom:
                fpa = torch.tensor(self.last_formation_energy_per_atom)
                fpa_f = fpa[torch.isfinite(fpa)]
                if fpa_f.numel() > 0:
                    metrics["energy/formation_per_atom_mean"] = fpa_f.mean().item()
                    metrics["energy/formation_per_atom_min"] = fpa_f.min().item()
                    metrics["energy/formation_per_atom_max"] = fpa_f.max().item()
                    metrics["energy/formation_per_atom_median"] = fpa_f.median().item()

            # E_hull metrics
            if getattr(self, 'last_e_hull_per_atom', None):
                eh = torch.tensor(self.last_e_hull_per_atom)
                eh_finite = eh[torch.isfinite(eh)]
                if eh_finite.numel() > 0:
                    metrics["energy/e_hull_per_atom_mean"] = eh_finite.mean().item()
                    metrics["energy/e_hull_per_atom_std"] = (
                        eh_finite.std().item() if eh_finite.numel() > 1 else 0.0
                    )
                    metrics["energy/e_hull_per_atom_min"] = eh_finite.min().item()
                    metrics["energy/e_hull_per_atom_max"] = eh_finite.max().item()
                    metrics["energy/e_hull_per_atom_median"] = eh_finite.median().item()
                    metrics["energy/e_hull_nonfinite_count"] = float(
                        (~torch.isfinite(eh)).sum().item()
                    )

        if getattr(self, 'last_e_hull_clamp_hit_fraction', None) is not None:
            metrics["reward/e_hull_clamp_hit_fraction"] = float(self.last_e_hull_clamp_hit_fraction)
        metrics["reward/ehull_untrusted_frac"] = float(getattr(self, 'last_ehull_untrusted_fraction', 0.0))
        metrics["reward/hull_lookup_fail_count"] = float(getattr(self, 'last_hull_lookup_failures', 0))
        # sparse-hull share, logged with the gate on or off
        metrics["reward/sparse_untrusted_frac"] = float(getattr(self, 'last_sparse_untrusted_fraction', 0.0))
        # abstaining share (sparse_route='neutral')
        metrics["reward/sparse_neutral_frac"] = float(getattr(self, 'last_sparse_neutral_fraction', 0.0))
        # single-element share (always) and the share penalized by the single-element guard
        # (nonzero only when route_elemental='worst')
        metrics["reward/elemental_frac"] = float(getattr(self, 'last_elemental_fraction', 0.0))
        metrics["reward/elemental_routed_frac"] = float(getattr(self, 'last_elemental_routed_fraction', 0.0))

        # Creativity metrics (only when w_creat > 0). The within-group std, the part GRPO can
        # learn from, is logged in grpo_lightning as creativity/wg_std (the reward does not know B/K).
        if self.w_creat > 0.0 and self._creativity is not None and self.last_creativity:
            ct = torch.tensor(self.last_creativity)
            _nc = ct.numel()
            c = self._creativity
            metrics["reward/creativity_mean"] = ct.mean().item()
            metrics["reward/creativity_std"] = ct.std(unbiased=False).item() if _nc > 1 else 0.0
            metrics["reward/creativity_unique_frac"] = (sum(c.last_unique) / _nc) if _nc else 0.0
            metrics["reward/creativity_novel_frac"] = (sum(c.last_novel) / _nc) if _nc else 0.0
            metrics["reward/creativity_amd_fallback_frac"] = (c.last_amd_count / _nc) if _nc else 0.0
            metrics["reward/creativity_timeout_frac"] = (c.last_timeout_count / _nc) if _nc else 0.0
            metrics["reward/creativity_amd_fail_count"] = float(c.last_amd_fail_count)
            metrics["reward/creativity_amd_clamp_count"] = float(c.last_amd_clamp_count)
            metrics["reward/creativity_sec"] = float(c.last_elapsed_sec)
            if self.creat_on_relaxed:
                metrics["reward/creativity_relax_fallback_frac"] = (
                    float(self.last_creat_relax_fallback) / _nc) if _nc else 0.0

        if getattr(self, 'last_degenerate_cell_fraction', None) is not None:
            metrics["reward/degenerate_cell_fraction"] = float(self.last_degenerate_cell_fraction)
        if getattr(self, 'last_small_cell_fraction', None) is not None:
            metrics["reward/small_cell_fraction"] = float(self.last_small_cell_fraction)
        if getattr(self, 'last_oblate_cell_fraction', None) is not None:
            metrics["reward/oblate_cell_fraction"] = float(self.last_oblate_cell_fraction)
        if getattr(self, 'last_huge_cell_fraction', None) is not None:
            metrics["reward/huge_cell_fraction"] = float(self.last_huge_cell_fraction)
        if getattr(self, 'last_masked_fraction', None) is not None:
            metrics["reward/masked_fraction"] = float(self.last_masked_fraction)

        # Relaxation-before-reward diagnostics (only populated on the e_hull
        # relax path): RMSD(gen, relaxed) and ΔE_relax = E(gen) − E(relaxed).
        rr = getattr(self, 'last_relax_rmsd', None)
        de = getattr(self, 'last_relax_delta_e', None)
        if rr:
            rr_a = np.array(rr, dtype=float); rr_a = rr_a[np.isfinite(rr_a)]
            if rr_a.size:
                metrics["relax/rmsd_median"] = float(np.median(rr_a))
                metrics["relax/rmsd_mean"] = float(rr_a.mean())
                metrics["relax/rmsd_max"] = float(rr_a.max())
                metrics["relax/n_relaxed"] = float(rr_a.size)
        if de:
            de_a = np.array(de, dtype=float); de_a = de_a[np.isfinite(de_a)]
            if de_a.size:
                metrics["relax/delta_e_median"] = float(np.median(de_a))
                metrics["relax/delta_e_mean"] = float(de_a.mean())
                metrics["relax/delta_e_max"] = float(de_a.max())

        return gathered_rewards, metrics

