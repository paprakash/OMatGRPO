# omg/grpo/grpo_lightning.py
from __future__ import annotations
import copy
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import torch
import lightning as L
import wandb


from omg.datamodule import OMGData
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.sampler.sampler import Sampler
from omg.model.model import Model

from omg.grpo.reward import OMatGRPOReward
from omg.grpo.mmd_diversity import build_comp_matrix, mmd_reward, scale_per_batch, COMP_DIM


class OMatGRPOModule(L.LightningModule):
    """
    GRPO fine-tuning of an OMatG generator (de novo generation, groups of K rollouts that share x0).

    Relies on two methods of the stochastic interpolants:
      - integrate_with_logprob(x0, model, model_ref, fields=..., stochastic=True, return_trajectory=True)
          -> gen: OMGData, traj: per-step record, logp_old_step: per-field log-probabilities
      - step_logprob(model, traj, fields=...)
          -> per-field log-probabilities recomputed under `model` on the same trajectory
    The policy can act on positions, the lattice and the discrete species channel.
    """

    def __init__(
        self,
        si: StochasticInterpolants,
        sampler: Sampler,
        model: Model,
        k: int = 4,
        reward_offset: float = 0.3,
        eps_clip: float = 0.2,
        beta_kl: float = 0.05,
        beta_kl_species: float = 0.0,
        lr: float = 1e-4,
        reward_cfg: Optional[dict] = None,
        fields: Tuple[str, ...] = ("pos", "cell"),
        num_inner_epochs: int = 1,
        dng_mode: bool = False,
        alpha_pos: float = 1.0,
        alpha_cell: float = 1.0,
        alpha_species: float = 1.0,
        use_diversity: bool = False,
        div_tol: int = 3,
        div_buff: int = 6,
        use_mmd_diversity: bool = False,
        w_mmd_diversity: float = 1.0,
        mmd_comp_reference: Optional[str] = None,
        mmd_kernel: str = "poly",
        mmd_poly_c: float = 1.0,
        mmd_poly_d: int = 3,
        mmd_max_reference: int = 10000,
        mmd_norm: str = "minmax",
    ) -> None:
        super().__init__()
        self.automatic_optimization = False # Manual opt for PPO inner epoch loop
        self.si = si
        self.sampler = sampler
        self.model = model
        self.reward_offset = reward_offset
        self.dng_mode = bool(dng_mode)

        # The closed-form species KL assumes species noise eta = 0. Check it here, before data
        # loading and the UMA download, instead of failing inside the first training step.
        # _compute_analytical_kl asserts the same condition at run time.
        if "species" in tuple(fields) and float(beta_kl_species) > 0.0:
            si_species = None
            try:
                si_species = next(
                    (
                        s
                        for df, s in zip(si._data_fields, si._stochastic_interpolants)
                        if df.name == "species"
                    ),
                    None,
                )
            except Exception:
                si_species = None
            if si_species is None:
                raise RuntimeError(
                    "--beta_kl_species > 0 but the model has no species interpolant."
                )
            si_eta = getattr(si_species, "_noise", None)
            if si_eta is None or float(si_eta) != 0.0:
                raise RuntimeError(
                    "--beta_kl_species={} needs species noise eta = 0, but the species "
                    "interpolant has noise {!r}. Set the species noise to 0 "
                    "(--species_eta_override=0.0); the closed-form species KL assumes it."
                    .format(beta_kl_species, si_eta)
                )

        # Frozen reference copy for KL regularization
        self.model_ref = copy.deepcopy(model).eval()
        for p in self.model_ref.parameters():
            p.requires_grad_(False)

        # GRPO hyperparameters
        self.k = int(k)
        self.eps_clip = float(eps_clip)
        # Per-channel KL weights. `beta_kl` is the position KL weight (kept under both names);
        # the lattice channel has no KL term.
        self.beta_kl = float(beta_kl)
        self.beta_kl_pos = float(beta_kl)
        self.beta_kl_species = float(beta_kl_species)
        self.lr = float(lr)
        self.fields = tuple(fields)
        self.num_inner_epochs = int(num_inner_epochs)
        # Per-channel weights of the clipped surrogate (alpha in the paper)
        self.alpha_pos = float(alpha_pos)
        self.alpha_cell = float(alpha_cell)
        self.alpha_species = float(alpha_species)

        # Occurrence discount (MatInvent's within-group composition filter): repeated compositions
        # inside a group are pulled toward the worst reward, which counters within-group collapse
        # onto one composition. See _apply_diversity_penalty for the signed-reward formulation.
        self.use_diversity = bool(use_diversity)
        self.div_tol = int(div_tol)
        self.div_buff = int(div_buff)
        if self.use_diversity:
            print(f"[diversity] within-group composition penalty ON: "
                  f"Tol={self.div_tol}, Buff={self.div_buff} (Occ<=Tol keep, "
                  f"Occ>=Buff kill; multiplicative around reward floor).")

        # Compositional MMD bonus, following the compositional diversity term of Chemeleon2 with
        # fractional element vectors in place of learned embeddings (omg/grpo/mmd_diversity.py).
        # Each structure gets a leave-one-out coverage credit (r_indiv), computed over the whole
        # B*K batch against an MP-20 composition reference. It counters narrowing of the element
        # distribution across groups, which the within-group occurrence discount cannot see.
        # r_indiv does not depend on the reference-only term, so r_term=0.0 is passed and the
        # N*N reference self-kernel is never built.
        self.use_mmd_diversity = bool(use_mmd_diversity)
        self.w_mmd_diversity = float(w_mmd_diversity)
        self.mmd_kernel = str(mmd_kernel)
        self.mmd_poly_c = float(mmd_poly_c)
        self.mmd_poly_d = int(mmd_poly_d)
        # Per-batch scaling of r_indiv (as in Chemeleon2): 'minmax' scales it to [0, 1] over the
        # B*K batch each step, 'zscore' to mean 0 / std 1, 'none' keeps the raw value. Raw r_indiv
        # is of order 1e-5, so scaling is what lets a weight of order 1 be meaningful.
        self.mmd_norm = str(mmd_norm)
        self._mmd_ref = None
        if self.use_mmd_diversity:
            if mmd_comp_reference is None:
                raise ValueError(
                    "use_mmd_diversity=True requires --mmd_comp_reference=<path.pt> "
                    "(python scripts/build_references.py mmd)."
                )
            blob = torch.load(mmd_comp_reference, map_location="cpu", weights_only=False)
            ref = blob["matrix"] if isinstance(blob, dict) else blob
            ref = ref.to(torch.float64)
            assert ref.dim() == 2 and ref.size(1) == COMP_DIM, \
                f"mmd_comp_reference must be [N,{COMP_DIM}], got {tuple(ref.shape)}"
            if ref.size(0) > int(mmd_max_reference):
                g = torch.Generator().manual_seed(0)
                idx = torch.randperm(ref.size(0), generator=g)[: int(mmd_max_reference)]
                ref = ref[idx].contiguous()
            self._mmd_ref = ref  # [Nref, 119] float64, CPU
            print(f"[mmd] compositional MMD-LOO diversity ON: ref={tuple(ref.shape)} "
                  f"kernel={self.mmd_kernel} (c={self.mmd_poly_c}, d={self.mmd_poly_d}) "
                  f"norm={self.mmd_norm} w_mmd_diversity={self.w_mmd_diversity} src={mmd_comp_reference}")

        # The reward (and its UMA potential) is built once and reused every step.
        reward_kwargs = {'weights': {'rmsd': 0.0, 'energy': 1.0}, 'reward_offset': reward_offset}
        reward_kwargs.update(reward_cfg or {})
        self.reward_fn = OMatGRPOReward(**reward_kwargs)  # used for mattersim rmsd


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """On --resume, Lightning restores the policy, the optimizer and the step counter from the
        checkpoint. The KL reference must stay the prior: refuse a checkpoint whose saved
        reference differs from the one built from --init_checkpoint."""
        saved = {k[len("model_ref."):]: v for k, v in checkpoint.get("state_dict", {}).items()
                 if k.startswith("model_ref.")}
        if not saved:
            return
        current = self.model_ref.state_dict()
        if saved.keys() != current.keys() or any(
                not torch.equal(saved[k].detach().cpu(), current[k].detach().cpu()) for k in saved):
            raise ValueError(
                "The --resume checkpoint was trained against a different KL reference than the prior "
                "given by --init_checkpoint. Resume with the --init_checkpoint of the original run.")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def _repeat_groups(self, x: OMGData, k: int) -> OMGData:
        """
        Repeat each structure in the batch k times (grouping by same x0).
        Preserves per-structure order: [s0]*k, [s1]*k, ...
        """
        # Unpack per-structure boundaries
        ptr = x.ptr           # (B+1,)
        B = len(x.n_atoms)    # number of structures
        assert ptr.numel() == B + 1

        # Prepare containers
        species_list, pos_list, batch_list = [], [], []
        n_atoms_list, cell_list = [], []

        for i in range(B):
            a, b = ptr[i].item(), ptr[i + 1].item()
            n = b - a
            # Atom-wise fields
            sp = x.species[a:b]
            ps = x.pos[a:b]
            # Structure-wise fields
            cell_i = x.cell[i:i+1]  # (1,3,3)

            for _ in range(k):
                species_list.append(sp)
                pos_list.append(ps)
                batch_list.append(torch.full((n,), len(n_atoms_list), device=x.batch.device, dtype=x.batch.dtype))
                n_atoms_list.append(torch.tensor(n, device=x.n_atoms.device, dtype=x.n_atoms.dtype))
                cell_list.append(cell_i)

        # Concatenate
        species = torch.cat(species_list, dim=0)
        pos = torch.cat(pos_list, dim=0)
        batch = torch.cat(batch_list, dim=0)
        n_atoms = torch.stack(n_atoms_list, dim=0)
        cell = torch.cat(cell_list, dim=0)

        # Rebuild ptr
        ptr_new = torch.zeros(len(n_atoms) + 1, device=n_atoms.device, dtype=ptr.dtype)
        torch.cumsum(n_atoms, dim=0, out=ptr_new[1:])

        # Carry over property dict if present (not used in de novo RL)
        prop = getattr(x, "property", None)

        # Repackage OMGData
        rep = OMGData()
        rep.species = species
        rep.pos = pos
        rep.batch = batch
        rep.n_atoms = n_atoms
        rep.cell = cell
        rep.ptr = ptr_new
        rep.property = prop
        # Consistency checks of the repeated batch
        assert rep.ptr[0].item() == 0
        assert rep.ptr[-1].item() == rep.pos.shape[0]
        assert rep.n_atoms.sum().item() == rep.pos.shape[0]
        assert rep.cell.shape[0] == rep.n_atoms.shape[0]
        assert rep.batch.shape[0] == rep.pos.shape[0]
        if rep.n_atoms.numel() > 0:
            assert rep.batch.max().item() == rep.n_atoms.numel() - 1
        assert torch.equal(rep.ptr[1:] - rep.ptr[:-1], rep.n_atoms.to(rep.ptr.dtype))

        return rep

    @torch.no_grad()
    def _rollout_groups(self, x1: OMGData) -> Tuple[OMGData, OMGData, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Build x0 from sampler, repeat each structure k times, run stochastic SI with logprob tracking.
        Returns:
          x0_rep: repeated initial states,
          gen: generated OMGData for all BK samples,
          traj: opaque trajectory record for replay,
          logp_old_step: FloatTensor [BK, T] stepwise log-prob under old policy
        """
        B = int(x1.n_atoms.numel())
        K = self.k

        if self.dng_mode:
            # Frozen composition, stage 1: sample one composition per group from the
            # frozen reference model (no grad, B samples).
            # x0: all species are the mask token (0) under the
            # OMatG MaskDistribution; pos/cell are random draws from the
            # configured lattice/pos distributions.
            x0_phase1 = self.sampler.sample_p_0(x1).to(self.device)
            p1_uniq = x0_phase1.species.unique().tolist()
            print(f"[frozen composition, stage 1] x0 species unique (expect [0]): {p1_uniq}")
            assert p1_uniq == [0], \
                f"[frozen composition, stage 1] x0_phase1.species not pure mask; got unique={p1_uniq}"

            # Run the base model's native integrator driven by the frozen
            # reference policy. This fires the species DFM on top of pos/cell
            # updates; we only consume gen_phase1.species downstream.
            def model_fn_ref(x, t):
                return self.model_ref(x, t)

            gen_phase1 = self.si.integrate(x0_phase1, model_fn_ref)

            # Structure-size invariants: integration must not reorder atoms
            # or change the number of atoms per structure.
            assert torch.equal(gen_phase1.ptr, x1.ptr), \
                f"[frozen composition, stage 1] ptr drift: gen={gen_phase1.ptr.tolist()} vs x1={x1.ptr.tolist()}"
            assert torch.equal(gen_phase1.n_atoms, x1.n_atoms), \
                f"[frozen composition, stage 1] n_atoms drift: gen={gen_phase1.n_atoms.tolist()} vs x1={x1.n_atoms.tolist()}"

            dng_species = gen_phase1.species.clone()
            # DFM must resolve every mask token to a real Z in [1, 118].
            assert (dng_species > 0).all(), \
                f"[frozen composition, stage 1] {int((dng_species == 0).sum())} mask tokens survived integration"
            assert (dng_species <= 118).all(), \
                f"[frozen composition, stage 1] species > 118 present: max={int(dng_species.max())}"

            gen_uniq = dng_species.unique().tolist()
            print(f"[frozen composition, stage 1] Generated species unique values: {gen_uniq}")
            for g in range(min(2, B)):
                a1 = int(x1.ptr[g].item()); b1 = int(x1.ptr[g + 1].item())
                x1_sp = x1.species[a1:b1].tolist()
                dng_sp = dng_species[a1:b1].tolist()
                print(f"[frozen composition, stage 1] Group {g}: DNG species={dng_sp}, x1 species={x1_sp}, "
                      f"match={x1_sp == dng_sp}")

            # Frozen composition, stage 2: RL on positions (and lattice) for the sampled
            # compositions (B*K samples). Fresh random pos/cell, independent of stage 1.
            x0_phase2 = self.sampler.sample_p_0(x1).to(self.device)
            assert torch.equal(x0_phase2.ptr, x1.ptr), \
                f"[frozen composition, stage 2] x0_phase2.ptr drift from x1: {x0_phase2.ptr.tolist()} vs {x1.ptr.tolist()}"
            # Inject the DFM-generated compositions, replacing mask tokens.
            x0_phase2.species = dng_species
            print(f"[frozen composition, stage 2] x0_phase2.species matches dng_species: "
                  f"{torch.equal(x0_phase2.species, dng_species)}")

            # Replicate each group K times — identical composition within a group.
            x0_rep = self._repeat_groups(x0_phase2, K)

            # x0_rep.species must be identical across the K members of a group
            # and must match dng_species for the corresponding group. We
            # slice via ptr (not view-reshape) because n_atoms varies per
            # group, so the K×n_g sub-view is only valid inside one group.
            for g in range(B):
                a1 = int(x1.ptr[g].item())
                b1 = int(x1.ptr[g + 1].item())
                n_g = b1 - a1
                dng_slice = dng_species[a1:b1]
                a_start = int(x0_rep.ptr[g * K].item())
                a_end = a_start + K * n_g
                group = x0_rep.species[a_start:a_end].view(K, n_g)
                assert (group == group[0:1]).all(), \
                    f"[A1-DNG] x0_rep.species not K-identical in group g={g} (n={n_g})"
                assert (group > 0).all(), \
                    f"[A1-DNG] mask tokens in x0_rep group g={g}"
                assert torch.equal(group[0], dng_slice), \
                    f"[A1-DNG] x0_rep group g={g} member 0 != dng_species[{a1}:{b1}]"

            # Composition diversity across groups (warning only).
            comp_set = set()
            for g in range(B):
                a_g = int(x0_rep.ptr[g * K].item())
                n_g = int(x0_rep.n_atoms[g * K].item())
                comp_set.add(tuple(sorted(x0_rep.species[a_g:a_g + n_g].tolist())))
            print(f"[A4] Unique compositions across {B} groups: {len(comp_set)}")
            if len(comp_set) == 1 and B > 1:
                print("[A4] WARNING: all B groups share a single composition — "
                      "possible DFM collapse; continuing but flag for review.")

            # Same integrator as the species-learning path, started from the fixed compositions.
            gen, traj, logp_old_step = self.si.integrate_with_logprob(
                x0_rep, self.model, self.model_ref,
                fields=self.fields, stochastic=True, return_trajectory=True,
            )

            # gen.species must still be identical within each group ("species" is not in
            # self.fields, so the species branch of integrate_with_logprob does not act), and
            # member 0 must still match dng_species (no atom reordering during integration).
            for g in range(B):
                a1 = int(x1.ptr[g].item())
                b1 = int(x1.ptr[g + 1].item())
                n_g = b1 - a1
                dng_slice = dng_species[a1:b1]
                g_start = int(gen.ptr[g * K].item())
                g_end = g_start + K * n_g
                gen_group = gen.species[g_start:g_end].view(K, n_g)
                assert (gen_group == gen_group[0:1]).all(), \
                    f"[A2-DNG] gen.species not K-identical in group g={g} (n={n_g})"
                assert torch.equal(gen_group[0], dng_slice), \
                    f"[A3-DNG] gen.species group g={g} member 0 != dng_species[{a1}:{b1}]"
            print("[frozen composition, stage 2] Assertions A1-DNG, A2-DNG, A3-DNG passed")

            return x0_rep, gen, traj, logp_old_step  # the code below is for the other paths only

        # Draw base samples (x0) with shapes defined by x1 (only the shapes of x1 are used)
        x0 = self.sampler.sample_p_0(x1).to(self.device)
        # When species is learned, the sampler's all-mask species must reach x0_rep so the
        # discrete channel generates the composition. Otherwise the composition is copied from
        # x1, so every member of a group shares a fixed composition.
        species_is_learned = "species" in self.fields
        if not species_is_learned:
            x0.species = x1.species.clone()
        # Repeat per-structure k times to form groups
        x0_rep = self._repeat_groups(x0, self.k)

        # Sanity check of _repeat_groups: the K members of a group start identical. When species
        # is learned the start is all-mask; otherwise member 0 equals the x1 composition.
        for g in range(B):
            a1 = int(x1.ptr[g].item())
            b1 = int(x1.ptr[g + 1].item())
            n_g = b1 - a1
            x1_slice = x1.species[a1:b1]
            a_start = int(x0_rep.ptr[g * K].item())
            a_end = a_start + K * n_g
            group = x0_rep.species[a_start:a_end].view(K, n_g)
            assert (group == group[0:1]).all(), \
                f"[A1] x0_rep.species not K-identical in group g={g} (n={n_g})"
            if species_is_learned:
                assert (group == 0).all(), \
                    f"[A1] expected an all-mask start in group g={g} (n={n_g}) when species is learned; " \
                    f"got non-mask tokens"
            else:
                assert torch.equal(group[0], x1_slice), \
                    f"[A1] x0_rep.species group g={g} member 0 != x1.species[{a1}:{b1}]"

        # Stochastic integration with per-step log-probabilities for every learned channel
        gen, traj, logp_old_step = self.si.integrate_with_logprob(
            x0_rep, self.model, self.model_ref, fields=self.fields, stochastic=True, return_trajectory=True
        )
        # Post-integration checks.
        #   Species learned: each member samples its own composition, so count the distinct
        #     compositions per group and warn below ceil(K/4).
        #   Species not learned: compositions must stay identical within a group [A2] and
        #     equal to x1 atom by atom [A3].
        diversity_threshold = (K + 3) // 4  # ceil(K / 4)
        unique_comps_counts = []
        for g in range(B):
            a1 = int(x1.ptr[g].item())
            b1 = int(x1.ptr[g + 1].item())
            n_g = b1 - a1
            x1_slice = x1.species[a1:b1]
            g_start = int(gen.ptr[g * K].item())
            g_end = g_start + K * n_g
            gen_group = gen.species[g_start:g_end].view(K, n_g)
            if species_is_learned:
                comps = {tuple(sorted(gen_group[i].tolist())) for i in range(K)}
                unique_comps_counts.append(len(comps))
                # A warning, not an error: the occurrence discount acts on collapsed groups,
                # so training continues and the count is logged.
                if len(comps) < diversity_threshold:
                    print(f"[A2 WARN] group g={g}: {len(comps)} unique compositions "
                          f"< {diversity_threshold} (K={K}) — within-group composition "
                          f"collapse; continuing (soft guard).")
            else:
                assert (gen_group == gen_group[0:1]).all(), \
                    (f"[A2] gen.species not K-identical in group g={g} (n={n_g}): species "
                     f"changed although species is not learned")
                assert torch.equal(gen_group[0], x1_slice), \
                    f"[A3] gen.species group g={g} member 0 != x1.species[{a1}:{b1}]: atom-order scrambled"
        # When species is learned: log the per-group distinct-composition counts and an
        # element-frequency histogram over all B*K structures.
        if species_is_learned and unique_comps_counts:
            counts_t = torch.tensor(unique_comps_counts, dtype=torch.float32)
            self.log("species/unique_comps_per_group_mean", counts_t.mean(),
                     on_step=True, prog_bar=False, batch_size=B)
            self.log("species/unique_comps_per_group_std", counts_t.std(unbiased=False),
                     on_step=True, prog_bar=False, batch_size=B)
            self.log("species/unique_comps_per_group_min", counts_t.min(),
                     on_step=True, prog_bar=False, batch_size=B)
            self.log("species/unique_comps_per_group_median", counts_t.median(),
                     on_step=True, prog_bar=False, batch_size=B)
            # number of groups below the ceil(K/4) warning threshold
            self.log("species/n_groups_below_a2", (counts_t < diversity_threshold).float().sum(),
                     on_step=True, prog_bar=False, batch_size=B)
            with torch.no_grad():
                unmasked_Z = gen.species[gen.species > 0]
                if unmasked_Z.numel() > 0:
                    elem_freq = torch.bincount(unmasked_Z.cpu(), minlength=119)
                    # Shannon entropy of the element frequencies over the B*K structures and the
                    # number of distinct elements. They measure narrowing of the element
                    # distribution directly; the fraction of unique structures does not.
                    p = elem_freq.to(torch.float64)
                    p = p[p > 0]
                    p = p / p.sum()
                    shannon = float(-(p * p.log()).sum())   # nats
                    n_distinct = int((elem_freq > 0).sum())
                    self.last_elem_shannon = shannon        # for the [watch] live monitor
                    self.last_n_distinct = n_distinct
                    self.log("diversity/elem_shannon_entropy", shannon,
                             on_step=True, prog_bar=False, batch_size=B)
                    self.log("diversity/n_distinct_elements", float(n_distinct),
                             on_step=True, prog_bar=False, batch_size=B)
                    if self.logger is not None:
                        self.logger.experiment.log(
                            {"species/element_freq_hist": wandb.Histogram(
                                np_histogram=(elem_freq.numpy(),
                                              torch.arange(120).numpy()))},
                            commit=False,
                        )
        # Shapes: gen is OMGData with BK structures; logp_old_step: [BK, T]
        return x0_rep, gen, traj, logp_old_step

    def _apply_diversity_penalty(self, rewards: torch.Tensor, gen, B: int) -> torch.Tensor:
        """MatInvent within-group composition-occurrence penalty (negative-reward-safe).

        Within each GRPO group of K, count how many times each structure's composition
        (full element multiset = sorted species list, the same key as the per-group composition
        count in _rollout_groups; within a group n_atoms is fixed, so this is the exact composition)
        occurs. Piecewise
        factor on occurrence Occ:
            Occ <= Tol           -> 1.0   (keep)
            Tol < Occ < Buff     -> (Buff - Occ) / (Buff - Tol)   (linear ramp)
            Occ >= Buff          -> 0.0   (kill)

        Signed rewards: MatInvent's `base*factor` assumes base >= 0, where factor=0 means the
        worst (zero) reward. Our reward can be negative, and scaling a negative reward toward 0
        would make a repeated bad structure better. So the factor is applied around the reward
        floor (the worst achievable reward) instead of around 0:
            final = floor + (base - floor) * factor
        This equals base*factor when floor=0 (MatInvent's case). For signed rewards it keeps
        final in [floor, base], with final == base only when factor == 1, so a repeated
        composition is always pulled toward the worst reward whatever the sign of base.
        floor = -(ehull_cap*w_ehull + rmsd_geom_clamp*w_rmsd_geom), lowered to the batch minimum
        if any reward sits below it.
        """
        K = self.k
        rf = self.reward_fn
        w_ehull = float(getattr(rf, 'w_ehull', 1.0))
        w_rmsd = float(getattr(rf, 'w_rmsd_geom', 0.0))
        rmsd_clamp = float(getattr(rf, 'rmsd_geom_clamp', 3.0))
        EH_CLAMP_MAX = float(getattr(rf, 'ehull_cap', 5.0))  # upper clamp of the E_hull term in reward.py
        floor = -(EH_CLAMP_MAX * w_ehull + rmsd_clamp * w_rmsd)  # weights['energy'] is a gate, not a scale
        floor = min(floor, float(rewards.min().item())) - 1e-6  # guard: base >= floor always

        tol, buff = float(self.div_tol), float(self.div_buff)
        device = rewards.device
        factors = torch.ones(B * K, device=device)
        gen_sp = gen.species
        n_penalized = n_killed = 0
        uniq_per_group = []
        for g in range(B):
            g_start = int(gen.ptr[g * K].item())
            n_g = int(gen.n_atoms[g * K].item())
            grp = gen_sp[g_start:g_start + K * n_g].view(K, n_g)
            keys = [tuple(sorted(grp[i].tolist())) for i in range(K)]
            counts = Counter(keys)
            uniq_per_group.append(len(counts))
            for i in range(K):
                occ = counts[keys[i]]
                if occ <= self.div_tol:
                    f = 1.0
                elif occ >= self.div_buff:
                    f = 0.0
                else:
                    f = (buff - occ) / (buff - tol)
                factors[g * K + i] = f
                if f < 1.0:
                    n_penalized += 1
                if f == 0.0:
                    n_killed += 1

        penalized = floor + (rewards - floor) * factors
        # Logging: how strongly the discount acts.
        self.log("diversity/n_penalized", float(n_penalized), on_step=True, prog_bar=False, batch_size=B)
        self.log("diversity/n_killed", float(n_killed), on_step=True, prog_bar=False, batch_size=B)
        self.log("diversity/mean_factor", factors.mean(), on_step=True, prog_bar=False, batch_size=B)
        self.log("diversity/min_factor", factors.min(), on_step=True, prog_bar=False, batch_size=B)
        print(f"[diversity] penalized={n_penalized}/{B*K} (killed={n_killed}) "
              f"mean_factor={factors.mean().item():.3f} floor={floor:.3f} "
              f"uniq_comps/group={uniq_per_group} (K={K})")
        return penalized

    def _mmd_diversity_bonus(self, gen, B: int) -> torch.Tensor:
        """Per-sample MMD leave-one-out coverage credit over the whole B*K batch.

        Builds the [BK, 119] fractional-element matrix from gen (same Z-slicing as
        _apply_diversity_penalty / process_data), computes the LOO marginal credit
        r_indiv against the MP-20 composition reference, and returns [BK] on CPU
        (the caller moves it to the rewards device). r_indiv > 0 => the structure
        improves the batch's coverage of the reference distribution; < 0 => redundant.

        Cheap (119-dim, BK ~ tens of rows, ref ~10k): we pass r_term=0.0 because
        r_indiv is independent of the reference-only term (see mmd_diversity.py).
        """
        z_gen = build_comp_matrix(
            gen.species.detach().cpu().long(),
            gen.n_atoms.detach().cpu().long(),
            dim=COMP_DIM,
        )  # [BK, 119] float64
        out = mmd_reward(
            z_gen, self._mmd_ref,
            kernel=self.mmd_kernel, c=self.mmd_poly_c, d=self.mmd_poly_d,
            r_term=0.0,
        )
        r_indiv = out["r_indiv"].to(torch.float32)  # [BK] raw LOO credit
        # Per-batch scaling (as in Chemeleon2), so a weight of order 1 is meaningful.
        r_scaled = scale_per_batch(r_indiv, self.mmd_norm)
        # Kept for logging in training_step: the scaled values are the ones added to the reward.
        self.last_mmd_r_indiv = r_scaled.detach()
        self.last_mmd_r_indiv_raw = r_indiv.detach()
        return r_scaled

    def _compute_group_advantages(self, rewards: torch.Tensor, B: int,
                                  neutral_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        K = self.k
        rewards_2d = rewards.view(B, K)

        # Abstention (sparse_route='neutral'). neutral_mask marks structures whose hull
        # reference is too sparse to trust; this includes structures that failed a guard or the
        # hull lookup, which have no reference entries. Deep-below-hull and single-element
        # structures are not in the mask; reward.py routes them to the penalty. Abstaining
        # structures get zero advantage: the group baseline and std use the trusted members only,
        # every member is normalized against them, and the abstaining ones are then set to 0.
        # A group with fewer than 2 trusted members has no usable baseline and is dead.
        # Without abstentions (neutral_mask None or all False) the plain path below is used,
        # which is the same math with every member trusted.
        use_neutral = neutral_mask is not None and bool(neutral_mask.any())
        if use_neutral:
            nm = neutral_mask.view(B, K).to(rewards_2d.dtype)        # 1.0 neutral, 0.0 trusted
            trusted = 1.0 - nm
            n_t = trusted.sum(dim=1, keepdim=True)                   # [B,1] trusted count
            denom = n_t.clamp(min=1.0)
            # Step 1: ±3σ clip using TRUSTED-only group stats.
            mean_t = (rewards_2d * trusted).sum(dim=1, keepdim=True) / denom
            var_t = (((rewards_2d - mean_t) ** 2) * trusted).sum(dim=1, keepdim=True) / denom
            std_t = var_t.sqrt()
            clip_lo = mean_t - 3.0 * (std_t + 1e-8)
            clip_hi = mean_t + 3.0 * (std_t + 1e-8)
            rc = rewards_2d.clamp(min=clip_lo, max=clip_hi)
            # Step 2: baseline/std over TRUSTED-only clipped members; normalize all.
            base = (rc * trusted).sum(dim=1, keepdim=True) / denom
            varc = (((rc - base) ** 2) * trusted).sum(dim=1, keepdim=True) / denom
            stdc = varc.sqrt()
            adv = (rc - base) / (stdc + 1e-8)
            adv = adv * trusted                                     # neutral members -> 0
            # Step 3: dead groups — <2 trusted members (no usable baseline) OR near-zero std.
            dead_mask = (n_t.squeeze(1) < 2.0) | (stdc.squeeze(1) < 1e-4)
            adv[dead_mask] = 0.0
            self.log("group/dead_count", dead_mask.sum().float(), prog_bar=False)
            self.log("group/dead_frac", dead_mask.float().mean(), prog_bar=True)
            self.log("group/neutral_frac", nm.mean(), on_step=True, prog_bar=False, batch_size=B)
            self.last_dead_frac = float(dead_mask.float().mean())
            self.last_neutral_frac = float(nm.mean())
            return adv.reshape(B * K)

        # Step 1: Clip rewards to ±3σ within each group
        group_mean = rewards_2d.mean(dim=1, keepdim=True)
        group_std = rewards_2d.std(dim=1, keepdim=True, unbiased=False)
        clip_lo = group_mean - 3.0 * (group_std + 1e-8)
        clip_hi = group_mean + 3.0 * (group_std + 1e-8)
        rewards_clipped = rewards_2d.clamp(min=clip_lo, max=clip_hi)

        # Step 2: Per-group advantage with std normalization (Eq. 9, both OMatG-IRL and Flow-GRPO)
        baseline = rewards_clipped.mean(dim=1, keepdim=True)
        std = rewards_clipped.std(dim=1, keepdim=True, unbiased=False)
        adv = (rewards_clipped - baseline) / (std + 1e-8)

        # Step 3: Dead group filtering — zero out groups with near-zero variance
        # (all K structures got ~identical rewards → no learning signal, just ÷1e-8 noise)
        dead_mask = (std.squeeze(1) < 1e-4)  # [B] boolean
        adv[dead_mask] = 0.0
        self.log("group/dead_count", dead_mask.sum().float(), prog_bar=False)
        self.log("group/dead_frac", dead_mask.float().mean(), prog_bar=True)
        self.last_dead_frac = float(dead_mask.float().mean())
        self.last_neutral_frac = 0.0

        return adv.reshape(B * K)


    def _ppo_grpo_loss(
        self,
        logp_old_step: Dict[str, Any],    # {"pos":[BK,T], "cell":[BK,T], "species":{"logp","step_idx","struct_idx"}}
        logp_new_step: Dict[str, Any],    # same shape as logp_old_step
        advantages: torch.Tensor,         # [BK,]
        analytical_kl: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Per-field PPO (GRPO) loss: separate clipped surrogate for each channel,
        weighted by α_pos, α_cell, α_species. Pos/cell are per-step-and-structure
        [BK, T]; species is event-level [E] with struct_idx for advantage
        broadcasting. KL-to-ref penalty shared across fields (analytical, pos/cell).
        """
        # Derive device from inputs (works even when `self` hasn't been placed
        # on a device yet, e.g. the bare-instance test fixture).
        device = advantages.device
        stats: Dict[str, torch.Tensor] = {}
        per_field_losses: Dict[str, torch.Tensor] = {}

        def _per_step_term(field_name: str):
            old = logp_old_step[field_name]
            new = logp_new_step[field_name]
            BK, T = old.shape
            A = advantages.view(BK, 1).expand(BK, T).detach()
            log_ratio = new - old
            ratio = torch.exp(log_ratio)
            unclipped = ratio * A
            clipped = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * A
            field_loss = -torch.mean(torch.minimum(unclipped, clipped))
            with torch.no_grad():
                clipfrac_f = torch.mean((torch.abs(ratio - 1.0) > self.eps_clip).float())
                approx_kl_f = 0.5 * torch.mean(log_ratio ** 2)
            stats[f"stats/ratio_mean_{field_name}"] = ratio.mean().detach()
            stats[f"stats/ratio_std_{field_name}"] = ratio.std().detach()
            stats[f"stats/log_ratio_absmax_{field_name}"] = log_ratio.abs().max().detach()
            stats[f"stats/clipfrac_{field_name}"] = clipfrac_f
            stats[f"stats/approx_kl_{field_name}"] = approx_kl_f
            return field_loss, ratio

        def _event_term():
            old_sp = logp_old_step["species"]
            new_sp = logp_new_step["species"]
            E = old_sp["logp"].shape[0]
            if E == 0:
                zero = torch.tensor(0.0, device=device)
                stats["stats/ratio_mean_species"] = zero.detach()
                stats["stats/ratio_std_species"] = zero.detach()
                stats["stats/log_ratio_absmax_species"] = zero.detach()
                stats["stats/clipfrac_species"] = zero.detach()
                stats["stats/approx_kl_species"] = zero.detach()
                stats["stats/species_event_count"] = torch.tensor(0.0, device=device)
                return zero, None
            A_ev = advantages[old_sp["struct_idx"]].detach()
            log_ratio = new_sp["logp"] - old_sp["logp"]
            ratio = torch.exp(log_ratio)
            unclipped = ratio * A_ev
            clipped = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * A_ev
            field_loss = -torch.mean(torch.minimum(unclipped, clipped))
            with torch.no_grad():
                clipfrac_f = torch.mean((torch.abs(ratio - 1.0) > self.eps_clip).float())
                approx_kl_f = 0.5 * torch.mean(log_ratio ** 2)
            stats["stats/ratio_mean_species"] = ratio.mean().detach()
            stats["stats/ratio_std_species"] = ratio.std().detach() if ratio.numel() > 1 else torch.tensor(0.0, device=device)
            stats["stats/log_ratio_absmax_species"] = log_ratio.abs().max().detach()
            stats["stats/clipfrac_species"] = clipfrac_f
            stats["stats/approx_kl_species"] = approx_kl_f
            stats["stats/species_event_count"] = torch.tensor(float(E), device=device)
            return field_loss, ratio

        if "pos" in self.fields and "pos" in logp_old_step:
            per_field_losses["pos"], _ = _per_step_term("pos")
        if "cell" in self.fields and "cell" in logp_old_step:
            per_field_losses["cell"], _ = _per_step_term("cell")
        if "species" in self.fields and "species" in logp_old_step:
            per_field_losses["species"], _ = _event_term()

        alpha_map = {"pos": self.alpha_pos, "cell": self.alpha_cell, "species": self.alpha_species}
        if per_field_losses:
            policy_loss = sum(alpha_map[k] * v for k, v in per_field_losses.items())
        else:
            policy_loss = torch.tensor(0.0, device=device)

        # Unweighted per-channel loss magnitudes and the alpha-weighted species/pos ratio, a cheap
        # proxy for the ratio of their gradient norms (an exact ratio needs one backward per term).
        with torch.no_grad():
            if "pos" in per_field_losses:
                stats["stats/loss_pos_raw"] = per_field_losses["pos"].detach().abs()
            if "species" in per_field_losses:
                stats["stats/loss_species_raw"] = per_field_losses["species"].detach().abs()
            if "pos" in per_field_losses and "species" in per_field_losses:
                pos_mag = per_field_losses["pos"].detach().abs()
                species_mag = per_field_losses["species"].detach().abs()
                stats["stats/grad_ratio_species_over_pos"] = (
                    self.alpha_species * species_mag / (self.alpha_pos * pos_mag + 1e-12)
                )
        # Per-channel KL with per-channel beta; analytical_kl is a dict {pos, species} (no KL on
        # the lattice). The getattr fallbacks let test fixtures built via __new__ set only
        # `beta_kl` (the position weight).
        beta_map = {
            "pos": getattr(self, "beta_kl_pos", getattr(self, "beta_kl", 0.0)),
            "species": getattr(self, "beta_kl_species", 0.0),
        }
        kl_field = {f: torch.tensor(0.0, device=device) for f in beta_map}
        if isinstance(analytical_kl, dict):
            for f in beta_map:
                if f in analytical_kl:
                    kl_field[f] = analytical_kl[f]
        elif analytical_kl is None:
            pass
        else:
            raise TypeError(
                "analytical_kl must be Dict[str, Tensor]; got "
                f"{type(analytical_kl).__name__}"
            )

        kl_total = sum(beta_map[f] * kl_field[f] for f in beta_map)

        loss = policy_loss + kl_total

        # Channel-agnostic keys (stats/ratio_mean etc.) report the position channel, or the
        # lattice or species channel when positions are not learned.
        primary = "pos" if "pos" in per_field_losses else (
            "cell" if "cell" in per_field_losses else (
                "species" if "species" in per_field_losses else None
            )
        )
        if primary is not None:
            stats["stats/ratio_mean"] = stats[f"stats/ratio_mean_{primary}"]
            stats["stats/ratio_std"] = stats[f"stats/ratio_std_{primary}"]
            stats["stats/log_ratio_absmax"] = stats[f"stats/log_ratio_absmax_{primary}"]
            stats["stats/clipfrac"] = stats[f"stats/clipfrac_{primary}"]
            stats["stats/approx_kl"] = stats[f"stats/approx_kl_{primary}"]
        else:
            zero = torch.tensor(0.0, device=device)
            for k in ("ratio_mean", "ratio_std", "log_ratio_absmax", "clipfrac", "approx_kl"):
                stats[f"stats/{k}"] = zero

        # |ratio - 1| distribution on the same channel
        if primary in ("pos", "cell"):
            with torch.no_grad():
                old_p = logp_old_step[primary]
                new_p = logp_new_step[primary]
                r_m1_abs = (torch.exp(new_p - old_p) - 1.0).abs()
            stats["stats/ratio_m1_abs_mean"] = r_m1_abs.mean()
            stats["stats/ratio_m1_abs_std"] = r_m1_abs.std() if r_m1_abs.numel() > 1 else torch.tensor(0.0, device=device)
            stats["stats/ratio_m1_abs_min"] = r_m1_abs.min()
            stats["stats/ratio_m1_abs_max"] = r_m1_abs.max()
        else:
            zero = torch.tensor(0.0, device=device)
            for k in ("ratio_m1_abs_mean", "ratio_m1_abs_std", "ratio_m1_abs_min", "ratio_m1_abs_max"):
                stats[f"stats/{k}"] = zero

        stats["loss/policy"] = policy_loss
        # Per-field raw (unweighted) KL.
        stats["loss/kl_pos"] = kl_field["pos"].detach()
        stats["loss/kl_species"] = kl_field["species"].detach()
        # beta-weighted total KL
        stats["loss/kl"] = kl_total.detach()
        stats["loss/total"] = loss
        # stats/analytical_kl is the position KL (same as stats/analytical_kl_pos).
        stats["stats/analytical_kl"] = kl_field["pos"].detach()
        stats["stats/analytical_kl_pos"] = kl_field["pos"].detach()
        stats["stats/analytical_kl_species"] = kl_field["species"].detach()
        # unweighted species KL (no beta scaling)
        stats["stats/analytical_kl_species_unweighted"] = kl_field["species"].detach()

        return loss, stats
    
    def _total_grad_norm(self) -> torch.Tensor:
        """L2 norm of all current parameter gradients (diagnostics)."""
        sq = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            if p.grad is not None:
                sq = sq + p.grad.detach().float().pow(2).sum()
        return sq.sqrt()

    def _compute_analytical_kl(
        self,
        drifts_new,
        drifts_ref,
        traj,
        n_atoms_float,
        BK,
        species_dist_new=None,
        species_dist_ref=None,
    ):
        """Per-channel analytical KL to the reference policy.

        Returns {pos, species}. Positions: Girsanov drift difference of the two SDEs. Species:
        closed-form categorical KL of the unmask events, which needs species noise eta = 0
        (the Bernoulli unmask probabilities then cancel). No KL term acts on the lattice.
        """
        field2si = {
            df.name: si_
            for df, si_ in zip(self.si._data_fields, self.si._stochastic_interpolants)
        }
        times = traj["times"]
        dt_vec = traj["dt"]
        atom_to_struct = traj["batch"]
        T = len(times)

        def _zero():
            return torch.tensor(0.0, device=self.device)

        kl_dict = {"pos": _zero(), "species": _zero()}

        # ── positions: Girsanov drift KL ──
        if "pos" in self.fields and "pos" in drifts_new and "pos" in drifts_ref:
            kl_pos = _zero()
            si_pos = field2si["pos"]
            for j in range(T):
                eps_t = si_pos._epsilon.epsilon(times[j])
                drift_diff = drifts_new["pos"][j] - drifts_ref["pos"][j]
                kl_per_atom = (
                    (drift_diff ** 2).sum(dim=-1) * dt_vec[j] / (4.0 * eps_t + 1e-12)
                )
                kl_per_struct = torch.zeros(BK, device=self.device, dtype=torch.float32)
                kl_per_struct.index_add_(
                    0, atom_to_struct, kl_per_atom.to(torch.float32)
                )
                kl_pos = kl_pos + (kl_per_struct / n_atoms_float).mean()
            kl_dict["pos"] = kl_pos / T

        # ── species: closed-form categorical KL at eta = 0 ──
        if (
            "species" in self.fields
            and species_dist_new is not None
            and species_dist_ref is not None
            and species_dist_new["logits"].numel() > 0
        ):
            si_sp = field2si.get("species", None)
            # Hard guard: η != 0 voids the Bernoulli cancellation; refuse to
            # silently miscompute. Caller must run with noise=0 if they want
            # species analytical KL.
            assert si_sp is not None and getattr(si_sp, "_noise", None) == 0.0, (
                "Species analytical KL assumes η=0 Bernoulli cancellation; "
                "got noise={!r}".format(getattr(si_sp, "_noise", None))
            )
            # Atom-set drift guard.
            assert torch.equal(
                species_dist_new["step_idx"], species_dist_ref["step_idx"]
            ), "species_dist step_idx mismatch between model and model_ref passes"
            assert torch.equal(
                species_dist_new["struct_idx"], species_dist_ref["struct_idx"]
            ), "species_dist struct_idx mismatch between model and model_ref passes"

            new_logits = species_dist_new["logits"]   # [E, S], grad via model
            ref_logits = species_dist_ref["logits"]   # [E, S], no grad
            log_p = torch.log_softmax(new_logits, dim=-1)
            log_q = torch.log_softmax(ref_logits, dim=-1)
            p = log_p.exp()
            kl_per_event = (p * (log_p - log_q)).sum(dim=-1)            # [E]
            weighted = species_dist_new["p_unmask"] * kl_per_event       # [E]

            kl_per_struct = torch.zeros(BK, device=self.device, dtype=torch.float32)
            kl_per_struct.index_add_(
                0, species_dist_new["struct_idx"], weighted.to(torch.float32)
            )
            kl_dict["species"] = (kl_per_struct / n_atoms_float).mean() / T

        return kl_dict
    
    
    def training_step(self, x1: OMGData, batch_idx: int):
        """
        De novo GRPO step with PPO inner epoch loop:
          1. Rollout + rewards + advantages (expensive, done once)
          2. Reference model drifts/logprobs (done once, frozen)
          3. Inner epoch loop: recompute logp_new, KL, loss, backward, step
        """
        import time
        import psutil
        opt = self.optimizers()

        # ── 1. Rollout (no grad) ──────────────────────────────────────
        with torch.no_grad():
            x0_rep, gen, traj, logp_old_step = self._rollout_groups(x1)

        # logp_old_step is a per-channel dict; pos and cell carry [BK, T].
        _primary_field = "pos" if "pos" in logp_old_step else "cell"
        BK, T = logp_old_step[_primary_field].shape
        B = len(x1.n_atoms)
        assert BK == B * self.k, f"Inconsistent BK: got {BK}, expected {B}*{self.k}"

        # ── 2. Rewards and advantages (fixed for all inner epochs) ────
        reward_start_time = time.time()
        try:
            rewards, reward_metrics = self.reward_fn(gen, aux=None, step=self.global_step)
            assert rewards.shape == (BK,)
            if self.use_diversity:        # occurrence discount (repeats within a group)
                rewards = self._apply_diversity_penalty(rewards, gen, B)
            if self.use_mmd_diversity:    # MMD coverage bonus over the whole batch
                rewards = rewards + self.w_mmd_diversity * self._mmd_diversity_bonus(gen, B).to(rewards.device)
            # Abstention: structures in the reward's sparse mask get zero advantage. The mask is
            # used only under sparse_route='neutral'; otherwise the plain advantage path runs.
            neutral_mask = None
            if getattr(self.reward_fn, 'sparse_route', 'worst') == 'neutral':
                m = getattr(self.reward_fn, 'last_sparse_neutral_mask', None)
                if m is not None:
                    neutral_mask = m.view(BK).to(rewards.device)
            advantages = self._compute_group_advantages(rewards, B, neutral_mask=neutral_mask)
            # [watch]: one stdout line per rollout with the main health numbers, easy to tail.
            try:
                rf = self.reward_fn
                wl = [f"step={self.global_step}",
                      f"dead_frac={getattr(self, 'last_dead_frac', float('nan')):.3f}",
                      f"neutral_frac={getattr(self, 'last_neutral_frac', 0.0):.3f}",
                      f"sparse_untrusted={getattr(rf, 'last_sparse_untrusted_fraction', float('nan')):.3f}",
                      f"H_elem={getattr(self, 'last_elem_shannon', float('nan')):.4f}",
                      f"n_elem={getattr(self, 'last_n_distinct', -1)}"]
                if self.use_mmd_diversity and getattr(self, 'last_mmd_r_indiv', None) is not None:
                    ri = self.last_mmd_r_indiv.float()
                    if ri.numel() == B * self.k:
                        mmd_wg = float((self.w_mmd_diversity * ri).view(B, self.k).std(dim=1, unbiased=False).mean())
                        wl.append(f"mmd_bonus_wg_std={mmd_wg:.4f}")
                eh = getattr(rf, 'last_ehull_clamped_term', None)
                if eh and len(eh) == B * self.k:
                    eht = torch.tensor(eh, dtype=torch.float)
                    if torch.isfinite(eht).all():
                        wl.append(f"ehull_term_wg_std={float(eht.view(B, self.k).std(dim=1, unbiased=False).mean()):.4f}")
                cv = getattr(rf, 'last_creativity', None)
                if float(getattr(rf, 'w_creat', 0.0)) > 0.0 and cv and len(cv) == B * self.k:
                    cvt = torch.tensor(cv, dtype=torch.float) * float(rf.w_creat)
                    wl.append(f"creativity_wg_std={float(cvt.view(B, self.k).std(dim=1, unbiased=False).mean()):.4f}")
                print("[watch] " + " ".join(wl), flush=True)
            except Exception as e:
                print(f"[watch] (skipped: {e})", flush=True)
        except TimeoutError:
            print("Reward timed out; skipping this training step.")
            return None
        reward_time = time.time() - reward_start_time

        if hasattr(self.reward_fn, '_get_current_fmax'):
            self.log("reward/current_fmax", self.reward_fn._get_current_fmax(), prog_bar=False)
            
        # Within-group spread of the energy: GRPO learns only from differences within a group.
        if hasattr(self.reward_fn, 'last_energy_per_atom') and self.reward_fn.last_energy_per_atom:
            epa = torch.tensor(self.reward_fn.last_energy_per_atom, device=self.device)
            if epa.numel() == B * self.k and torch.isfinite(epa).all():
                epa_2d = epa.view(B, self.k)
                within_std = epa_2d.std(dim=1, unbiased=False)     # [B]
                within_range = epa_2d.max(dim=1).values - epa_2d.min(dim=1).values  # [B]
                self.log("energy/within_group_std_mean", within_std.mean(),
                        on_step=True, prog_bar=False, batch_size=B)
                self.log("energy/within_group_std_min", within_std.min(),
                        on_step=True, prog_bar=False, batch_size=B)
                self.log("energy/within_group_range_mean", within_range.mean(),
                        on_step=True, prog_bar=False, batch_size=B)

        # With the residual geometry term (w_rmsd_geom > 0, reward-hacking appendix): within-group std and
        # mean of the E_hull and geometry terms, to see how their balance moves during training.
        if float(getattr(self.reward_fn, 'w_rmsd_geom', 0.0)) > 0.0:
            for tag, attr in (("ehull_term", "last_ehull_clamped_term"),
                              ("rmsd_term", "last_rmsd_geom_clamped")):
                vals = getattr(self.reward_fn, attr, None)
                if vals and len(vals) == B * self.k:
                    t = torch.tensor(vals, device=self.device, dtype=torch.float)
                    if torch.isfinite(t).all():
                        t2d = t.view(B, self.k)
                        self.log(f"combined/{tag}_within_group_std_mean",
                                 t2d.std(dim=1, unbiased=True).mean(),
                                 on_step=True, prog_bar=False, batch_size=B)
                        self.log(f"combined/{tag}_mean", t.mean(),
                                 on_step=True, prog_bar=False, batch_size=B)

        # MMD bonus magnitudes (scaled bonus w_mmd * r_indiv). The within-group std is the part
        # that survives advantage normalization; a constant within a group is subtracted out.
        if self.use_mmd_diversity and getattr(self, 'last_mmd_r_indiv', None) is not None:
            ri = self.last_mmd_r_indiv.to(self.device).float()
            if ri.numel() == B * self.k:
                bonus = self.w_mmd_diversity * ri
                b2d = bonus.view(B, self.k)
                self.log("mmd/r_indiv_mean", ri.mean(), on_step=True, prog_bar=False, batch_size=B)
                self.log("mmd/r_indiv_std", ri.std(unbiased=False), on_step=True, prog_bar=False, batch_size=B)
                self.log("mmd/bonus_abs_mean", bonus.abs().mean(), on_step=True, prog_bar=False, batch_size=B)
                self.log("mmd/bonus_within_group_std_mean", b2d.std(dim=1, unbiased=False).mean(),
                         on_step=True, prog_bar=False, batch_size=B)

        # Creativity term: mean and within-group std of the weighted term. A term that is
        # constant within groups is removed by the group baseline and gives no gradient, so
        # creativity/wg_std shows whether the term can act at all. Only when w_creat > 0.
        if float(getattr(self.reward_fn, 'w_creat', 0.0)) > 0.0:
            vals = getattr(self.reward_fn, 'last_creativity', None)
            if vals and len(vals) == B * self.k:
                ct = torch.tensor(vals, device=self.device, dtype=torch.float)
                bonus2d = (float(self.reward_fn.w_creat) * ct).view(B, self.k)
                self.log("creativity/mean", ct.mean(),
                         on_step=True, prog_bar=False, batch_size=B)
                self.log("creativity/wg_std", bonus2d.std(dim=1, unbiased=False).mean(),
                         on_step=True, prog_bar=False, batch_size=B)

        # ── Monitoring (once per step, not per epoch) ──────────────────
        ram_gb = psutil.virtual_memory().used / (1024**3)
        ram_percent = psutil.virtual_memory().percent
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0

        monitor_log = getattr(self, "monitor_log", None)   # set by omg.grpo.train (<output_dir>/monitor.log)
        if monitor_log:
            with open(monitor_log, "a") as f:
                f.write(f"STEP={self.global_step} | "
                        f"RAM={ram_gb:.2f}GB ({ram_percent:.1f}%) | "
                        f"GPU_alloc={gpu_mem_allocated:.2f}GB | "
                        f"GPU_reserved={gpu_mem_reserved:.2f}GB | "
                        f"reward_time={reward_time:.1f}s | "
                        f"mean_reward={rewards.mean().item():.4f}\n")

        print(f"[monitor] step {self.global_step}: RAM={ram_gb:.1f}GB ({ram_percent:.0f}%), "
              f"GPU={gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB, "
              f"Reward_time={reward_time:.0f}s")

        # ── 3. Reference model on the same trajectory (frozen, once) ──
        self.model.train()
        self.model_ref.eval()

        species_kl_active = (
            "species" in self.fields and self.beta_kl_species > 0.0
        )
        with torch.no_grad():
            if species_kl_active:
                logp_ref_step, drifts_ref, species_dist_ref = self.si.step_logprob(
                    self.model_ref, traj, fields=self.fields,
                    return_drift=True, return_species_dist=True,
                )
            else:
                logp_ref_step, drifts_ref = self.si.step_logprob(
                    self.model_ref, traj, fields=self.fields, return_drift=True,
                )
                species_dist_ref = None

        n_atoms_float = x0_rep.n_atoms.float()

        # Memory snapshot before the inner epochs. The peak counter is reset here, so the peak
        # logged at the end covers this step's inner-epoch replay only.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_pre_alloc_gb = torch.cuda.memory_allocated() / 1e9
        else:
            mem_pre_alloc_gb = 0.0

        # ── 4. Inner epochs (clipped surrogate + KL, manual optimization) ──
        for epoch_i in range(self.num_inner_epochs):
            opt.zero_grad()

            if species_kl_active:
                logp_new_step, drifts_new, species_dist_new = self.si.step_logprob(
                    self.model, traj, fields=self.fields,
                    return_drift=True, return_species_dist=True,
                )
            else:
                logp_new_step, drifts_new = self.si.step_logprob(
                    self.model, traj, fields=self.fields, return_drift=True,
                )
                species_dist_new = None

            # Per-channel analytical KL {pos, species}
            analytical_kl = self._compute_analytical_kl(
                drifts_new, drifts_ref, traj, n_atoms_float, BK,
                species_dist_new=species_dist_new,
                species_dist_ref=species_dist_ref,
            )

            # PPO clipped loss (clipping matters on epoch 1+ when ratio ≠ 1.0)
            loss, stats = self._ppo_grpo_loss(
                logp_old_step, logp_new_step, advantages, analytical_kl=analytical_kl)

            # Diagnostics: gradient norms of the policy term and of the KL term separately.
            # Every 10th global step, inner epoch 0 only, to keep the overhead small. The extra
            # backward passes end with opt.zero_grad(), so they do not change the update.
            diag_grad_this = (self.global_step % 10 == 0) and (epoch_i == 0)
            if diag_grad_this:
                policy_loss_t = stats["loss/policy"]
                # Rebuild the KL total from analytical_kl: stats["loss/kl"] is detached, and a
                # backward through it would silently do nothing
                # (tests/test_species_analytical_kl.py::test_diagnostic_kl_total_has_grad).
                kl_total_t = (
                    self.beta_kl_pos * analytical_kl["pos"]
                    + self.beta_kl_species * analytical_kl["species"]
                )
                # Policy-only grad norm (retain graph for subsequent passes)
                opt.zero_grad()
                self.manual_backward(policy_loss_t, retain_graph=True)
                policy_grad_norm_val = self._total_grad_norm()
                # KL-total grad norm (already β-weighted; do NOT remultiply).
                opt.zero_grad()
                self.manual_backward(kl_total_t, retain_graph=True)
                kl_grad_norm_val = self._total_grad_norm()
                # Species-only KL gradient norm: one extra backward every 10 steps.
                if species_kl_active:
                    opt.zero_grad()
                    species_kl_term = self.beta_kl_species * analytical_kl["species"]
                    self.manual_backward(species_kl_term, retain_graph=True)
                    kl_species_grad_norm_val = self._total_grad_norm()
                else:
                    kl_species_grad_norm_val = torch.tensor(0.0, device=self.device)
                opt.zero_grad()
                self.log("grad/policy_only_norm", policy_grad_norm_val,
                         on_step=True, prog_bar=False, batch_size=B)
                self.log("grad/kl_only_norm", kl_grad_norm_val,
                         on_step=True, prog_bar=False, batch_size=B)
                self.log("grad/kl_species_only_norm", kl_species_grad_norm_val,
                         on_step=True, prog_bar=False, batch_size=B)

            self.manual_backward(loss)
            # Gradient norms of the two species heads: type_out gives the element logits,
            # type_out_2 the unmask rate. Their grads are None when species is not learned or
            # no unmask event happened, hence the checks.
            if self.model.encoder.type_out.weight.grad is not None:
                self.log("grad/type_out_weight_norm",
                         self.model.encoder.type_out.weight.grad.norm(),
                         on_step=True, prog_bar=False, batch_size=B)
            if self.model.encoder.type_out_2.weight.grad is not None:
                self.log("grad/type_out_2_weight_norm",
                         self.model.encoder.type_out_2.weight.grad.norm(),
                         on_step=True, prog_bar=False, batch_size=B)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.log("grad/norm_pre_clip", grad_norm, on_step=True, prog_bar=False, batch_size=B)
            opt.step()

            # Per-epoch stdout line every 50 optimizer steps, and for the first steps of a run.
            if self.global_step % 50 == 0 or self.global_step <= 12:
                print(f"  inner epoch {epoch_i}/{self.num_inner_epochs}: "
                      f"loss={loss.item():.6f}, ratio_mean={stats['stats/ratio_mean'].item():.6f}, "
                      f"clipfrac={stats['stats/clipfrac'].item():.4f}, "
                      f"kl={stats['stats/analytical_kl'].item():.6f}")


        # ── 5. Logging (stats of the last inner epoch) ────────────────
        # The per-channel cell ratio stats are not logged separately; when the lattice is the
        # first learned channel they appear under the channel-agnostic keys.
        _DROP_CELL_STAT_KEYS = {
            "stats/ratio_mean_cell", "stats/ratio_std_cell", "stats/log_ratio_absmax_cell",
            "stats/clipfrac_cell", "stats/approx_kl_cell",
        }
        self.log_dict(
            {
                "reward/mean": rewards.mean(),
                "reward/std": rewards.std(),
                "adv/mean": advantages.mean(),
                "adv/std": advantages.std(),
                **{k: v for k, v in stats.items() if k not in _DROP_CELL_STAT_KEYS},
            },
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=B
        )

        rewards_2d = rewards.view(B, self.k)
        advantages_2d = advantages.view(B, self.k)
        group_reward_means = rewards_2d.mean(dim=1)
        group_reward_stds = rewards_2d.std(dim=1)
        group_adv_stds = advantages_2d.std(dim=1)

        self.log_dict({
            "group/reward_mean_avg": group_reward_means.mean(),
            "group/reward_mean_std": group_reward_means.std(),
            "group/reward_std_avg": group_reward_stds.mean(),
            "group/adv_std_avg": group_adv_stds.mean(),
            "group/adv_std_min": group_adv_stds.min(),
            "adv/max": advantages.max(),
            "adv/min": advantages.min(),
            "policy/lr": opt.param_groups[0]['lr'],
        }, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=B)

        if isinstance(reward_metrics, dict):
            safe = {}
            for k, v in reward_metrics.items():
                kk = k if k.startswith(("reward/", "energy/")) else f"reward/{k}"
                if kk in {"reward/mean", "reward/std"}:
                    continue
                safe[kk] = torch.as_tensor(v, device=self.device)
            if safe:
                self.log_dict(safe, on_step=True, on_epoch=False,
                              prog_bar=False, sync_dist=True, batch_size=B)

        # Memory after the step and the peak during the inner epochs, once per training step
        # (a steady climb would indicate a leak).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_post_alloc_gb = torch.cuda.memory_allocated() / 1e9
            mem_post_reserved_gb = torch.cuda.memory_reserved() / 1e9
            mem_peak_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            mem_post_alloc_gb = 0.0
            mem_post_reserved_gb = 0.0
            mem_peak_gb = 0.0

        self.log_dict({
            "stats/gpu_mem_peak_gb": mem_peak_gb,
        }, on_step=True, on_epoch=False, prog_bar=False, batch_size=B)

        print(f"[memory] step {self.global_step}: "
              f"pre_alloc={mem_pre_alloc_gb:.2f}GB "
              f"post_alloc={mem_post_alloc_gb:.2f}GB "
              f"peak={mem_peak_gb:.2f}GB "
              f"reserved_post={mem_post_reserved_gb:.2f}GB")

        # Manual optimization returns None (Lightning ignores return value)