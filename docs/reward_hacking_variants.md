# Reward variants from the reward-hacking study

The paper's reward-hacking appendix catalogs ways in which earlier versions of the reward were exploited. Modes 3
and 5 and penalty routing are switches of the main code (see the [guard table](reward.md#reward-terms-and-guards)).
This branch adds the code of the other variants, so the failure modes can be studied.
`tests/test_reward_variants.py` checks each variant's arithmetic on CPU with a stand-in for the potential, and each
variant ran for two rollouts without error on a GPU.

The original runs used an earlier command-line interface, whose defaults differ from the defaults here (no
sparse-hull gate, penalty routing, no floor at zero, cap 5, no cell relaxation, no occurrence discount, no MMD bonus,
no creativity term, α = 1 for positions and lattice). The commands below write those settings out in today's flags.
The runs also used older versions of the code, so a rerun repeats the settings but not the run bit for bit.

All commands below share these flags:

```bash
COMMON="--fields pos,cell,species --species_eta 0 --num_groups 4 --group_size 16 --time_grid 64 --inner_epochs 3 \
  --lr 1e-4 --alpha_pos 1 --alpha_cell 1 --alpha_species 0.1 --beta_kl_pos 0.01 --beta_kl_species 0 \
  --relax_cell false --stability_floor_at_zero false --stability_cap 5 --deep_below_hull -0.1 --sparse_gate false \
  --sparse_min_refs 12 --sparse_route penalty --single_element_guard off --occurrence_discount false --w_mmd 0 \
  --w_creat 0 --creat_on_relaxed false"
```

**Displacement reward (mode 1).** Flags `--w_rmsd` (weight), `--fmax` (force tolerance of its relaxation, eV/Å),
`--fmax_schedule` (tolerance by training step, for example `0:10,1000:5`) and `--reward_offset`. The term is
log(1 + offset) - log(1 + RMSD) between the generated structure and its FIRE relaxation (at most 1,000 steps),
weighted by `--w_rmsd`. It adds to the energy reward chosen by `--reward_type`. This code has no switch for a
displacement-only reward. Mode 1 was found with an early version of the reward that used a different relaxation
engine, so its numbers cannot be reproduced exactly with this code. The original runs are not recorded in this
repository.

```bash
python -m omg.grpo.train $COMMON --run_name displacement --rollouts 750 --relax false --reward_type absolute --w_rmsd 1
```

**Formation-energy reward (mode 2).** Flag `--reward_type formation`. The reward is minus the formation energy
per atom, clipped to [-10, 10] eV/atom, with the bulk-crystal element references of LeMat-GenBench. Original runs:
`apr05_R6_poscellspecies_350r_formfix` (all three channels, 350 rollouts) and its siblings R3 to R5 on fewer
channels.

```bash
python -m omg.grpo.train $COMMON --run_name R6_poscellspecies_350r_formfix --rollouts 350 --relax false \
    --reward_type formation
```

**Absolute-energy reward.** Flag `--reward_type absolute`. The reward is minus the UMA energy per atom, clipped to
[-20, 20] eV/atom. This was the default reward of the earlier interface. Its original runs are not recorded in
this repository.

```bash
python -m omg.grpo.train $COMMON --run_name absolute --rollouts 750 --relax false --reward_type absolute
```

**Residual geometry term (mode 4).** Flags `--w_rmsd_geom` (weight) and `--rmsd_geom_clamp` (Å). The term
subtracts the weight times the RMSD between the generated structure and its relaxed structure, clipped to
[0, clamp], from the stability term. It reuses the relaxation before scoring, so it needs `--relax true`. Original
runs: `ehull_rmsd_combined_3field_750` (stability and geometry terms) and `ehull_rmsdonly_3field_750` (geometry
term only, `--w_stability 0`).

```bash
python -m omg.grpo.train $COMMON --run_name ehull_rmsd_combined_3field_750 --rollouts 750 --relax true \
    --relax_steps 100 --w_stability 1 --w_rmsd_geom 1 --rmsd_geom_clamp 3
python -m omg.grpo.train $COMMON --run_name ehull_rmsdonly_3field_750 --rollouts 750 --relax true \
    --relax_steps 100 --w_stability 0 --w_rmsd_geom 1 --rmsd_geom_clamp 3
```
