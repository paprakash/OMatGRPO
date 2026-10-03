# Training and reward

This page describes the GRPO loss, the reward and its guards. [flags.md](flags.md) lists every flag.

## How training works

Each rollout samples B groups of K structures (`--num_groups 4`, `--group_size 16`). The members of a group share
their number of atoms. Generation integrates the interpolants on a grid of 64 time points, which gives 63 steps.
The species noise η is 0, which makes the species KL to the prior available in closed form.

The loss has one PPO-clipped surrogate per channel (clip range 0.2), weighted by α_pos, α_cell and α_species, plus
KL terms to the prior. The position KL (weight β_pos) comes from the Girsanov drift difference of the two SDEs. The
species KL (weight β_species) is the categorical KL of the unmasking events. No KL term acts on the lattice. Each
rollout is followed by three PPO epochs (`--inner_epochs 3`) with AdamW, learning rate 1e-4, gradient-norm
clipping at 0.5 and bf16 mixed precision.

The reward of structure i, as `calculate_rewards`, `_apply_diversity_penalty` and `_mmd_diversity_bonus` assemble
it, is

```
r_base  = -w_stab * clip(E_hull_routed, 0, cap) + w_creat * creativity
r_occ   = r_pen + c_occ * (r_base - r_pen)        r_pen = min(-cap * w_stab, min over the batch of r_base) - 1e-6
r       = r_occ + w_mmd * r_mmd
```

With the OMatGRPO settings (w_stab = 1, cap = 1, w_creat = 1, w_mmd = 0.4) this is the paper's form
r = r_pen + c_occ (r_stab + r_creat - r_pen) + 0.4 r_mmd with r_stab = -clip(E_hull, 0, 1) and r_pen = -1. In the
code r_pen is the lower of -1 and the lowest r_base of the batch, minus 1e-6, which keeps every r_base above it.
E_hull_routed is the E_hull after the guards below. c_occ is the occurrence discount (1 for a
composition seen at most 3 times in its group, falling linearly to 0 at 6). r_mmd is the leave-one-out MMD credit
of the structure's composition against MP-20, scaled to [0, 1] within the batch.

Advantages are computed per group. Rewards are clipped to three standard deviations around the group mean, and
then centered and divided by the group standard deviation. Under abstention routing the mean and standard
deviation use the trusted members only, and abstaining members get advantage 0. A group with fewer than two
trusted members, or with a standard deviation below 1e-4, contributes nothing.

## Reward terms and guards

Each guard exists because a run without it found a way to collect reward that the evaluation does not confirm. The
paper catalogs these reward-hacking modes in its reward-hacking appendix.

| name in the paper | what it does | why it exists | flag (OMatGRPO value) | off |
|---|---|---|---|---|
| relaxation before scoring | relaxes every structure with FIRE and UMA (at most 100 steps, force tolerance 0.05 eV/Å), with the cell through a Frechet filter, and scores the relaxed structure | a raw sample is rewarded for where it lands after a short relaxation, which is also what the evaluation scores | `--relax true --relax_cell true --relax_steps 100` | `--relax false` |
| cell and masked-species guards | cells that are collapsed, very flat or huge, and structures that still carry a mask token, are not scored. After a relaxation with the cell, the cell is checked again | UMA cannot score such cells, and a flat or huge cell can exhaust GPU memory | always on | none |
| floor at zero and cap | the stability term is -clip(E_hull, 0, cap) | without the floor the policy chases ever lower E_hull below the hull (mode 3); the cap bounds the bad tail and is the value given to penalized structures | `--stability_floor_at_zero true --stability_cap 1` | `--stability_floor_at_zero false --stability_cap 5` reproduces mode 3 |
| deep-below-hull floor | E_hull below -0.1 eV/atom is penalized with the cap | a structure far below the hull is more likely an error of the potential or of the hull than a discovery | `--deep_below_hull -0.1` | `--deep_below_hull=-inf` |
| sparse-hull abstention | a structure whose chemical system has fewer than 12 hull reference entries gets advantage 0 | a hull with few entries is not trusted for either sign of E_hull; without the gate the policy drifts into poorly sampled chemical systems | `--sparse_gate true --sparse_min_refs 12 --sparse_route abstain` | `--sparse_gate false` |
| penalty routing | the alternative to abstention: sparse structures get the cap value | used by the penalty-routing runs | `--sparse_route penalty` | |
| single-element guard | single-element structures are penalized with the cap | single-element compositions sit on sparse hulls, so abstention never penalizes them and the policy drifts toward them (mode 5) | `--single_element_guard on` | `--single_element_guard off` reproduces mode 5 |
| relaxation failure | a structure whose cell fails the guard after relaxation, or whose hull lookup fails, has no hull entry. It abstains under abstention routing and gets the cap under penalty routing. If the relaxation of a whole batch fails, the unrelaxed energies are used, and five failures in a row stop the run | a failed structure carries no trustworthy energy | always on | none |
| occurrence discount | pulls a composition repeated more than 3 times in a group toward r_pen, completely at 6 | stops a group from collapsing onto one composition | `--occurrence_discount true --occurrence_tol 3 --occurrence_zero 6` | `--occurrence_discount false` |
| MMD bonus | adds 0.4 times the batch-scaled leave-one-out MMD credit of the composition against MP-20 | rewards compositions that improve the batch's coverage of MP-20 chemistry | `--w_mmd 0.4` | `--w_mmd 0` |
| creativity term | adds 1 for a structure that is unique in its batch and novel against MP-20 train, 0 for neither, and an AMD distance in between, scored on the relaxed structure. The AMD distance comes from the `average-minimum-distance` package, which is licensed CC BY-NC-SA 4.0 (non-commercial), while OMatGRPO's code is MIT | rewards new structures instead of rediscovered ones | `--w_creat 1 --creat_on_relaxed true` | `--w_creat 0` |
| KL terms | KL of positions and species to the prior | keeps the policy close to the prior | `--beta_kl_pos 0.01 --beta_kl_species 0.05` | set to 0 |

In OMatGRPO's training run, guard, relaxation and hull-lookup failures affected 137 of 48,000 scored structures
(750 rollouts of 64), 0.18 per rollout on average and at most 6 in one rollout. Under abstention routing these
structures abstain, except single-element ones, which the single-element guard penalizes.

The creativity term needs `average-minimum-distance` (CC BY-NC-SA 4.0). Its non-commercial terms apply to every
run with `--w_creat > 0`, although OMatGRPO's own code is MIT licensed.

**This branch.** The paper's reward-hacking appendix also studies reward variants that no run in `configs/runs/`
uses: the displacement reward (mode 1), the formation-energy reward (mode 2), the absolute-energy reward and the
residual geometry term (mode 4). This branch has them, with their flags and tests. See
[reward_hacking_variants.md](reward_hacking_variants.md).

## Frozen-composition control

The frozen-composition control (`configs/runs/frozen_control.yaml`) samples one composition per group from the
prior and learns only positions and lattice: `--fields pos,cell --freeze_composition true --alpha_pos 1
--alpha_cell 1 --beta_kl_species 0 --relax_cell false --single_element_guard off --occurrence_discount false
--w_mmd 0 --w_creat 0 --creat_on_relaxed false`. All members of a group share a composition there, so the
occurrence discount would set every reward to the floor. The code refuses that combination.
