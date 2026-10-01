# OMatGRPO

OMatGRPO fine-tunes an [OMatG](https://github.com/FERMat-ML/OMatG) crystal generator with group relative policy
optimization (GRPO). The policy acts on all three channels of the generator. Positions and the lattice are
stochastic differential equations, and the species channel is discrete flow matching, so the composition itself is
learned by reinforcement learning. The reward pays for thermodynamic stability, measured as the energy above the
convex hull (E_hull) of the relaxed structure with the UMA machine-learning potential. It adds a creativity term
for structures that are unique and novel with respect to the MP-20 training set, and a compositional coverage
bonus. Guards stop the policy from collecting reward through errors of the potential or of the hull reference.

## Paper

"Reinforcement Learning on the Discrete Composition Channel of a Crystal Generator: Validated Gains and Reward
Hacking." arXiv link coming soon.

Pawan Prakash (University of Florida; Oak Ridge National Laboratory), Philipp Höllmer (New York University), Addis
Fuhr (Oak Ridge National Laboratory), Peter Hirschfeld (University of Florida), P. Ganesh (Oak Ridge National
Laboratory), Stefano Martiniani (New York University), Richard Hennig (University of Florida).

## Relation to OMatG

The `omg` package in this repository is a copy of OMatG at commit
[`9172203`](https://github.com/FERMat-ML/OMatG/tree/9172203a9026af41732247cc62aaf4d903010e9c) (2025-08-06).
Upstream OMatG has changed since then. This copy is self-contained and does not track upstream.

OMatGRPO adds the package `omg/grpo/` and changes three upstream files.

| file | change | reason |
|---|---|---|
| `omg/si/stochastic_interpolants.py` | adds `integrate_with_logprob`, `integrate_consistent` and `step_logprob` | GRPO needs the log-probability of every integration step of a sampled trajectory, and the drifts of the policy and of the reference model for the KL terms |
| `omg/si/single_stochastic_interpolant.py` | adds `DifferentialEquationType.DISCRETE` | marks the species channel, which is neither an ODE nor an SDE |
| `omg/si/discrete_flow_matching_mask.py` | sets that type on the masked species interpolant | lets the trainer admit the species channel as a policy channel |

OMatG is MIT licensed, and so is OMatGRPO (`LICENSE`). `THIRD_PARTY_NOTICES.md` lists the other code, models and
data this project uses and their licenses.

Install OMatGRPO into a fresh environment. An installed upstream OMatG also provides a package named `omg`, and the
two cannot coexist.

## Setup

The weights and the evaluated structures are on Hugging Face (`paprakash/OMatGRPO`), not in this repository.
`scripts/download_assets.py` fetches them. The two MP-20 references of the reward are built locally.

```bash
# 1. clone
git clone https://github.com/paprakash/OMatGRPO && cd OMatGRPO
# 2. create the environment (Python 3.12, PyTorch 2.8 with CUDA 12.8, the versions of the paper's runs)
conda env create -f environment.yml
conda activate omatgrpo
pip install -e ".[dev]"
# 3. download the prior and the MP-20 LMDBs (add "models" and "structures" for the paper's weights and sets)
python scripts/download_assets.py
# 4. build the two references of the reward
python scripts/build_references.py mmd
python scripts/build_references.py creativity
```

The reward and the evaluation use the UMA potential `uma-s-1p2` from `facebook/UMA` on Hugging Face. The model is
gated. Request access on its Hugging Face page, accept the license, and log in once with `hf auth login`.

The hull reference (`LeMaterial/LeMat-Bulk-MLIP-Hull`, pinned to revision `70d505bb`) is downloaded into the data
directory the first time the reward runs.

LeMat-GenBench, which produces the paper's tables, runs in its own environment. Clone
[lemat-genbench](https://github.com/LeMaterial/lemat-genbench), check out commit
`58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6`, and install it as its README describes.

## Assets

```bash
python scripts/download_assets.py                         # prior and MP-20 LMDBs (the default)
python scripts/download_assets.py models                  # the trained weights of the seven runs
python scripts/download_assets.py structures              # the nine structure sets LeMat-GenBench scored
python scripts/download_assets.py models --models arityguard_creatrelax   # or selected ones
```

Files go into the data directory, which is `omg/data` unless `--data_dir` or `OMATGRPO_DATA_DIR` says otherwise.
Every file is checked against the sha256 listed in `scripts/download_assets.py` (for the CIFs of a structure set,
against the repository's `SHA256SUMS`, whose entries for the set's summary and result file must match the pinned
values). A file that is already present with the right hash is skipped. A file with a wrong hash stops the script
and is left in place.

```
prior/prior.safetensors, prior/train.yaml     the pretrained OMatG model every run starts from      (Hugging Face)
mp_20/{train,val,test}.lmdb                   MP-20, from upstream OMatG at commit 9172203        (GitHub)
models/<identifier>/final_model.safetensors   trained weights, and resolved_config.json            (Hugging Face, optional)
structures/<identifier>/                      CIFs, structures_summary.csv, LeMat-GenBench JSON    (Hugging Face, optional)
references/mp20_comp_reference.pt             composition matrix of the MMD bonus                  (build_references.py)
references/mp20_train_ref.json.gz             MP-20 training structures by formula, for creativity (build_references.py)
```

The structure sets are `R0` (the prior), `bestofn48k_top2500` (the best-of-N baseline) and the seven run
identifiers. They are the exact sets behind the paper's LeMat-GenBench numbers, not regenerations.

The references come from public data. `python scripts/build_references.py mmd` reproduces the MMD reference byte
for byte from the MP-20 training LMDB. `python scripts/build_references.py creativity` builds the creativity
reference from the MP-20 training CSV of DiffCSP at a pinned commit. Its formulas, structure counts, species and
coordinates equal the reference used in the paper, and 2,650 of 27,136 lattice matrices differ by at most
6.2e-15 Å from floating-point rounding.

**The prior.** Every run starts from our own OMatG model, pretrained on MP-20 for de novo generation, and uses it
as the KL reference. It is released with the code. Positions and lattice are SDE interpolants, and species use
masked discrete flow matching. Its species and position channels, the number of integration steps (710), the
optimizer and the trainer settings are those of the released OMatG `MP-20-DNG/Linear-SDE-Gamma` model. Its lattice
channel is identical to that of `Trig-ODE-Gamma`, and its relative loss weights are those of `VPSBD-SDE`. It was
trained with OMatG at commit `9172203` (AdamW, learning rate 1.97e-4, batch size 32, up to 2,000 epochs), and the
checkpoint with the lowest validation `dng_eval` was kept (epoch index 1599). `configs/prior/train.yaml` holds the
full configuration. The hardware, wall time and seed of the pretraining were not recorded.

## Quick start

Train OMatGRPO. The recipe sets every value of the paper's run, and command-line flags override it.

```bash
python -m omg.grpo.train --recipe configs/recipes/arityguard_creatrelax.yaml
python -m omg.grpo.train --recipe configs/recipes/arityguard_creatrelax.yaml --print_config   # show the settings
```

Generate 2,500 structures from the final weights with the paper's evaluation protocol, relax them and score them.

```bash
python scripts/eval/generate.py --checkpoint outputs/arityguard_creatrelax/final_model.safetensors \
    --out_dir outputs/eval/arityguard_creatrelax --n 2500
python scripts/eval/export_cifs.py --run_dir outputs/eval/arityguard_creatrelax \
    --dest outputs/eval/arityguard_creatrelax/export
```

Evaluate them with LeMat-GenBench and report the counts and the mSUN decomposition.

```bash
export LEMAT_GENBENCH_ROOT=/path/to/lemat-genbench
scripts/eval/run_lemat_genbench.sh outputs/eval/arityguard_creatrelax/export/cifs arityguard_creatrelax
python scripts/eval/lgb_report.py --name arityguard_creatrelax \
    --cifs outputs/eval/arityguard_creatrelax/export/cifs \
    --summary outputs/eval/arityguard_creatrelax/export/structures_summary.csv
```

`scripts/slurm/` has SLURM templates for these three steps. `scripts/eval/novelty.py` is an optional internal
check of uniqueness and novelty against MP-20 and Alex-MP-20.

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

**The `reward-hacking` branch.** The paper's reward-hacking appendix also studies reward variants that no recipe
here uses. The branch `reward-hacking` is this code plus those variants, with their flags, tests and a README
section: the displacement reward (mode 1, `--w_rmsd`, `--fmax`, `--fmax_schedule`, `--reward_offset`), the
formation-energy reward (mode 2, `--reward_type formation`), the absolute-energy reward (`--reward_type absolute`)
and the residual geometry term (mode 4, `--w_rmsd_geom`, `--rmsd_geom_clamp`). Use `main` for the paper's runs.

## Recipes

Each identifier is the run name in the paper's table of all runs and a file in `configs/recipes/`.

| identifier | paper name | change from OMatGRPO |
|---|---|---|
| `arityguard_creatrelax` | OMatGRPO | |
| `canonical_creatrelax` | discovery | `--single_element_guard off` |
| `sparseworst_creatrelax` | penalty routing | `--single_element_guard off --sparse_route penalty` |
| `arityguard` | guarded, pre-creativity | `--w_creat 0 --creat_on_relaxed false` |
| `canonical` | discovery, pre-creativity | `--single_element_guard off --w_creat 0 --creat_on_relaxed false` |
| `sparseworst` | penalty routing, pre-creativity | `--single_element_guard off --sparse_route penalty --w_creat 0 --creat_on_relaxed false` |
| `frozen_control` | frozen-composition control | see below |

The frozen-composition control samples one composition per group from the prior and learns only positions and
lattice: `--fields pos,cell --freeze_composition true --alpha_pos 1 --alpha_cell 1 --beta_kl_species 0
--relax_cell false --single_element_guard off --occurrence_discount false --w_mmd 0 --w_creat 0
--creat_on_relaxed false`. All members of a group share a composition there, so the occurrence discount would
set every reward to the floor. The code refuses that combination.

## Flag reference

Generated from the parser (`omg.grpo.train.flag_table_markdown()`). `python -m omg.grpo.train --help` prints the
same. Defaults are the OMatGRPO recipe. An unknown flag is an error.

**run**

| flag | default | description |
|---|---|---|
| `--recipe` | `None` | YAML file with flag values (keys = flag names without --); command-line flags override it |
| `--print_config` |  | print the resolved settings and exit |
| `--run_name` | `omatgrpo` | name of the run (output folder and wandb run name) |
| `--output_dir` | `None` | folder for checkpoints, the final model and the resolved config; None means outputs/<run_name> |
| `--data_dir` | `None` | data folder (MP-20 LMDBs, references, hull cache, prior); None means OMATGRPO_DATA_DIR or omg/data |
| `--model_config` | `configs/prior/train.yaml` | OMatG config of the prior (model, interpolants, data) |
| `--init_checkpoint` | `None` | weights of the prior; the policy starts from them and they are the KL reference. None means <data_dir>/prior/prior.safetensors |
| `--resume` | `None` | periodic checkpoint of an interrupted run to resume (policy, optimizer, step) |
| `--seed` | `0` | global seed (the paper runs used 0; B200 kernels are not bit-reproducible anyway) |
| `--checkpoint_every` | `50` | save a periodic checkpoint every N rollouts |
| `--wandb_project` | `omatgrpo` | wandb project |
| `--wandb_mode` | `offline` | wandb mode: offline (logs stay in <output_dir>/wandb; upload later with `wandb sync`), online, or disabled (online, offline, disabled) |

**rollouts and optimization**

| flag | default | description |
|---|---|---|
| `--fields` | `pos,cell,species` | channels the policy learns: pos, pos,cell or pos,cell,species (pos, pos,cell, pos,cell,species) |
| `--freeze_composition` | `False` | sample one composition per group from the frozen prior and learn only positions and lattice (the frozen-composition control); needs --fields pos,cell and --occurrence_discount false |
| `--num_groups` | `4` | B, groups per rollout |
| `--group_size` | `16` | K, structures per group (members of a group have the same number of atoms) |
| `--time_grid` | `64` | points of the integration time grid (steps = points - 1); capped at the value in the model config |
| `--species_eta` | `0.0` | noise eta of the discrete species channel (the closed-form species KL needs 0) |
| `--rollouts` | `750` | number of rollouts (optimizer steps = rollouts x inner_epochs) |
| `--inner_epochs` | `3` | PPO epochs per rollout |
| `--lr` | `0.0001` | AdamW learning rate |
| `--alpha_pos` | `0.1` | weight of the position surrogate |
| `--alpha_cell` | `0.1` | weight of the lattice surrogate |
| `--alpha_species` | `1.0` | weight of the species surrogate |
| `--beta_kl_pos` | `0.01` | weight of the position KL to the prior (0 = off) |
| `--beta_kl_species` | `0.05` | weight of the species KL to the prior (0 = off) |

**reward: stability term and guards**

| flag | default | description |
|---|---|---|
| `--w_stability` | `1.0` | weight of the stability term -clip(E_hull); 0 turns it off together with every guard penalty |
| `--relax` | `True` | relax structures (FIRE, UMA) before scoring |
| `--relax_cell` | `True` | include the cell in the relaxation (Frechet filter), with a post-relaxation cell guard |
| `--relax_steps` | `100` | maximum FIRE steps of the relaxation |
| `--stability_floor_at_zero` | `True` | clip E_hull at 0 from below, so depth below the hull earns nothing |
| `--stability_cap` | `1.0` | upper clip of E_hull (eV/atom), also the value of penalized and failed structures |
| `--deep_below_hull` | `-0.1` | E_hull below this counts as deep below the hull and is penalized; --deep_below_hull=-inf turns the guard off |
| `--sparse_gate` | `True` | treat structures on sparse hulls (fewer than --sparse_min_refs reference entries) as untrusted for either E_hull sign |
| `--sparse_min_refs` | `12` | minimum hull reference entries of a trusted E_hull |
| `--sparse_route` | `abstain` | what untrusted sparse structures get: abstain (zero advantage) or penalty (the cap value) (abstain, penalty) |
| `--single_element_guard` | `on` | penalize single-element structures (on, off) |

**reward: diversity and creativity**

| flag | default | description |
|---|---|---|
| `--occurrence_discount` | `True` | pull repeated compositions within a group toward the worst reward |
| `--occurrence_tol` | `3` | occurrences in a group with no discount |
| `--occurrence_zero` | `6` | occurrences in a group at which the discount is complete |
| `--w_mmd` | `0.4` | weight of the compositional MMD coverage bonus (0 = off) |
| `--mmd_comp_reference` | `None` | MMD composition reference; None means <data_dir>/references/mp20_comp_reference.pt |
| `--mmd_kernel` | `poly` | MMD kernel (poly, linear) |
| `--mmd_poly_c` | `1.0` | polynomial kernel offset c |
| `--mmd_poly_d` | `3` | polynomial kernel degree d |
| `--mmd_max_reference` | `10000` | rows of the MMD reference used (seeded subsample) |
| `--mmd_norm` | `minmax` | per-batch scaling of the MMD credit (minmax, zscore, none) |
| `--w_creat` | `1.0` | weight of the creativity term (unique and novel vs MP-20 train; 0 = off) |
| `--creat_on_relaxed` | `True` | score creativity on the relaxed structures (needs --relax true) |
| `--creativity_reference` | `None` | creativity reference; None means <data_dir>/references/mp20_train_ref.json.gz |
| `--creativity_sm_timeout` | `3.0` | StructureMatcher timeout per structure (s, rounded to an integer >= 1); a timeout scores 0 |

wandb logs offline by default (`<output_dir>/wandb`, upload later with `wandb sync`). Use `--wandb_mode online`
to log live, or `--wandb_mode disabled` to turn it off.

## Evaluation protocol and LeMat-GenBench

`scripts/eval/generate.py` follows the paper's protocol. It uses the consistent sampler (64 time points, species
noise 0), draws 2,500 structures in chunks of 100 with seed 42, and takes the number of atoms of each structure from
the MP-20 validation set. It relaxes every structure that passes the guards with FIRE and UMA, including the cell,
for at most 500 steps, and computes E_hull against the LeMat-Bulk UMA hull. `export_cifs.py` writes the valid
structures as CIFs.

The tables of the paper come from LeMat-GenBench at commit `58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6` with the
preset `comprehensive_multi_mlip_hull`. Stability there comes from an ensemble of three potentials (ORB, MACE and
UMA). UMA is gated, and without a Hugging Face token LeMat-GenBench silently drops the UMA leg and scores with two
potentials. `run_lemat_genbench.sh` therefore refuses to start without a token, and `lgb_report.py` checks that all
three legs scored the same number of structures.

LeMat-GenBench counts structures on the hull as SUN and excludes them from mSUN (metastable, unique and novel).
Unique, novel and metastable rates are over the valid structures, as LeMat-GenBench reports them. mSUN is reported
over the nominal 2,500 generated structures. `lgb_report.py` prints both kinds of rates. With `--cifs` and
`--summary` it also splits mSUN into single-element and sparse-hull structures. That split needs a join between
LeMat-GenBench's structure indices and the CIF files, and the script proves the join element by element before it
reports.

## Reproducing the paper

Train the recipe, generate 2,500 structures from `final_model.safetensors`, and evaluate them as in the quick start.
The trained weights of all seven runs are also available (`python scripts/download_assets.py models`), and so are
the exact structure sets the paper's numbers come from (`python scripts/download_assets.py structures`), which
`lgb_report.py` can decompose without regenerating anything.

| identifier | command | mSUN (% of 2,500) | single-element in mSUN | sparse hull in mSUN | compounds in mSUN |
|---|---|---|---|---|---|
| `arityguard_creatrelax` | `python -m omg.grpo.train --recipe configs/recipes/arityguard_creatrelax.yaml` | 1138 (45.5) | 4 (0.4%) | 23.5% | 1134 |
| `canonical_creatrelax` | same with `canonical_creatrelax.yaml` | 939 (37.6) | 502 (53.5%) | 63.4% | 437 |
| `sparseworst_creatrelax` | same with `sparseworst_creatrelax.yaml` | 841 (33.6) | 1 (0.1%) | 3.6% | 840 |
| `arityguard` | same with `arityguard.yaml` | 982 (39.3) | 20 (2.0%) | 14.1% | 962 |
| `canonical` | same with `canonical.yaml` | 1063 (42.5) | 528 (49.7%) | 65.8% | 535 |
| `sparseworst` | same with `sparseworst.yaml` | 789 (31.6) | 1 (0.1%) | 0.9% | 788 |

| run | valid % | unique % | novel % | metastable % | SUN | mSUN % (count) | single-element in mSUN | sparse hull % of mSUN |
|---|---|---|---|---|---|---|---|---|
| prior | 94.1 | 99.5 | 82.1 | 27.2 | 8 | 13.4 (336) | 7 | 21.7 |
| `frozen_control` | 93.5 | 99.5 | 87.7 | 18.7 | 9 | 9.8 (245) | 1 | 12.2 |
| `arityguard_creatrelax` (OMatGRPO) | 69.8 | 98.7 | 88.3 | 75.8 | 7 | 45.5 (1138) | 4 | 23.5 |

The paper also compares with a best-of-N baseline (mSUN 192) and Chemeleon2 (mSUN 915). Their code is not part of
this repository.

Each number comes from one training run with seed 0. GPU training and generation are not bit-reproducible, so a
rerun gives different structures and somewhat different counts.

Compute: one run is 750 rollouts on one GPU. The SLURM jobs of the three creativity runs took between 7 h 41 min and
8 h 35 min on one NVIDIA B200. Generating and relaxing 200 structures took about 150 s in our
tests, so 2,500 take about half an hour. LeMat-GenBench runs on CPUs and takes several hours for 2,500 structures.

## Hardware

The runs used one NVIDIA B200 (183 GB). GPU memory grows with the number of structures per rollout and the length
of the time grid (B·K·T). B = 4 groups is the paper's setting because B = 8 ran out of memory on that GPU. The
training jobs requested 120 to 178 GB of host memory, and SLURM recorded a maximum resident set size of about 3.9 GB
for the three creativity runs.

## Outputs and logged metrics

A run writes into `--output_dir` (default `outputs/<run_name>`): `resolved_config.json` with every setting, the git
commit and the package versions, `checkpoints/` (every `--checkpoint_every` rollouts, and the last one), and at the
end `final_model.ckpt` and `final_model.safetensors`. `--resume <checkpoint>` continues an interrupted run. It
refuses a checkpoint whose KL reference differs from the prior given by `--init_checkpoint`.

Main logged metrics (wandb, and one `[watch]` line per rollout on stdout):

| metric | meaning |
|---|---|
| `energy/e_hull_per_atom_*` | E_hull of the scored structures |
| `reward/ehull_untrusted_frac` | share penalized by the guards |
| `reward/sparse_neutral_frac`, `group/neutral_frac` | share abstaining |
| `reward/elemental_frac` | share of single-element structures |
| `reward/hull_lookup_fail_count` | failed hull lookups (all failing in one batch stops the run) |
| `reward/creativity_*` | creativity term, unique and novel shares |
| `diversity/*`, `mmd/*` | occurrence discount and MMD bonus |
| `group/dead_frac` | share of groups without learning signal |
| `reward/relax/*` | displacement and energy change of the relaxation |
| `energy/formation_per_atom_*` | formation energy (monitoring only) |

## Known limitations

- The creativity term and the novelty of the evaluation both compare against MP-20, so the training signal and the
  metric share their definition of novelty.
- Every result is a single training run with seed 0.
- B200 kernels are not bit-deterministic. Generation is reproducible only with `--deterministic`, and those
  numbers are not the paper's. UMA's scatter operation has no deterministic CUDA implementation, so the
  relaxation stays nondeterministic even then.
- The hull reference and the reward use the same potential, and the hull has few entries in some chemical systems.
  The sparse-hull gate and the single-element guard exist for this reason.
- The timeouts of the relaxation and of StructureMatcher use SIGALRM. They work only on the main thread under Unix.
- The frozen-composition control needs the occurrence discount off.
- The creativity term gives StructureMatcher 3 seconds per structure (`--creativity_sm_timeout`), and a timeout
  scores 0. Creativity scores therefore depend on the speed of the machine, and the same batch can score
  differently on a slower or busier node. `reward/creativity_timeout_frac` logs the share of timeouts.

## Tests

```bash
pytest -m "not gpu and not uma and not network"     # what CI runs, CPU only
pytest                                               # everything, with a GPU, UMA access and network
```

Tests marked `gpu` need a CUDA device, `uma` needs access to the gated UMA model, and `network` downloads data.
The creativity tests are skipped until the creativity reference is in the data directory. The upstream OMatG
tests in `omg/tests` run too. Fifteen of them already failed at the pinned upstream commit and are marked as
expected failures (`omg/tests/conftest.py`).

## License and acknowledgments

MIT (`LICENSE`). OMatGRPO builds on OMatG by the FERMat project. `omg/grpo/lemat_hull.py` (adapted) and
`omg/grpo/element_chem_pot.json` (copied) come from LeMat-GenBench and stay under the Apache License 2.0. The
creativity term and the MMD bonus follow Chemeleon2 (MIT). The `average-minimum-distance` package used by the
creativity term is CC BY-NC-SA 4.0, while OMatGRPO's code is MIT. The UMA model has its own license.
`THIRD_PARTY_NOTICES.md` has the license texts and the details.

Data. The MP-20 structures come from the Materials Project (A. Jain et al., 2013,
https://doi.org/10.1063/1.4812323), whose data are licensed CC BY 4.0. MP-20 was introduced by T. Xie et al.
(2021, https://arxiv.org/abs/2110.06197). The stability term uses the UMA convex hull of LeMat-Bulk-MLIP-Hull
(LeMaterial, https://huggingface.co/datasets/LeMaterial/LeMat-Bulk-MLIP-Hull, revision `70d505bb`). It is
downloaded on first use and not redistributed here.

## Citation

Please cite the paper and OMatG:

```bibtex
@article{prakash2026omatgrpo,
    title={Reinforcement Learning on the Discrete Composition Channel of a Crystal Generator: Validated Gains
    and Reward Hacking},
    author={Pawan Prakash and Philipp H{\"o}llmer and Addis Fuhr and Peter Hirschfeld and P. Ganesh and
    Stefano Martiniani and Richard Hennig},
    journal={arXiv link coming soon},
    year={2026},
}

@article{hoellmer2025,
    title={Open Materials Generation with Stochastic Interpolants},
    author={Philipp H{\"o}llmer and Thomas Egg and Maya Martirossyan and Eric
    Fuemmeler and Zeren Shui and Amit Gupta and Pawan Prakash and Adrian
    Roitberg and Mingjie Liu and George Karypis and Mark Transtrum and Richard
    Hennig and Ellad B. Tadmor and Stefano Martiniani},
    journal={arXiv preprint arXiv:2502.02582},
    year={2025},
}
```
