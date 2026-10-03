# Reproducing the paper

This page has the details behind the [README](../README.md): the assets, the prior, the evaluation protocol, the
results of every run, compute, outputs, limitations and tests.

## Setup details

The weights and the evaluated structures are on Hugging Face (`paprakash/OMatGRPO`), not in this repository.
`scripts/download_assets.py` fetches them. The two MP-20 references of the reward are built locally.
`environment.yml` installs Python 3.12 and PyTorch 2.8 with CUDA 12.8, the versions of the paper's runs.

The reward and the evaluation use the UMA potential `uma-s-1p2` from `facebook/UMA` on Hugging Face. The model is
gated. Request access on its Hugging Face page, accept the license, and log in once with `hf auth login`.

The hull reference (`LeMaterial/LeMat-Bulk-MLIP-Hull`, pinned to revision `70d505bb`) is downloaded into the data
directory the first time the reward runs.

LeMat-GenBench, which produces the paper's tables, runs in its own environment. Clone
[lemat-genbench](https://github.com/LeMaterial/lemat-genbench), check out commit
`58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6`, and install it as its README describes.

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

## The prior

Every run starts from our own OMatG model, pretrained on MP-20 for de novo generation, and uses it
as the KL reference. It is released with the code. Positions and lattice are SDE interpolants, and species use
masked discrete flow matching. Its species and position channels, the number of integration steps (710), the
optimizer and the trainer settings are those of the released OMatG `MP-20-DNG/Linear-SDE-Gamma` model. Its lattice
channel is identical to that of `Trig-ODE-Gamma`, and its relative loss weights are those of `VPSBD-SDE`. It was
trained with OMatG at commit `9172203` (AdamW, learning rate 1.97e-4, batch size 32, up to 2,000 epochs), and the
checkpoint with the lowest validation `dng_eval` was kept (epoch index 1599). `configs/prior/train.yaml` holds the
full configuration. The hardware, wall time and seed of the pretraining were not recorded.

## Evaluation protocol and LeMat-GenBench

`scripts/eval/generate.py` follows the paper's protocol. It uses the consistent sampler (64 time points, species
noise 0), draws 2,500 structures in chunks of 100 with seed 42, and takes the number of atoms of each structure from
the MP-20 validation set. It relaxes every structure that passes the guards with FIRE and UMA, including the cell,
for at most 500 steps, and computes E_hull against the LeMat-Bulk UMA hull. `export_cifs.py` writes the valid
structures as CIFs. `scripts/eval/novelty.py` is an optional internal check of uniqueness and novelty against MP-20
and Alex-MP-20.

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

## Results of all runs

Train the recipe, generate 2,500 structures from `final_model.safetensors`, and evaluate them as in the
[quick start](../README.md#quick-start).
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

## Compute and hardware

One run is 750 rollouts on one GPU. The SLURM jobs of the three creativity runs took between 7 h 41 min and
8 h 35 min on one NVIDIA B200. Generating and relaxing 200 structures took about 150 s in our
tests, so 2,500 take about half an hour. LeMat-GenBench runs on CPUs and takes several hours for 2,500 structures.

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

## Acknowledgments and data

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
