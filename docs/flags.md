# Flag reference

Generated from the parser (`omg.grpo.train.flag_table_markdown()`). `python -m omg.grpo.train --help` prints the
same. Defaults are the settings of the OMatGRPO run. An unknown flag is an error.

**run**

| flag | default | description |
|---|---|---|
| `--config` | `None` | YAML file with flag values (keys = flag names without --); command-line flags override it |
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
