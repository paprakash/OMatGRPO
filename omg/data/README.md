# Data directory

`scripts/download_assets.py` (from Hugging Face and upstream OMatG) and `scripts/build_references.py` fill
this folder. Nothing in it is tracked except this file. Set `OMATGRPO_DATA_DIR` or pass `--data_dir` to use
another location.

```
prior/prior.safetensors, prior/train.yaml     the OMatG pretrain every run starts from
mp_20/{train,val,test}.lmdb                   MP-20, from upstream OMatG
models/<identifier>/                          trained weights and resolved config of the paper's runs (optional)
structures/<identifier>/                      the structure sets LeMat-GenBench scored for the paper (optional)
references/mp20_comp_reference.pt             MMD composition reference (scripts/build_references.py)
references/mp20_train_ref.json.gz             creativity reference (scripts/build_references.py)
convex_hulls/, .cache/                        LeMat-Bulk-MLIP-Hull files, downloaded on first use
```
