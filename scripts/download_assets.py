"""Download the assets of OMatGRPO into the data directory and check their sha256.

The weights and the evaluated structures are on Hugging Face (repository HF_REPO below), not in this
repository. The MP-20 LMDBs come from upstream OMatG. The two MP-20 references of the reward are not
downloaded: build them with scripts/build_references.py after this script.

  prior       <data_dir>/prior/prior.safetensors and train.yaml (our OMatG pretrain on MP-20; the
              starting point and KL reference of every run)
  mp20        <data_dir>/mp_20/{train,val,test}.lmdb, from upstream OMatG at the commit this
              package is copied from
  models      <data_dir>/models/<identifier>/final_model.safetensors and resolved_config.json for the
              seven runs of the paper (or the ones given with --models)
  structures  <data_dir>/structures/<identifier>/: the 2,500-structure sets that LeMat-GenBench scored
              for the paper (cifs.zip, unpacked into cifs/, structures_summary.csv and the LeMat-GenBench
              result JSON), for the prior (R0), best-of-N and the seven runs (or the ones given with
              --structures)

Default: prior and mp20. The data directory is --data_dir, else $OMATGRPO_DATA_DIR, else omg/data.
Files already present with the right sha256 are skipped; a file with the wrong sha256 is an error and
is not overwritten.

Usage:
  python scripts/download_assets.py [prior mp20 models structures] [--models ID ...]
                                    [--structures ID ...] [--data_dir DIR]
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omg.grpo import paths  # noqa: E402

HF_REPO = "paprakash/OMatGRPO"
HF_REVISION = "main"

OMATG_COMMIT = "9172203a9026af41732247cc62aaf4d903010e9c"
OMATG_RAW = f"https://raw.githubusercontent.com/FERMat-ML/OMatG/{OMATG_COMMIT}/omg/data/mp_20"

# path in the data directory (= path in the Hugging Face repository) -> sha256
PRIOR_FILES = {
    "prior/prior.safetensors": "c9a66c0cdd572c59cc1e8b6554458adef2d32ac142e93cbd1b746f45a81bffca",
    "prior/train.yaml": "85922da23e400240012194956e5a6d6b9d7de814d1fd28b902678c3b0f4077f8",
}
# identifier -> file in models/<identifier>/ -> sha256
MODEL_FILES = {
    "arityguard_creatrelax": {
        "final_model.safetensors": "2c65da63c2d89eb1db791f9d693fd108595fca135a0284b4f2c4e029d3cad088",
        "resolved_config.json": "06dc0f6fdead5608dc7160ab2be3acee5c23035415ab7ea192414eb8c8290fb7",
    },
    "canonical_creatrelax": {
        "final_model.safetensors": "f0bfd5fdebae329013ecb387ca7da5d2998641bcab743c7f08a150a474049ec1",
        "resolved_config.json": "71fd1e7f043ec70e4e85147b832a85a914333e599f383d3b27297fd8a987bb7f",
    },
    "sparseworst_creatrelax": {
        "final_model.safetensors": "ae6f60cd8258d70da6c415709730a3a6afdf76db88426bb813a79a2d7736df74",
        "resolved_config.json": "e807aa3c89097d7e8dc9f276c54b379f5491b9a2872495363357413900d222b0",
    },
    "arityguard": {
        "final_model.safetensors": "13d24de97e1674cb6eb27c1ac1873d71c1c408b7a8524d9348b55f967c289bfa",
        "resolved_config.json": "4d83fe66650ad8bb47af42500794ebdfe31253f47cbf8e1a002ff630a0b6c116",
    },
    "canonical": {
        "final_model.safetensors": "e855d4b9fc98dec8ed55cb0e64e99e7ee3f378cf61faa8954f01913e9c8f1e6e",
        "resolved_config.json": "2698cd027a015646f78c393cc097158fd75fa864868818b3463c46c9ace801cb",
    },
    "sparseworst": {
        "final_model.safetensors": "dbe6119cc4f987d2d0d12f0d8be04be1d3d365719cef522c6fa56ebf157935d6",
        "resolved_config.json": "d0af403b763fdd0ab3afaa12ac6e964892860a4cfc68cd91cfd99c9d53afb8fc",
    },
    "frozen_control": {
        "final_model.safetensors": "1de39ff8e84783f8332e7fa60a42aef9a427ac786dee8e038bc8665ba3e94188",
        "resolved_config.json": "f9db6397cfef0eba68260b6e8b93c795f8c560b01e8d581af48af279f121bbea",
    },
}
# identifier -> file in structures/<identifier>/ -> sha256 (structures/<identifier>/cifs.zip is checked
# against the repository's SHA256SUMS, whose entries for these files must match the values here)
STRUCTURE_FILES = {
    "R0": {
        "R0_comprehensive_multi_mlip_hull_20260807_034204.json": "d7788416f35a0429667db097efc575fa5607a20a8d80124233c9afb4d127a26e",
        "structures_summary.csv": "b20d8790bede4d1ac8297b4b414a04e649d9587204b3e3fa0cd561d051b254b4",
    },
    "bestofn48k_top2500": {
        "bestofn48k_top2500_comprehensive_multi_mlip_hull_20260820_131617.json": "d63b3d439ef9df00b2cad999da3acfe1fc1d818b04b6a45d4e68eda51672683c",
        "structures_summary.csv": "514efdddf5b3f45bcf980cecb0127bbadcb76eb753dd7b286dc9f4853a6240af",
    },
    "arityguard_creatrelax": {
        "arityguard_creatrelax_r750_comprehensive_multi_mlip_hull_20260806_202703.json": "dec64e8b93fd86989af2ee1e5d1d78d50e7f0f418dad9c3741a17a07b11ebe75",
        "structures_summary.csv": "9974995a64d7bede024218e932a4dbab8024b3654a9d2b507bcb163778948c62",
    },
    "canonical_creatrelax": {
        "canonical_creatrelax_r750_comprehensive_multi_mlip_hull_20260806_232418.json": "c8d3460c2b328e8c699a7620cd1b1cc66c5977399a2490fd9b261668e95e778f",
        "structures_summary.csv": "890f2a5c4de185e2c077cc5f5165489de180bba780be919daf742af7a37f6cfa",
    },
    "sparseworst_creatrelax": {
        "sparseworst_creatrelax_r750_comprehensive_multi_mlip_hull_20260806_204926.json": "d7b753e9deb5b899c26b828d43ee9469d924a497b5c0e9d23fa37781cdf4e285",
        "structures_summary.csv": "14c8627f5d9df4ad3dcd37719cb5f460a8f83deaf7ef4471a2cf1e2551f1a2fa",
    },
    "arityguard": {
        "arityguard_r750_comprehensive_multi_mlip_hull_20260806_103812.json": "6b594ef6e8cd87b0dc97c699c70a688e873ba965cd64be4359a20fe67d6ca7ba",
        "structures_summary.csv": "c4d397dd742325b9a87de63f39bd9fd3a96008f85983d7c58e885c5c273726b8",
    },
    "canonical": {
        "canonical_r750_comprehensive_multi_mlip_hull_20260806_143220.json": "3e29213623d4f6d72c8f9c1cd1551d31ec8ae7575db3482ab24c0e72b72de276",
        "structures_summary.csv": "cdd5419377ca02cc79720da2859bf654da46d87e1cd60f4519e1867028fd2a6f",
    },
    "sparseworst": {
        "sparseworst_r750_comprehensive_multi_mlip_hull_20260806_151650.json": "637efbdef33345a1fca20508ad633f7489ae85672297ac640953e6813af4edd5",
        "structures_summary.csv": "cdb686450092325ab5c3dd55e894309f01c55adfe1f9506d1253e9246439bd32",
    },
    "frozen_control": {
        "frozen_control_r750_comprehensive_multi_mlip_hull_20260820_051609.json": "8947b2ad9181a606d8645ed67fa6ecdbd68519455dd2a2650fdd6b3f84a55b81",
        "structures_summary.csv": "8f782858e55cf82f188f209e4e2505169a846be4c672f3e7730f8ce6863b9b55",
    },
}
MP20_LMDB = {
    "mp_20/train.lmdb": "ca21166f6d0ecaac7278629ef813eac734797bcbe98f34d580993a2ae14c545b",
    "mp_20/val.lmdb": "14f73bbea446909f92af23bc06902bb3d2f08bede34c1ca08837cff62671cd43",
    "mp_20/test.lmdb": "74223481f135274e54375c52e38ace5ab7a2403f367a4dd8352cfd1874b2986d",
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def present(dest: Path, expected: str) -> bool:
    """True if dest exists with the expected hash; raise if it exists with another one."""
    if not dest.exists():
        return False
    got = sha256(dest)
    if got != expected:
        raise SystemExit(f"{dest} exists with sha256 {got}, expected {expected}. Move it away and rerun.")
    print(f"ok (present)  {dest}")
    return True


def install(tmp: Path, dest: Path, expected: str, source: str):
    got = sha256(tmp)
    if got != expected:
        raise SystemExit(f"{source}: sha256 {got}, expected {expected}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(dest))
    print(f"ok            {dest}")


def fetch_hf(rel: str, expected: str, root: Path, repo: str, revision: str):
    dest = root / rel
    if present(dest, expected):
        return
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory(dir=root) as tmpdir:
        tmp = Path(hf_hub_download(repo_id=repo, filename=rel, revision=revision, local_dir=tmpdir))
        install(tmp, dest, expected, f"{repo}/{rel}")


def fetch_url(url: str, rel: str, expected: str, root: Path):
    dest = root / rel
    if present(dest, expected):
        return
    with tempfile.TemporaryDirectory(dir=root) as tmpdir:
        tmp = Path(tmpdir) / Path(rel).name
        print(f"downloading   {url}")
        urllib.request.urlretrieve(url, tmp)
        install(tmp, dest, expected, url)


def read_sha256sums(path) -> dict:
    sums = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            sums[rel.strip()] = digest
    return sums


def fetch_structures(ident: str, root: Path, repo: str, revision: str):
    """One structure set: cifs.zip, unpacked into cifs/, and the pinned summary and result JSON.

    The sha256 of cifs.zip comes from the repository's SHA256SUMS, whose entries for the pinned files must
    match the values in STRUCTURE_FILES. The CIFs are zipped because a Hugging Face repository holds at
    most 20,000 files.
    """
    from huggingface_hub import hf_hub_download
    prefix = f"structures/{ident}/"
    with tempfile.TemporaryDirectory(dir=root) as tmpdir:
        sums = read_sha256sums(hf_hub_download(repo_id=repo, filename="SHA256SUMS", revision=revision,
                                               local_dir=tmpdir))
    for name, expected in STRUCTURE_FILES[ident].items():
        if sums.get(prefix + name) != expected:
            raise SystemExit(f"{repo} SHA256SUMS does not list {prefix + name} with the pinned sha256")
    if prefix + "cifs.zip" not in sums:
        raise SystemExit(f"{repo} SHA256SUMS does not list {prefix}cifs.zip")
    fetch_hf(prefix + "cifs.zip", sums[prefix + "cifs.zip"], root, repo, revision)
    for name, expected in STRUCTURE_FILES[ident].items():
        fetch_hf(prefix + name, expected, root, repo, revision)
    unpack_cifs(root / prefix)


def unpack_cifs(set_dir: Path):
    """Unpack set_dir/cifs.zip (members cifs/<name>.cif) into set_dir/cifs/, replacing an existing cifs/."""
    with zipfile.ZipFile(set_dir / "cifs.zip") as z:
        names = [n for n in z.namelist() if n != "cifs/"]
        bad = [n for n in names if not (n.startswith("cifs/") and n.endswith(".cif") and n.count("/") == 1)]
        if bad:
            raise SystemExit(f"{set_dir / 'cifs.zip'}: unexpected members {bad[:3]}")
        with tempfile.TemporaryDirectory(dir=set_dir) as tmpdir:
            z.extractall(tmpdir, members=names)
            dest = set_dir / "cifs"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(Path(tmpdir) / "cifs"), str(dest))
    print(f"ok            {dest} ({len(names)} CIFs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="*", choices=["prior", "mp20", "models", "structures"],
                    help="what to download (default: prior mp20)")
    ap.add_argument("--models", nargs="+", choices=sorted(MODEL_FILES), default=None,
                    help="the runs whose weights to download (default with 'models': all seven)")
    ap.add_argument("--structures", nargs="+", choices=sorted(STRUCTURE_FILES), default=None,
                    help="the structure sets to download (default with 'structures': all nine)")
    ap.add_argument("--data_dir", default=None, help="data directory (default: $OMATGRPO_DATA_DIR, else omg/data)")
    ap.add_argument("--repo", default=HF_REPO, help="Hugging Face repository")
    ap.add_argument("--revision", default=HF_REVISION, help="revision of the Hugging Face repository")
    args = ap.parse_args()

    what = args.what or ["prior", "mp20"]
    if args.models and "models" not in what:
        what.append("models")
    if args.structures and "structures" not in what:
        what.append("structures")
    root = paths.set_data_dir(args.data_dir) if args.data_dir else paths.data_dir()
    root.mkdir(parents=True, exist_ok=True)
    print(f"data directory: {root}")

    if "prior" in what:
        for rel, expected in PRIOR_FILES.items():
            fetch_hf(rel, expected, root, args.repo, args.revision)
    if "mp20" in what:
        for rel, expected in MP20_LMDB.items():
            fetch_url(f"{OMATG_RAW}/{Path(rel).name}", rel, expected, root)
    if "models" in what:
        for ident in args.models or sorted(MODEL_FILES):
            for name, expected in MODEL_FILES[ident].items():
                fetch_hf(f"models/{ident}/{name}", expected, root, args.repo, args.revision)
    if "structures" in what:
        for ident in args.structures or sorted(STRUCTURE_FILES):
            fetch_structures(ident, root, args.repo, args.revision)
    if not (root / "references").exists():
        print("next: python scripts/build_references.py mmd && python scripts/build_references.py creativity")


if __name__ == "__main__":
    main()
