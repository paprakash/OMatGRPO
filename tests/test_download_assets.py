"""Hash checks of scripts/download_assets.py (no downloads)."""
import hashlib
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import download_assets  # noqa: E402


def test_present_file_with_matching_hash_is_kept(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert download_assets.present(p, hashlib.sha256(b"abc").hexdigest())


def test_missing_file_is_not_present(tmp_path):
    assert not download_assets.present(tmp_path / "missing", "0" * 64)


def test_present_file_with_wrong_hash_is_an_error_and_is_kept(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    with pytest.raises(SystemExit, match="expected"):
        download_assets.present(p, "0" * 64)
    assert p.read_bytes() == b"abc"


def test_downloaded_file_with_wrong_hash_is_not_installed(tmp_path):
    tmp = tmp_path / "download"
    tmp.write_bytes(b"abc")
    dest = tmp_path / "data" / "f.bin"
    with pytest.raises(SystemExit, match="sha256"):
        download_assets.install(tmp, dest, "0" * 64, "test")
    assert not dest.exists()


def test_identifiers_match_recipes():
    recipes = {p.stem for p in (REPO / "configs" / "recipes").glob("*.yaml")}
    assert set(download_assets.MODEL_FILES) == recipes
    assert set(download_assets.STRUCTURE_FILES) == recipes | {"R0", "bestofn48k_top2500"}


def _fake_hub(monkeypatch, repo_dir):
    """hf_hub_download and snapshot_download served from a local folder laid out like the repository."""
    import huggingface_hub

    def hf_hub_download(repo_id, filename, revision, local_dir):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_dir / filename, dest)
        return str(dest)

    def snapshot_download(repo_id, revision, allow_patterns, local_dir):
        prefix = allow_patterns[0].rstrip("*")
        for src in repo_dir.rglob("*"):
            rel = src.relative_to(repo_dir).as_posix()
            if src.is_file() and rel.startswith(prefix):
                dest = Path(local_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)


def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    files = {"structures/S/structures_summary.csv": b"idx\n0\n", "structures/S/result.json": b"{}",
             "structures/S/cifs/0000_X.cif": b"cif"}
    for rel, data in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    sums = {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()}
    (repo / "SHA256SUMS").write_text("".join(f"{h}  {rel}\n" for rel, h in sums.items()))
    pinned = {"structures_summary.csv": sums["structures/S/structures_summary.csv"],
              "result.json": sums["structures/S/result.json"]}
    return repo, pinned


def test_structure_set_is_installed_and_checked(tmp_path, monkeypatch):
    repo, pinned = _fake_repo(tmp_path)
    _fake_hub(monkeypatch, repo)
    monkeypatch.setattr(download_assets, "STRUCTURE_FILES", {"S": pinned})
    data = tmp_path / "data"
    data.mkdir()
    download_assets.fetch_structures("S", data, "x/y", "main")
    assert (data / "structures/S/cifs/0000_X.cif").read_bytes() == b"cif"
    download_assets.fetch_structures("S", data, "x/y", "main")      # present: skipped


def test_structure_file_with_wrong_hash_is_refused(tmp_path, monkeypatch):
    repo, pinned = _fake_repo(tmp_path)
    (repo / "structures/S/cifs/0000_X.cif").write_bytes(b"changed")
    _fake_hub(monkeypatch, repo)
    monkeypatch.setattr(download_assets, "STRUCTURE_FILES", {"S": pinned})
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(SystemExit, match="sha256"):
        download_assets.fetch_structures("S", data, "x/y", "main")
    assert not (data / "structures/S").exists()
