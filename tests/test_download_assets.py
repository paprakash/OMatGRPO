"""Hash checks of scripts/download_assets.py (no downloads)."""
import hashlib
import shutil
import sys
import zipfile
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


def test_identifiers_match_runs():
    runs = {p.stem for p in (REPO / "configs" / "runs").glob("*.yaml")}
    assert set(download_assets.MODEL_FILES) == runs
    assert set(download_assets.STRUCTURE_FILES) == runs | {"R0", "bestofn48k_top2500"}


def _fake_hub(monkeypatch, repo_dir):
    """hf_hub_download served from a local folder laid out like the repository."""
    import huggingface_hub

    def hf_hub_download(repo_id, filename, revision, local_dir):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_dir / filename, dest)
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)


def _write_zip(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


def _fake_repo(tmp_path, members=None):
    repo = tmp_path / "repo"
    files = {"structures/S/structures_summary.csv": b"idx\n0\n", "structures/S/result.json": b"{}"}
    for rel, data in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    _write_zip(repo / "structures/S/cifs.zip", members or {"cifs/0000_X.cif": b"cif"})
    rels = [*files, "structures/S/cifs.zip"]
    sums = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in rels}
    (repo / "SHA256SUMS").write_text("".join(f"{h}  {rel}\n" for rel, h in sums.items()))
    pinned = {"structures_summary.csv": sums["structures/S/structures_summary.csv"],
              "result.json": sums["structures/S/result.json"]}
    return repo, pinned


def _setup(tmp_path, monkeypatch, members=None):
    repo, pinned = _fake_repo(tmp_path, members)
    _fake_hub(monkeypatch, repo)
    monkeypatch.setattr(download_assets, "STRUCTURE_FILES", {"S": pinned})
    data = tmp_path / "data"
    data.mkdir()
    return repo, data


def test_structure_set_is_installed_and_unpacked(tmp_path, monkeypatch):
    _, data = _setup(tmp_path, monkeypatch)
    download_assets.fetch_structures("S", data, "x/y", "main")
    assert (data / "structures/S/cifs/0000_X.cif").read_bytes() == b"cif"
    assert (data / "structures/S/structures_summary.csv").exists()
    download_assets.fetch_structures("S", data, "x/y", "main")      # present: skipped, unpacked again
    assert sorted(p.name for p in (data / "structures/S/cifs").iterdir()) == ["0000_X.cif"]


def test_zip_with_wrong_hash_is_refused(tmp_path, monkeypatch):
    repo, data = _setup(tmp_path, monkeypatch)
    _write_zip(repo / "structures/S/cifs.zip", {"cifs/0000_X.cif": b"changed"})
    with pytest.raises(SystemExit, match="sha256"):
        download_assets.fetch_structures("S", data, "x/y", "main")
    assert not (data / "structures/S").exists()


def test_pinned_file_missing_from_sha256sums_is_refused(tmp_path, monkeypatch):
    repo, data = _setup(tmp_path, monkeypatch)
    lines = (repo / "SHA256SUMS").read_text().splitlines()
    (repo / "SHA256SUMS").write_text("\n".join(l for l in lines if "result.json" not in l) + "\n")
    with pytest.raises(SystemExit, match="pinned"):
        download_assets.fetch_structures("S", data, "x/y", "main")


def test_zip_with_unexpected_members_is_refused(tmp_path, monkeypatch):
    _, data = _setup(tmp_path, monkeypatch, members={"cifs/0000_X.cif": b"cif", "other/evil.txt": b"x"})
    with pytest.raises(SystemExit, match="unexpected members"):
        download_assets.fetch_structures("S", data, "x/y", "main")
    assert not (data / "structures/S/cifs").exists()
