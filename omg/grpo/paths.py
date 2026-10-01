"""Location of downloaded and built data (MP-20 LMDBs, references, hull cache).

Resolution order: the --data_dir flag of omg.grpo.train (which sets OMATGRPO_DATA_DIR for the
process), then the OMATGRPO_DATA_DIR environment variable, then omg/data inside the repository.
"""
import os
from pathlib import Path

ENV_VAR = "OMATGRPO_DATA_DIR"


def data_dir() -> Path:
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data"


def set_data_dir(path) -> Path:
    """Make `path` the data directory for this process (and its children)."""
    path = Path(path).expanduser().resolve()
    os.environ[ENV_VAR] = str(path)
    return path


def hull_dir() -> Path:
    """Local copies of the LeMat-Bulk-MLIP-Hull files, if any (checked before downloading)."""
    return data_dir() / "convex_hulls"


def hf_cache_dir() -> Path:
    """Download cache for the LeMat-Bulk-MLIP-Hull files."""
    return data_dir() / ".cache"


def references_dir() -> Path:
    """MMD composition reference and creativity reference."""
    return data_dir() / "references"
