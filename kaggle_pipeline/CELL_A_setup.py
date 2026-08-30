"""
CELL A: SETUP (MEGACOMPACT)
Initializes the run: seed, run directory, telemetry, and repo detection.

Run this FIRST in the Kaggle notebook (or locally) before Cell B.

Defines for all later cells:
  PRIMARY_SEED  int      global random seed (42)
  RUN_DIR       Path     per-run output directory
  telemetry     object   JSONL event logger
  REPO_PATH     Path|None  location of the cloned repository (if found)

No torch / transformers / GPU required — the Megacompact DeFi pipeline is
CPU-only (numpy + pandas + scikit-learn).
"""

import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

print("=" * 70)
print("CELL A: SETUP (MEGACOMPACT)")
print("=" * 70)

# ===== Step 1: Locate the repository (if cloned) =====
def _detect_repo_root():
    """Find the repo root that contains kaggle_pipeline/CELL_A_setup.py."""
    candidates = []

    env_repo = os.environ.get("MEGACOMPACT_REPO")
    if env_repo:
        candidates.append(Path(env_repo))

    # Common Kaggle / manual clone locations
    candidates.extend([
        Path("/kaggle/working/megacompact-defi-kaggle-pipeline"),
        Path("/kaggle/working/repo"),
        Path("/tmp/repo"),
        Path.cwd(),
    ])
    # Anything already on sys.path (e.g. scripts/ helpers)
    candidates.extend(Path(p) for p in list(sys.path) if p)
    # Parents of cwd (running from a subdirectory of the repo)
    candidates.extend(Path.cwd().parents)

    for cand in candidates:
        try:
            if (cand / "kaggle_pipeline" / "CELL_A_setup.py").exists():
                return cand.resolve()
        except (OSError, TypeError):
            continue
    return None

REPO_PATH = _detect_repo_root()
if REPO_PATH is not None:
    print(f"✓ Repository found: {REPO_PATH}")
else:
    print("⚠ Repository not found (running standalone).")
    print("  If you cloned the repo, set REPO_PATH or MEGACOMPACT_REPO before this cell.")

# ===== Step 2: Seed =====
PRIMARY_SEED = 42
random.seed(PRIMARY_SEED)
os.environ["PYTHONHASHSEED"] = str(PRIMARY_SEED)
try:
    import numpy as np
    np.random.seed(PRIMARY_SEED)
    print(f"✓ Seeds set (random + numpy): {PRIMARY_SEED}")
except ImportError:
    print(f"✓ Seeds set (random only): {PRIMARY_SEED}")

# ===== Step 3: Run directory =====
WORK_BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "runs"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = WORK_BASE / f"megacompact_run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "data").mkdir(exist_ok=True)
(RUN_DIR / "checkpoints").mkdir(exist_ok=True)
print(f"✓ Run directory: {RUN_DIR}")

# ===== Step 4: Telemetry =====
class SimpleTelemetry:
    """Minimal JSONL event logger shared by all cells."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type, **kwargs):
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event_type}
        entry.update(kwargs)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

telemetry = SimpleTelemetry(RUN_DIR / "telemetry.jsonl")

# ===== Step 5: Make repo modules importable =====
if REPO_PATH is not None:
    for p in (str(REPO_PATH), str(REPO_PATH / "megacompact_core")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Copy core.py / engineers.py next to the working dir so that code which
    # does `import core` / `import engineers` (the old %%writefile workflow)
    # keeps working when the real framework files are in megacompact_core/.
    for name in ("core.py", "engineers.py"):
        src = REPO_PATH / "megacompact_core" / name
        if src.exists():
            try:
                shutil.copy2(src, WORK_BASE / name)
                print(f"✓ Copied {src.name} -> {WORK_BASE / name}")
            except OSError as e:
                print(f"⚠ Could not copy {src.name}: {e}")

# ===== Step 6: Provenance (repo commit, if available) =====
run_config = {
    "run_id": RUN_ID,
    "seed": PRIMARY_SEED,
    "run_dir": str(RUN_DIR),
    "repo_path": str(REPO_PATH) if REPO_PATH else None,
    "python": sys.version.split()[0],
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
if REPO_PATH is not None:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_PATH), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if commit.returncode == 0:
            run_config["repo_commit"] = commit.stdout.strip()
            print(f"✓ Repo commit: {run_config['repo_commit'][:12]}…")
    except (OSError, subprocess.SubprocessError):
        pass

with open(RUN_DIR / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

telemetry.write("CELL_A_SETUP_COMPLETE", **run_config)

print()
print(f"✓ CELL A complete — run ID: megacompact_run_{RUN_ID}")
print("Next: run CELL_B_data_generation.py")
