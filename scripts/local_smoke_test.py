#!/usr/bin/env python3
"""
Local smoke test for the Megacompact DeFi pipeline.

Runs all five pipeline cells (A, B, DATA_INSPECTION, C, D) in order — exactly
the way the Kaggle notebook runs them — inside a temporary directory, then
verifies every expected artifact exists.

Usage:
    python3 scripts/local_smoke_test.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ["MEGACOMPACT_REPO"] = str(REPO)
# Use the real MegaCompact16 engine on a tiny config so the wiring can be
# verified quickly. Remove this to run the full 30-day research config.
os.environ["MEGACOMPACT_FAST_SMOKE"] = "1"

CELLS = [
    "CELL_A_setup.py",
    "CELL_B_data_generation.py",
    "DATA_INSPECTION.py",
    "CELL_C_megacompact_training.py",
    "CELL_D_megacompact_evaluation.py",
]

EXPECTED_ARTIFACTS = [
    "data/decision_packets.jsonl",
    "data/outcome_labels.json",
    "checkpoints/prime_model.pkl",
    "checkpoints/phantom_model.pkl",
    "checkpoints/scaler.pkl",
    "checkpoints/metadata.json",
    "checkpoints/evaluation_results.json",
    "telemetry.jsonl",
    "run_config.json",
]


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="megacompact_smoke_"))
    os.chdir(workdir)
    print(f"Smoke test working directory: {workdir}")

    # Shared namespace, mirroring a single notebook kernel session
    namespace = {}

    for cell in CELLS:
        path = REPO / "kaggle_pipeline" / cell
        print()
        print("=" * 70)
        print(f">>> EXECUTING {cell}")
        print("=" * 70)
        source = path.read_text()
        exec(compile(source, str(path), "exec"), namespace)  # noqa: S102

    run_dir = Path(namespace["RUN_DIR"])
    print()
    print("=" * 70)
    print("ARTIFACT CHECK")
    print("=" * 70)
    missing = []
    for rel in EXPECTED_ARTIFACTS:
        target = run_dir / rel
        status = "✓" if target.exists() and target.stat().st_size > 0 else "❌ MISSING"
        if status.startswith("❌"):
            missing.append(rel)
        print(f"  {status} {rel}")

    if missing:
        print(f"\nSMOKE TEST FAILED — missing: {missing}")
        return 1

    print()
    print("=" * 70)
    print("✓ SMOKE TEST PASSED — full pipeline ran end to end")
    print(f"  Run dir: {run_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
