# Megacompact DeFi Kaggle Pipeline

A GitHub-linked Kaggle pipeline for generating synthetic DeFi decision
packets, simulating outcomes, training PnL-prediction models, and evaluating
profitability prediction.

- **GitHub** stores and versions the code
- **Kaggle** executes the pipeline (CPU is enough — no GPU required)
- **You** download the trained models and evaluation results when a run finishes

## Pipeline

1. **Cell A** — initializes the run (seed 42, run directory, telemetry, repo detection)
2. **Cell B** — generates decision packets and outcome labels
3. **Data inspection** — optional data-quality report
4. **Cell C** — trains PRIME and PHANTOM
5. **Cell D** — evaluates both models on a time-based test set and recommends a winner
6. Download the artifacts from the Kaggle output panel

## Models

| Model | Algorithm | Role |
|-------|-----------|------|
| **PRIME** | Ridge regression | Conservative, stable, interpretable baseline |
| **PHANTOM** | Random Forest | Exploratory, captures non-linear relationships |

Cell C extracts **20 features** from each real megacompact `DecisionPacket`
(block/timestamp, objective horizon & capital, aggregate token prices,
`base_fee_gwei`, `total_liquidity`, action-candidate size/slippage/deadline
aggregates, and trade constraints), splits the data **by block number**
(70/15/15 — no future leakage), and predicts `net_pnl_usd`.

Cell D reports R², RMSE, MAE, **profit precision/recall**, direction
accuracy, and rank correlation, then recommends the stronger model. The
winner is chosen from the actual test results of the current run — it is not
hardcoded.

## Quick start on Kaggle

1. Create a new Kaggle notebook and turn **Settings → Internet → On**
2. First cell — clone this repo:

```python
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ifelaniya-byte/megacompact-defi-kaggle-pipeline.git"
REPO_PATH = Path("/kaggle/working/megacompact-defi-kaggle-pipeline")

if not REPO_PATH.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_PATH)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_PATH), "pull"], check=True)

sys.path.insert(0, str(REPO_PATH))
print("Repository ready at:", REPO_PATH)
```

3. Then one cell per pipeline step, in this order:

```python
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_A_setup.py").read())
```
```python
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_B_data_generation.py").read())
```
```python
exec(open(f"{REPO_PATH}/kaggle_pipeline/DATA_INSPECTION.py").read())   # optional
```
```python
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_C_megacompact_training.py").read())
```
```python
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_D_megacompact_evaluation.py").read())
```

4. Zip and download the run folder:

```python
import shutil
print(shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name))
```

Full walkthrough: [docs/MEGACOMPACT_KAGGLE_GUIDE.md](docs/MEGACOMPACT_KAGGLE_GUIDE.md)

## Final artifacts

After a complete run, `/kaggle/working/megacompact_run_<RUN_ID>/` contains:

```text
data/
├── decision_packets.jsonl      # packets produced by Cell B
└── outcome_labels.json         # simulated PnL outcomes
checkpoints/
├── prime_model.pkl             # trained Ridge model
├── phantom_model.pkl           # trained Random Forest model
├── scaler.pkl                  # REQUIRED with the model for inference
├── metadata.json               # training details
└── evaluation_results.json     # test metrics + recommended winner
run_config.json                 # seed, repo commit, paths
telemetry.jsonl                 # event log
```

The practical deliverable is **the winning model + its scaler +
`evaluation_results.json`**.

## Repository structure

```text
kaggle_pipeline/     all Kaggle cell code (A, B, C, D, data inspection)
megacompact_core/    real MegaCompact16 engine (core.py) + double-pass
                     verification gate (engineers.py)
configs/             default_config.json — documented run/model defaults
docs/                MEGACOMPACT_KAGGLE_GUIDE.md
scripts/             local_smoke_test.py — run the whole pipeline locally
results/             small metric summaries (never models or raw data)
```

### About Cell B and `megacompact_core/`

`megacompact_core/core.py` is the **real MegaCompact16 engine** (144 KB) and
`megacompact_core/engineers.py` is the **real double-pass verification gate**.
Cell B drives the engine directly — `SyntheticMarketGenerator` →
`DataValidator` → `TimeCausalFeatureStore` + `DecisionPacketBuilder` →
`LabelBuilder` (AMM / quote / execution / cost / outcome) → `DoublePassEngineerGate`
— then persists `decision_packets.jsonl` and `outcome_labels.json`, which
Cells C and D consume. Cell A copies `core.py` / `engineers.py` into the
working directory so the classic `import core as mc` /
`from engineers import DoublePassEngineerGate` flow works on Kaggle too.

The default run reproduces the real research scale:
`events=121671`, `valid=121653`, `packets=1199`, `labels=3597`,
`allowed=100/100`.

## Local testing

```bash
pip install -r requirements.txt
python scripts/local_smoke_test.py
```

Runs all five cells in a temp directory and verifies every artifact.

## Model selection

Use `evaluation_results.json` to compare R², RMSE, MAE, profit precision,
profit recall, direction accuracy, and rank correlation. Do not assume a
particular model always wins — pick the winner from the current run's test
results.

## Important

This project is for research and simulation only. It does not execute real
trades and should not be connected to real funds without additional testing,
monitoring, paper trading, and risk controls.
