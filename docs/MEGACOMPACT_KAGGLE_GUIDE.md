# Megacompact DeFi Kaggle Guide

Complete guide for running the Megacompact DeFi PnL-prediction pipeline on
Kaggle from this GitHub repository.

## Overview

The pipeline:

1. Generates synthetic DeFi market events
2. Builds **decision packets** (potential trading decisions)
3. Simulates outcomes with realistic market dynamics
4. Creates **outcome labels** (actual PnL results)
5. Trains two ML models to predict net PnL from decision packets
6. Evaluates both models and recommends a winner

The two competing models:

| Model | Algorithm | Role |
|-------|-----------|------|
| **PRIME** | Ridge regression | Conservative, stable, interpretable baseline |
| **PHANTOM** | Random Forest | Exploratory, captures non-linear relationships |

> The winner is **not** predetermined. Cell D selects it from the actual test
> results of the current run.

## Repository structure

```text
megacompact-defi-kaggle-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── kaggle_pipeline/
│   ├── CELL_A_setup.py                 # seed, run dir, telemetry, repo detection
│   ├── CELL_B_data_generation.py       # decision packets + outcome labels
│   ├── CELL_C_megacompact_training.py  # PRIME + PHANTOM training
│   ├── CELL_D_megacompact_evaluation.py# test-set evaluation + recommendation
│   └── DATA_INSPECTION.py              # optional data quality report
├── megacompact_core/
│   ├── __init__.py
│   ├── core.py                         # DecisionPacket/OutcomeLabel schema (reference)
│   └── engineers.py                    # verification gate (reference)
├── configs/
│   └── default_config.json             # documented defaults (seed, split, models)
├── docs/
│   └── MEGACOMPACT_KAGGLE_GUIDE.md     # this file
├── scripts/
│   └── local_smoke_test.py             # run the whole pipeline locally
└── results/                            # small metric summaries only
```

**Note on Cell B:** the shipped `CELL_B_data_generation.py` is a
self-contained *reference generator* that produces data in the exact
megacompact DecisionPacket / OutcomeLabel schema. Cells C and D depend only
on the two files it writes (`data/decision_packets.jsonl` and
`data/outcome_labels.json`), so you can replace Cell B (and drop the real
`megacompact_core.py` / `megacompact_engineers.py` into `megacompact_core/`)
without touching anything downstream.

## Kaggle setup (step by step)

### Step 1: Create the notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Name it e.g. `Megacompact DeFi PnL Training`
3. **Settings → Internet → On** (required to `git clone` from GitHub)
4. Accelerator: none needed — the pipeline is CPU-only (Ridge + Random Forest)

### Step 2: Clone the repository (first cell)

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

The repository is public, so no token or authentication is needed for cloning.

### Step 3: Run the pipeline cells (in order)

One Kaggle cell per pipeline script:

```python
# Cell: A — setup (seed, run dir, telemetry)
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_A_setup.py").read())
```

```python
# Cell: B — generate decision packets + outcome labels
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_B_data_generation.py").read())
```

```python
# Cell: optional — inspect the data before training
exec(open(f"{REPO_PATH}/kaggle_pipeline/DATA_INSPECTION.py").read())
```

```python
# Cell: C — train PRIME and PHANTOM
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_C_megacompact_training.py").read())
```

```python
# Cell: D — evaluate both models on the test set
exec(open(f"{REPO_PATH}/kaggle_pipeline/CELL_D_megacompact_evaluation.py").read())
```

### Step 4: Download the artifacts

```python
import shutil

zip_path = shutil.make_archive(
    str(RUN_DIR), "zip",
    root_dir=RUN_DIR.parent,
    base_dir=RUN_DIR.name,
)
print(f"Created: {zip_path}")
```

Then download the ZIP from the notebook **Output** panel.

## Expected output

### Cell B (data)

```text
1. GENERATE SYNTHETIC MARKET EVENTS
   events=12000
2. VALIDATE (time-causality, schema)
   valid=12000/12000
3. BUILD DECISION PACKETS
   packets=1199
4. BUILD OUTCOME LABELS
   labels=3597
✓ CELL B (data) complete
```

### Cell C (training)

```text
✓ Features extracted: shape=(1199, 20)
✓ Time-based split (by block_number):
  Train: 839 samples | Val: 179 samples | Test: 181 samples
1. TRAINING PRIME MODEL (Ridge Regression, alpha=1.0)
   ✓ PRIME R²: Train=0.6993 | Val=0.7063
   ✓ 5-Fold CV: Mean=0.6772 ± 0.0186
2. TRAINING PHANTOM MODEL (Random Forest, n_estimators=100)
   ✓ PHANTOM R²: Train=0.9214 | Val=0.6183
   ✓ 5-Fold CV: Mean=0.6601 ± 0.0250
3. SAVING MODELS AND METADATA
✓ CELL C (training) COMPLETE
```

(Exact numbers depend on the data; these are from the reference generator.)

### Cell D (evaluation)

```text
EVALUATION RESULTS (Test Set - Real PnL Prediction)

1. PRIME MODEL (Conservative - Ridge Regression)
   R² Score:                 0.6689
   RMSE (USD):               59.64
   MAE (USD):                44.40
   Profitability Precision:  80.00%
   Profitability Recall:     78.35%
   Direction Accuracy:       77.90%
   Rank Correlation:         0.8107

2. PHANTOM MODEL (Exploratory - Random Forest)
   R² Score:                 0.6260
   RMSE (USD):               63.38
   ...
✓ Saved evaluation results to .../checkpoints/evaluation_results.json
✓ CELL D (evaluation) COMPLETE
```

## File structure after a complete run

```text
/kaggle/working/megacompact_run_<RUN_ID>/
├── data/
│   ├── decision_packets.jsonl     # 1199 packets from Cell B
│   └── outcome_labels.json        # 3597 labels from Cell B
├── checkpoints/
│   ├── prime_model.pkl            # trained Ridge model
│   ├── phantom_model.pkl          # trained Random Forest model
│   ├── scaler.pkl                 # StandardScaler (required for inference)
│   ├── metadata.json              # training info from Cell C
│   └── evaluation_results.json    # test metrics from Cell D
├── run_config.json                # seed, repo commit, paths (Cell A)
└── telemetry.jsonl                # event log (all cells)
```

## Understanding the metrics

| Metric | What it measures | Target |
|--------|------------------|--------|
| R² | Variance in PnL explained by the model | 0.5–0.8 good, 0.8+ excellent |
| RMSE | Average prediction error in USD | Lower is better |
| MAE | Mean absolute deviation in USD | Lower is better |
| Profit precision | Of trades *predicted* profitable, fraction that actually were | 70%+ |
| Profit recall | Of *actually* profitable trades, fraction correctly identified | 70%+ |
| Direction accuracy | Did the prediction get the sign (win/lose) right | 65%+ |
| Rank correlation | Does the model order trades by profitability correctly | 0.5+ |

Example: 78% profit precision means "when we predict a trade is profitable,
it actually is 78% of the time".

## The 20 features (Cell C / Cell D)

| Group | Features |
|-------|----------|
| Temporal | `block_number`, `timestamp_ms` |
| Market microstructure | `spot_price`, `mid_price`, `bid_price`, `ask_price`, `spread_pct` |
| Volume | `base_volume_24h`, `quote_volume_24h` |
| Volatility | `volatility_pct` |
| Execution | `estimated_slippage_usd`, `estimated_gas_usd`, `estimated_include_prob` |
| Horizon | `horizon_blocks` |
| Action candidates | `num_candidates`, `max_trade_size`, `avg_expected_output`, `max_expected_output` |
| Constraints | `max_loss_usd`, `max_gas_usd` |

The split is **time-based** (sorted by `block_number`, 70/15/15), so there is
no future leakage between train, validation, and test.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `git clone` fails in Kaggle | Internet toggle is off | Settings → Internet → On |
| `❌ Cell A not run yet` | Cells run out of order | Run Cell A first |
| `❌ Cell B data not found` | Cell B did not complete | Run Cell B, check its output |
| `❌ Cell C models not found` | Cell C did not complete | Run Cell C, check its output |
| Low R² (< 0.4) | Features don't explain PnL | Add features (see below) |
| High RMSE | PnL variance very high | Filter outliers, add data |
| Low profit precision | Many false positives | Raise the prediction threshold |
| Low profit recall | Missing real winners | Lower the prediction threshold |

## Advanced: adding features

Edit `extract_features_from_packet()` in **both** Cell C and Cell D (they must
stay identical), e.g.:

```python
def extract_features_from_packet(packet):
    features = {}
    # ... existing features ...

    market = packet.get('market', {})

    # Technical indicators (if present in your packets)
    features['rsi_14'] = float(market.get('rsi_14', 50))
    features['macd'] = float(market.get('macd', 0))

    # Order-book imbalance
    bid_vol = float(market.get('bid_volume', 1))
    ask_vol = float(market.get('ask_volume', 1))
    features['bid_ask_ratio'] = bid_vol / max(ask_vol, 1e-9)

    # Time-of-day (cyclical)
    ts_ms = float(packet.get('as_of', {}).get('timestamp_ms', 0))
    features['hour_of_day'] = (ts_ms // (1000 * 60 * 60)) % 24
    ...
```

Then add the new names to `FEATURE_ORDER` and re-run Cells C and D.

## Advanced: hyperparameter tuning

In Cell C:

```python
# PRIME: more regularization for better generalization
prime_model = Ridge(alpha=10.0, solver='auto')

# PHANTOM: more/deeper trees
phantom_model = RandomForestRegressor(
    n_estimators=200,      # was 100
    max_depth=15,          # was 10
    min_samples_split=3,   # was 5
    min_samples_leaf=1,    # was 2
    random_state=PRIMARY_SEED,
    n_jobs=-1,
)
```

## Using a trained model for pre-trade filtering

```python
import pickle

with open("phantom_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# X_new: rows built with the SAME extract_features_from_packet()
predicted_pnl = model.predict(scaler.transform(X_new))

threshold = 50  # only trade if predicted PnL exceeds this
for i, pnl in enumerate(predicted_pnl):
    if pnl > threshold:
        execute(packets[i])
```

The scaler is **required** — new data must be normalized exactly like
training data.

## Local testing

The whole pipeline runs anywhere Python + numpy/pandas/scikit-learn are
installed:

```bash
pip install -r requirements.txt
python scripts/local_smoke_test.py
```

## What this project is NOT

This is a research and simulation pipeline. It does not connect to live
chains, exchanges, or wallets, and it should never be connected to real funds
without additional testing, monitoring, paper trading, and risk controls.
