"""
CELL C: MODEL TRAINING (MEGACOMPACT - PRODUCTION)
Trains PRIME (Ridge) and PHANTOM (RandomForest) to predict net PnL from the
REAL MegaCompact16 DecisionPacket / OutcomeLabel schema produced by Cell B.

Run after Cell A (setup) and Cell B (data generation).

Inputs (written by Cell B):
  RUN_DIR/data/decision_packets.jsonl   one DecisionPacket (model_dump) per line
  RUN_DIR/data/outcome_labels.json      {decision_id: [OutcomeLabel, ...]}

Features (20) are extracted from the real packet structure:
  as_of             -> block_number, decision_timestamp_ms
  objective         -> horizon_blocks, capital_usd
  market            -> price_* aggregates (first/mean/std/min/max)
  execution         -> base_fee_gwei, total_liquidity
  action_candidates -> count + trade-size / slippage / deadline aggregates
  constraints       -> max_price_impact_bps, max_gas_usd, max_loss_usd

Target: mean net_pnl_usd across the candidate outcomes of each decision.
Split: time-based on block_number (no future leakage).
"""

import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

print("=" * 70)
print("CELL C: MODEL TRAINING (MEGACOMPACT)")
print("=" * 70)

# ===== Step 0: Verify Cell A & B completed =====
try:
    assert PRIMARY_SEED is not None
    assert RUN_DIR.exists()
    assert telemetry is not None
    print(f"✓ Cell A variables available (seed={PRIMARY_SEED}, dir={RUN_DIR.name})")
except NameError as e:
    print(f"❌ Cell A not run yet: {e}")
    exit(1)

# ===== Step 1: Load data from Cell B =====
data_dir = RUN_DIR / "data"
packets_file = data_dir / "decision_packets.jsonl"
labels_file = data_dir / "outcome_labels.json"

if not packets_file.exists() or not labels_file.exists():
    print(f"❌ Cell B data not found at {data_dir}")
    print(" Run Cell B first to generate data")
    exit(1)

# Load packets (JSONL format - one packet per line)
packets = []
with open(packets_file) as f:
    for line in f:
        if line.strip():
            packets.append(json.loads(line))

# Load labels (JSON format - dict of decision_id -> [outcomes])
with open(labels_file) as f:
    labels_data = json.load(f)

print(f"✓ Loaded {len(packets)} packets")
print(f"✓ Loaded {len(labels_data)} decisions with outcome labels")

# ===== Step 2: Extract features from real megacompact DecisionPackets =====
"""
Real DecisionPacket structure (from megacompact_core/core.py):

  as_of             {chain_id, block_number, block_hash, decision_timestamp_ms, max_feature_age_ms}
  objective         {horizon_blocks, capital_usd, min_net_edge_usd, max_loss_usd, risk_aversion, confidence_level}
  market            {price_token_0, price_token_1, ...}          (one per token)
  execution         {base_fee_gwei, total_liquidity}
  graph             {nodes, edges}
  action_candidates [{action_id, route, trade_size_usd, borrow, slippage_limit_bps, gas_policy, deadline_block, private_submission}]
  constraints       {max_price_impact_bps, max_gas_usd, gas_uncertainty_buffer_usd, max_quote_age_ms, max_loss_usd}
  provenance        {feature_versions, feature_source_timestamps_ms, ...}
"""

FEATURE_ORDER = [
    # as_of
    'block_number', 'decision_timestamp_ms',
    # objective
    'horizon_blocks', 'capital_usd',
    # market (aggregates over all token prices -> fixed dimensionality)
    'price_first', 'price_mean', 'price_std', 'price_min', 'price_max',
    # execution
    'base_fee_gwei', 'total_liquidity',
    # action candidates
    'num_candidates', 'max_trade_size_usd', 'mean_trade_size_usd',
    'min_slippage_limit_bps', 'max_slippage_limit_bps', 'min_deadline_block',
    # constraints
    'max_price_impact_bps', 'max_gas_usd', 'max_loss_usd',
]


def extract_features_from_packet(packet: Dict[str, Any]) -> np.ndarray:
    """Extract a fixed-length feature vector from a real DecisionPacket dict."""
    f: Dict[str, float] = {}

    as_of = packet.get('as_of', {})
    f['block_number'] = float(as_of.get('block_number', 0))
    f['decision_timestamp_ms'] = float(as_of.get('decision_timestamp_ms', 0))

    objective = packet.get('objective', {})
    f['horizon_blocks'] = float(objective.get('horizon_blocks', 0))
    f['capital_usd'] = float(objective.get('capital_usd', 0))

    market = packet.get('market', {})
    prices = [float(v) for k, v in market.items()
              if k.startswith('price_') and v is not None]
    if prices:
        arr = np.asarray(prices, dtype=np.float32)
        f['price_first'] = float(arr[0])
        f['price_mean'] = float(arr.mean())
        f['price_std'] = float(arr.std())
        f['price_min'] = float(arr.min())
        f['price_max'] = float(arr.max())
    else:
        f['price_first'] = f['price_mean'] = f['price_std'] = 0.0
        f['price_min'] = f['price_max'] = 0.0

    execution = packet.get('execution', {})
    f['base_fee_gwei'] = float(execution.get('base_fee_gwei', 0))
    f['total_liquidity'] = float(execution.get('total_liquidity', 0))

    candidates = packet.get('action_candidates', [])
    if candidates:
        sizes = [float(c.get('trade_size_usd', 0.0)) for c in candidates]
        slips = [float(c.get('slippage_limit_bps', 0.0)) for c in candidates]
        deadlines = [float(c.get('deadline_block', 0.0)) for c in candidates]
        f['num_candidates'] = float(len(candidates))
        f['max_trade_size_usd'] = float(max(sizes))
        f['mean_trade_size_usd'] = float(np.mean(sizes))
        f['min_slippage_limit_bps'] = float(min(slips))
        f['max_slippage_limit_bps'] = float(max(slips))
        f['min_deadline_block'] = float(min(deadlines))
    else:
        f['num_candidates'] = 0.0
        f['max_trade_size_usd'] = 0.0
        f['mean_trade_size_usd'] = 0.0
        f['min_slippage_limit_bps'] = 0.0
        f['max_slippage_limit_bps'] = 0.0
        f['min_deadline_block'] = 0.0

    constraints = packet.get('constraints', {})
    f['max_price_impact_bps'] = float(constraints.get('max_price_impact_bps', 0))
    f['max_gas_usd'] = float(constraints.get('max_gas_usd', 0))
    f['max_loss_usd'] = float(constraints.get('max_loss_usd', 0))

    return np.array([f.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32)


# Build feature matrix
try:
    X = np.array([extract_features_from_packet(p) for p in packets], dtype=np.float32)
    print(f"✓ Features extracted: shape={X.shape}")

    print(f"  Features: {', '.join(FEATURE_ORDER[:5])}... ({len(FEATURE_ORDER)} total)")

    if np.isnan(X).any():
        print("⚠ Warning: NaN values in features, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isinf(X).any():
        print("⚠ Warning: Infinite values in features, clamping")
        X = np.clip(X, -1e9, 1e9)

    print(f"  Value range: [{X.min():.4f}, {X.max():.4f}]")
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 3: Build target vector from real OutcomeLabel =====
"""
Each decision has one OutcomeLabel per action candidate. The target for a
decision is the mean net_pnl_usd across its candidate outcomes (the same
aggregation used throughout the reference pipeline).
"""
try:
    decision_pnl = {}
    if isinstance(labels_data, dict):
        for decision_id, outcomes in labels_data.items():
            if isinstance(outcomes, list):
                pnls = [float(o.get('net_pnl_usd', 0.0)) for o in outcomes]
            else:
                pnls = [float(outcomes.get('net_pnl_usd', 0.0))]
            if pnls:
                decision_pnl[decision_id] = np.mean(pnls)

    y = []
    for packet in packets:
        decision_id = packet.get('decision_id')
        pnl = decision_pnl.get(decision_id, 0.0)
        y.append(pnl)

    y = np.array(y, dtype=np.float32)
    print(f"✓ Targets built: shape={y.shape}")
    print(f"  Range: [{y.min():.2f}, {y.max():.2f}] USD")
    print(f"  Mean: {y.mean():.2f} USD, Std: {y.std():.2f} USD")
    print(f"  Profitability: {(y > 0).sum()}/{len(y)} positive outcomes")
except Exception as e:
    print(f"❌ Target building failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 4: Train/val/test split (time-based) =====
try:
    block_numbers = X[:, 0]  # First feature is block_number
    sort_indices = np.argsort(block_numbers)

    X = X[sort_indices]
    y = y[sort_indices]

    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"✓ Time-based split (by block_number):")
    print(f"  Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples | Test: {X_test.shape[0]} samples")
except Exception as e:
    print(f"❌ Data split failed: {e}")
    exit(1)

# ===== Step 5: Normalize features =====
try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"✓ Scaler fitted on training data")
except Exception as e:
    print(f"❌ Scaler fitting failed: {e}")
    exit(1)

# ===== Step 6: Train PRIME Model (Conservative, Ridge) =====
print("\n1. TRAINING PRIME MODEL (Ridge Regression, α=1.0)")
print(" → Conservative, stable, interpretable")

try:
    prime_model = Ridge(alpha=1.0, solver='auto', random_state=PRIMARY_SEED)
    prime_model.fit(X_train_scaled, y_train)

    prime_train_r2 = prime_model.score(X_train_scaled, y_train)
    prime_val_r2 = prime_model.score(X_val_scaled, y_val)

    print(f"   ✓ PRIME R²: Train={prime_train_r2:.4f} | Val={prime_val_r2:.4f}")

    cv_scores = cross_val_score(prime_model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"   ✓ 5-Fold CV: Mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
except Exception as e:
    print(f" ❌ PRIME training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 7: Train PHANTOM Model (Exploratory, Random Forest) =====
print("\n2. TRAINING PHANTOM MODEL (Random Forest, n_estimators=100)")
print(" → Exploratory, captures non-linearity, higher variance")

try:
    phantom_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=PRIMARY_SEED,
        n_jobs=-1
    )
    phantom_model.fit(X_train_scaled, y_train)

    phantom_train_r2 = phantom_model.score(X_train_scaled, y_train)
    phantom_val_r2 = phantom_model.score(X_val_scaled, y_val)

    print(f"   ✓ PHANTOM R²: Train={phantom_train_r2:.4f} | Val={phantom_val_r2:.4f}")

    cv_scores = cross_val_score(phantom_model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"   ✓ 5-Fold CV: Mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
except Exception as e:
    print(f" ❌ PHANTOM training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 8: Save models and metadata =====
print("\n3. SAVING MODELS AND METADATA")

try:
    checkpoints_dir = RUN_DIR / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with open(checkpoints_dir / "prime_model.pkl", "wb") as f:
        pickle.dump(prime_model, f)
    print(f"   ✓ Saved {checkpoints_dir / 'prime_model.pkl'}")

    with open(checkpoints_dir / "phantom_model.pkl", "wb") as f:
        pickle.dump(phantom_model, f)
    print(f"   ✓ Saved {checkpoints_dir / 'phantom_model.pkl'}")

    with open(checkpoints_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Saved {checkpoints_dir / 'scaler.pkl'}")

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": PRIMARY_SEED,
        "n_features": X.shape[1],
        "feature_names": FEATURE_ORDER,
        "n_samples": len(X),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "y_range": [float(y.min()), float(y.max())],
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "positive_outcomes_pct": float((y > 0).sum() / len(y) * 100),
        "prime_model": {
            "type": "Ridge",
            "alpha": 1.0,
            "train_r2": float(prime_train_r2),
            "val_r2": float(prime_val_r2),
        },
        "phantom_model": {
            "type": "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": 10,
            "train_r2": float(phantom_train_r2),
            "val_r2": float(phantom_val_r2),
        }
    }

    with open(checkpoints_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved {checkpoints_dir / 'metadata.json'}")

    telemetry.write("CELL_C_TRAINING_COMPLETE",
                    prime_val_r2=float(prime_val_r2),
                    phantom_val_r2=float(phantom_val_r2),
                    n_features=X.shape[1],
                    n_samples=len(X),
                    y_range=[float(y.min()), float(y.max())])
except Exception as e:
    print(f" ❌ Save failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✓ CELL C (training) COMPLETE")
print("=" * 70)
print(f"Next: Run CELL D (evaluation) to compare models on test data")
