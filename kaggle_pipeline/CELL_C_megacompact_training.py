"""
CELL C: MODEL TRAINING (MEGACOMPACT - PRODUCTION)
Works with REAL DecisionPacket and OutcomeLabel format from megacompact.
Run after Cell A (setup) and Cell B (data generation).

Trains two competing models to predict net_pnl_usd from decision packets:
  PRIME   -> Ridge Regression       (conservative, stable, interpretable)
  PHANTOM -> Random Forest          (exploratory, captures non-linearity)

Saves to RUN_DIR/checkpoints/:
  prime_model.pkl, phantom_model.pkl, scaler.pkl, metadata.json
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
    print("✓ Cell A variables available (seed={}, dir={})".format(PRIMARY_SEED, RUN_DIR.name))
except NameError as e:
    print(f"❌ Cell A not run yet: {e}")
    exit(1)

# ===== Step 1: Load data from Cell B =====
data_dir = RUN_DIR / "data"
packets_file = data_dir / "decision_packets.jsonl"
labels_file = data_dir / "outcome_labels.json"

if not packets_file.exists() or not labels_file.exists():
    print(f"❌ Cell B data not found at {data_dir}")
    print("   Run Cell B first to generate data")
    exit(1)

# Load packets (JSONL format - one packet per line)
packets = []
with open(packets_file) as f:
    for line in f:
        if line.strip():
            packets.append(json.loads(line))

# Load labels (JSON format - dict of decision_id -> outcomes)
with open(labels_file) as f:
    labels_data = json.load(f)

print(f"✓ Loaded {len(packets)} packets")
print(f"✓ Loaded {len(labels_data)} outcome labels")

# ===== Step 2: Extract features from megacompact DecisionPackets =====
"""
Features extracted from DecisionPacket structure:

as_of:
    block_number:      Current block
    timestamp_ms:      Current timestamp
market:
    spot_price_usd:    Current spot price
    mid_price_usd:     Mid price
    bid_price_usd:     Bid price
    ask_price_usd:     Ask price
    base_volume_24h:   Volume in past 24h
    quote_volume_24h:  Quote volume
    volatility_pct:    Historical volatility
execution:
    estimated_slippage_usd:        Slippage estimate
    estimated_gas_usd:             Gas estimate
    estimated_include_probability: Probability of inclusion
objective:
    horizon_blocks:    Decision time horizon
action_candidates:
    trade_size_usd:        Size of trades
    expected_output_usd:   Expected PnL
"""

FEATURE_ORDER = [
    'block_number', 'timestamp_ms', 'spot_price', 'mid_price',
    'bid_price', 'ask_price', 'spread_pct', 'base_volume_24h',
    'quote_volume_24h', 'volatility_pct', 'estimated_slippage_usd',
    'estimated_gas_usd', 'estimated_include_prob', 'horizon_blocks',
    'num_candidates', 'max_trade_size', 'avg_expected_output',
    'max_expected_output', 'max_loss_usd', 'max_gas_usd'
]
feature_names = FEATURE_ORDER


def extract_features_from_packet(packet: Dict[str, Any]) -> np.ndarray:
    """Extract feature vector from a megacompact DecisionPacket."""
    features = {}

    # === as_of features ===
    as_of = packet.get('as_of', {})
    features['block_number'] = float(as_of.get('block_number', 0))
    features['timestamp_ms'] = float(as_of.get('timestamp_ms', 0))

    # === market features ===
    market = packet.get('market', {})
    features['spot_price'] = float(market.get('spot_price_usd', 1.0))
    features['mid_price'] = float(market.get('mid_price_usd', 1.0))
    features['bid_price'] = float(market.get('bid_price_usd', 1.0))
    features['ask_price'] = float(market.get('ask_price_usd', 1.0))

    # Spread as percentage
    bid = float(market.get('bid_price_usd', 1.0))
    ask = float(market.get('ask_price_usd', 1.0))
    if ask > 0:
        features['spread_pct'] = (ask - bid) / ask * 100
    else:
        features['spread_pct'] = 0.0

    # Volume features (24h)
    features['base_volume_24h'] = float(market.get('base_volume_24h', 0.0))
    features['quote_volume_24h'] = float(market.get('quote_volume_24h', 0.0))

    # Volatility
    features['volatility_pct'] = float(market.get('volatility_pct', 0.0))

    # === execution features ===
    execution = packet.get('execution', {})
    features['estimated_slippage_usd'] = float(execution.get('estimated_slippage_usd', 0.0))
    features['estimated_gas_usd'] = float(execution.get('estimated_gas_usd', 0.0))
    features['estimated_include_prob'] = float(execution.get('estimated_include_probability', 0.5))

    # === objective features ===
    objective = packet.get('objective', {})
    features['horizon_blocks'] = float(objective.get('horizon_blocks', 12))

    # === action candidates features (aggregate statistics) ===
    candidates = packet.get('action_candidates', [])
    if candidates:
        trade_sizes = [float(c.get('trade_size_usd', 0.0)) for c in candidates]
        expected_outputs = [float(c.get('expected_output_usd', 0.0)) for c in candidates]

        features['num_candidates'] = float(len(candidates))
        features['max_trade_size'] = float(max(trade_sizes)) if trade_sizes else 0.0
        features['avg_expected_output'] = float(np.mean(expected_outputs)) if expected_outputs else 0.0
        features['max_expected_output'] = float(max(expected_outputs)) if expected_outputs else 0.0
    else:
        features['num_candidates'] = 0.0
        features['max_trade_size'] = 0.0
        features['avg_expected_output'] = 0.0
        features['max_expected_output'] = 0.0

    # === constraints features ===
    constraints = packet.get('constraints', {})
    features['max_loss_usd'] = float(constraints.get('max_loss_usd', -1000.0))
    features['max_gas_usd'] = float(constraints.get('max_gas_usd', 100.0))

    # Return as ordered numpy array
    return np.array([features.get(f, 0.0) for f in FEATURE_ORDER], dtype=np.float32)


# Build feature matrix
try:
    X = np.array([extract_features_from_packet(p) for p in packets], dtype=np.float32)
    print(f"✓ Features extracted: shape={X.shape}")
    print(f"  Features: {', '.join(feature_names[:5])}... (20 total)")

    # Validate
    if np.isnan(X).any():
        print("⚠ Warning: NaN values in features, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isinf(X).any():
        print("⚠ Warning: Infinite values in features, clamping")
        X = np.clip(X, -1e6, 1e6)

    print(f"  Value range: [{X.min():.4f}, {X.max():.4f}]")
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 3: Build target vector from OutcomeLabel =====
"""
Target is net_pnl_usd from OutcomeLabel for each decision.
We use the average net PnL across all outcomes for each decision.
"""

try:
    # Group outcomes by decision_id and compute average net PnL
    decision_pnl = {}

    if isinstance(labels_data, dict):
        # Format: {decision_id: [outcomes]}
        for decision_id, outcomes in labels_data.items():
            if isinstance(outcomes, list):
                pnls = [float(o.get('net_pnl_usd', 0.0)) for o in outcomes]
            else:
                # Single outcome
                pnls = [float(outcomes.get('net_pnl_usd', 0.0))]

            if pnls:
                decision_pnl[decision_id] = np.mean(pnls)

    # Build y array aligned with packets
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
"""
Time-based split using block_number to ensure no future leakage.
"""
try:
    # Extract block numbers for sorting
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

    print("✓ Time-based split (by block_number):")
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

    print("✓ Scaler fitted on training data")
    print(f"  Mean (train): {X_train_scaled.mean(axis=0)[:3]}... (first 3)")
    print(f"  Std (train):  {X_train_scaled.std(axis=0)[:3]}... (first 3)")
except Exception as e:
    print(f"❌ Scaler fitting failed: {e}")
    exit(1)

# ===== Step 6: Train PRIME Model (Conservative, Ridge) =====
print()
print("1. TRAINING PRIME MODEL (Ridge Regression, alpha=1.0)")
print("   → Conservative, stable, interpretable")

try:
    np.random.seed(PRIMARY_SEED)
    prime_model = Ridge(alpha=1.0, solver='auto', random_state=PRIMARY_SEED)
    prime_model.fit(X_train_scaled, y_train)

    prime_train_r2 = prime_model.score(X_train_scaled, y_train)
    prime_val_r2 = prime_model.score(X_val_scaled, y_val)

    print(f"   ✓ PRIME R²: Train={prime_train_r2:.4f} | Val={prime_val_r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(prime_model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"   ✓ 5-Fold CV: Mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
except Exception as e:
    print(f"   ❌ PRIME training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 7: Train PHANTOM Model (Exploratory, Random Forest) =====
print()
print("2. TRAINING PHANTOM MODEL (Random Forest, n_estimators=100)")
print("   → Exploratory, captures non-linearity, higher variance")

try:
    phantom_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=PRIMARY_SEED,
        n_jobs=-1  # Use all CPU cores
    )
    phantom_model.fit(X_train_scaled, y_train)

    phantom_train_r2 = phantom_model.score(X_train_scaled, y_train)
    phantom_val_r2 = phantom_model.score(X_val_scaled, y_val)

    print(f"   ✓ PHANTOM R²: Train={phantom_train_r2:.4f} | Val={phantom_val_r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(phantom_model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"   ✓ 5-Fold CV: Mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
except Exception as e:
    print(f"   ❌ PHANTOM training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 8: Save models and metadata =====
print()
print("3. SAVING MODELS AND METADATA")

try:
    checkpoints_dir = RUN_DIR / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    with open(checkpoints_dir / "prime_model.pkl", "wb") as f:
        pickle.dump(prime_model, f)
    print(f"   ✓ Saved {checkpoints_dir / 'prime_model.pkl'}")

    with open(checkpoints_dir / "phantom_model.pkl", "wb") as f:
        pickle.dump(phantom_model, f)
    print(f"   ✓ Saved {checkpoints_dir / 'phantom_model.pkl'}")

    # Save scaler
    with open(checkpoints_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Saved {checkpoints_dir / 'scaler.pkl'}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": PRIMARY_SEED,
        "n_features": X.shape[1],
        "feature_names": feature_names,
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

    # Log to telemetry
    telemetry.write("CELL_C_TRAINING_COMPLETE",
                    prime_val_r2=float(prime_val_r2),
                    phantom_val_r2=float(phantom_val_r2),
                    n_features=X.shape[1],
                    n_samples=len(X),
                    y_range=[float(y.min()), float(y.max())])
except Exception as e:
    print(f"   ❌ Save failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("=" * 70)
print("✓ CELL C (training) COMPLETE")
print("=" * 70)
print("Next: Run CELL D (evaluation) to compare models on test data")
