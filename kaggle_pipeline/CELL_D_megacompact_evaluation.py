"""
CELL D: MODEL EVALUATION (MEGACOMPACT - PRODUCTION)
Evaluates trained models on the test set from real megacompact data.
Run after Cell A, B, and C complete successfully.

Loads the models trained by Cell C and reports:
  R², RMSE, MAE, Profit Precision/Recall, Direction Accuracy,
  Rank Correlation, and a PRIME vs PHANTOM recommendation.

Saves: RUN_DIR/checkpoints/evaluation_results.json
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("CELL D: MODEL EVALUATION (MEGACOMPACT)")
print("=" * 70)

# ===== Step 0: Verify all previous cells completed =====
try:
    assert PRIMARY_SEED is not None
    assert RUN_DIR.exists()
    assert telemetry is not None
    print("✓ Cell A variables available")
except NameError as e:
    print(f"❌ Cell A not run: {e}")
    exit(1)

# Check data exists
data_dir = RUN_DIR / "data"
packets_file = data_dir / "decision_packets.jsonl"
labels_file = data_dir / "outcome_labels.json"

if not packets_file.exists() or not labels_file.exists():
    print("❌ Cell B data not found")
    exit(1)
print("✓ Cell B data available")

# Check models exist
checkpoints_dir = RUN_DIR / "checkpoints"
if not (checkpoints_dir / "prime_model.pkl").exists():
    print(f"❌ Cell C models not found at {checkpoints_dir}")
    print("   Run Cell C first to train models")
    exit(1)
print("✓ Cell C models available")

# ===== Step 1: Load data =====
packets = []
with open(packets_file) as f:
    for line in f:
        if line.strip():
            packets.append(json.loads(line))

with open(labels_file) as f:
    labels_data = json.load(f)

print(f"✓ Loaded {len(packets)} packets, {len(labels_data)} outcome labels")

# ===== Step 2: Rebuild features (same as Cell C) =====
def extract_features_from_packet(packet: Dict[str, Any]) -> np.ndarray:
    """Extract feature vector from a megacompact DecisionPacket."""
    features = {}

    as_of = packet.get('as_of', {})
    features['block_number'] = float(as_of.get('block_number', 0))
    features['timestamp_ms'] = float(as_of.get('timestamp_ms', 0))

    market = packet.get('market', {})
    features['spot_price'] = float(market.get('spot_price_usd', 1.0))
    features['mid_price'] = float(market.get('mid_price_usd', 1.0))
    features['bid_price'] = float(market.get('bid_price_usd', 1.0))
    features['ask_price'] = float(market.get('ask_price_usd', 1.0))

    bid = float(market.get('bid_price_usd', 1.0))
    ask = float(market.get('ask_price_usd', 1.0))
    if ask > 0:
        features['spread_pct'] = (ask - bid) / ask * 100
    else:
        features['spread_pct'] = 0.0

    features['base_volume_24h'] = float(market.get('base_volume_24h', 0.0))
    features['quote_volume_24h'] = float(market.get('quote_volume_24h', 0.0))
    features['volatility_pct'] = float(market.get('volatility_pct', 0.0))

    execution = packet.get('execution', {})
    features['estimated_slippage_usd'] = float(execution.get('estimated_slippage_usd', 0.0))
    features['estimated_gas_usd'] = float(execution.get('estimated_gas_usd', 0.0))
    features['estimated_include_prob'] = float(execution.get('estimated_include_probability', 0.5))

    objective = packet.get('objective', {})
    features['horizon_blocks'] = float(objective.get('horizon_blocks', 12))

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

    constraints = packet.get('constraints', {})
    features['max_loss_usd'] = float(constraints.get('max_loss_usd', -1000.0))
    features['max_gas_usd'] = float(constraints.get('max_gas_usd', 100.0))

    feature_order = [
        'block_number', 'timestamp_ms', 'spot_price', 'mid_price',
        'bid_price', 'ask_price', 'spread_pct', 'base_volume_24h',
        'quote_volume_24h', 'volatility_pct', 'estimated_slippage_usd',
        'estimated_gas_usd', 'estimated_include_prob', 'horizon_blocks',
        'num_candidates', 'max_trade_size', 'avg_expected_output',
        'max_expected_output', 'max_loss_usd', 'max_gas_usd'
    ]

    return np.array([features.get(f, 0.0) for f in feature_order], dtype=np.float32)


# Build feature matrix
X = np.array([extract_features_from_packet(p) for p in packets], dtype=np.float32)

# Build target vector
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

# Recreate split (same as Cell C)
block_numbers = X[:, 0]
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

print(f"✓ Features rebuilt: Test set size = {len(X_test)}")

# ===== Step 3: Load scaler and transform test data =====
with open(checkpoints_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)
print("✓ Scaler loaded and applied to test data")

# ===== Step 4: Load trained models =====
with open(checkpoints_dir / "prime_model.pkl", "rb") as f:
    prime_model = pickle.load(f)

with open(checkpoints_dir / "phantom_model.pkl", "rb") as f:
    phantom_model = pickle.load(f)

print("✓ Models loaded: PRIME and PHANTOM")

# ===== Step 5: Evaluate on test set =====
def evaluate_model(model, X_test, y_test, name, seed):
    """Comprehensive evaluation on test set."""
    y_pred = model.predict(X_test)

    # Basic regression metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Profitability metrics
    pred_positive = (y_pred > 0).sum()
    actual_positive = (y_test > 0).sum()
    correct_positive = ((y_pred > 0) & (y_test > 0)).sum()

    # Precision and recall for profitable predictions
    pred_precision = correct_positive / max(pred_positive, 1)  # Of predicted profitable, how many were
    pred_recall = correct_positive / max(actual_positive, 1)   # Of actual profitable, how many we caught

    # Direction accuracy (did we predict the right sign?)
    direction_correct = (np.sign(y_pred) == np.sign(y_test)).sum()
    direction_accuracy = direction_correct / len(y_test)

    # PnL correlation
    avg_pred_pnl = y_pred.mean()
    avg_actual_pnl = y_test.mean()

    # Sorted rank correlation (Spearman-like)
    pred_ranks = np.argsort(y_pred).argsort()
    actual_ranks = np.argsort(y_test).argsort()
    rank_corr = np.corrcoef(pred_ranks, actual_ranks)[0, 1]
    if not np.isfinite(rank_corr):
        rank_corr = 0.0

    return {
        "model": name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "pred_positive_pct": (pred_positive / len(X_test)) * 100,
        "actual_positive_pct": (actual_positive / len(y_test)) * 100,
        "profit_precision": pred_precision,
        "profit_recall": pred_recall,
        "direction_accuracy": direction_accuracy,
        "rank_correlation": rank_corr,
        "avg_pred_pnl": avg_pred_pnl,
        "avg_actual_pnl": avg_actual_pnl,
        "n_samples": len(X_test)
    }


print()
print("=" * 70)
print("EVALUATION RESULTS (Test Set - Real PnL Prediction)")
print("=" * 70)

prime_results = evaluate_model(prime_model, X_test_scaled, y_test, "PRIME", PRIMARY_SEED)
phantom_results = evaluate_model(phantom_model, X_test_scaled, y_test, "PHANTOM", PRIMARY_SEED)

# ===== Step 6: Print results =====
print()
print("1. PRIME MODEL (Conservative - Ridge Regression)")
print(f"   R² Score:                 {prime_results['r2']:.4f}")
print(f"   RMSE (USD):               {prime_results['rmse']:.2f}")
print(f"   MAE (USD):                {prime_results['mae']:.2f}")
print(f"   Profitability Precision:  {prime_results['profit_precision']:.2%}")
print(f"   Profitability Recall:     {prime_results['profit_recall']:.2%}")
print(f"   Direction Accuracy:       {prime_results['direction_accuracy']:.2%}")
print(f"   Rank Correlation:         {prime_results['rank_correlation']:.4f}")
print(f"   Avg Predicted PnL:        ${prime_results['avg_pred_pnl']:.2f}")
print(f"   Avg Actual PnL:           ${prime_results['avg_actual_pnl']:.2f}")

print()
print("2. PHANTOM MODEL (Exploratory - Random Forest)")
print(f"   R² Score:                 {phantom_results['r2']:.4f}")
print(f"   RMSE (USD):               {phantom_results['rmse']:.2f}")
print(f"   MAE (USD):                {phantom_results['mae']:.2f}")
print(f"   Profitability Precision:  {phantom_results['profit_precision']:.2%}")
print(f"   Profitability Recall:     {phantom_results['profit_recall']:.2%}")
print(f"   Direction Accuracy:       {phantom_results['direction_accuracy']:.2%}")
print(f"   Rank Correlation:         {phantom_results['rank_correlation']:.4f}")
print(f"   Avg Predicted PnL:        ${phantom_results['avg_pred_pnl']:.2f}")
print(f"   Avg Actual PnL:           ${phantom_results['avg_actual_pnl']:.2f}")

# ===== Step 7: Comparative analysis =====
print()
print("=" * 70)
print("COMPARATIVE SUMMARY (Test Data)")
print("=" * 70)

comparison_data = {
    "Metric": [
        "R² Score", "RMSE (USD)", "MAE (USD)",
        "Profit Precision", "Profit Recall", "Direction Acc"
    ],
    "PRIME": [
        f"{prime_results['r2']:.4f}",
        f"{prime_results['rmse']:.2f}",
        f"{prime_results['mae']:.2f}",
        f"{prime_results['profit_precision']:.2%}",
        f"{prime_results['profit_recall']:.2%}",
        f"{prime_results['direction_accuracy']:.2%}"
    ],
    "PHANTOM": [
        f"{phantom_results['r2']:.4f}",
        f"{phantom_results['rmse']:.2f}",
        f"{phantom_results['mae']:.2f}",
        f"{phantom_results['profit_precision']:.2%}",
        f"{phantom_results['profit_recall']:.2%}",
        f"{phantom_results['direction_accuracy']:.2%}"
    ],
    "Winner": [
        "PHANTOM" if phantom_results['r2'] > prime_results['r2'] else "PRIME",
        "PHANTOM" if phantom_results['rmse'] < prime_results['rmse'] else "PRIME",
        "PHANTOM" if phantom_results['mae'] < prime_results['mae'] else "PRIME",
        "PHANTOM" if phantom_results['profit_precision'] > prime_results['profit_precision'] else "PRIME",
        "PHANTOM" if phantom_results['profit_recall'] > prime_results['profit_recall'] else "PRIME",
        "PHANTOM" if phantom_results['direction_accuracy'] > prime_results['direction_accuracy'] else "PRIME",
    ]
}

print()
print(f"{'Metric':<25} {'PRIME':<18} {'PHANTOM':<18} {'Winner':<10}")
print("-" * 71)
for i, metric in enumerate(comparison_data["Metric"]):
    print(f"{metric:<25} {comparison_data['PRIME'][i]:<18} {comparison_data['PHANTOM'][i]:<18} {comparison_data['Winner'][i]:<10}")

# ===== Step 8: Recommendation =====
print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Multi-criteria recommendation
scores = {
    "PRIME": sum([1 for w in comparison_data["Winner"] if w == "PRIME"]),
    "PHANTOM": sum([1 for w in comparison_data["Winner"] if w == "PHANTOM"])
}

if scores["PHANTOM"] >= scores["PRIME"]:
    print()
    print("✓ PHANTOM MODEL RECOMMENDED")
    print(f"  Wins on {scores['PHANTOM']} out of 6 metrics")
    print(f"  Especially strong on:")
    if phantom_results['r2'] > prime_results['r2']:
        print(f"    - R² ({phantom_results['r2']:.4f} vs {prime_results['r2']:.4f})")
    if phantom_results['profit_precision'] > prime_results['profit_precision']:
        print(f"    - Profit Precision ({phantom_results['profit_precision']:.2%} vs {prime_results['profit_precision']:.2%})")
    if phantom_results['direction_accuracy'] > prime_results['direction_accuracy']:
        print(f"    - Direction Accuracy ({phantom_results['direction_accuracy']:.2%} vs {prime_results['direction_accuracy']:.2%})")
else:
    print()
    print("✓ PRIME MODEL RECOMMENDED")
    print(f"  Wins on {scores['PRIME']} out of 6 metrics")
    print(f"  Conservative approach with stable performance")

# ===== Step 9: Save evaluation results =====
eval_results = {
    "timestamp": str(RUN_DIR.name),
    "seed": PRIMARY_SEED,
    "test_set_size": len(X_test),
    "y_stats": {
        "mean": float(y_test.mean()),
        "std": float(y_test.std()),
        "min": float(y_test.min()),
        "max": float(y_test.max()),
        "positive_pct": float((y_test > 0).sum() / len(y_test) * 100)
    },
    "prime_model": {
        "type": "Ridge",
        "test_r2": float(prime_results['r2']),
        "test_rmse": float(prime_results['rmse']),
        "test_mae": float(prime_results['mae']),
        "profit_precision": float(prime_results['profit_precision']),
        "profit_recall": float(prime_results['profit_recall']),
        "direction_accuracy": float(prime_results['direction_accuracy']),
        "rank_correlation": float(prime_results['rank_correlation']),
        "avg_pred_pnl": float(prime_results['avg_pred_pnl']),
    },
    "phantom_model": {
        "type": "RandomForestRegressor",
        "test_r2": float(phantom_results['r2']),
        "test_rmse": float(phantom_results['rmse']),
        "test_mae": float(phantom_results['mae']),
        "profit_precision": float(phantom_results['profit_precision']),
        "profit_recall": float(phantom_results['profit_recall']),
        "direction_accuracy": float(phantom_results['direction_accuracy']),
        "rank_correlation": float(phantom_results['rank_correlation']),
        "avg_pred_pnl": float(phantom_results['avg_pred_pnl']),
    },
    "winner": "PHANTOM" if scores["PHANTOM"] >= scores["PRIME"] else "PRIME",
    "winner_score": max(scores["PHANTOM"], scores["PRIME"])
}

results_file = checkpoints_dir / "evaluation_results.json"
with open(results_file, "w") as f:
    json.dump(eval_results, f, indent=2)
print()
print(f"✓ Saved evaluation results to {results_file}")

# Log to telemetry
telemetry.write("CELL_D_EVALUATION_COMPLETE",
                prime_r2=float(prime_results['r2']),
                phantom_r2=float(phantom_results['r2']),
                winner="PHANTOM" if scores["PHANTOM"] >= scores["PRIME"] else "PRIME",
                prime_profit_precision=float(prime_results['profit_precision']),
                phantom_profit_precision=float(phantom_results['profit_precision']))

print()
print("=" * 70)
print("✓ CELL D (evaluation) COMPLETE")
print("=" * 70)
print()
print(f"Artifacts saved to: {checkpoints_dir}/")
print("Download from /kaggle/working/ output tab when done")
