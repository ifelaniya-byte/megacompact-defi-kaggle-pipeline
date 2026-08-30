"""
DATA INSPECTION SCRIPT (MEGACOMPACT)
Run this AFTER Cell B to understand your decision packets and labels.
Helps identify available features and data quality issues.

Optional but recommended before Cell C.
"""

import json
from pathlib import Path
from typing import Dict, Any

print("=" * 70)
print("MEGACOMPACT DATA INSPECTION")
print("=" * 70)

# ===== Step 1: Locate data files =====
try:
    RUN_DIR  # noqa: B018 - defined by Cell A
except NameError:
    print("❌ RUN_DIR not defined — run Cell A (and Cell B) first.")
    exit(1)

try:
    data_dir = RUN_DIR / "data"
    packets_file = data_dir / "decision_packets.jsonl"
    labels_file = data_dir / "outcome_labels.json"

    assert packets_file.exists(), f"Packets not found at {packets_file}"
    assert labels_file.exists(), f"Labels not found at {labels_file}"

    print(f"✓ Found packets: {packets_file}")
    print(f"✓ Found labels: {labels_file}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Make sure Cell B completed successfully")
    exit(1)

# ===== Step 2: Load and inspect packets =====
print()
print("=" * 70)
print("DECISION PACKETS")
print("=" * 70)

packets = []
try:
    with open(packets_file) as f:
        for i, line in enumerate(f):
            if line.strip():
                packets.append(json.loads(line))
    print(f"✓ Loaded {len(packets)} packets")
except Exception as e:
    print(f"❌ Error loading packets: {e}")
    exit(1)

if packets:
    # Inspect first packet
    first_packet = packets[0]
    print(f"\nFIRST PACKET (decision_id: {first_packet.get('decision_id')})")
    print(f"  Top-level keys: {list(first_packet.keys())}")

    # Inspect each main section
    for section in ['as_of', 'market', 'execution', 'objective', 'constraints']:
        if section in first_packet:
            data = first_packet[section]
            print(f"\n  [{section}]")
            if isinstance(data, dict):
                for key, value in list(data.items())[:5]:  # Show first 5
                    print(f"    {key}: {type(value).__name__} = {value}")
                if len(data) > 5:
                    print(f"    ... and {len(data) - 5} more keys")
            else:
                print(f"    {type(data).__name__} with {len(data)} items")

    # Action candidates
    if 'action_candidates' in first_packet:
        candidates = first_packet['action_candidates']
        print(f"\n  [action_candidates] ({len(candidates)} candidates)")
        if candidates:
            for key, value in list(candidates[0].items())[:5]:
                print(f"    {key}: {type(value).__name__} = {value}")
            if len(candidates[0]) > 5:
                print(f"    ... and {len(candidates[0]) - 5} more keys")

# ===== Step 3: Analyze all packets for available features =====
print()
print("=" * 70)
print("FEATURE AVAILABILITY ANALYSIS")
print("=" * 70)

# Collect all keys across all packets
all_keys = {
    'as_of': set(),
    'market': set(),
    'execution': set(),
    'objective': set(),
    'constraints': set()
}

for packet in packets:
    for section in all_keys.keys():
        if section in packet and isinstance(packet[section], dict):
            all_keys[section].update(packet[section].keys())

for section, keys in all_keys.items():
    print(f"\n[{section}] ({len(keys)} fields)")
    for key in sorted(keys):
        # Sample value from first packet with this key
        sample_value = None
        for packet in packets:
            if section in packet and key in packet[section]:
                sample_value = packet[section][key]
                break

        value_type = type(sample_value).__name__
        print(f"  • {key}: {value_type} (example: {sample_value})")

# ===== Step 4: Load and inspect labels =====
print()
print("=" * 70)
print("OUTCOME LABELS")
print("=" * 70)

labels_data = None
try:
    with open(labels_file) as f:
        labels_data = json.load(f)

    if isinstance(labels_data, dict):
        print(f"✓ Loaded {len(labels_data)} decision outcomes")

        # Sample a decision_id
        sample_decision_id = list(labels_data.keys())[0]
        sample_outcomes = labels_data[sample_decision_id]

        print(f"\nFIRST DECISION OUTCOMES (decision_id: {sample_decision_id})")
        print(f"  Type: {type(sample_outcomes).__name__}")

        if isinstance(sample_outcomes, list):
            print(f"  Number of outcomes: {len(sample_outcomes)}")
            if sample_outcomes:
                first_outcome = sample_outcomes[0]
                print(f"  First outcome keys: {list(first_outcome.keys())}")
                for key, value in list(first_outcome.items())[:8]:
                    print(f"    {key}: {type(value).__name__} = {value}")
        else:
            print(f"  Single outcome with keys: {list(sample_outcomes.keys())}")
            for key, value in list(sample_outcomes.items())[:8]:
                print(f"    {key}: {type(value).__name__} = {value}")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 5: Label statistics =====
print()
print("=" * 70)
print("OUTCOME LABEL STATISTICS")
print("=" * 70)

all_pnls = []
if isinstance(labels_data, dict):
    # Analyze PnL distribution
    for decision_id, outcomes in labels_data.items():
        if isinstance(outcomes, list):
            for outcome in outcomes:
                pnl = float(outcome.get('net_pnl_usd', 0.0))
                all_pnls.append(pnl)
        else:
            pnl = float(outcomes.get('net_pnl_usd', 0.0))
            all_pnls.append(pnl)

if all_pnls:
    import numpy as np
    all_pnls = np.array(all_pnls)

    print("net_pnl_usd distribution:")
    print(f"  Count: {len(all_pnls)}")
    print(f"  Min: ${all_pnls.min():.2f}")
    print(f"  Q25: ${np.percentile(all_pnls, 25):.2f}")
    print(f"  Median: ${np.median(all_pnls):.2f}")
    print(f"  Q75: ${np.percentile(all_pnls, 75):.2f}")
    print(f"  Max: ${all_pnls.max():.2f}")
    print(f"  Mean: ${all_pnls.mean():.2f}")
    print(f"  Std: ${all_pnls.std():.2f}")
    print(f"  Profitable (PnL > 0): {(all_pnls > 0).sum()} ({(all_pnls > 0).sum() / len(all_pnls) * 100:.1f}%)")
    print(f"  Unprofitable (PnL < 0): {(all_pnls < 0).sum()} ({(all_pnls < 0).sum() / len(all_pnls) * 100:.1f}%)")
    print(f"  Breakeven (PnL ≈ 0): {(np.abs(all_pnls) < 1).sum()}")

# ===== Step 6: Data quality checks =====
print()
print("=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

issues = []

# Check packet count
if len(packets) < 100:
    issues.append(f"⚠ Low packet count: {len(packets)} (expected 500+)")
else:
    print(f"✓ Packet count: {len(packets)}")

# Check for NaN/None values
nan_count = 0
for packet in packets[:50]:  # Sample first 50
    for section in ['as_of', 'market', 'execution']:
        if section in packet:
            for key, value in packet[section].items():
                if value is None:
                    nan_count += 1

if nan_count > 10:
    issues.append("⚠ Many None/NaN values detected")
else:
    print("✓ No significant None/NaN values in sample")

# Check label coverage
decision_ids_in_packets = {p.get('decision_id') for p in packets}
decision_ids_in_labels = set(labels_data.keys()) if isinstance(labels_data, dict) else set()
if decision_ids_in_packets:
    coverage = len(decision_ids_in_labels & decision_ids_in_packets) / len(decision_ids_in_packets)
else:
    coverage = 0.0

if coverage < 0.8:
    issues.append(f"⚠ Low label coverage: {coverage * 100:.1f}% (expected 95%+)")
else:
    print(f"✓ Label coverage: {coverage * 100:.1f}%")

# Check for duplicate decision_ids
unique_decision_ids = len({p.get('decision_id') for p in packets})
if unique_decision_ids < len(packets):
    issues.append("⚠ Duplicate decision_ids detected")
else:
    print("✓ All decision_ids unique")

# ===== Step 7: Feature suitability assessment =====
print()
print("=" * 70)
print("FEATURE ENGINEERING RECOMMENDATIONS")
print("=" * 70)

recommendations = []

# Price features
if 'spot_price_usd' in all_keys['market']:
    recommendations.append("✓ Price data available → Use price features")
else:
    recommendations.append("✗ Price data missing → Consider adding")

# Volume features
if 'base_volume_24h' in all_keys['market'] or 'quote_volume_24h' in all_keys['market']:
    recommendations.append("✓ Volume data available → Use volume features")
else:
    recommendations.append("✗ Volume data missing → Consider adding")

# Volatility
if 'volatility_pct' in all_keys['market'] or 'volatility' in all_keys['market']:
    recommendations.append("✓ Volatility data available → Use volatility features")
else:
    recommendations.append("✗ Volatility data missing → Consider adding")

# Execution metrics
if 'estimated_gas_usd' in all_keys['execution'] or 'estimated_slippage_usd' in all_keys['execution']:
    recommendations.append("✓ Execution metrics available → Critical for PnL prediction")
else:
    recommendations.append("⚠ Limited execution metrics → May limit accuracy")

# Constraints
if len(all_keys['constraints']) > 2:
    recommendations.append("✓ Rich constraints available → Use as predictors")
else:
    recommendations.append("✗ Few constraints → Limited constraint feature opportunities")

for rec in recommendations:
    print(f"  {rec}")

# ===== Step 8: Summary =====
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print()
print("📊 Data Summary:")
print(f"  Packets: {len(packets)}")
print(f"  Feature dimensions: {len(all_keys['as_of'])} + {len(all_keys['market'])} + {len(all_keys['execution'])} + {len(all_keys['objective'])} + {len(all_keys['constraints'])}")
if isinstance(labels_data, dict):
    print(f"  Total labels: {sum(len(v) if isinstance(v, list) else 1 for v in labels_data.values())}")
if len(all_pnls):
    print(f"  PnL target range: ${all_pnls.min():.2f} to ${all_pnls.max():.2f}")

if issues:
    print()
    print(f"⚠️ Issues detected ({len(issues)}):")
    for issue in issues:
        print(f"  {issue}")
else:
    print()
    print("✅ No major quality issues detected")

print()
print("🚀 Ready for training:")
print(f"  - Cell C will extract features from the {len(packets)} packets")
print("  - PRIME and PHANTOM models will compete")
print(f"  - Test set: ~{int(len(packets) * 0.15)} samples")
print("  - Target metric: Predict net_pnl_usd")

print()
print("=" * 70)
print("✓ INSPECTION COMPLETE")
print("=" * 70)
