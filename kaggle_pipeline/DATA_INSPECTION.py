"""
DATA INSPECTION SCRIPT (REAL MEGACOMPACT SCHEMA)
Run this AFTER Cell B to understand the real MegaCompact16 decision packets
and outcome labels before training. Helps identify available features and
data-quality issues.
"""

import json
from pathlib import Path
from typing import Dict, Any

print("=" * 70)
print("MEGACOMPACT DATA INSPECTION")
print("=" * 70)

# ===== Step 1: Locate data files =====
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
    print(" Make sure Cell B completed successfully")
    exit(1)

# ===== Step 2: Load and inspect packets =====
print("\n" + "=" * 70)
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
    first_packet = packets[0]
    print(f"\nFIRST PACKET (decision_id: {first_packet.get('decision_id')})")
    print(f" Top-level keys: {list(first_packet.keys())}")

    for section in ['as_of', 'objective', 'market', 'execution', 'graph', 'constraints']:
        if section in first_packet:
            data = first_packet[section]
            print(f"\n  [{section}]")
            if isinstance(data, dict):
                for key, value in list(data.items())[:8]:
                    print(f"    {key}: {type(value).__name__} = {value}")
                if len(data) > 8:
                    print(f"    ... and {len(data) - 8} more keys")

    if 'action_candidates' in first_packet:
        candidates = first_packet['action_candidates']
        print(f"\n  [action_candidates] ({len(candidates)} candidates)")
        if candidates:
            for key, value in list(candidates[0].items())[:8]:
                print(f"    {key}: {type(value).__name__} = {value}")
            if len(candidates[0]) > 8:
                print(f"    ... and {len(candidates[0]) - 8} more keys")

# ===== Step 3: Analyze all packets for available features =====
print("\n" + "=" * 70)
print("FEATURE AVAILABILITY ANALYSIS")
print("=" * 70)

sections = ['as_of', 'objective', 'market', 'execution', 'graph', 'constraints']
all_keys = {s: set() for s in sections}

for packet in packets:
    for section in sections:
        if section in packet and isinstance(packet[section], dict):
            all_keys[section].update(packet[section].keys())

for section in sections:
    if all_keys[section]:
        print(f"\n[{section}] ({len(all_keys[section])} fields)")
        for key in sorted(all_keys[section]):
            sample_value = None
            for packet in packets:
                if section in packet and key in packet[section]:
                    sample_value = packet[section][key]
                    break
            print(f"  • {key}: {type(sample_value).__name__} (example: {sample_value})")

# Candidate-level fields
cand_keys = set()
for packet in packets:
    for c in packet.get('action_candidates', []):
        cand_keys.update(c.keys())
if cand_keys:
    print(f"\n[action_candidates] ({len(cand_keys)} fields)")
    for key in sorted(cand_keys):
        sample_value = None
        for packet in packets:
            for c in packet.get('action_candidates', []):
                if key in c:
                    sample_value = c[key]
                    break
            if sample_value is not None:
                break
        print(f"  • {key}: {type(sample_value).__name__} (example: {sample_value})")

# ===== Step 4: Load and inspect labels =====
print("\n" + "=" * 70)
print("OUTCOME LABELS")
print("=" * 70)

try:
    with open(labels_file) as f:
        labels_data = json.load(f)

    if isinstance(labels_data, dict):
        print(f"✓ Loaded {len(labels_data)} decision outcomes")

        sample_decision_id = list(labels_data.keys())[0]
        sample_outcomes = labels_data[sample_decision_id]

        print(f"\nFIRST DECISION OUTCOMES (decision_id: {sample_decision_id})")
        print(f"  Type: {type(sample_outcomes).__name__}")

        if isinstance(sample_outcomes, list):
            print(f"  Number of outcomes: {len(sample_outcomes)}")
            if sample_outcomes:
                first_outcome = sample_outcomes[0]
                print(f"  First outcome keys: {list(first_outcome.keys())}")
                for key, value in list(first_outcome.items())[:12]:
                    print(f"    {key}: {type(value).__name__} = {value}")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===== Step 5: Label statistics =====
print("\n" + "=" * 70)
print("OUTCOME LABEL STATISTICS")
print("=" * 70)

if isinstance(labels_data, dict):
    all_pnls = []
    n_reverted = 0
    n_included = 0
    n_labels = 0
    for decision_id, outcomes in labels_data.items():
        items = outcomes if isinstance(outcomes, list) else [outcomes]
        for outcome in items:
            n_labels += 1
            all_pnls.append(float(outcome.get('net_pnl_usd', 0.0)))
            if outcome.get('reverted'):
                n_reverted += 1
            if outcome.get('included_before_deadline'):
                n_included += 1

    if all_pnls:
        import numpy as np
        all_pnls = np.array(all_pnls)

        print(f"net_pnl_usd distribution ({n_labels} labels):")
        print(f"  Count: {len(all_pnls)}")
        print(f"  Min: ${all_pnls.min():.2f}")
        print(f"  Q25: ${np.percentile(all_pnls, 25):.2f}")
        print(f"  Median: ${np.median(all_pnls):.2f}")
        print(f"  Q75: ${np.percentile(all_pnls, 75):.2f}")
        print(f"  Max: ${all_pnls.max():.2f}")
        print(f"  Mean: ${all_pnls.mean():.2f}")
        print(f"  Std: ${all_pnls.std():.2f}")
        print(f"  Profitable (PnL > 0): {(all_pnls > 0).sum()} ({(all_pnls > 0).sum()/len(all_pnls)*100:.1f}%)")
        print(f"  Unprofitable (PnL < 0): {(all_pnls < 0).sum()} ({(all_pnls < 0).sum()/len(all_pnls)*100:.1f}%)")
        print(f"  Reverted labels: {n_reverted} ({n_reverted/n_labels*100:.1f}%)")
        print(f"  Included before deadline: {n_included} ({n_included/n_labels*100:.1f}%)")

# ===== Step 6: Data quality checks =====
print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

issues = []

if len(packets) < 100:
    issues.append(f"⚠ Low packet count: {len(packets)} (expected 500+)")
else:
    print(f"✓ Packet count: {len(packets)}")

nan_count = 0
for packet in packets[:50]:
    for section in ['as_of', 'objective', 'market', 'execution']:
        if section in packet:
            for key, value in packet[section].items():
                if value is None:
                    nan_count += 1

if nan_count > 10:
    issues.append(f"⚠ Many None values detected")
else:
    print(f"✓ No significant None values in sample")

decision_ids_in_packets = {p.get('decision_id') for p in packets}
decision_ids_in_labels = set(labels_data.keys()) if isinstance(labels_data, dict) else set()
if decision_ids_in_packets:
    coverage = len(decision_ids_in_labels & decision_ids_in_packets) / len(decision_ids_in_packets)
    if coverage < 0.8:
        issues.append(f"⚠ Low label coverage: {coverage*100:.1f}% (expected 95%+)")
    else:
        print(f"✓ Label coverage: {coverage*100:.1f}%")

unique_decision_ids = len({p.get('decision_id') for p in packets})
if unique_decision_ids < len(packets):
    issues.append(f"⚠ Duplicate decision_ids detected")
else:
    print(f"✓ All decision_ids unique")

# ===== Step 7: Feature suitability assessment =====
print("\n" + "=" * 70)
print("FEATURE ENGINEERING RECOMMENDATIONS")
print("=" * 70)

recommendations = []

if any(k.startswith('price_') for k in all_keys['market']):
    recommendations.append("✓ Price data available → Use price features")
else:
    recommendations.append("✗ Price data missing → Consider adding")

if 'total_liquidity' in all_keys['execution']:
    recommendations.append("✓ Liquidity available → Use liquidity features")
else:
    recommendations.append("✗ Liquidity data missing")

if 'base_fee_gwei' in all_keys['execution']:
    recommendations.append("✓ Gas data available → Critical for PnL prediction")
else:
    recommendations.append("✗ Gas data missing → May limit accuracy")

if 'horizon_blocks' in all_keys['objective']:
    recommendations.append("✓ Horizon available → Use as predictor")
else:
    recommendations.append("✗ Horizon missing")

for rec in recommendations:
    print(f" {rec}")

# ===== Step 8: Summary =====
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n📊 Data Summary:")
print(f" Packets: {len(packets)}")
print(f" Decisions with labels: {len(labels_data)}")
if all_pnls is not None and len(all_pnls) > 0:
    print(f" Total labels: {len(all_pnls)}")
    print(f" PnL target range: ${all_pnls.min():.2f} to ${all_pnls.max():.2f}")

if issues:
    print(f"\n⚠️ Issues detected ({len(issues)}):")
    for issue in issues:
        print(f" {issue}")
else:
    print(f"\n✅ No major quality issues detected")

print(f"\n🚀 Ready for training:")
print(f" - Cell C will extract features from the {len(packets)} packets")
print(f" - PRIME and PHANTOM models will compete")
print(f" - Test set: ~{int(len(packets)*0.15)} samples")
print(f" - Target metric: Predict net_pnl_usd")

print("\n" + "=" * 70)
print("✓ INSPECTION COMPLETE")
print("=" * 70)
