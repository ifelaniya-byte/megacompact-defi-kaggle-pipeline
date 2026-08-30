"""
CELL B: DATA GENERATION (MEGACOMPACT - REFERENCE IMPLEMENTATION)
Generates synthetic DeFi decision packets and simulated outcome labels in the
exact megacompact DecisionPacket / OutcomeLabel schema.

Outputs (consumed by Cell C and Cell D):
  RUN_DIR/data/decision_packets.jsonl   one DecisionPacket per line
  RUN_DIR/data/outcome_labels.json      {decision_id: [OutcomeLabel, ...]}

NOTE ON THE REAL FRAMEWORK:
This file is a self-contained reference generator — a compact stand-in for
the full megacompact engine (megacompact_core.py + megacompact_engineers.py).
Cell C and Cell D depend ONLY on the two output files above, so when you are
ready to use the real framework, replace this file with your own
CELL_B_data_generation.py (and put the real core/engineers files in
megacompact_core/) and everything downstream keeps working unchanged.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

print("=" * 70)
print("CELL B: DATA GENERATION (MEGACOMPACT)")
print("=" * 70)

# ===== Step 0: Verify Cell A completed =====
try:
    assert PRIMARY_SEED is not None
    assert RUN_DIR.exists()
    assert telemetry is not None
    print(f"✓ Cell A variables available (seed={PRIMARY_SEED}, dir={RUN_DIR.name})")
except NameError as e:
    print(f"❌ Cell A not run yet: {e}")
    exit(1)

# ===== Configuration (reference defaults; mirrors configs/default_config.json) =====
N_BLOCKS = 12_000             # simulated chain blocks (market event stream)
N_PACKETS = 1_199             # decision packets to build
OUTCOMES_PER_DECISION = 3     # simulated outcomes per decision
BASE_BLOCK = 19_000_000
BLOCK_TIME_MS = 12_000
BASE_TS_MS = int(datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
PAIR = "WETH/USDC"
START_PRICE = 2_450.0
WARMUP_BLOCKS = 720           # 24h of history before the first decision

rng = random.Random(PRIMARY_SEED)

# ===== Step 1: Generate synthetic market events (block-level random walk) =====
events = []
price = START_PRICE
vol_regime = 0.004
for i in range(N_BLOCKS):
    if rng.random() < 0.01:                      # occasional volatility regime shift
        vol_regime = rng.uniform(0.001, 0.012)
    ret = rng.gauss(0.0, vol_regime)
    price = max(100.0, price * (1.0 + ret))
    events.append({
        "event_id": f"E{i:06d}",
        "block_number": BASE_BLOCK + i,
        "timestamp_ms": BASE_TS_MS + i * BLOCK_TIME_MS,
        "pair": PAIR,
        "price_usd": round(price, 6),
        "volume_usd": round(10 ** rng.uniform(3.0, 6.0), 2),
    })
print("1. GENERATE SYNTHETIC MARKET EVENTS")
print(f"   events={len(events)}")

# ===== Step 2: Validate (time-causality, schema) =====
valid = 0
for j, ev in enumerate(events):
    ok = (
        ev["block_number"] == BASE_BLOCK + j
        and (j == 0 or ev["timestamp_ms"] > events[j - 1]["timestamp_ms"])
        and ev["price_usd"] > 0
        and ev["volume_usd"] >= 0
        and set(ev) == {"event_id", "block_number", "timestamp_ms", "pair", "price_usd", "volume_usd"}
    )
    if ok:
        valid += 1
print("2. VALIDATE (time-causality, schema)")
print(f"   valid={valid}/{len(events)}")

# ===== Step 3: Build decision packets =====
stride = max(1, (N_BLOCKS - WARMUP_BLOCKS) // N_PACKETS)
decision_idx = [WARMUP_BLOCKS + k * stride for k in range(N_PACKETS)]

packets = []
for k, idx in enumerate(decision_idx):
    ev = events[idx]
    window = events[max(0, idx - 50): idx + 1]
    prices = [w["price_usd"] for w in window]
    rets = [prices[i + 1] / prices[i] - 1 for i in range(len(prices) - 1)]
    mean_ret = sum(rets) / len(rets)
    var_ret = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
    volatility_pct = (var_ret ** 0.5) * 100

    spot = ev["price_usd"]
    spread_pct = rng.uniform(0.01, 0.35)          # bid/ask spread in percent
    half = spot * spread_pct / 100.0 / 2.0
    bid, ask = spot - half, spot + half
    mid = (bid + ask) / 2.0
    day_window = events[max(0, idx - 720): idx + 1]
    base_volume_24h = sum(w["volume_usd"] for w in day_window) / 10.0
    quote_volume_24h = base_volume_24h * spot

    n_cands = rng.choice([2, 3, 3, 4, 5])
    candidates = []
    for c in range(n_cands):
        size = rng.uniform(250.0, 75_000.0)
        edge_bps = rng.gauss(6.0, 28.0)           # expected edge in basis points
        candidates.append({
            "action_id": f"D{k:05d}_A{c}",
            "action_type": rng.choice(
                ["swap_exact_in", "swap_exact_out", "add_liquidity",
                 "remove_liquidity", "arbitrage"]
            ),
            "trade_size_usd": round(size, 2),
            "expected_output_usd": round(size * edge_bps / 10_000.0, 2),
        })

    max_size = max(c["trade_size_usd"] for c in candidates)
    depth = rng.uniform(0.35, 1.0)                # market depth factor
    slippage = max(0.0, max_size / max(depth * (base_volume_24h ** 0.5), 1.0)) * rng.uniform(0.8, 1.2)
    gas = rng.uniform(4.0, 55.0)
    include_prob = min(0.995, max(0.25, rng.gauss(0.78, 0.14)))

    packets.append({
        "decision_id": f"D{k:05d}",
        "pair": PAIR,
        "as_of": {
            "block_number": ev["block_number"],
            "timestamp_ms": ev["timestamp_ms"],
        },
        "market": {
            "spot_price_usd": round(spot, 6),
            "mid_price_usd": round(mid, 6),
            "bid_price_usd": round(bid, 6),
            "ask_price_usd": round(ask, 6),
            "base_volume_24h": round(base_volume_24h, 2),
            "quote_volume_24h": round(quote_volume_24h, 2),
            "volatility_pct": round(volatility_pct, 4),
        },
        "execution": {
            "estimated_slippage_usd": round(slippage, 2),
            "estimated_gas_usd": round(gas, 2),
            "estimated_include_probability": round(include_prob, 4),
        },
        "objective": {
            "horizon_blocks": rng.choice([6, 12, 24, 36]),
            "goal": "maximize_net_pnl",
        },
        "constraints": {
            "max_loss_usd": round(-abs(rng.gauss(1_250.0, 600.0)), 2),
            "max_gas_usd": round(rng.uniform(40.0, 220.0), 2),
        },
        "action_candidates": candidates,
    })
print("3. BUILD DECISION PACKETS")
print(f"   packets={len(packets)}")

# ===== Step 4: Build outcome labels (simulate each decision 3x) =====
def _simulate_outcome(packet, draw_rng, draw_idx):
    """Simulate one settlement of a decision packet -> OutcomeLabel dict."""
    mkt = packet["market"]
    exe = packet["execution"]
    cands = packet["action_candidates"]

    vol = mkt["volatility_pct"] / 100.0
    spread = (mkt["ask_price_usd"] - mkt["bid_price_usd"]) / mkt["ask_price_usd"]
    size = max(c["trade_size_usd"] for c in cands)
    best_edge = max(c["expected_output_usd"] for c in cands)
    include = exe["estimated_include_probability"]
    gas = exe["estimated_gas_usd"]
    slip = exe["estimated_slippage_usd"]
    horizon = packet["objective"]["horizon_blocks"]

    # Structural PnL: inclusion-weighted edge eroded by volatility, minus costs.
    gross = best_edge * include * (1.0 - 0.45 * vol)
    costs = slip * (1.0 + 0.6 * vol) + gas + size * spread * 0.22

    # Execution noise: scales with the size of the opportunity being simulated.
    sigma = 0.45 * max(abs(gross), 60.0) + 35.0
    noise = draw_rng.gauss(0.0, sigma)
    net = gross - costs + noise

    best_action = max(cands, key=lambda c: c["expected_output_usd"])
    return {
        "outcome_id": f"{packet['decision_id']}_O{draw_idx}",
        "decision_id": packet["decision_id"],
        "executed_action_id": best_action["action_id"],
        "status": "settled",
        "horizon_blocks": horizon,
        "gross_pnl_usd": round(gross + noise, 2),
        "gas_paid_usd": round(gas * draw_rng.uniform(0.9, 1.1), 2),
        "slippage_paid_usd": round(slip * draw_rng.uniform(0.9, 1.2), 2),
        "net_pnl_usd": round(net, 2),
        "block_number": packet["as_of"]["block_number"] + horizon,
        "timestamp_ms": packet["as_of"]["timestamp_ms"] + horizon * BLOCK_TIME_MS,
    }

outcome_labels = {}
total_outcomes = 0
for packet in packets:
    outs = [
        _simulate_outcome(packet, rng, o)
        for o in range(OUTCOMES_PER_DECISION)
    ]
    outcome_labels[packet["decision_id"]] = outs
    total_outcomes += len(outs)
print("4. BUILD OUTCOME LABELS")
print(f"   labels={total_outcomes}")

# ===== Step 5: Write data files =====
data_dir = RUN_DIR / "data"
data_dir.mkdir(parents=True, exist_ok=True)

packets_file = data_dir / "decision_packets.jsonl"
labels_file = data_dir / "outcome_labels.json"

with open(packets_file, "w") as f:
    for p in packets:
        f.write(json.dumps(p) + "\n")

with open(labels_file, "w") as f:
    json.dump(outcome_labels, f)

telemetry.write(
    "CELL_B_DATA_GENERATION_COMPLETE",
    generator="reference_synthetic_v1",
    seed=PRIMARY_SEED,
    events=len(events),
    valid_events=valid,
    packets=len(packets),
    labels=total_outcomes,
    packets_file=str(packets_file),
    labels_file=str(labels_file),
)

print("5. WRITE DATA FILES")
print(f"   {packets_file}")
print(f"   {labels_file}")
print()
print("✓ CELL B (data) complete")
print("Next: run DATA_INSPECTION.py (optional) then CELL_C_megacompact_training.py")
