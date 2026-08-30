# ============================================================
# Ω-MEGA v5.3 — CELL B: REAL MEGACOMPACT DATA GENERATION
# ============================================================
# Runs the REAL MegaCompact16 engine (megacompact_core/core.py) to generate
# synthetic market events, validate them, build decision packets, simulate
# outcome labels, and run the double-pass engineer gate.
#
# Module resolution (in order of preference):
#   1. REPO_PATH/megacompact_core  (set by Cell A when the repo is cloned)
#   2. /kaggle/working             (the %%writefile core.py / engineers.py flow)
#
# Cell A copies megacompact_core/{core,engineers}.py into the working dir, so
# both `import core as mc` and `from engineers import ...` resolve either way.
#
# PERSISTENCE: this cell writes the two files that Cells C and D consume:
#   RUN_DIR/data/decision_packets.jsonl   one DecisionPacket per line
#   RUN_DIR/data/outcome_labels.json      {decision_id: [OutcomeLabel, ...]}
#
# FAST SMOKE MODE: set env MEGACOMPACT_FAST_SMOKE=1 to run a tiny config
# (used by scripts/local_smoke_test.py so the wiring can be verified quickly).
# The default is the real 30-day research config.
# ============================================================
import os
import sys
import json

import numpy as np

# ---- module path: prefer the cloned repo, then the Kaggle working dir ----
if "REPO_PATH" in globals() and REPO_PATH is not None:
    sys.path.insert(0, str(REPO_PATH / "megacompact_core"))
sys.path.insert(0, "/kaggle/working")

import core as mc
from engineers import DoublePassEngineerGate

# ---- config (real research default; tiny override for the smoke test) ----
cfg = mc.Config(seed=PRIMARY_SEED)
if os.environ.get("MEGACOMPACT_FAST_SMOKE") == "1":
    cfg.synthetic = mc.SyntheticConfig(
        days=3, blocks_per_day=200, num_tokens=3, num_pools=4, num_venues=2,
        seed=PRIMARY_SEED,
    )
else:
    cfg.synthetic = mc.SyntheticConfig(
        days=30, blocks_per_day=200, num_tokens=6, num_pools=10, num_venues=4,
        seed=PRIMARY_SEED,
    )

gate = DoublePassEngineerGate(ledger_path=str(RUN_DIR / "verification_ledger.jsonl"))

print("1. GENERATE SYNTHETIC MARKET EVENTS")
events = mc.SyntheticMarketGenerator(cfg.synthetic).generate()
print(f"   events={len(events)}")

print("2. VALIDATE (time-causality, schema)")
valid = mc.DataValidator(cfg).validate_events(events)
print(f"   valid={len(valid)}")

print("3. BUILD DECISION PACKETS")
fs = mc.TimeCausalFeatureStore(valid, cfg.max_feature_age_ms)
packets = mc.DecisionPacketBuilder(cfg, fs).build_packets(valid)
print(f"   packets={len(packets)}")

print("4. BUILD REAL OUTCOME LABELS")
amm = mc.ConstantProductAMM()
market_replay = mc.MarketReplay(valid)
quote_engine = mc.QuoteEngine(amm)
exec_engine = mc.ExecutionEngine(cfg)
cost_engine = mc.CostEngine()
outcome_engine = mc.OutcomeEngine()
label_builder = mc.LabelBuilder(market_replay, quote_engine, exec_engine, cost_engine, outcome_engine)
rng = np.random.default_rng(PRIMARY_SEED)
labels = label_builder.build_labels(packets, rng)
print(f"   labels={len(labels)}")

print("5. ENGINEER DOUBLE-PASS GATE (audit a sample, log to telemetry)")
eng_n = min(100, len(packets))
audit = gate.run_batch([(p.model_dump(), p.decision_id, "DecisionPacket") for p in packets[:eng_n]])
print(f"   allowed={audit['n_allowed_for_llm']}/{audit['n_total']}")

# ---- 6. PERSIST the data files Cells C and D read ----
data_dir = RUN_DIR / "data"
data_dir.mkdir(parents=True, exist_ok=True)

packets_file = data_dir / "decision_packets.jsonl"
labels_file = data_dir / "outcome_labels.json"

with open(packets_file, "w") as f:
    for p in packets:
        f.write(json.dumps(p.model_dump(), default=str) + "\n")

outcome_labels = {}
for label in labels:
    outcome_labels.setdefault(label.decision_id, []).append(label.model_dump())

with open(labels_file, "w") as f:
    json.dump(outcome_labels, f, default=str)

telemetry.write("megacompact_data_generated", n_events=len(events), n_valid=len(valid),
                n_packets=len(packets), n_labels=len(labels), engineer_audit=audit,
                packets_file=str(packets_file), labels_file=str(labels_file))
print(f"   wrote {packets_file}")
print(f"   wrote {labels_file}")
print("✓ CELL B (data) complete")
