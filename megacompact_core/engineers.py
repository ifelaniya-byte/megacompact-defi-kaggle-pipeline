"""
megacompact_core/engineers.py — REFERENCE VERIFICATION GATE (placeholder)

Lightweight stand-in for the real megacompact_engineers.py double-pass
verification gate. Provides:

  - validate_packet(packet)      schema check for one DecisionPacket
  - time_causality_check(events) monotonic block/timestamp check
  - double_pass_gate(...)        pass 1: schema, pass 2: label alignment

>>> REPLACE THIS FILE <<<
Overwrite this file with your real megacompact_engineers.py to use the full
engineering gate. Cell C and Cell D do not import this module; it exists so
the real framework drops straight into this package layout.
"""

from typing import Dict, List, Tuple

REQUIRED_SECTIONS = ("as_of", "market", "execution", "objective", "constraints", "action_candidates")


def validate_packet(packet: dict) -> Tuple[bool, List[str]]:
    """Pass 1: check a packet dict has the required sections and keys."""
    errors = []
    for section in REQUIRED_SECTIONS:
        if section not in packet:
            errors.append(f"missing section: {section}")

    if "decision_id" not in packet:
        errors.append("missing decision_id")

    market = packet.get("market", {})
    for key in ("spot_price_usd", "bid_price_usd", "ask_price_usd"):
        if key in market and float(market[key]) <= 0:
            errors.append(f"market.{key} must be positive")

    candidates = packet.get("action_candidates", [])
    if not isinstance(candidates, list) or len(candidates) == 0:
        errors.append("action_candidates must be a non-empty list")

    return (len(errors) == 0), errors


def time_causality_check(events: List[dict]) -> Tuple[bool, List[int]]:
    """Check block_number and timestamp_ms are strictly increasing."""
    bad = []
    for i in range(1, len(events)):
        prev, cur = events[i - 1], events[i]
        if cur.get("block_number", 0) <= prev.get("block_number", 0):
            bad.append(i)
        elif cur.get("timestamp_ms", 0) <= prev.get("timestamp_ms", 0):
            bad.append(i)
    return (len(bad) == 0), bad


def double_pass_gate(packets: List[dict], labels: Dict[str, list]) -> dict:
    """Pass 1: schema validation. Pass 2: label alignment + coverage."""
    allowed = 0
    schema_failures = []

    for packet in packets:
        ok, errors = validate_packet(packet)
        if ok:
            allowed += 1
        else:
            schema_failures.append({"decision_id": packet.get("decision_id"), "errors": errors})

    labeled_ids = {p.get("decision_id") for p in packets if p.get("decision_id") in labels}
    coverage = len(labeled_ids) / len(packets) if packets else 0.0

    return {
        "total_packets": len(packets),
        "allowed": allowed,
        "schema_failures": schema_failures,
        "labels_available": len(labels),
        "label_coverage": round(coverage, 4),
        "passed": allowed == len(packets) and coverage >= 0.95,
    }
