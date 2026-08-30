"""
megacompact_core — schema reference package.

This package documents the DecisionPacket / OutcomeLabel schema the pipeline
consumes. The `core.py` and `engineers.py` files here are lightweight
reference placeholders: replace them with the real megacompact_core.py and
megacompact_engineers.py to use the full simulation engine.
"""

from .core import (
    ActionCandidate,
    AsOf,
    Constraints,
    DecisionPacket,
    ExecutionEstimate,
    MarketState,
    Objective,
    OutcomeLabel,
)
from .engineers import double_pass_gate, time_causality_check, validate_packet

__all__ = [
    "ActionCandidate",
    "AsOf",
    "Constraints",
    "DecisionPacket",
    "ExecutionEstimate",
    "MarketState",
    "Objective",
    "OutcomeLabel",
    "validate_packet",
    "time_causality_check",
    "double_pass_gate",
]
