"""
megacompact_core/core.py — REFERENCE SCHEMA (placeholder)

This file documents, as executable dataclasses, the DecisionPacket and
OutcomeLabel schema that the Kaggle pipeline (Cell C / Cell D) consumes.

>>> REPLACE THIS FILE <<<
When you have the real megacompact framework, overwrite this file with your
full megacompact_core.py (and megacompact_engineers.py -> engineers.py).
Cell C and Cell D only read the two JSON data files produced by Cell B, so
they will keep working with the real engine unchanged.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class AsOf:
    """Chain position the decision was evaluated at."""
    block_number: int
    timestamp_ms: int


@dataclass
class MarketState:
    """Market microstructure snapshot."""
    spot_price_usd: float
    mid_price_usd: float
    bid_price_usd: float
    ask_price_usd: float
    base_volume_24h: float = 0.0
    quote_volume_24h: float = 0.0
    volatility_pct: float = 0.0


@dataclass
class ExecutionEstimate:
    """Pre-trade execution cost estimates."""
    estimated_slippage_usd: float
    estimated_gas_usd: float
    estimated_include_probability: float = 0.5


@dataclass
class Objective:
    """What the decision is optimizing for."""
    horizon_blocks: int
    goal: str = "maximize_net_pnl"


@dataclass
class Constraints:
    """Risk limits for the decision."""
    max_loss_usd: float
    max_gas_usd: float


@dataclass
class ActionCandidate:
    """One executable action under consideration."""
    action_id: str
    action_type: str
    trade_size_usd: float
    expected_output_usd: float


@dataclass
class DecisionPacket:
    """A single trading decision to be scored by the PnL models."""
    decision_id: str
    as_of: AsOf
    market: MarketState
    execution: ExecutionEstimate
    objective: Objective
    constraints: Constraints
    action_candidates: List[ActionCandidate] = field(default_factory=list)
    pair: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OutcomeLabel:
    """A simulated settlement of one decision."""
    outcome_id: str
    decision_id: str
    executed_action_id: str
    status: str
    net_pnl_usd: float
    gross_pnl_usd: float
    gas_paid_usd: float
    slippage_paid_usd: float
    horizon_blocks: int
    block_number: int
    timestamp_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


# The 20 features Cell C extracts from a DecisionPacket (in order).
FEATURE_ORDER = [
    'block_number', 'timestamp_ms', 'spot_price', 'mid_price',
    'bid_price', 'ask_price', 'spread_pct', 'base_volume_24h',
    'quote_volume_24h', 'volatility_pct', 'estimated_slippage_usd',
    'estimated_gas_usd', 'estimated_include_prob', 'horizon_blocks',
    'num_candidates', 'max_trade_size', 'avg_expected_output',
    'max_expected_output', 'max_loss_usd', 'max_gas_usd'
]
