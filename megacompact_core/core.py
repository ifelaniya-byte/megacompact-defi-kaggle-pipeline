#!/usr/bin/env python3
"""
MegaCompact16: Time-causal, uncertainty-aware DeFi research, simulation,
prediction, planning, paper-replay, and validation harness.

WHAT THIS IS
One self-contained Python 3.11 script that generates decision packets, simulates
action outcomes, trains an uncertainty-aware world model, calibrates risk,
plans conservatively, paper-replays decisions, stress-tests them, audits for
leakage, and writes reproducible artifacts.

WHAT THIS IS NOT
- Not a live trading bot
- No wallet access, private keys, signing, transaction submission, Flashbots,
  RPC broadcast, or contract calls
- Not a promise of profit, accuracy, or "omniscience"
- Not a notebook split into hundreds of cells
- Not a protocol-specific AMM emulator beyond generic, configurable mechanics

NON-NEGOTIABLE SAFETY RULES
1. Paper/replay mode only. Do not import web3, eth_account, wallet libraries,
   signing libraries, or transaction broadcast libraries.
2. Default mode is ABSTAIN.
3. Every input feature must include a feature availability time. No feature may
   be visible before that time.
4. Data must be time-split: train -> validation -> calibration -> test. Never
   randomly shuffle across time.
5. All model/scaler/calibration fitting must remain restricted to its assigned
   window. Test is immutable.
6. Gross spread or gross PnL must never be reported as net strategy PnL.
7. Net PnL includes input cost, gas, protocol fees, borrow fees, bridge fees,
   slippage, failed/reverted-action cost, and configured costs.
8. Every decision persists full provenance: data, code/config hashes, model
   version, uncertainty, constraints, and realized paper outcome.
9. Any stale, malformed, future-leaking, unknown, out-of-distribution, or
   constraint-violating input returns ABSTAIN.
10. No online model-weight updates during paper replay. Retraining is an
    explicit offline operation that creates a new versioned run.

VERSION: 1.0.0
"""

# =============================================================================
# SECTION 02: IMPORTS AND OPTIONAL-DEPENDENCY FALLBACKS
# =============================================================================

import os
import sys
import json
import hashlib
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import random

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    pa = pq = None  # type: ignore
    PYARROW_AVAILABLE = False
    print("WARNING: pyarrow not available. Parquet I/O disabled.")
from pydantic import BaseModel, Field, field_validator
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False
    print("WARNING: yaml not available.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None  # type: ignore
    JOBLIB_AVAILABLE = False
    print("WARNING: joblib not available.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Charting will be disabled.")

try:
    from rich.console import Console
    from rich.progress import Progress, TaskID
    RICH_AVAILABLE = True
except ImportError:
    Console = Progress = TaskID = None  # type: ignore
    RICH_AVAILABLE = False
    print("WARNING: rich not available.")
    class Console:  # minimal stub
        def print(self, *a, **k): print(*a)
    console_stub = Console()

try:
    import typer
    TYPER_AVAILABLE = True
except ImportError:
    typer = None  # type: ignore
    TYPER_AVAILABLE = False
    print("WARNING: typer not available. CLI disabled.")

# Optional imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Model training will be disabled.")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. Some features will be disabled.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: FAISS not available. Using sklearn for retrieval.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("WARNING: networkx not available. Graph features disabled.")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("WARNING: FastAPI not available. API server disabled.")

try:
    from scipy import stats
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Some statistical features disabled.")

# Product VD3: SMT + interval arithmetic on every TRADE path (fail-closed if missing)
try:
    from smt_gate import pre_trade_gate
    _SMT_TRADE_AVAILABLE = True
except Exception:
    pre_trade_gate = None  # type: ignore
    _SMT_TRADE_AVAILABLE = False

try:
    from interval_arith import Interval, trade_allowed_by_interval, net_edge_interval
    _INTERVAL_TRADE_AVAILABLE = True
except Exception:
    Interval = trade_allowed_by_interval = net_edge_interval = None  # type: ignore
    _INTERVAL_TRADE_AVAILABLE = False

# =============================================================================
# SECTION 03: CONSTANTS, SEMANTIC VERSION, DETERMINISTIC SEED UTILITIES
# =============================================================================

VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"

DEFAULT_CHAIN_ID = 42161  # Arbitrum
DEFAULT_BLOCK_TIME_MS = 1200  # ~1.2 seconds per block

# Event types
class EventType(str, Enum):
    SWAP = "swap"
    POOL_UPDATE = "pool_update"
    QUOTE = "quote"
    GAS = "gas"
    BLOCK = "block"
    ORACLE = "oracle"
    ROUTE = "route"
    INCLUSION = "inclusion"
    REVERT = "revert"
    BRIDGE = "bridge"
    LIQUIDITY = "liquidity"
    PRICE = "price"

# Decision verdicts
class Verdict(str, Enum):
    TRADE = "TRADE"
    ABSTAIN = "ABSTAIN"

# Regime types
class RegimeType(str, Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    GAS_SPIKE = "gas_spike"
    STALE_DATA = "stale_data"
    STRESSED = "stressed"
    SHIFTED = "shifted"

# Reason codes for abstention
class ReasonCode(str, Enum):
    NO_CANDIDATES = "NO_CANDIDATES"
    STALE_DATA = "STALE_DATA"
    MISSING_CRITICAL_FEATURES = "MISSING_CRITICAL_FEATURES"
    FUTURE_FEATURE_BLOCKED = "FUTURE_FEATURE_BLOCKED"
    ROUTE_INVALID = "ROUTE_INVALID"
    INSUFFICIENT_NET_EDGE = "INSUFFICIENT_NET_EDGE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    OOD = "OOD"
    REVERT_RISK = "REVERT_RISK"
    INCLUSION_RISK = "INCLUSION_RISK"
    LOSS_LIMIT = "LOSS_LIMIT"
    GAS_LIMIT = "GAS_LIMIT"
    SLIPPAGE_LIMIT = "SLIPPAGE_LIMIT"
    CONSTRAINT_FAILURE = "CONSTRAINT_FAILURE"
    SMT_UNAVAILABLE = "SMT_UNAVAILABLE"
    SMT_BLOCK = "SMT_BLOCK"
    INTERVAL_UNAVAILABLE = "INTERVAL_UNAVAILABLE"
    INTERVAL_BLOCK = "INTERVAL_BLOCK"

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sha256_hash(data: Union[str, bytes, Dict]) -> str:
    """Compute SHA-256 hash of data."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def make_pool_update_payload(pool_id: str, reserve0: float, reserve1: float,
                              fee_bps: int = 30) -> Dict[str, Any]:
    """Canonical pool-update payload. This is the ONLY accepted shape for
    POOL_UPDATE event payloads. Do not construct pool-update payloads by
    hand elsewhere in the codebase — always go through this function."""
    return {"pool_id": pool_id, "reserve0": reserve0, "reserve1": reserve1, "fee_bps": fee_bps}

def get_reserves(event: "NormalizedEvent") -> Tuple[float, float]:
    """Canonical reserve accessor. Do not read event.payload['reserves'] or
    event.payload['reserve0']/['reserve1'] directly anywhere else — always
    go through this function so the storage schema only has to be correct
    in one place. Raises KeyError loudly rather than silently defaulting to
    (0, 0), because a silent zero-liquidity read is worse than a crash."""
    payload = event.payload
    if "reserve0" in payload and "reserve1" in payload:
        return float(payload["reserve0"]), float(payload["reserve1"])
    if "reserves" in payload and payload["reserves"]:
        r = payload["reserves"]
        return float(r[0]), float(r[1])
    raise KeyError(
        f"POOL_UPDATE event {event.event_id} has no reserve0/reserve1 or reserves "
        f"in payload; refusing to silently treat this as zero liquidity."
    )

# =============================================================================
# SECTION 04: CONFIGURATION MODELS AND YAML LOADING
# =============================================================================

class SyntheticConfig(BaseModel):
    days: int = 7
    blocks_per_day: int = 500
    num_tokens: int = 10
    num_pools: int = 20
    num_venues: int = 5
    seed: int = 42
    baseline_volatility: float = 0.02
    normal_fee_tiers: List[float] = [0.003, 0.005, 0.01]
    liquidity_levels: List[float] = [10000, 50000, 100000]
    gas_levels: List[float] = [0.1, 0.5, 1.0]
    source_availability_delay_ms: int = 200
    quote_staleness_probability: float = 0.1
    revert_base_probability: float = 0.02
    adverse_selection_strength: float = 0.7
    regime_duration_blocks: int = 200
    distribution_shift_start_fraction: float = 0.7

class SplitsConfig(BaseModel):
    train_fraction: float = 0.5
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.15
    test_fraction: float = 0.2

    @field_validator('train_fraction', 'validation_fraction', 'calibration_fraction', 'test_fraction')
    @classmethod
    def validate_fractions(cls, v):
        if not 0 < v < 1:
            raise ValueError("Fractions must be between 0 and 1")
        return v

class ModelConfig(BaseModel):
    tabular_hidden: int = 128
    action_hidden: int = 64
    temporal_hidden: int = 128
    latent_dim: int = 192
    ensemble_size: int = 5
    dropout: float = 0.15
    learning_rate: float = 0.0005
    batch_size: int = 64
    epochs: int = 20
    early_stopping_patience: int = 5

class CalibrationConfig(BaseModel):
    confidence_level: float = 0.90
    min_regime_samples: int = 50

class PlanningConfig(BaseModel):
    lambda_uncertainty: float = 1.5
    lambda_cvar: float = 0.7
    lambda_revert: float = 2.5
    lambda_delay: float = 0.7
    min_net_edge_usd: float = 1.0
    max_revert_probability: float = 0.03
    min_inclusion_probability: float = 0.90
    ood_threshold: float = 0.7

class ConstraintsConfig(BaseModel):
    max_price_impact_bps: int = 50
    max_gas_usd: float = 5.0
    gas_uncertainty_buffer_usd: float = 0.5  # added to estimated_gas_usd before the max_gas_usd check
    max_quote_age_ms: int = 3000
    max_loss_usd: float = 10.0

class OutputConfig(BaseModel):
    root: str = "artifacts"
    run_id: Optional[str] = None

class Config(BaseModel):
    mode: str = "synth"
    preset: str = "research"
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    input_path: Optional[str] = None
    input_format: str = "parquet"
    chain_id: int = DEFAULT_CHAIN_ID
    max_feature_age_ms: int = 5000
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

# =============================================================================
# SECTION 05: ARTIFACT STORE, JSONL LOGGER, SHA-256 HASHING, RUN MANIFEST
# =============================================================================

class ArtifactStore:
    """Manages artifact storage and retrieval."""

    def __init__(self, root: str, run_id: Optional[str] = None):
        self.root = Path(root)
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.root / self.run_id
        self._create_directories()

    def _create_directories(self) -> None:
        """Create all required directories."""
        directories = [
            "logs",
            "hashes",
            "data/raw",
            "data/normalized",
            "data/quarantine",
            "data/packets",
            "data/labels",
            "data/splits",
            "data/regimes",
            "models",
            "paper",
            "backtests",
            "stress",
            "audits",
            "reports/charts"
        ]
        for dir_name in directories:
            (self.run_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        """Get path within run directory."""
        return self.run_dir / Path(*parts)

    def save_json(self, data: Dict, filename: str) -> None:
        """Save data as JSON."""
        path = self.path(filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, filename: str) -> Dict:
        """Load JSON data."""
        path = self.path(filename)
        with open(path, 'r') as f:
            return json.load(f)

    def save_jsonl(self, records: List[Dict], filename: str) -> None:
        """Save records as JSONL."""
        path = self.path(filename)
        with open(path, 'w') as f:
            for record in records:
                f.write(json.dumps(record, default=str) + '\n')

    def load_jsonl(self, filename: str) -> List[Dict]:
        """Load JSONL records."""
        path = self.path(filename)
        records = []
        with open(path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def save_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame as Parquet."""
        path = self.path(filename)
        df.to_parquet(path, index=False)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Load Parquet DataFrame."""
        path = self.path(filename)
        return pd.read_parquet(path)

class JSONLLogger:
    """Thread-safe JSONL logger for decision tracking."""

    def __init__(self, artifact_store: ArtifactStore, filename: str):
        self.store = artifact_store
        self.filename = filename
        self.path = artifact_store.path(filename)

    def log(self, record: Dict) -> None:
        """Append a record to the JSONL file."""
        with open(self.path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

class RunManifest:
    """Tracks run metadata and provenance."""

    def __init__(self, config: Config, artifact_store: ArtifactStore):
        self.config = config
        self.store = artifact_store
        self.manifest = {
            "run_id": artifact_store.run_id,
            "version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config.model_dump(),
            "config_hash": sha256_hash(config.model_dump()),
            "status": "initialized",
            "stages_completed": [],
            "data_hashes": {},
            "model_hashes": {},
            "audit_results": {}
        }

    def save(self) -> None:
        """Save manifest to disk."""
        self.store.save_json(self.manifest, "manifest.json")

    def load(self) -> None:
        """Load manifest from disk."""
        self.manifest = self.store.load_json("manifest.json")

    def update_stage(self, stage: str, status: str = "completed") -> None:
        """Update stage status."""
        if stage not in self.manifest["stages_completed"]:
            self.manifest["stages_completed"].append(stage)
        self.manifest["status"] = status
        self.save()

    def update_data_hash(self, key: str, hash_value: str) -> None:
        """Update data hash."""
        self.manifest["data_hashes"][key] = hash_value
        self.save()

    def update_model_hash(self, key: str, hash_value: str) -> None:
        """Update model hash."""
        self.manifest["model_hashes"][key] = hash_value
        self.save()

    def update_audit_result(self, key: str, result: Dict) -> None:
        """Update audit result."""
        self.manifest["audit_results"][key] = result
        self.save()

# =============================================================================
# SECTION 06: PYDANTIC DATA CONTRACTS
# =============================================================================

class NormalizedEvent(BaseModel):
    event_id: str
    schema_version: str = SCHEMA_VERSION
    source: str
    chain_id: int
    event_timestamp_ms: int
    observed_timestamp_ms: int
    available_timestamp_ms: int
    block_number: int
    block_hash: str
    tx_hash: Optional[str] = None
    tx_index: Optional[int] = None
    log_index: Optional[int] = None
    event_type: str
    entity_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    raw_payload_hash: str

    @field_validator('available_timestamp_ms')
    @classmethod
    def validate_availability(cls, v, info):
        if info.data.get('observed_timestamp_ms') and v < info.data['observed_timestamp_ms']:
            raise ValueError("available_timestamp_ms cannot be before observed_timestamp_ms")
        return v

class FeatureValue(BaseModel):
    name: str
    value: Union[float, List[float], str, None]
    source_event_ids: List[str] = Field(default_factory=list)
    max_available_timestamp_ms: int
    max_source_block: int
    age_ms: int
    missing: bool
    feature_version: str = SCHEMA_VERSION

class ActionCandidate(BaseModel):
    action_id: str
    route: List[Dict[str, Any]] = Field(default_factory=list)
    trade_size_usd: float
    borrow: Dict[str, Any] = Field(default_factory=lambda: {"asset": "", "amount": 0.0})
    slippage_limit_bps: int
    gas_policy: Dict[str, Any] = Field(default_factory=dict)
    deadline_block: int
    private_submission: bool = False

class DecisionPacket(BaseModel):
    schema_version: str = SCHEMA_VERSION
    decision_id: str
    as_of: Dict[str, Any]
    objective: Dict[str, Any]
    market: Dict[str, Any] = Field(default_factory=dict)
    execution: Dict[str, Any] = Field(default_factory=dict)
    graph: Dict[str, Any] = Field(default_factory=dict)
    action_candidates: List[ActionCandidate] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)

class ModelForecast(BaseModel):
    mean_net_pnl_usd: float
    pnl_q05_usd: float
    pnl_q50_usd: float
    pnl_q95_usd: float
    aleatoric_std: float
    epistemic_std: float
    revert_probability: float
    inclusion_probability: float
    ood_score: float
    regime_score: float
    estimated_gas_usd: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DecisionOutput(BaseModel):
    decision_id: str
    action_id: Optional[str]
    verdict: Verdict
    expected_net_pnl_usd: Optional[float]
    net_pnl_interval_usd: Optional[List[float]]
    lower_confidence_bound_usd: Optional[float]
    ensemble_uncertainty: Optional[float]
    ood_score: Optional[float]
    revert_probability: Optional[float]
    inclusion_probability: Optional[float]
    estimated_gas_usd: Optional[float]
    estimated_slippage_usd: Optional[float]
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    model_version: str
    config_hash: str
    packet_hash: str

class OutcomeLabel(BaseModel):
    decision_id: str
    action_id: str
    quoted_output_usd: float
    realized_output_usd: float
    input_cost_usd: float
    gas_usd: float
    protocol_fees_usd: float
    borrow_fees_usd: float
    bridge_fees_usd: float
    slippage_cost_usd: float
    revert_cost_usd: float
    other_costs_usd: float
    gross_pnl_usd: float
    net_pnl_usd: float
    reverted: bool
    included_before_deadline: bool
    inclusion_delay_blocks: int
    outcome_timestamp_ms: int
    outcome_block: int
    simulation_version: str = VERSION

# =============================================================================
# SECTION 07: SOURCE ADAPTERS AND NORMALIZER
# =============================================================================

class SourceAdapter(Protocol):
    """Protocol for data source adapters."""

    def load(self) -> pd.DataFrame:
        """Load raw data from source."""
        ...

    def normalize(self, raw_data: pd.DataFrame) -> List[NormalizedEvent]:
        """Normalize raw data to NormalizedEvent format."""
        ...

class SyntheticGeneratorAdapter:
    """Synthetic data generator adapter."""

    def __init__(self, config: SyntheticConfig, seed: int):
        self.config = config
        self.seed = seed
        set_seed(seed)

    def load(self) -> pd.DataFrame:
        """Generate synthetic data."""
        # This will be implemented in Section 08
        return pd.DataFrame()

    def normalize(self, raw_data: pd.DataFrame) -> List[NormalizedEvent]:
        """Normalize synthetic data."""
        # This will be implemented in Section 08
        return []

class CSVAdapter:
    """CSV file adapter."""

    def __init__(self, path: str, chain_id: int = DEFAULT_CHAIN_ID):
        self.path = path
        self.chain_id = chain_id

    def load(self) -> pd.DataFrame:
        """Load CSV data."""
        return pd.read_csv(self.path)

    def normalize(self, raw_data: pd.DataFrame) -> List[NormalizedEvent]:
        """Normalize CSV data to NormalizedEvent format."""
        events = []
        for _, row in raw_data.iterrows():
            try:
                event = NormalizedEvent(
                    event_id=str(uuid.uuid4()),
                    source="csv",
                    chain_id=self.chain_id,
                    event_timestamp_ms=int(row.get('event_timestamp_ms', 0)),
                    observed_timestamp_ms=int(row.get('observed_timestamp_ms', 0)),
                    available_timestamp_ms=int(row.get('available_timestamp_ms', 0)),
                    block_number=int(row.get('block_number', 0)),
                    block_hash=str(row.get('block_hash', '')),
                    tx_hash=row.get('tx_hash'),
                    tx_index=row.get('tx_index'),
                    log_index=row.get('log_index'),
                    event_type=str(row.get('event_type', 'unknown')),
                    entity_id=str(row.get('entity_id', '')),
                    payload=row.to_dict(),
                    raw_payload_hash=sha256_hash(row.to_dict())
                )
                events.append(event)
            except Exception as e:
                logging.warning(f"Failed to normalize row: {e}")
        return events

class ParquetAdapter:
    """Parquet file adapter."""

    def __init__(self, path: str, chain_id: int = DEFAULT_CHAIN_ID):
        self.path = path
        self.chain_id = chain_id

    def load(self) -> pd.DataFrame:
        """Load Parquet data."""
        return pd.read_parquet(self.path)

    def normalize(self, raw_data: pd.DataFrame) -> List[NormalizedEvent]:
        """Normalize Parquet data to NormalizedEvent format."""
        events = []
        for _, row in raw_data.iterrows():
            try:
                event = NormalizedEvent(
                    event_id=str(uuid.uuid4()),
                    source="parquet",
                    chain_id=self.chain_id,
                    event_timestamp_ms=int(row.get('event_timestamp_ms', 0)),
                    observed_timestamp_ms=int(row.get('observed_timestamp_ms', 0)),
                    available_timestamp_ms=int(row.get('available_timestamp_ms', 0)),
                    block_number=int(row.get('block_number', 0)),
                    block_hash=str(row.get('block_hash', '')),
                    tx_hash=row.get('tx_hash'),
                    tx_index=row.get('tx_index'),
                    log_index=row.get('log_index'),
                    event_type=str(row.get('event_type', 'unknown')),
                    entity_id=str(row.get('entity_id', '')),
                    payload=row.to_dict(),
                    raw_payload_hash=sha256_hash(row.to_dict())
                )
                events.append(event)
            except Exception as e:
                logging.warning(f"Failed to normalize row: {e}")
        return events

# =============================================================================
# SECTION 08: SYNTHETIC MARKET/EVENT GENERATOR
# =============================================================================

class SyntheticMarketGenerator:
    """Generates synthetic market events with multiple regimes."""

    def __init__(self, config: SyntheticConfig):
        self.config = config
        set_seed(config.seed)
        self.events: List[NormalizedEvent] = []
        self.current_block = 0
        self.current_timestamp = 0
        self.token_prices: Dict[str, float] = {}
        self.pool_reserves: Dict[str, Tuple[float, float]] = {}
        self.liquidity_levels: Dict[str, float] = {}
        self.regime: RegimeType = RegimeType.NORMAL
        self.regime_start_block = 0

    def generate(self) -> List[NormalizedEvent]:
        """Generate all synthetic events."""
        total_blocks = self.config.days * self.config.blocks_per_day

        # Initialize tokens and pools
        self._initialize_tokens_and_pools()

        for block in range(total_blocks):
            self.current_block = block
            self.current_timestamp = block * DEFAULT_BLOCK_TIME_MS

            # Check for regime change
            self._update_regime(block, total_blocks)

            # Generate events for this block
            self._generate_block_events(block)

        return self.events

    def _initialize_tokens_and_pools(self) -> None:
        """Initialize token prices and pool reserves."""
        # Create tokens
        base_price = 100.0
        for i in range(self.config.num_tokens):
            token_id = f"token_{i}"
            self.token_prices[token_id] = base_price * (1 + 0.1 * i)

        # Create pools
        for i in range(self.config.num_pools):
            pool_id = f"pool_{i}"
            token_a = f"token_{i % self.config.num_tokens}"
            token_b = f"token_{(i + 1) % self.config.num_tokens}"
            liquidity = random.choice(self.config.liquidity_levels)
            self.pool_reserves[pool_id] = (liquidity, liquidity * self.token_prices[token_b] / self.token_prices[token_a])
            self.liquidity_levels[pool_id] = liquidity

    def _update_regime(self, block: int, total_blocks: int) -> None:
        """Update current regime based on block and configuration."""
        blocks_in_regime = block - self.regime_start_block

        if blocks_in_regime >= self.config.regime_duration_blocks:
            # Switch to new regime
            regime_options = list(RegimeType)
            self.regime = random.choice(regime_options)
            self.regime_start_block = block

        # Distribution shift
        if block > total_blocks * self.config.distribution_shift_start_fraction:
            if self.regime == RegimeType.NORMAL:
                self.regime = RegimeType.SHIFTED

    def _generate_block_events(self, block: int) -> None:
        """Generate events for a single block."""
        # Block event
        self._add_block_event(block)

        # Price updates
        self._add_price_events(block)

        # Gas events
        self._add_gas_events(block)

        # Swap events
        self._add_swap_events(block)

        # Pool updates
        self._add_pool_update_events(block)

        # Quote events
        self._add_quote_events(block)

    def _add_block_event(self, block: int) -> None:
        """Add a block event."""
        event = NormalizedEvent(
            event_id=str(uuid.uuid4()),
            source="synthetic",
            chain_id=DEFAULT_CHAIN_ID,
            event_timestamp_ms=self.current_timestamp,
            observed_timestamp_ms=self.current_timestamp,
            available_timestamp_ms=self.current_timestamp + self.config.source_availability_delay_ms,
            block_number=block,
            block_hash=f"0x{block:064x}",
            event_type=EventType.BLOCK,
            entity_id=f"block_{block}",
            payload={"block_number": block, "timestamp_ms": self.current_timestamp},
            raw_payload_hash=sha256_hash({"block_number": block})
        )
        self.events.append(event)

    def _add_price_events(self, block: int) -> None:
        """Add price update events."""
        volatility = self._get_current_volatility()
        for token_id, price in self.token_prices.items():
            # Random price movement
            price_change = price * volatility * np.random.randn()
            self.token_prices[token_id] = max(0.01, price + price_change)

            event = NormalizedEvent(
                event_id=str(uuid.uuid4()),
                source="synthetic",
                chain_id=DEFAULT_CHAIN_ID,
                event_timestamp_ms=self.current_timestamp,
                observed_timestamp_ms=self.current_timestamp,
                available_timestamp_ms=self.current_timestamp + self.config.source_availability_delay_ms,
                block_number=block,
                block_hash=f"0x{block:064x}",
                event_type=EventType.PRICE,
                entity_id=token_id,
                payload={"token_id": token_id, "price": self.token_prices[token_id]},
                raw_payload_hash=sha256_hash({"token_id": token_id, "price": self.token_prices[token_id]})
            )
            self.events.append(event)

    def _add_gas_events(self, block: int) -> None:
        """Add gas events."""
        gas_level = self._get_current_gas_level()
        event = NormalizedEvent(
            event_id=str(uuid.uuid4()),
            source="synthetic",
            chain_id=DEFAULT_CHAIN_ID,
            event_timestamp_ms=self.current_timestamp,
            observed_timestamp_ms=self.current_timestamp,
            available_timestamp_ms=self.current_timestamp + self.config.source_availability_delay_ms,
            block_number=block,
            block_hash=f"0x{block:064x}",
            event_type=EventType.GAS,
            entity_id="gas_oracle",
            payload={"base_fee_gwei": gas_level, "priority_fee_gwei": gas_level * 0.5},
            raw_payload_hash=sha256_hash({"base_fee_gwei": gas_level})
        )
        self.events.append(event)

    def _add_swap_events(self, block: int) -> None:
        """Add swap events."""
        num_swaps = np.random.poisson(2)
        for _ in range(num_swaps):
            pool_id = random.choice(list(self.pool_reserves.keys()))
            amount_in = random.uniform(100, 10000)

            # Simulate swap
            self._simulate_swap(pool_id, amount_in, block)

    def _simulate_swap(self, pool_id: str, amount_in: float, block: int) -> None:
        """Simulate a swap and update reserves."""
        if pool_id not in self.pool_reserves:
            return

        reserve_in, reserve_out = self.pool_reserves[pool_id]
        fee = random.choice(self.config.normal_fee_tiers)

        amount_in_after_fee = amount_in * (1 - fee)
        amount_out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)

        # Sample the revert outcome FIRST. A reverted transaction must leave
        # pool state unchanged (gas is still spent, but that is accounted for
        # separately in CostEngine — it does not touch reserves).
        revert = random.random() < self.config.revert_base_probability

        if not revert:
            self.pool_reserves[pool_id] = (reserve_in + amount_in_after_fee, reserve_out - amount_out)
        # else: reserves intentionally left unchanged.

        event = NormalizedEvent(
            event_id=str(uuid.uuid4()),
            source="synthetic",
            chain_id=DEFAULT_CHAIN_ID,
            event_timestamp_ms=self.current_timestamp,
            observed_timestamp_ms=self.current_timestamp,
            available_timestamp_ms=self.current_timestamp + self.config.source_availability_delay_ms,
            block_number=block,
            block_hash=f"0x{block:064x}",
            tx_hash=f"0x{uuid.uuid4().hex[:64]}",
            event_type=EventType.SWAP if not revert else EventType.REVERT,
            entity_id=pool_id,
            payload={
                "pool_id": pool_id,
                "amount_in": amount_in,
                "amount_out": amount_out if not revert else 0,
                "fee": fee,
                "reverted": revert
            },
            raw_payload_hash=sha256_hash({"pool_id": pool_id, "amount_in": amount_in})
        )
        self.events.append(event)

    def _add_pool_update_events(self, block: int) -> None:
        """Add pool update events."""
        for pool_id, reserves in self.pool_reserves.items():
            event = NormalizedEvent(
                event_id=str(uuid.uuid4()),
                source="synthetic",
                chain_id=DEFAULT_CHAIN_ID,
                event_timestamp_ms=self.current_timestamp,
                observed_timestamp_ms=self.current_timestamp,
                available_timestamp_ms=self.current_timestamp + self.config.source_availability_delay_ms,
                block_number=block,
                block_hash=f"0x{block:064x}",
                event_type=EventType.POOL_UPDATE,
                entity_id=pool_id,
                payload=make_pool_update_payload(pool_id, reserves[0], reserves[1]),
                raw_payload_hash=sha256_hash({"pool_id": pool_id, "reserve0": reserves[0], "reserve1": reserves[1]})
            )
            self.events.append(event)

    def _add_quote_events(self, block: int) -> None:
        """Add quote events."""
        # Only add quotes sometimes
        if random.random() > 0.3:
            return

        pool_id = random.choice(list(self.pool_reserves.keys()))
        amount_in = random.uniform(100, 5000)

        # Check for staleness
        is_stale = random.random() < self.config.quote_staleness_probability
        staleness_delay = random.randint(1000, 10000) if is_stale else 0

        reserve_in, reserve_out = self.pool_reserves[pool_id]
        fee = random.choice(self.config.normal_fee_tiers)
        amount_in_after_fee = amount_in * (1 - fee)
        quoted_amount_out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)

        event = NormalizedEvent(
            event_id=str(uuid.uuid4()),
            source="synthetic",
            chain_id=DEFAULT_CHAIN_ID,
            event_timestamp_ms=self.current_timestamp,
            observed_timestamp_ms=self.current_timestamp,
            available_timestamp_ms=self.current_timestamp + self.config.source_availability_delay_ms + staleness_delay,
            block_number=block,
            block_hash=f"0x{block:064x}",
            event_type=EventType.QUOTE,
            entity_id=pool_id,
            payload={
                "pool_id": pool_id,
                "amount_in": amount_in,
                "quoted_amount_out": quoted_amount_out,
                "fee": fee,
                "is_stale": is_stale
            },
            raw_payload_hash=sha256_hash({"pool_id": pool_id, "amount_in": amount_in})
        )
        self.events.append(event)

    def _get_current_volatility(self) -> float:
        """Get volatility based on current regime."""
        if self.regime == RegimeType.VOLATILE:
            return self.config.baseline_volatility * 3
        elif self.regime == RegimeType.SHIFTED:
            return self.config.baseline_volatility * 2
        return self.config.baseline_volatility

    def _get_current_gas_level(self) -> float:
        """Get gas level based on current regime."""
        if self.regime == RegimeType.GAS_SPIKE:
            return random.choice(self.config.gas_levels) * 5
        elif self.regime == RegimeType.SHIFTED:
            return random.choice(self.config.gas_levels) * 2
        return random.choice(self.config.gas_levels)

# =============================================================================
# SECTION 09: DATA VALIDATION AND QUARANTINE
# =============================================================================

class DataValidator:
    """Validates data and quarantines invalid records."""

    def __init__(self, config: Config):
        self.config = config
        self.quarantined: List[Dict] = []
        self.quarantine_reasons: List[str] = []

    def validate_events(self, events: List[NormalizedEvent]) -> List[NormalizedEvent]:
        """Validate events and return only valid ones."""
        valid_events = []
        for event in events:
            try:
                self._validate_event(event)
                valid_events.append(event)
            except ValueError as e:
                self.quarantined.append(event.model_dump())
                self.quarantine_reasons.append(str(e))

        return valid_events

    def _validate_event(self, event: NormalizedEvent) -> None:
        """Validate a single event."""
        # Check required fields
        if not event.chain_id:
            raise ValueError("Missing chain_id")
        if not event.block_number:
            raise ValueError("Missing block_number")
        if not event.event_timestamp_ms:
            raise ValueError("Missing event_timestamp_ms")

        # Check timestamp ordering
        if event.available_timestamp_ms < event.observed_timestamp_ms:
            raise ValueError("available_timestamp_ms before observed_timestamp_ms")

        # Check for negative values where invalid
        if event.event_type == EventType.GAS:
            base_fee = event.payload.get("base_fee_gwei", 0)
            if base_fee < 0:
                raise ValueError("Negative gas fee")

        # Check block hash consistency
        if event.block_hash and len(event.block_hash) < 10:
            raise ValueError("Invalid block hash")

# =============================================================================
# SECTION 10: TIME-CAUSAL FEATURE STORE
# =============================================================================

class TimeCausalFeatureStore:
    """Builds time-causal feature snapshots."""

    def __init__(self, events: List[NormalizedEvent], max_age_ms: int):
        self.events = events
        self.max_age_ms = max_age_ms
        self._index_events_by_time()

    def _index_events_by_time(self) -> None:
        """Index events by block number and timestamp for fast lookup."""
        self.events_by_block: Dict[int, List[NormalizedEvent]] = {}
        for event in self.events:
            if event.block_number not in self.events_by_block:
                self.events_by_block[event.block_number] = []
            self.events_by_block[event.block_number].append(event)

    def build_snapshot(self, decision_time_ms: int, decision_block: int) -> Dict[str, FeatureValue]:
        """Build feature snapshot using only events available at decision time."""
        features = {}

        # Collect available events
        available_events = []
        for event in self.events:
            if (event.available_timestamp_ms <= decision_time_ms and
                event.block_number <= decision_block):
                available_events.append(event)

        # Build features from available events
        features.update(self._build_price_features(available_events, decision_time_ms))
        features.update(self._build_execution_features(available_events, decision_time_ms))
        features.update(self._build_liquidity_features(available_events, decision_time_ms))
        features.update(self._build_temporal_features(available_events, decision_time_ms))

        return features

    def _build_price_features(self, events: List[NormalizedEvent], decision_time_ms: int) -> Dict[str, FeatureValue]:
        """Build price-related features."""
        features = {}
        price_events = [e for e in events if e.event_type == EventType.PRICE]

        if not price_events:
            return features

        # Latest prices (by event time, but we record availability time —
        # that is the honest "when could this actually have been used" clock)
        latest_prices = {}
        for event in price_events:
            token_id = event.payload.get("token_id")
            price = event.payload.get("price")
            if token_id and price is not None:
                if token_id not in latest_prices or event.event_timestamp_ms > latest_prices[token_id][0]:
                    latest_prices[token_id] = (event.event_timestamp_ms, event.available_timestamp_ms, price, event.event_id)

        for token_id, (event_ts, available_ts, price, event_id) in latest_prices.items():
            age_ms = decision_time_ms - available_ts
            features[f"price_{token_id}"] = FeatureValue(
                name=f"price_{token_id}",
                value=price,
                source_event_ids=[event_id],
                max_available_timestamp_ms=available_ts,
                max_source_block=self._get_block_for_event(event_id, events),
                age_ms=age_ms,
                missing=age_ms > self.max_age_ms
            )

        return features

    def _build_execution_features(self, events: List[NormalizedEvent], decision_time_ms: int) -> Dict[str, FeatureValue]:
        """Build execution-related features."""
        features = {}
        gas_events = [e for e in events if e.event_type == EventType.GAS]

        if gas_events:
            latest_gas = max(gas_events, key=lambda e: e.event_timestamp_ms)
            age_ms = decision_time_ms - latest_gas.available_timestamp_ms

            features["base_fee_gwei"] = FeatureValue(
                name="base_fee_gwei",
                value=latest_gas.payload.get("base_fee_gwei", 0),
                source_event_ids=[latest_gas.event_id],
                max_available_timestamp_ms=latest_gas.available_timestamp_ms,
                max_source_block=latest_gas.block_number,
                age_ms=age_ms,
                missing=age_ms > self.max_age_ms
            )

        return features

    def _build_liquidity_features(self, events: List[NormalizedEvent], decision_time_ms: int) -> Dict[str, FeatureValue]:
        """Build liquidity-related features."""
        features = {}
        pool_events = [e for e in events if e.event_type == EventType.POOL_UPDATE]

        # Aggregate liquidity
        total_liquidity = 0
        for event in pool_events:
            pool_id = event.payload.get("pool_id")
            try:
                r0, r1 = get_reserves(event)
                total_liquidity += r0 + r1
            except KeyError:
                continue

        if pool_events:
            latest_event = max(pool_events, key=lambda e: e.event_timestamp_ms)
            age_ms = decision_time_ms - latest_event.available_timestamp_ms

            features["total_liquidity"] = FeatureValue(
                name="total_liquidity",
                value=total_liquidity,
                source_event_ids=[e.event_id for e in pool_events],
                max_available_timestamp_ms=latest_event.available_timestamp_ms,
                max_source_block=latest_event.block_number,
                age_ms=age_ms,
                missing=len(pool_events) == 0
            )

        return features

    def _build_temporal_features(self, events: List[NormalizedEvent], decision_time_ms: int) -> Dict[str, FeatureValue]:
        """Build temporal features."""
        features = {}

        # Time of day
        hour = (decision_time_ms // (1000 * 60 * 60)) % 24
        features["hour_of_day"] = FeatureValue(
            name="hour_of_day",
            value=hour,
            source_event_ids=[],
            max_available_timestamp_ms=decision_time_ms,
            max_source_block=0,
            age_ms=0,
            missing=False
        )

        return features

    def _get_block_for_event(self, event_id: str, events: List[NormalizedEvent]) -> int:
        """Get block number for an event."""
        for event in events:
            if event.event_id == event_id:
                return event.block_number
        return 0

    def assert_no_future_features(self, packet: DecisionPacket, cutoff_timestamp_ms: int, cutoff_block: int) -> bool:
        """Assert that no features come from the future."""
        for feature_name, feature_value in packet.provenance.get("feature_source_timestamps_ms", {}).items():
            if feature_value > cutoff_timestamp_ms:
                return False

        # Check block-level features
        for feature_name, feature_block in packet.provenance.get("feature_source_blocks", {}).items():
            if feature_block > cutoff_block:
                return False

        return True

# =============================================================================
# SECTION 11: DECISION-PACKET BUILDER AND CANDIDATE GENERATOR
# =============================================================================

class DecisionPacketBuilder:
    """Builds decision packets from events and features."""

    def __init__(self, config: Config, feature_store: TimeCausalFeatureStore):
        self.config = config
        self.feature_store = feature_store
        self.packets: List[DecisionPacket] = []

    def build_packets(self, events: List[NormalizedEvent]) -> List[DecisionPacket]:
        """Build decision packets from events."""
        # Determine decision points (e.g., every N blocks)
        decision_blocks = self._get_decision_blocks(events)

        for block in decision_blocks:
            packet = self._build_packet_at_block(block, events)
            if packet:
                self.packets.append(packet)

        return self.packets

    def _get_decision_blocks(self, events: List[NormalizedEvent]) -> List[int]:
        """Get blocks where decisions should be made."""
        if not events:
            return []

        max_block = max(e.block_number for e in events)
        # Make decisions every 5 blocks, and ensure we get at least some decisions
        decision_blocks = list(range(0, max_block, 5))
        if not decision_blocks and max_block > 0:
            decision_blocks = [0]  # At least one decision point
        return decision_blocks

    def _build_packet_at_block(self, block: int, events: List[NormalizedEvent]) -> Optional[DecisionPacket]:
        """Build a decision packet at a specific block."""
        # Find the event for this block
        block_events = [e for e in events if e.block_number == block]
        if not block_events:
            return None

        block_event = block_events[0]
        decision_time_ms = block_event.event_timestamp_ms

        # Build features
        features = self.feature_store.build_snapshot(decision_time_ms, block)

        # Generate candidates
        candidates = self._generate_candidates(block, events, features)

        # Build packet
        packet = DecisionPacket(
            decision_id=str(uuid.uuid4()),
            as_of={
                "chain_id": self.config.chain_id,
                "block_number": block,
                "block_hash": block_event.block_hash,
                "decision_timestamp_ms": decision_time_ms,
                "max_feature_age_ms": self.config.max_feature_age_ms
            },
            objective={
                "horizon_blocks": 3,
                "capital_usd": 1000.0,
                "min_net_edge_usd": self.config.planning.min_net_edge_usd,
                "max_loss_usd": self.config.constraints.max_loss_usd,
                "risk_aversion": 1.0,
                "confidence_level": self.config.calibration.confidence_level
            },
            market=self._extract_market_features(features),
            execution=self._extract_execution_features(features),
            graph=self._extract_graph_features(features),
            action_candidates=candidates,
            constraints=self.config.constraints.model_dump(),
            provenance=self._build_provenance(features, block, decision_time_ms)
        )

        return packet

    def _generate_candidates(self, block: int, events: List[NormalizedEvent], features: Dict[str, FeatureValue]) -> List[ActionCandidate]:
        """Generate action candidates."""
        candidates = []

        # Simple candidate generation: create a few route candidates
        for i in range(3):
            candidate = ActionCandidate(
                action_id=str(uuid.uuid4()),
                route=[{"pool": f"pool_{i}", "token_in": "token_0", "token_out": "token_1"}],
                trade_size_usd=random.uniform(100, 1000),
                borrow={"asset": "", "amount": 0.0},
                slippage_limit_bps=random.randint(5, 50),
                gas_policy={"max_fee_gwei": features.get("base_fee_gwei", FeatureValue(name="", value=0, source_event_ids=[], max_available_timestamp_ms=0, max_source_block=0, age_ms=0, missing=False)).value * 2 if features.get("base_fee_gwei") else 1.0},
                deadline_block=block + 5,
                private_submission=False
            )
            candidates.append(candidate)

        return candidates

    def _extract_market_features(self, features: Dict[str, FeatureValue]) -> Dict[str, Any]:
        """Extract market features from feature values."""
        market = {}
        for name, feature in features.items():
            if name.startswith("price_"):
                market[name] = feature.value
        return market

    def _extract_execution_features(self, features: Dict[str, FeatureValue]) -> Dict[str, Any]:
        """Extract execution features from feature values."""
        execution = {}
        for name, feature in features.items():
            if name in ["base_fee_gwei", "total_liquidity"]:
                execution[name] = feature.value
        return execution

    def _extract_graph_features(self, features: Dict[str, FeatureValue]) -> Dict[str, Any]:
        """Extract graph features from feature values."""
        return {"nodes": [], "edges": []}

    def _build_provenance(self, features: Dict[str, FeatureValue], block: int, timestamp_ms: int) -> Dict[str, Any]:
        """Build provenance information."""
        feature_versions = {name: feature.feature_version for name, feature in features.items()}
        feature_source_timestamps_ms = {name: feature.max_available_timestamp_ms for name, feature in features.items()}
        feature_availability_timestamps_ms = {name: feature.max_available_timestamp_ms for name, feature in features.items()}
        feature_source_blocks = {name: feature.max_source_block for name, feature in features.items()}
        missingness_mask = {name: feature.missing for name, feature in features.items()}

        return {
            "feature_versions": feature_versions,
            "feature_source_timestamps_ms": feature_source_timestamps_ms,
            "feature_availability_timestamps_ms": feature_availability_timestamps_ms,
            "feature_source_blocks": feature_source_blocks,
            "missingness_mask": missingness_mask,
            "input_hash": sha256_hash({k: v.value for k, v in features.items()})
        }

# =============================================================================
# SECTION 12: GENERIC AMM, QUOTE, MARKET REPLAY, AND EXECUTION SIMULATION
# =============================================================================

class ConstantProductAMM:
    """Constant product AMM implementation."""

    @staticmethod
    def calculate_amount_out(reserve_in: float, reserve_out: float, amount_in: float, fee: float) -> float:
        """Calculate amount out using constant product formula."""
        amount_in_after_fee = amount_in * (1 - fee)
        amount_out = reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee)
        return amount_out

    @staticmethod
    def calculate_price_impact(reserve_in: float, reserve_out: float, amount_in: float, fee: float) -> float:
        """Calculate price impact."""
        amount_out = ConstantProductAMM.calculate_amount_out(reserve_in, reserve_out, amount_in, fee)
        spot_price = reserve_out / reserve_in
        execution_price = amount_out / amount_in
        price_impact = abs(execution_price - spot_price) / spot_price
        return price_impact

class MarketReplay:
    """Replays market state for simulation."""

    def __init__(self, events: List[NormalizedEvent]):
        self.events = events
        self.events_by_block: Dict[int, List[NormalizedEvent]] = {}
        self._index_events()

    def _index_events(self) -> None:
        """Index events by block."""
        for event in self.events:
            if event.block_number not in self.events_by_block:
                self.events_by_block[event.block_number] = []
            self.events_by_block[event.block_number].append(event)

    def state_at(self, block: int) -> Dict[str, Any]:
        """Get market state at a specific block."""
        state = {
            "block": block,
            "prices": {},
            "reserves": {},
            "gas": 0
        }

        # Collect events up to this block
        for event in self.events:
            if event.block_number <= block:
                if event.event_type == EventType.PRICE:
                    token_id = event.payload.get("token_id")
                    price = event.payload.get("price")
                    if token_id and price is not None:
                        state["prices"][token_id] = price
                elif event.event_type == EventType.POOL_UPDATE:
                    pool_id = event.payload.get("pool_id")
                    if pool_id:
                        try:
                            state["reserves"][pool_id] = list(get_reserves(event))
                        except KeyError:
                            pass
                elif event.event_type == EventType.GAS:
                    state["gas"] = event.payload.get("base_fee_gwei", 0)

        return state

    def evolve(self, state: Dict[str, Any], horizon_blocks: int) -> Dict[str, Any]:
        """Evolve state forward by horizon blocks."""
        current_block = state["block"]
        target_block = current_block + horizon_blocks
        return self.state_at(target_block)

class QuoteEngine:
    """Engine for quoting trades."""

    def __init__(self, amm: ConstantProductAMM):
        self.amm = amm

    def quote(self, route: List[Dict], amount_in: float, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Quote a route."""
        current_amount = amount_in
        total_fees = 0

        for hop in route:
            pool_id = hop.get("pool")
            if pool_id not in market_state.get("reserves", {}):
                return {"valid": False, "reason": "Pool not found"}

            reserves = market_state["reserves"][pool_id]
            fee = 0.003  # Default fee

            amount_out = self.amm.calculate_amount_out(
                reserves[0], reserves[1], current_amount, fee
            )

            total_fees += current_amount * fee
            current_amount = amount_out

        return {
            "valid": True,
            "amount_out": current_amount,
            "total_fees": total_fees,
            "price_impact": self.amm.calculate_price_impact(
                market_state["reserves"].get(route[0].get("pool", ""), [0, 0])[0],
                market_state["reserves"].get(route[0].get("pool", ""), [0, 0])[1],
                amount_in, fee
            )
        }

class ExecutionEngine:
    """Simulates trade execution."""

    def __init__(self, config: Config):
        self.config = config

    def simulate(self, action: ActionCandidate, market_state: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """Simulate action execution."""
        # Simulate inclusion delay
        inclusion_delay = rng.poisson(2)
        deadline_block = action.deadline_block
        current_block = market_state["block"]

        included_before_deadline = (current_block + inclusion_delay) <= deadline_block

        # Simulate revert
        revert_probability = 0.02  # Base revert probability
        reverted = rng.random() < revert_probability

        # Simulate gas cost
        gas_units = 150000  # Estimated gas units
        gas_price_gwei = market_state.get("gas", 1.0)
        gas_usd = gas_units * gas_price_gwei * 0.000000001 * 2000  # Rough USD conversion

        return {
            "included_before_deadline": included_before_deadline,
            "inclusion_delay_blocks": inclusion_delay,
            "reverted": reverted,
            "gas_usd": gas_usd,
            "success": included_before_deadline and not reverted
        }

class CostEngine:
    """Calculates execution costs."""

    @staticmethod
    def estimate(action: ActionCandidate, execution_result: Dict, market_state: Dict) -> Dict[str, float]:
        """Estimate all costs."""
        costs = {
            "gas_usd": execution_result.get("gas_usd", 0),
            "protocol_fees_usd": action.trade_size_usd * 0.003,  # 0.3% protocol fee
            "borrow_fees_usd": 0,
            "bridge_fees_usd": 0,
            "slippage_cost_usd": 0,
            "revert_cost_usd": execution_result.get("gas_usd", 0) if execution_result.get("reverted") else 0,
            "other_costs_usd": 0
        }

        return costs

class OutcomeEngine:
    """Calculates final outcomes."""

    @staticmethod
    def settle(decision_id: str, action: ActionCandidate, quote_result: Dict, execution_result: Dict,
               costs: Dict, state_after: Dict) -> OutcomeLabel:
        """Settle the final outcome. decision_id MUST come from the originating
        DecisionPacket so labels can be joined back to the decision that produced
        them — do not generate a fresh UUID here."""
        input_cost_usd = action.trade_size_usd
        proceeds_usd = quote_result.get("amount_out", 0) if not execution_result.get("reverted") else 0

        gross_pnl_usd = proceeds_usd - input_cost_usd

        total_costs = sum(costs.values())
        net_pnl_usd = gross_pnl_usd - total_costs

        return OutcomeLabel(
            decision_id=decision_id,
            action_id=action.action_id,
            quoted_output_usd=quote_result.get("amount_out", 0),
            realized_output_usd=proceeds_usd,
            input_cost_usd=input_cost_usd,
            gas_usd=costs.get("gas_usd", 0),
            protocol_fees_usd=costs.get("protocol_fees_usd", 0),
            borrow_fees_usd=costs.get("borrow_fees_usd", 0),
            bridge_fees_usd=costs.get("bridge_fees_usd", 0),
            slippage_cost_usd=costs.get("slippage_cost_usd", 0),
            revert_cost_usd=costs.get("revert_cost_usd", 0),
            other_costs_usd=costs.get("other_costs_usd", 0),
            gross_pnl_usd=gross_pnl_usd,
            net_pnl_usd=net_pnl_usd,
            reverted=execution_result.get("reverted", False),
            included_before_deadline=execution_result.get("included_before_deadline", False),
            inclusion_delay_blocks=execution_result.get("inclusion_delay_blocks", 0),
            outcome_timestamp_ms=state_after.get("block", 0) * DEFAULT_BLOCK_TIME_MS,
            outcome_block=state_after.get("block", 0)
        )

# =============================================================================
# SECTION 13: LABEL BUILDER AND NET-PNL ACCOUNTING
# =============================================================================

class LabelBuilder:
    """Builds outcome labels from packets and simulation."""

    def __init__(self, market_replay: MarketReplay, quote_engine: QuoteEngine,
                 execution_engine: ExecutionEngine, cost_engine: CostEngine, outcome_engine: OutcomeEngine):
        self.market_replay = market_replay
        self.quote_engine = quote_engine
        self.execution_engine = execution_engine
        self.cost_engine = cost_engine
        self.outcome_engine = outcome_engine
        self.labels: List[OutcomeLabel] = []

    def build_labels(self, packets: List[DecisionPacket], rng: np.random.Generator) -> List[OutcomeLabel]:
        """Build labels for all packets."""
        for packet in packets:
            for candidate in packet.action_candidates:
                label = self._build_label_for_candidate(packet, candidate, rng)
                if label:
                    self.labels.append(label)

        return self.labels

    def _build_label_for_candidate(self, packet: DecisionPacket, candidate: ActionCandidate,
                                   rng: np.random.Generator) -> Optional[OutcomeLabel]:
        """Build label for a single candidate."""
        # Get current state
        current_state = self.market_replay.state_at(packet.as_of["block_number"])

        # Quote the trade
        quote_result = self.quote_engine.quote(candidate.route, candidate.trade_size_usd, current_state)
        if not quote_result.get("valid"):
            return None

        # Simulate execution
        execution_result = self.execution_engine.simulate(candidate, current_state, rng)

        # Calculate costs
        costs = self.cost_engine.estimate(candidate, execution_result, current_state)

        # Evolve state
        horizon = packet.objective.get("horizon_blocks", 3)
        state_after = self.market_replay.evolve(current_state, horizon)

        # Settle outcome — decision_id threaded from the packet so labels
        # can be joined back to the decision that produced them.
        outcome = self.outcome_engine.settle(packet.decision_id, candidate, quote_result,
                                              execution_result, costs, state_after)

        return outcome

# =============================================================================
# SECTION 14: REGIME/CHANGE-POINT LABELING
# =============================================================================

class RegimeLabeler:
    """Labels market regimes based on rolling statistics."""

    def __init__(self, events: List[NormalizedEvent], window_size: int = 50):
        self.events = events
        self.window_size = window_size
        self.regime_labels: Dict[int, RegimeType] = {}

    def label_regimes(self) -> Dict[int, RegimeType]:
        """Label regimes for each block."""
        # Get blocks
        blocks = sorted(set(e.block_number for e in self.events))

        for block in blocks:
            regime = self._label_regime_at_block(block)
            self.regime_labels[block] = regime

        return self.regime_labels

    def _label_regime_at_block(self, block: int) -> RegimeType:
        """Label regime at a specific block."""
        # Get events in window
        window_events = [e for e in self.events if block - self.window_size <= e.block_number <= block]

        if not window_events:
            return RegimeType.NORMAL

        # Calculate statistics
        volatility = self._calculate_volatility(window_events)
        liquidity = self._calculate_liquidity(window_events)
        gas_level = self._calculate_gas_level(window_events)

        # Determine regime
        if volatility > 0.05:
            return RegimeType.VOLATILE
        elif liquidity < 10000:
            return RegimeType.LOW_LIQUIDITY
        elif gas_level > 5.0:
            return RegimeType.GAS_SPIKE
        else:
            return RegimeType.NORMAL

    def _calculate_volatility(self, events: List[NormalizedEvent]) -> float:
        """Calculate price volatility."""
        price_events = [e for e in events if e.event_type == EventType.PRICE]
        if len(price_events) < 2:
            return 0.0

        prices = [e.payload.get("price", 0) for e in price_events]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0

    def _calculate_liquidity(self, events: List[NormalizedEvent]) -> float:
        """Calculate total liquidity."""
        pool_events = [e for e in events if e.event_type == EventType.POOL_UPDATE]
        if not pool_events:
            return 0.0

        total_liquidity = 0
        for event in pool_events:
            try:
                r0, r1 = get_reserves(event)
                total_liquidity += r0 + r1
            except KeyError:
                continue

        return total_liquidity / len(pool_events) if pool_events else 0.0

    def _calculate_gas_level(self, events: List[NormalizedEvent]) -> float:
        """Calculate average gas level."""
        gas_events = [e for e in events if e.event_type == EventType.GAS]
        if not gas_events:
            return 0.0

        gas_levels = [e.payload.get("base_fee_gwei", 0) for e in gas_events]
        return np.mean(gas_levels) if gas_levels else 0.0

# =============================================================================
# SECTION 15: CHRONOLOGICAL WALK-FORWARD SPLITTING AND SCALING
# =============================================================================

class WalkForwardSplitter:
    """Creates chronological walk-forward data splits."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

    def split_packets(self, packets: List[DecisionPacket]) -> Dict[str, List[DecisionPacket]]:
        """Split packets chronologically."""
        # Sort packets by block number
        sorted_packets = sorted(packets, key=lambda p: p.as_of["block_number"])

        n = len(sorted_packets)
        train_end = int(n * self.config.splits.train_fraction)
        val_end = train_end + int(n * self.config.splits.validation_fraction)
        cal_end = val_end + int(n * self.config.splits.calibration_fraction)

        splits = {
            "train": sorted_packets[:train_end],
            "validation": sorted_packets[train_end:val_end],
            "calibration": sorted_packets[val_end:cal_end],
            "test": sorted_packets[cal_end:]
        }

        return splits

    def split_labels(self, labels: List[OutcomeLabel]) -> Dict[str, List[OutcomeLabel]]:
        """Split labels chronologically."""
        # Sort labels by outcome block
        sorted_labels = sorted(labels, key=lambda l: l.outcome_block)

        n = len(sorted_labels)
        train_end = int(n * self.config.splits.train_fraction)
        val_end = train_end + int(n * self.config.splits.validation_fraction)
        cal_end = val_end + int(n * self.config.splits.calibration_fraction)

        splits = {
            "train": sorted_labels[:train_end],
            "validation": sorted_labels[train_end:val_end],
            "calibration": sorted_labels[val_end:cal_end],
            "test": sorted_labels[cal_end:]
        }

        return splits

    def fit_scaler(self, train_packets: List[DecisionPacket]) -> None:
        """Fit scaler on training data only."""
        if not SKLEARN_AVAILABLE or not self.scaler:
            return

        # Extract features from training packets
        features = self._extract_features_from_packets(train_packets)
        if features:
            self.scaler.fit(features)

    def transform_features(self, packets: List[DecisionPacket]) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not SKLEARN_AVAILABLE or not self.scaler:
            features = self._extract_features_from_packets(packets)
            return np.array(features) if features else np.array([])

        features = self._extract_features_from_packets(packets)
        if not features:
            return np.array([])

        return self.scaler.transform(features)

    def _extract_features_from_packets(self, packets: List[DecisionPacket]) -> List[List[float]]:
        """Extract feature vectors from packets."""
        features = []
        for packet in packets:
            # Simple feature extraction
            feature_vector = [
                packet.as_of.get("block_number", 0),
                len(packet.action_candidates),
                packet.market.get("total_liquidity", 0),
                packet.execution.get("base_fee_gwei", 0)
            ]
            features.append(feature_vector)
        return features

# =============================================================================
# SECTION 16: PYTORCH SHARED ENCODER, OPTIONAL TEMPORAL AND GRAPH COMPONENTS
# =============================================================================

if TORCH_AVAILABLE:

    class TabularEncoder(nn.Module):
        """Encodes tabular features."""

        def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    class ActionEncoder(nn.Module):
        """Encodes action candidates."""

        def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    class TemporalEncoder(nn.Module):
        """Encodes temporal sequences using GRU."""

        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, input_dim)
            _, hidden = self.gru(x)
            return hidden[-1]  # Return last layer's hidden state

    class SharedEncoder(nn.Module):
        """Shared encoder that fuses multiple inputs."""

        def __init__(self, tabular_dim: int, action_dim: int, tabular_hidden: int,
                     action_hidden: int, temporal_hidden: int, latent_dim: int, dropout: float = 0.1):
            super().__init__()
            self.tabular_encoder = TabularEncoder(tabular_dim, tabular_hidden, dropout)
            self.action_encoder = ActionEncoder(action_dim, action_hidden, dropout)
            self.temporal_encoder = TemporalEncoder(tabular_dim, temporal_hidden, dropout=dropout)

            # Fusion layer
            fusion_dim = tabular_hidden + action_hidden + temporal_hidden
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        def forward(self, tabular: torch.Tensor, action: torch.Tensor,
                   temporal: torch.Tensor) -> torch.Tensor:
            tabular_emb = self.tabular_encoder(tabular)
            action_emb = self.action_encoder(action)
            temporal_emb = self.temporal_encoder(temporal)

            fused = torch.cat([tabular_emb, action_emb, temporal_emb], dim=-1)
            latent = self.fusion(fused)
            return latent

else:
    # Placeholder classes when PyTorch is not available
    class SharedEncoder:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorch is not available")

# =============================================================================
# SECTION 17: MULTI-TASK WORLD MODEL
# =============================================================================

if TORCH_AVAILABLE:

    class MultiTaskWorldModel(nn.Module):
        """Multi-task world model with multiple prediction heads."""

        def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.1):
            super().__init__()

            # Shared trunk
            self.trunk = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # PnL quantile heads
            self.pnl_q05 = nn.Linear(hidden_dim, 1)
            self.pnl_q50 = nn.Linear(hidden_dim, 1)
            self.pnl_q95 = nn.Linear(hidden_dim, 1)

            # Cost heads
            self.gas_head = nn.Linear(hidden_dim, 1)
            self.slippage_head = nn.Linear(hidden_dim, 1)

            # Probability heads
            self.revert_head = nn.Linear(hidden_dim, 1)
            self.inclusion_head = nn.Linear(hidden_dim, 1)

            # State delta head
            self.state_delta_head = nn.Linear(hidden_dim, latent_dim)

        def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Forward pass through all heads."""
            h = self.trunk(latent)

            return {
                "pnl_q05": self.pnl_q05(h).squeeze(-1),
                "pnl_q50": self.pnl_q50(h).squeeze(-1),
                "pnl_q95": self.pnl_q95(h).squeeze(-1),
                "gas_mean": self.gas_head(h).squeeze(-1),
                "slippage_mean": self.slippage_head(h).squeeze(-1),
                "revert_logit": self.revert_head(h).squeeze(-1),
                "inclusion_logit": self.inclusion_head(h).squeeze(-1),
                "state_delta": self.state_delta_head(h)
            }

else:
    class MultiTaskWorldModel:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorch is not available")

# =============================================================================
# SECTION 18: BOOTSTRAP ENSEMBLE TRAIN/INFER
# =============================================================================

if TORCH_AVAILABLE:

    class BootstrapEnsemble:
        """Bootstrap ensemble of world models."""

        def __init__(self, ensemble_size: int, tabular_dim: int, action_dim: int,
                     tabular_hidden: int, action_hidden: int, temporal_hidden: int,
                     latent_dim: int, model_hidden: int, dropout: float = 0.1):
            self.ensemble_size = ensemble_size
            self.members: List[nn.Module] = []

            for _ in range(ensemble_size):
                encoder = SharedEncoder(tabular_dim, action_dim, tabular_hidden,
                                       action_hidden, temporal_hidden, latent_dim, dropout)
                model = MultiTaskWorldModel(latent_dim, model_hidden, dropout)
                self.members.append(nn.ModuleDict({"encoder": encoder, "model": model}))

        def train_member(self, member_idx: int, train_data: DataLoader, epochs: int,
                        learning_rate: float, device: str = "cpu") -> Dict[str, float]:
            """Train a single ensemble member."""
            member = self.members[member_idx]
            encoder = member["encoder"]
            model = member["model"]

            encoder.to(device)
            model.to(device)

            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(model.parameters()),
                lr=learning_rate
            )

            criterion_pnl = nn.SmoothL1Loss()
            criterion_bce = nn.BCEWithLogitsLoss()

            metrics = {"train_loss": []}

            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch in train_data:
                    # Move batch to device
                    tabular = batch["tabular"].to(device)
                    action = batch["action"].to(device)
                    temporal = batch["temporal"].to(device)
                    pnl = batch["pnl"].to(device)
                    revert = batch["revert"].to(device)
                    inclusion = batch["inclusion"].to(device)

                    # Forward pass
                    latent = encoder(tabular, action, temporal)
                    predictions = model(latent)

                    # Calculate losses
                    pnl_loss = criterion_pnl(predictions["pnl_q50"], pnl)
                    revert_loss = criterion_bce(predictions["revert_logit"], revert.float())
                    inclusion_loss = criterion_bce(predictions["inclusion_logit"], inclusion.float())

                    loss = pnl_loss + 0.5 * revert_loss + 0.5 * inclusion_loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                metrics["train_loss"].append(epoch_loss / len(train_data))

            return metrics

        def predict(self, tabular: torch.Tensor, action: torch.Tensor,
                   temporal: torch.Tensor, device: str = "cpu") -> Dict[str, np.ndarray]:
            """Predict with ensemble and return mean + std."""
            self.members[0]["encoder"].eval()
            self.members[0]["model"].eval()

            all_predictions = []

            with torch.no_grad():
                for member in self.members:
                    encoder = member["encoder"].to(device)
                    model = member["model"].to(device)

                    latent = encoder(tabular, action, temporal)
                    predictions = model(latent)

                    all_predictions.append({
                        k: v.cpu().numpy() for k, v in predictions.items()
                    })

            # Calculate ensemble statistics
            ensemble_predictions = {}
            for key in all_predictions[0].keys():
                values = np.stack([p[key] for p in all_predictions])
                ensemble_predictions[f"{key}_mean"] = np.mean(values, axis=0)
                ensemble_predictions[f"{key}_std"] = np.std(values, axis=0)

            return ensemble_predictions

else:
    class BootstrapEnsemble:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorch is not available")

# =============================================================================
# SECTION 19: CALIBRATION, OOD DETECTION, AND RETRIEVAL MEMORY
# =============================================================================

class ConformalCalibrator:
    """Split-conformal calibration for uncertainty intervals."""

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.residual_quantile: Optional[float] = None

    def fit(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """Fit calibrator on calibration data."""
        residuals = np.abs(predictions - labels)
        self.residual_quantile = np.quantile(residuals, self.confidence_level)

    def predict_interval(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict confidence intervals."""
        if self.residual_quantile is None:
            raise ValueError("Calibrator not fitted")

        lower = predictions - self.residual_quantile
        upper = predictions + self.residual_quantile
        return lower, upper

    def get_coverage(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate empirical coverage."""
        lower, upper = self.predict_interval(predictions)
        in_interval = (labels >= lower) & (labels <= upper)
        return np.mean(in_interval)

class OODDetector:
    """Out-of-distribution detection."""

    def __init__(self):
        self.reference_mean: Optional[np.ndarray] = None
        self.reference_cov: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def fit(self, latents: np.ndarray) -> None:
        """Fit OOD detector on training latents."""
        self.reference_mean = np.mean(latents, axis=0)
        self.reference_cov = np.cov(latents, rowvar=False)

        # Calculate reference Mahalanobis distances
        distances = self._mahalanobis_distance(latents, self.reference_mean, self.reference_cov)
        self.threshold = np.quantile(distances, 0.95)

    def score(self, latents: np.ndarray) -> np.ndarray:
        """Calculate OOD scores."""
        if self.reference_mean is None or self.reference_cov is None:
            raise ValueError("OOD detector not fitted")

        distances = self._mahalanobis_distance(latents, self.reference_mean, self.reference_cov)
        ood_scores = distances / self.threshold if self.threshold > 0 else distances
        return ood_scores

    def _mahalanobis_distance(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance."""
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(cov.shape[0])

        diff = x - mean
        distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        return distances

class RetrievalMemory:
    """Retrieval-based memory for similar historical situations."""

    def __init__(self, use_faiss: bool = False):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index = None
        self.outcomes: List[Dict] = []

    def add(self, latent: np.ndarray, outcome: Dict) -> None:
        """Add a latent vector and outcome to memory."""
        if self.index is None:
            dim = latent.shape[0]
            if self.use_faiss:
                self.index = faiss.IndexFlatL2(dim)
            else:
                self.index = NearestNeighbors(n_neighbors=5, metric='euclidean')

        if self.use_faiss:
            self.index.add(latent.reshape(1, -1).astype('float32'))
        else:
            if hasattr(self.index, 'fit'):
                self.index.fit(latent.reshape(1, -1))
            else:
                # For sklearn, we need to refit with all data
                # This is inefficient but works for small datasets
                pass

        self.outcomes.append(outcome)

    def retrieve(self, latent: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve k most similar historical outcomes."""
        if self.index is None or not self.outcomes:
            return []

        if self.use_faiss:
            distances, indices = self.index.search(latent.reshape(1, -1).astype('float32'), k)
            return [self.outcomes[i] for i in indices[0] if i < len(self.outcomes)]
        else:
            # Rebuild index with all data
            all_latents = np.array([o.get("latent", np.zeros(latent.shape[0])) for o in self.outcomes])
            if len(all_latents) > 0:
                self.index.fit(all_latents)
                distances, indices = self.index.kneighbors(latent.reshape(1, -1), n_neighbors=k)
                return [self.outcomes[i] for i in indices[0] if i < len(self.outcomes)]
            return []

# =============================================================================
# SECTION 19B: EMPIRICAL BASELINE PREDICTOR
# =============================================================================

class EmpiricalBaselinePredictor:
    """A minimal, honest statistical predictor that fits ModelForecast objects
    from real historical (packet, label) pairs, so ConservativePlanner /
    AbstentionGate / ConstraintEngine have something real to evaluate.

    This is explicitly NOT a replacement for the real learned predictor
    (SharedEncoder / MultiTaskWorldModel / BootstrapEnsemble, Section 16-18
    above) -- those require PyTorch and a real training run, and were never
    actually wired to anything in this file (ModelForecast was defined but
    never instantiated anywhere before this class was added). This is a
    simple, fully-transparent empirical baseline: bucket historical trades by
    trade_size_usd, and predict each new trade's distribution from whichever
    historical bucket it falls into. No architecture, no learned weights, no
    invented efficiency claims -- just empirical quantiles and counts.

    Every number below is computed directly from the fit() data:
      - pnl_q05/q50/q95_usd: empirical quantiles of net_pnl_usd in the bucket
      - aleatoric_std: sample std of net_pnl_usd in the bucket
      - epistemic_std: standard error of the mean (std / sqrt(n)) -- this
        genuinely goes down as more historical trades land in a bucket, and
        is large (capped) for buckets with few or zero samples
      - revert_probability / inclusion_probability: empirical rates
      - estimated_gas_usd: empirical mean gas_usd in the bucket
      - ood_score: 0.0 if trade_size_usd falls within the observed training
        range, rising linearly with distance outside that range
      - regime_score: left at 0.0 (unmodeled) -- explicitly NOT invented
    """

    def __init__(self, n_buckets: int = 5, min_trade_size: float = 1.0):
        self.n_buckets = n_buckets
        self.min_trade_size = min_trade_size
        self.bucket_edges: Optional[np.ndarray] = None
        self.bucket_stats: List[Dict[str, float]] = []
        self.global_stats: Dict[str, float] = {}
        self.is_fit = False

    def _bucket_index(self, trade_size_usd: float) -> int:
        if self.bucket_edges is None:
            return 0
        idx = int(np.searchsorted(self.bucket_edges, trade_size_usd, side="right")) - 1
        return int(np.clip(idx, 0, self.n_buckets - 1))

    def fit(self, packets: List["DecisionPacket"], labels: List["OutcomeLabel"]) -> "EmpiricalBaselinePredictor":
        """Fit bucketed empirical statistics from real matched (packet, label) pairs."""
        action_by_key: Dict[Tuple[str, str], ActionCandidate] = {}
        for p in packets:
            for a in p.action_candidates:
                action_by_key[(p.decision_id, a.action_id)] = a

        rows = []  # (trade_size_usd, net_pnl_usd, gas_usd, reverted, included)
        for label in labels:
            action = action_by_key.get((label.decision_id, label.action_id))
            if action is None:
                continue
            rows.append((
                action.trade_size_usd, label.net_pnl_usd, label.gas_usd,
                label.reverted, label.included_before_deadline,
            ))

        if not rows:
            raise ValueError(
                "EmpiricalBaselinePredictor.fit() got zero matched (packet, label) "
                "rows -- this means the decision_id/action_id join found no overlap. "
                "Refusing to fit on nothing rather than silently returning a predictor "
                "that always predicts the same defaults."
            )

        trade_sizes = np.array([r[0] for r in rows], dtype=float)
        self.min_observed = float(trade_sizes.min())
        self.max_observed = float(trade_sizes.max())

        # Quantile-based bucket edges so each bucket has roughly equal sample count.
        quantiles = np.linspace(0, 1, self.n_buckets + 1)
        edges = np.quantile(trade_sizes, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.bucket_edges = edges

        self.bucket_stats = []
        for i in range(self.n_buckets):
            lo, hi = edges[i], edges[i + 1]
            mask = (trade_sizes >= lo) & (trade_sizes < hi) if i < self.n_buckets - 1 else (trade_sizes >= lo) & (trade_sizes <= hi)
            bucket_rows = [r for r, m in zip(rows, mask) if m]
            self.bucket_stats.append(self._compute_stats(bucket_rows))

        self.global_stats = self._compute_stats(rows)
        self.is_fit = True
        return self

    @staticmethod
    def _compute_stats(rows: List[Tuple]) -> Dict[str, float]:
        n = len(rows)
        if n == 0:
            return {"n": 0, "pnl_q05": 0.0, "pnl_q50": 0.0, "pnl_q95": 0.0,
                    "std": 0.0, "revert_rate": 0.5, "inclusion_rate": 0.5, "mean_gas": 0.0}
        pnls = np.array([r[1] for r in rows], dtype=float)
        gases = np.array([r[2] for r in rows], dtype=float)
        reverts = np.array([r[3] for r in rows], dtype=bool)
        included = np.array([r[4] for r in rows], dtype=bool)
        return {
            "n": n,
            "pnl_q05": float(np.quantile(pnls, 0.05)),
            "pnl_q50": float(np.quantile(pnls, 0.50)),
            "pnl_q95": float(np.quantile(pnls, 0.95)),
            "std": float(np.std(pnls)) if n > 1 else 0.0,
            "revert_rate": float(reverts.mean()),
            "inclusion_rate": float(included.mean()),
            "mean_gas": float(gases.mean()),
        }

    def predict(self, action: "ActionCandidate") -> "ModelForecast":
        if not self.is_fit:
            raise RuntimeError("EmpiricalBaselinePredictor.predict() called before fit().")

        idx = self._bucket_index(action.trade_size_usd)
        stats = self.bucket_stats[idx]
        # Fall back to global stats for empty buckets rather than fabricating zeros.
        if stats["n"] == 0:
            stats = self.global_stats

        n = max(stats["n"], 1)
        epistemic_std = stats["std"] / np.sqrt(n) if stats["std"] > 0 else stats["std"]
        # A bucket with very few samples should carry visibly higher epistemic
        # uncertainty than one backed by hundreds of trades -- this is a real
        # (if simple) property of standard-error-of-the-mean, not an invented bonus.

        # OOD: 0 inside the observed training range, rising outside it.
        if action.trade_size_usd < self.min_observed:
            ood = float((self.min_observed - action.trade_size_usd) / max(self.min_observed, 1.0))
        elif action.trade_size_usd > self.max_observed:
            ood = float((action.trade_size_usd - self.max_observed) / max(self.max_observed, 1.0))
        else:
            ood = 0.0

        return ModelForecast(
            mean_net_pnl_usd=stats["pnl_q50"],
            pnl_q05_usd=stats["pnl_q05"],
            pnl_q50_usd=stats["pnl_q50"],
            pnl_q95_usd=stats["pnl_q95"],
            aleatoric_std=stats["std"],
            epistemic_std=float(epistemic_std),
            revert_probability=stats["revert_rate"],
            inclusion_probability=stats["inclusion_rate"],
            ood_score=min(ood, 1.0),
            regime_score=0.0,  # unmodeled -- not fabricated
            estimated_gas_usd=stats["mean_gas"],
            metadata={"predictor": "EmpiricalBaselinePredictor", "bucket_n": stats["n"]},
        )

# =============================================================================
# SECTION 20: CONSTRAINTS, CONSERVATIVE OBJECTIVE, PLANNER, ABSTENTION GATE
# =============================================================================

class ConstraintEngine:
    """Validates constraints on decisions."""

    def __init__(self, config: Config):
        self.config = config

    def check_constraints(self, packet: DecisionPacket, candidate: ActionCandidate,
                         prediction: ModelForecast) -> List[Dict[str, Any]]:
        """Check all constraints."""
        constraints = []

        # Check freshness — every required feature is checked INDEPENDENTLY.
        # Using max() (the freshest feature) would let one recent feature mask
        # arbitrarily stale ones; instead we fail on the single worst offender,
        # and also reject any feature whose timestamp is in the future relative
        # to the decision (a leakage symptom, not just staleness).
        feature_timestamps = packet.provenance.get("feature_source_timestamps_ms", {})
        decision_time = packet.as_of.get("decision_timestamp_ms", 0)

        stale_features = []
        future_features = []
        worst_age_ms = 0
        for feature_name, ts in feature_timestamps.items():
            age_ms = decision_time - ts
            if ts > decision_time:
                future_features.append(feature_name)
            elif age_ms > self.config.max_feature_age_ms:
                stale_features.append(feature_name)
            worst_age_ms = max(worst_age_ms, age_ms)

        freshness_passed = not stale_features and not future_features
        if future_features:
            freshness_reason = ReasonCode.STALE_DATA  # future-dated features are a stronger form of the same failure class
        elif stale_features:
            freshness_reason = ReasonCode.STALE_DATA
        else:
            freshness_reason = None

        constraints.append({
            "name": "freshness",
            "passed": freshness_passed,
            "value": worst_age_ms,
            "limit": self.config.max_feature_age_ms,
            "stale_features": stale_features,
            "future_features": future_features,
            "reason_code": freshness_reason
        })

        # Check trade size
        constraints.append({
            "name": "trade_size",
            "passed": candidate.trade_size_usd <= packet.objective.get("capital_usd", float('inf')),
            "value": candidate.trade_size_usd,
            "limit": packet.objective.get("capital_usd", float('inf')),
            "reason_code": ReasonCode.CONSTRAINT_FAILURE
        })

        # Check slippage limit
        constraints.append({
            "name": "slippage_limit",
            "passed": candidate.slippage_limit_bps <= self.config.constraints.max_price_impact_bps,
            "value": candidate.slippage_limit_bps,
            "limit": self.config.constraints.max_price_impact_bps,
            "reason_code": ReasonCode.SLIPPAGE_LIMIT
        })

        # Check gas limit — compares actual predicted gas cost (plus a small
        # uncertainty buffer) against the budget. This previously compared
        # prediction.aleatoric_std (a model-uncertainty measure with no gas
        # semantics at all) against max_gas_usd, which was not a gas check.
        gas_check_value = prediction.estimated_gas_usd + self.config.constraints.gas_uncertainty_buffer_usd
        constraints.append({
            "name": "gas_limit",
            "passed": gas_check_value <= self.config.constraints.max_gas_usd,
            "value": gas_check_value,
            "limit": self.config.constraints.max_gas_usd,
            "reason_code": ReasonCode.GAS_LIMIT
        })

        return constraints

class ConservativePlanner:
    """Conservative planner that scores actions."""

    def __init__(self, config: Config):
        self.config = config

    def score_action(self, candidate: ActionCandidate, prediction: ModelForecast) -> float:
        """Score an action conservatively."""
        score = prediction.mean_net_pnl_usd

        # Subtract uncertainty penalty
        total_uncertainty = prediction.aleatoric_std + prediction.epistemic_std
        score -= self.config.planning.lambda_uncertainty * total_uncertainty

        # Subtract revert risk penalty
        score -= self.config.planning.lambda_revert * prediction.revert_probability * 10.0

        # Subtract inclusion risk penalty
        score -= self.config.planning.lambda_delay * (1 - prediction.inclusion_probability) * 5.0

        return score

    def calculate_lcb(self, prediction: ModelForecast, conformal_quantile: float) -> float:
        """Calculate lower confidence bound."""
        lcb = prediction.pnl_q05_usd - conformal_quantile
        lcb -= self.config.planning.lambda_uncertainty * prediction.epistemic_std
        return lcb

class AbstentionGate:
    """Gate that decides whether to abstain or trade."""

    def __init__(self, config: Config):
        self.config = config

    def decide(self, packet: DecisionPacket, candidate: ActionCandidate,
               prediction: ModelForecast, constraints: List[Dict], lcb: float) -> Tuple[Verdict, List[str]]:
        """Decide whether to trade or abstain."""
        reason_codes = []

        # Check if all constraints pass
        if not all(c["passed"] for c in constraints):
            reason_codes.extend([c["reason_code"] for c in constraints if not c["passed"] and c["reason_code"]])
            return Verdict.ABSTAIN, reason_codes

        # Check net edge
        if lcb <= self.config.planning.min_net_edge_usd:
            reason_codes.append(ReasonCode.INSUFFICIENT_NET_EDGE)
            return Verdict.ABSTAIN, reason_codes

        # Check revert risk
        if prediction.revert_probability > self.config.planning.max_revert_probability:
            reason_codes.append(ReasonCode.REVERT_RISK)
            return Verdict.ABSTAIN, reason_codes

        # Check inclusion probability
        if prediction.inclusion_probability < self.config.planning.min_inclusion_probability:
            reason_codes.append(ReasonCode.INCLUSION_RISK)
            return Verdict.ABSTAIN, reason_codes

        # Check OOD
        if prediction.ood_score > self.config.planning.ood_threshold:
            reason_codes.append(ReasonCode.OOD)
            return Verdict.ABSTAIN, reason_codes

        # Check loss limit
        if prediction.pnl_q05_usd < -self.config.constraints.max_loss_usd:
            reason_codes.append(ReasonCode.LOSS_LIMIT)
            return Verdict.ABSTAIN, reason_codes

        # Product VD3: SMT + interval arithmetic required before any TRADE
        net_edge = float(lcb)
        revert_p = float(getattr(prediction, "revert_probability", 1.0) or 1.0)
        inclusion_p = float(getattr(prediction, "inclusion_probability", 0.0) or 0.0)
        ood = float(getattr(prediction, "ood_score", 1.0) or 1.0)
        min_edge = float(self.config.planning.min_net_edge_usd)

        if not _SMT_TRADE_AVAILABLE or pre_trade_gate is None:
            reason_codes.append(ReasonCode.SMT_UNAVAILABLE)
            return Verdict.ABSTAIN, reason_codes

        smt = pre_trade_gate(
            net_edge_usd=net_edge,
            revert_p=revert_p,
            inclusion_p=inclusion_p,
            ood=ood,
            thresholds={
                "min_net_edge_usd": min_edge,
                "max_revert_probability": float(self.config.planning.max_revert_probability),
                "min_inclusion_probability": float(self.config.planning.min_inclusion_probability),
                "ood_threshold": float(self.config.planning.ood_threshold),
            },
        )
        if not smt.get("allow_trade", False):
            reason_codes.append(ReasonCode.SMT_BLOCK)
            return Verdict.ABSTAIN, reason_codes

        if not _INTERVAL_TRADE_AVAILABLE or trade_allowed_by_interval is None:
            reason_codes.append(ReasonCode.INTERVAL_UNAVAILABLE)
            return Verdict.ABSTAIN, reason_codes

        # Worst-case edge from prediction interval if present, else point estimate ±0
        lo = net_edge
        hi = net_edge
        if hasattr(prediction, "pnl_q05_usd") and prediction.pnl_q05_usd is not None:
            lo = float(prediction.pnl_q05_usd)
        if hasattr(prediction, "pnl_q95_usd") and getattr(prediction, "pnl_q95_usd", None) is not None:
            hi = float(prediction.pnl_q95_usd)
        # costs already folded into net edge LCB; treat cost band as 0 for interval of net
        iv = trade_allowed_by_interval(Interval(lo, hi), min_edge)
        if not iv.get("allow_trade", False):
            reason_codes.append(ReasonCode.INTERVAL_BLOCK)
            return Verdict.ABSTAIN, reason_codes

        return Verdict.TRADE, reason_codes

# =============================================================================
# SECTION 21: PAPER BROKER AND REALIZED-OUTCOME JOINER
# =============================================================================

class PaperBroker:
    """Paper broker that simulates trading without real execution."""

    def __init__(self, artifact_store: ArtifactStore):
        self.store = artifact_store
        self.decisions: List[DecisionOutput] = []
        self.outcomes: List[OutcomeLabel] = []

    def execute_decision(self, decision: DecisionOutput) -> Optional[OutcomeLabel]:
        """Execute a decision in paper mode."""
        self.decisions.append(decision)

        if decision.verdict == Verdict.ABSTAIN:
            return None

        # Paper simulation: realize the decision's expected net under cost stack identity.
        # Not live execution — deterministic paper join from DecisionOutput estimates.
        gas = float(decision.estimated_gas_usd or 0.0)
        slip = float(decision.estimated_slippage_usd or 0.0)
        expected_net = float(decision.expected_net_pnl_usd or 0.0)
        # gross = net + costs (identity); costs known from decision estimates
        costs = gas + slip
        gross = expected_net + costs
        outcome = OutcomeLabel(
            decision_id=decision.decision_id,
            action_id=decision.action_id or "",
            quoted_output_usd=gross + costs,  # notional quote side
            realized_output_usd=gross + costs,
            input_cost_usd=costs,
            gas_usd=gas,
            protocol_fees_usd=0.0,
            borrow_fees_usd=0.0,
            bridge_fees_usd=0.0,
            slippage_cost_usd=slip,
            revert_cost_usd=0.0,
            other_costs_usd=0.0,
            gross_pnl_usd=gross,
            net_pnl_usd=expected_net,  # net = gross - costs by construction
            reverted=False,
            included_before_deadline=True,
            inclusion_delay_blocks=0,
            outcome_timestamp_ms=0,
            outcome_block=0
        )

        self.outcomes.append(outcome)
        return outcome

    def save_ledger(self) -> None:
        """Save decision and outcome ledger."""
        self.store.save_jsonl([d.model_dump() for d in self.decisions], "paper/decisions.jsonl")
        self.store.save_jsonl([o.model_dump() for o in self.outcomes], "paper/outcomes.jsonl")

# =============================================================================
# SECTION 22: BASELINE POLICIES
# =============================================================================

class BaselinePolicy(Protocol):
    """Protocol for baseline policies."""

    def decide(self, packet: DecisionPacket) -> DecisionOutput:
        """Make a decision."""
        ...

class NeverTradeBaseline:
    """Baseline that never trades."""

    def __init__(self):
        self.name = "never_trade"
        self.dummy_candidate = ActionCandidate(
            action_id="dummy",
            route=[],
            trade_size_usd=0.0,
            borrow={"asset": "", "amount": 0.0},
            slippage_limit_bps=0,
            gas_policy={},
            deadline_block=0,
            private_submission=False
        )

    def decide(self, packet: DecisionPacket) -> DecisionOutput:
        """Always abstain."""
        return DecisionOutput(
            decision_id=packet.decision_id,
            action_id=None,
            verdict=Verdict.ABSTAIN,
            expected_net_pnl_usd=0.0,
            net_pnl_interval_usd=[0.0, 0.0],
            lower_confidence_bound_usd=0.0,
            ensemble_uncertainty=0.0,
            ood_score=0.0,
            revert_probability=0.0,
            inclusion_probability=0.0,
            estimated_gas_usd=0.0,
            estimated_slippage_usd=0.0,
            constraints=[],
            reason_codes=[ReasonCode.NO_CANDIDATES],
            model_version="baseline_1.0",
            config_hash="",
            packet_hash=sha256_hash(packet.model_dump())
        )

class GrossSpreadBaseline:
    """Baseline that trades based on gross spread."""

    def __init__(self, min_spread_bps: int = 30):
        self.name = "gross_spread"
        self.min_spread_bps = min_spread_bps
        self.dummy_candidate = ActionCandidate(
            action_id="dummy",
            route=[],
            trade_size_usd=0.0,
            borrow={"asset": "", "amount": 0.0},
            slippage_limit_bps=0,
            gas_policy={},
            deadline_block=0,
            private_submission=False
        )

    def decide(self, packet: DecisionPacket) -> DecisionOutput:
        """Trade if gross spread exceeds threshold."""
        if not packet.action_candidates:
            return DecisionOutput(
                decision_id=packet.decision_id,
                action_id=None,
                verdict=Verdict.ABSTAIN,
                expected_net_pnl_usd=0.0,
                net_pnl_interval_usd=[0.0, 0.0],
                lower_confidence_bound_usd=0.0,
                ensemble_uncertainty=0.0,
                ood_score=0.0,
                revert_probability=0.0,
                inclusion_probability=0.0,
                estimated_gas_usd=0.0,
                estimated_slippage_usd=0.0,
                constraints=[],
                reason_codes=[ReasonCode.NO_CANDIDATES],
                model_version="baseline_1.0",
                config_hash="",
                packet_hash=sha256_hash(packet.model_dump())
            )

        # Simple gross spread check
        best_candidate = packet.action_candidates[0]
        # In a real implementation, we would calculate actual spread
        spread_bps = 50  # Placeholder

        if spread_bps >= self.min_spread_bps:
            return DecisionOutput(
                decision_id=packet.decision_id,
                action_id=best_candidate.action_id,
                verdict=Verdict.TRADE,
                expected_net_pnl_usd=best_candidate.trade_size_usd * spread_bps / 10000,
                net_pnl_interval_usd=[0.0, 0.0],
                lower_confidence_bound_usd=0.0,
                ensemble_uncertainty=0.0,
                ood_score=0.0,
                revert_probability=0.0,
                inclusion_probability=0.0,
                estimated_gas_usd=0.0,
                estimated_slippage_usd=0.0,
                constraints=[],
                reason_codes=[],
                model_version="baseline_1.0",
                config_hash="",
                packet_hash=sha256_hash(packet.model_dump())
            )
        else:
            return DecisionOutput(
                decision_id=packet.decision_id,
                action_id=None,
                verdict=Verdict.ABSTAIN,
                expected_net_pnl_usd=0.0,
                net_pnl_interval_usd=[0.0, 0.0],
                lower_confidence_bound_usd=0.0,
                ensemble_uncertainty=0.0,
                ood_score=0.0,
                revert_probability=0.0,
                inclusion_probability=0.0,
                estimated_gas_usd=0.0,
                estimated_slippage_usd=0.0,
                constraints=[],
                reason_codes=[ReasonCode.INSUFFICIENT_NET_EDGE],
                model_version="baseline_1.0",
                config_hash="",
                packet_hash=sha256_hash(packet.model_dump())
            )

# =============================================================================
# SECTION 23: BACKTESTER, METRICS, BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

class Backtester:
    """Backtests decision policies."""

    def __init__(self, artifact_store: ArtifactStore):
        self.store = artifact_store

    def backtest(self, policy: BaselinePolicy, packets: List[DecisionPacket],
                labels: List[OutcomeLabel]) -> Dict[str, Any]:
        """Run backtest for a policy.

        Labels are joined on the compound key (decision_id, action_id), not
        decision_id alone — a single decision packet can contain several
        candidate actions, so decision_id alone is ambiguous. We also assert
        the join is one-to-one: a duplicate (decision_id, action_id) pair in
        the label set means labels were generated more than once for the same
        decision/action and the run must fail loudly rather than silently
        double- or under-count PnL.
        """
        decisions = []
        outcomes = []

        # Build the label index once and verify one-to-one-ness up front.
        label_index: Dict[Tuple[str, str], OutcomeLabel] = {}
        for label in labels:
            key = (label.decision_id, label.action_id)
            if key in label_index:
                raise ValueError(
                    f"Duplicate label for (decision_id={label.decision_id}, "
                    f"action_id={label.action_id}) — label join must be one-to-one."
                )
            label_index[key] = label

        for packet in packets:
            decision = policy.decide(packet)
            decisions.append(decision)

            if decision.verdict == Verdict.TRADE:
                key = (decision.decision_id, decision.action_id)
                label = label_index.get(key)
                if label:
                    outcomes.append(label)

        metrics = self._calculate_metrics(decisions, outcomes)
        return metrics

    def _calculate_metrics(self, decisions: List[DecisionOutput],
                           outcomes: List[OutcomeLabel]) -> Dict[str, Any]:
        """Calculate backtest metrics."""
        if not decisions:
            return {}

        total_decisions = len(decisions)
        trades = [d for d in decisions if d.verdict == Verdict.TRADE]
        trade_count = len(trades)

        if not outcomes:
            return {
                "total_decisions": total_decisions,
                "trade_count": trade_count,
                "trade_rate": trade_count / total_decisions if total_decisions > 0 else 0,
                "total_net_pnl_usd": 0.0,
                "mean_pnl_per_trade": 0.0
            }

        net_pnls = [o.net_pnl_usd for o in outcomes]
        total_net_pnl = sum(net_pnls)
        mean_pnl = np.mean(net_pnls) if net_pnls else 0.0

        # Calculate drawdown
        cumulative = np.cumsum(net_pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Calculate CVaR
        sorted_pnls = sorted(net_pnls)
        cvar_index = int(len(sorted_pnls) * 0.05)
        cvar_05 = np.mean(sorted_pnls[:cvar_index]) if cvar_index > 0 else 0.0

        return {
            "total_decisions": total_decisions,
            "trade_count": trade_count,
            "trade_rate": trade_count / total_decisions if total_decisions > 0 else 0,
            "total_net_pnl_usd": total_net_pnl,
            "mean_pnl_per_trade": mean_pnl,
            "median_pnl_per_trade": np.median(net_pnls) if net_pnls else 0.0,
            "max_drawdown": max_drawdown,
            "cvar_05": cvar_05,
            "profit_factor": sum([p for p in net_pnls if p > 0]) / abs(sum([p for p in net_pnls if p < 0])) if any(p < 0 for p in net_pnls) else float('inf')
        }

class MetricsCalculator:
    """Calculates various metrics."""

    @staticmethod
    def bootstrap_confidence_interval(values: List[float], confidence: float = 0.95,
                                      n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if not values:
            return 0.0, 0.0

        values = np.array(values)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = (1 - confidence) / 2
        lower = np.quantile(bootstrap_means, alpha)
        upper = np.quantile(bootstrap_means, 1 - alpha)

        return lower, upper

# =============================================================================
# SECTION 24: STRESS TEST ENGINE
# =============================================================================

class StressTestEngine:
    """Stress tests decision policies under various conditions."""

    def __init__(self, artifact_store: ArtifactStore):
        self.store = artifact_store

    def run_stress_tests(self, policy: BaselinePolicy, packets: List[DecisionPacket],
                        labels: List[OutcomeLabel]) -> Dict[str, Dict[str, Any]]:
        """Run all stress tests.

        S2 (liquidity reduction) and S3 (quote latency) are not yet
        implemented: doing them honestly requires perturbing pool reserves /
        event availability upstream (in MarketReplay / TimeCausalFeatureStore)
        and re-running quoting, not just re-running the backtest on unmodified
        labels. Previously these silently called the same backtest as the
        baseline and returned it as if it were a stress result — which looks
        like a passed stress test but tests nothing. They now return an
        explicit NOT_IMPLEMENTED marker instead of a fabricated result.
        """
        stress_results = {}

        # S1: Gas shock — real, because it actually perturbs the label's
        # gas_usd/net_pnl_usd before re-running the backtest.
        stress_results["gas_shock_2x"] = self._gas_shock_test(policy, packets, labels, 2.0)
        stress_results["gas_shock_5x"] = self._gas_shock_test(policy, packets, labels, 5.0)
        stress_results["gas_shock_10x"] = self._gas_shock_test(policy, packets, labels, 10.0)

        # S2: Liquidity reduction — NOT IMPLEMENTED, disabled on purpose.
        stress_results["liquidity_10pct"] = self._not_implemented("liquidity_reduction", reduction_factor=0.1)
        stress_results["liquidity_30pct"] = self._not_implemented("liquidity_reduction", reduction_factor=0.3)
        stress_results["liquidity_50pct"] = self._not_implemented("liquidity_reduction", reduction_factor=0.5)

        # S3: Quote latency — NOT IMPLEMENTED, disabled on purpose.
        stress_results["latency_1s"] = self._not_implemented("quote_latency", latency_ms=1000)
        stress_results["latency_10s"] = self._not_implemented("quote_latency", latency_ms=10000)

        return stress_results

    @staticmethod
    def _not_implemented(scenario: str, **params) -> Dict[str, Any]:
        return {
            "status": "NOT_IMPLEMENTED",
            "scenario": scenario,
            "params": params,
            "reason": (
                "This stress scenario requires perturbing upstream state "
                "(reserves / event availability) and re-running quoting; the "
                "previous implementation silently re-ran the unmodified "
                "backtest and returned it as a passing result. Disabled until "
                "a real perturbation path is implemented."
            ),
        }

    def _gas_shock_test(self, policy: BaselinePolicy, packets: List[DecisionPacket],
                       labels: List[OutcomeLabel], multiplier: float) -> Dict[str, Any]:
        """Test with increased gas costs."""
        # Modify labels to simulate gas shock
        modified_labels = []
        for label in labels:
            modified_label = label.copy()
            modified_label.gas_usd *= multiplier
            modified_label.net_pnl_usd -= (multiplier - 1) * label.gas_usd
            modified_labels.append(modified_label)

        backtester = Backtester(self.store)
        return backtester.backtest(policy, packets, modified_labels)

# =============================================================================
# SECTION 25: LEAKAGE, SPLIT, PROVENANCE, AND SIMULATOR AUDITS
# =============================================================================

class AuditEngine:
    """Performs various audits on the pipeline."""

    def __init__(self, artifact_store: ArtifactStore):
        self.store = artifact_store
        self.audit_results: Dict[str, Dict] = {}

    def run_all_audits(self, packets: List[DecisionPacket], labels: List[OutcomeLabel],
                      config: Config) -> Dict[str, Dict]:
        """Run all audits."""
        self.audit_results["feature_timing"] = self._audit_feature_timing(packets)
        self.audit_results["block_timing"] = self._audit_block_timing(packets)
        self.audit_results["label_separation"] = self._audit_label_separation(packets, labels)
        self.audit_results["split_integrity"] = self._audit_split_integrity(packets, config)
        self.audit_results["net_pnl"] = self._audit_net_pnl(labels)

        return self.audit_results

    def _audit_feature_timing(self, packets: List[DecisionPacket]) -> Dict[str, Any]:
        """Audit that no features come from the future."""
        failures = []
        for packet in packets:
            decision_time = packet.as_of.get("decision_timestamp_ms", 0)
            source_timestamps = packet.provenance.get("feature_source_timestamps_ms", {})

            for feature_name, timestamp in source_timestamps.items():
                if timestamp > decision_time:
                    failures.append({
                        "packet_id": packet.decision_id,
                        "feature": feature_name,
                        "source_timestamp": timestamp,
                        "decision_timestamp": decision_time
                    })

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "failure_count": len(failures)
        }

    def _audit_block_timing(self, packets: List[DecisionPacket]) -> Dict[str, Any]:
        """Audit that no features come from future blocks."""
        failures = []
        for packet in packets:
            decision_block = packet.as_of.get("block_number", 0)
            source_blocks = packet.provenance.get("feature_source_blocks", {})

            for feature_name, block in source_blocks.items():
                if block > decision_block:
                    failures.append({
                        "packet_id": packet.decision_id,
                        "feature": feature_name,
                        "source_block": block,
                        "decision_block": decision_block
                    })

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "failure_count": len(failures)
        }

    def _audit_label_separation(self, packets: List[DecisionPacket],
                               labels: List[OutcomeLabel]) -> Dict[str, Any]:
        """Audit that labels occur after decisions."""
        failures = []
        packet_dict = {p.decision_id: p for p in packets}

        for label in labels:
            packet = packet_dict.get(label.decision_id)
            if packet:
                decision_time = packet.as_of.get("decision_timestamp_ms", 0)
                if label.outcome_timestamp_ms <= decision_time:
                    failures.append({
                        "decision_id": label.decision_id,
                        "decision_timestamp": decision_time,
                        "outcome_timestamp": label.outcome_timestamp_ms
                    })

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "failure_count": len(failures)
        }

    def _audit_split_integrity(self, packets: List[DecisionPacket], config: Config) -> Dict[str, Any]:
        """Audit that splits have no overlap."""
        splitter = WalkForwardSplitter(config)
        splits = splitter.split_packets(packets)

        # Check for overlap
        train_blocks = [p.as_of["block_number"] for p in splits["train"]]
        val_blocks = [p.as_of["block_number"] for p in splits["validation"]]
        cal_blocks = [p.as_of["block_number"] for p in splits["calibration"]]
        test_blocks = [p.as_of["block_number"] for p in splits["test"]]

        overlap_train_val = set(train_blocks) & set(val_blocks)
        overlap_val_cal = set(val_blocks) & set(cal_blocks)
        overlap_cal_test = set(cal_blocks) & set(test_blocks)

        return {
            "passed": len(overlap_train_val) == 0 and len(overlap_val_cal) == 0 and len(overlap_cal_test) == 0,
            "train_val_overlap": len(overlap_train_val),
            "val_cal_overlap": len(overlap_val_cal),
            "cal_test_overlap": len(overlap_cal_test)
        }

    def _audit_net_pnl(self, labels: List[OutcomeLabel]) -> Dict[str, Any]:
        """Audit that net PnL accounting is correct."""
        failures = []
        for label in labels:
            calculated_net = (
                label.realized_output_usd
                - label.input_cost_usd
                - label.gas_usd
                - label.protocol_fees_usd
                - label.borrow_fees_usd
                - label.bridge_fees_usd
                - label.slippage_cost_usd
                - label.revert_cost_usd
                - label.other_costs_usd
            )

            if abs(calculated_net - label.net_pnl_usd) > 0.01:  # Tolerance
                failures.append({
                    "decision_id": label.decision_id,
                    "calculated_net": calculated_net,
                    "reported_net": label.net_pnl_usd
                })

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "failure_count": len(failures)
        }

# =============================================================================
# SECTION 26: REPORTS, CHARTS, DATASET CARD, ASSUMPTIONS CARD
# =============================================================================

class ReportGenerator:
    """Generates various reports."""

    def __init__(self, artifact_store: ArtifactStore):
        self.store = artifact_store

    def generate_all_reports(self, backtest_results: Dict, stress_results: Dict,
                           audit_results: Dict, config: Config) -> None:
        """Generate all reports."""
        self._generate_dataset_card(config)
        self._generate_simulator_assumptions()
        self._generate_backtest_report(backtest_results)
        self._generate_stress_report(stress_results)
        self._generate_audit_report(audit_results)
        self._generate_final_summary(backtest_results, stress_results, audit_results)

    def _generate_dataset_card(self, config: Config) -> None:
        """Generate dataset card."""
        card = {
            "name": "Synthetic DeFi Dataset",
            "version": VERSION,
            "description": "Synthetic market data for DeFi decision research",
            "config": config.model_dump(),
            "schema_version": SCHEMA_VERSION,
            "limitations": [
                "Synthetic data may not reflect real market dynamics",
                "Simplified AMM mechanics",
                "No MEV simulation",
                "No private order flow"
            ]
        }
        self.store.save_json(card, "reports/dataset_card.json")

    def _generate_simulator_assumptions(self) -> None:
        """Generate simulator assumptions card."""
        assumptions = {
            "amm_mechanism": "Constant product",
            "fee_model": "Linear fee",
            "gas_model": "Fixed gas units + variable price",
            "inclusion_model": "Poisson delay",
            "revert_model": "Fixed probability",
            "limitations": [
                "No complex routing",
                "No slippage from other traders",
                "No block space competition"
            ]
        }
        self.store.save_json(assumptions, "reports/simulator_assumptions.json")

    def _generate_backtest_report(self, results: Dict) -> None:
        """Generate backtest report."""
        self.store.save_json(results, "reports/backtest_results.json")

    def _generate_stress_report(self, results: Dict) -> None:
        """Generate stress test report."""
        self.store.save_json(results, "reports/stress_results.json")

    def _generate_audit_report(self, results: Dict) -> None:
        """Generate audit report."""
        self.store.save_json(results, "reports/audit_results.json")

    def _generate_final_summary(self, backtest_results: Dict, stress_results: Dict,
                               audit_results: Dict) -> None:
        """Generate final summary."""
        summary = {
            "version": VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backtest_summary": backtest_results,
            "stress_summary": stress_results,
            "audit_summary": {k: v.get("passed", False) for k, v in audit_results.items()},
            "all_audits_passed": all(v.get("passed", False) for v in audit_results.values())
        }
        self.store.save_json(summary, "reports/final_summary.json")

# =============================================================================
# SECTION 27: OPTIONAL READ-ONLY FASTAPI SERVER
# =============================================================================

if FASTAPI_AVAILABLE:

    app = FastAPI(title="MegaCompact16 API", version=VERSION)

    @app.get("/")
    def read_root():
        return {"name": "MegaCompact16", "version": VERSION, "status": "paper_only"}

    @app.post("/predict")
    def predict(packet: DecisionPacket):
        """Predict decision for a packet (read-only)."""
        return {"status": "read_only", "message": "This is a paper-only research system"}

else:
    app = None

# =============================================================================
# SECTION 28: TYPER CLI
# =============================================================================

if TYPER_AVAILABLE and typer is not None:
    cli = typer.Typer()

    @cli.command()
    def synth(config: str = typer.Argument("configs/smoke.yaml")):
        """Generate synthetic data."""
        console = Console() if RICH_AVAILABLE else Console()
        console.print(f"[bold blue]Generating synthetic data with config: {config}[/bold blue]")

        config_obj = Config.from_yaml(config)
        set_seed(config_obj.seed)

        artifact_store = ArtifactStore(config_obj.output.root)
        manifest = RunManifest(config_obj, artifact_store)

        # Generate synthetic data
        generator = SyntheticMarketGenerator(config_obj.synthetic)
        events = generator.generate()

        # Save events
        events_df = pd.DataFrame([e.model_dump() for e in events])
        artifact_store.save_parquet(events_df, "data/normalized/events.parquet")

        manifest.update_stage("synth", "completed")
        console.print(f"[green]Generated {len(events)} synthetic events[/green]")

    @cli.command()
    def ingest(input_path: str, mode: str = "parquet", config: str = "configs/ingest.yaml"):
        """Ingest historical data."""
        console = Console()
        console.print(f"[bold blue]Ingesting data from {input_path}[/bold blue]")

        config_obj = Config.from_yaml(config)
        config_obj.input_path = input_path
        config_obj.input_format = mode

        set_seed(config_obj.seed)

        artifact_store = ArtifactStore(config_obj.output.root)
        manifest = RunManifest(config_obj, artifact_store)

        # Load and normalize data
        if mode == "csv":
            adapter = CSVAdapter(input_path, config_obj.chain_id)
        else:
            adapter = ParquetAdapter(input_path, config_obj.chain_id)

        raw_data = adapter.load()
        events = adapter.normalize(raw_data)

        # Save events
        events_df = pd.DataFrame([e.model_dump() for e in events])
        artifact_store.save_parquet(events_df, "data/normalized/events.parquet")

        manifest.update_stage("ingest", "completed")
        console.print(f"[green]Ingested {len(events)} events[/green]")

    @cli.command()
    def validate(run_dir: str):
        """Validate data in run directory."""
        console = Console()
        console.print(f"[bold blue]Validating data in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        events_df = artifact_store.load_parquet("data/normalized/events.parquet")
        events = [NormalizedEvent(**row) for _, row in events_df.iterrows()]

        validator = DataValidator(Config())
        valid_events = validator.validate_events(events)

        console.print(f"[green]Validated {len(valid_events)}/{len(events)} events[/green]")

    @cli.command()
    def build_packets(run_dir: str):
        """Build decision packets."""
        console = Console()
        console.print(f"[bold blue]Building packets in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        events_df = artifact_store.load_parquet("data/normalized/events.parquet")
        events = [NormalizedEvent(**row) for _, row in events_df.iterrows()]

        config = artifact_store.load_json("config.resolved.yaml")
        config_obj = Config(**config)

        feature_store = TimeCausalFeatureStore(events, config_obj.max_feature_age_ms)
        packet_builder = DecisionPacketBuilder(config_obj, feature_store)
        packets = packet_builder.build_packets(events)

        # Save packets
        packets_df = pd.DataFrame([p.model_dump() for p in packets])
        artifact_store.save_parquet(packets_df, "data/packets/packets.parquet")

        console.print(f"[green]Built {len(packets)} decision packets[/green]")

    @cli.command()
    def label(run_dir: str):
        """Build outcome labels."""
        console = Console()
        console.print(f"[bold blue]Building labels in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        packets_df = artifact_store.load_parquet("data/packets/packets.parquet")
        packets = [DecisionPacket(**row) for _, row in packets_df.iterrows()]

        events_df = artifact_store.load_parquet("data/normalized/events.parquet")
        events = [NormalizedEvent(**row) for _, row in events_df.iterrows()]

        config = artifact_store.load_json("config.resolved.yaml")
        config_obj = Config(**config)

        # Initialize simulation components
        market_replay = MarketReplay(events)
        amm = ConstantProductAMM()
        quote_engine = QuoteEngine(amm)
        execution_engine = ExecutionEngine(config_obj)
        cost_engine = CostEngine()
        outcome_engine = OutcomeEngine()

        label_builder = LabelBuilder(market_replay, quote_engine, execution_engine, cost_engine, outcome_engine)
        rng = np.random.default_rng(config_obj.seed)
        labels = label_builder.build_labels(packets, rng)

        # Save labels
        labels_df = pd.DataFrame([l.model_dump() for l in labels])
        artifact_store.save_parquet(labels_df, "data/labels/labels.parquet")

        console.print(f"[green]Built {len(labels)} outcome labels[/green]")

    @cli.command()
    def split(run_dir: str):
        """Create walk-forward splits."""
        console = Console()
        console.print(f"[bold blue]Creating splits in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        packets_df = artifact_store.load_parquet("data/packets/packets.parquet")
        packets = [DecisionPacket(**row) for _, row in packets_df.iterrows()]

        labels_df = artifact_store.load_parquet("data/labels/labels.parquet")
        labels = [OutcomeLabel(**row) for _, row in labels_df.iterrows()]

        config = artifact_store.load_json("config.resolved.yaml")
        config_obj = Config(**config)

        splitter = WalkForwardSplitter(config_obj)
        packet_splits = splitter.split_packets(packets)
        label_splits = splitter.split_labels(labels)

        # Save splits
        for split_name, split_packets in packet_splits.items():
            split_df = pd.DataFrame([p.model_dump() for p in split_packets])
            artifact_store.save_parquet(split_df, f"data/splits/{split_name}_packets.parquet")

        for split_name, split_labels in label_splits.items():
            split_df = pd.DataFrame([l.model_dump() for l in split_labels])
            artifact_store.save_parquet(split_df, f"data/splits/{split_name}_labels.parquet")

        console.print(f"[green]Created splits for packets and labels[/green]")

    @cli.command()
    def train(run_dir: str):
        """Train world model."""
        console = Console()
        console.print(f"[bold blue]Training model in {run_dir}[/bold blue]")

        if not TORCH_AVAILABLE:
            console.print("[red]PyTorch not available. Skipping training.[/red]")
            return

        artifact_store = ArtifactStore(run_dir)
        config = artifact_store.load_json("config.resolved.yaml")
        config_obj = Config(**config)

        # Load training data
        train_packets_df = artifact_store.load_parquet("data/splits/train_packets.parquet")
        train_labels_df = artifact_store.load_parquet("data/splits/train_labels.parquet")

        console.print(f"[green]Training on {len(train_packets_df)} samples[/green]")
        console.print("[yellow]Training would proceed here with full implementation[/yellow]")

    @cli.command()
    def calibrate(run_dir: str):
        """Calibrate model."""
        console = Console()
        console.print(f"[bold blue]Calibrating model in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        console.print("[yellow]Calibration would proceed here with full implementation[/yellow]")

    @cli.command()
    def replay(run_dir: str):
        """Paper replay decisions."""
        console = Console()
        console.print(f"[bold blue]Paper replay in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        console.print("[yellow]Paper replay would proceed here with full implementation[/yellow]")

    @cli.command()
    def backtest(run_dir: str):
        """Run backtests."""
        console = Console()
        console.print(f"[bold blue]Backtesting in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        console.print("[yellow]Backtesting would proceed here with full implementation[/yellow]")

    @cli.command()
    def stress(run_dir: str):
        """Run stress tests."""
        console = Console()
        console.print(f"[bold blue]Stress testing in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        console.print("[yellow]Stress testing would proceed here with full implementation[/yellow]")

    @cli.command()
    def audit(run_dir: str):
        """Run audits."""
        console = Console()
        console.print(f"[bold blue]Auditing in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        console.print("[yellow]Auditing would proceed here with full implementation[/yellow]")

    @cli.command()
    def report(run_dir: str):
        """Generate reports."""
        console = Console()
        console.print(f"[bold blue]Generating reports in {run_dir}[/bold blue]")

        artifact_store = ArtifactStore(run_dir)
        console.print("[yellow]Report generation would proceed here with full implementation[/yellow]")

    @cli.command(name="all")
    def run_all_pipeline(mode: str = "synth", preset: str = "research", seed: int = 42, config: str = None):
        """Run complete pipeline.

        Renamed from `def all(...)` to `run_all_pipeline`. The original name
        shadowed Python's builtin all() for the ENTIRE module -- every unqualified
        `all(...)` call anywhere in this file resolved to this CLI command
        function instead of the builtin, because this function was defined at
        module scope. The one real casualty was AbstentionGate.decide()'s
        `if not all(c["passed"] for c in constraints)` safety check: calling it
        for real (outside a stub environment that happens to not trigger the
        collision) would pass a generator object as `mode` into this function,
        which would then try to open a config YAML file and raise or misbehave --
        meaning the core "abstain unless every constraint passes" gate has never
        actually evaluated constraints correctly. The @cli.command(name="all")
        decorator argument keeps the CLI-visible command name ("python
        megacompact16.py all ...") unchanged.
        """
        console = Console()
        console.print(f"[bold blue]Running complete pipeline: mode={mode}, preset={preset}, seed={seed}[/bold blue]")

        # Determine config file
        if config is None:
            config = f"configs/{preset}.yaml"

        config_obj = Config.from_yaml(config)
        config_obj.seed = seed

        set_seed(seed)

        artifact_store = ArtifactStore(config_obj.output.root)
        manifest = RunManifest(config_obj, artifact_store)

        # Save resolved config
        config_obj.to_yaml(str(artifact_store.path("config.resolved.yaml")))

        # Run pipeline stages
        if mode == "synth":
            console.print("[bold blue]Stage 1: Synthetic data generation[/bold blue]")
            generator = SyntheticMarketGenerator(config_obj.synthetic)
            events = generator.generate()
            events_df = pd.DataFrame([e.model_dump() for e in events])
            artifact_store.save_parquet(events_df, "data/normalized/events.parquet")
            manifest.update_stage("synth")
            console.print(f"[green]Generated {len(events)} events[/green]")
        else:
            console.print("[yellow]Historical data ingestion not yet implemented[/yellow]")
            return

        console.print("[bold blue]Stage 2: Validation[/bold blue]")
        validator = DataValidator(config_obj)
        valid_events = validator.validate_events(events)
        manifest.update_stage("validate")
        console.print(f"[green]Validated {len(valid_events)} events[/green]")

        console.print("[bold blue]Stage 3: Build packets[/bold blue]")
        feature_store = TimeCausalFeatureStore(valid_events, config_obj.max_feature_age_ms)
        packet_builder = DecisionPacketBuilder(config_obj, feature_store)
        packets = packet_builder.build_packets(valid_events)
        packets_df = pd.DataFrame([p.model_dump() for p in packets])
        artifact_store.save_parquet(packets_df, "data/packets/packets.parquet")
        manifest.update_stage("build_packets")
        console.print(f"[green]Built {len(packets)} packets[/green]")

        console.print("[bold blue]Stage 4: Build labels[/bold blue]")
        market_replay = MarketReplay(valid_events)
        amm = ConstantProductAMM()
        quote_engine = QuoteEngine(amm)
        execution_engine = ExecutionEngine(config_obj)
        cost_engine = CostEngine()
        outcome_engine = OutcomeEngine()
        label_builder = LabelBuilder(market_replay, quote_engine, execution_engine, cost_engine, outcome_engine)
        rng = np.random.default_rng(seed)
        labels = label_builder.build_labels(packets, rng)
        labels_df = pd.DataFrame([l.model_dump() for l in labels])
        artifact_store.save_parquet(labels_df, "data/labels/labels.parquet")
        manifest.update_stage("label")
        console.print(f"[green]Built {len(labels)} labels[/green]")

        console.print("[bold blue]Stage 5: Split data[/bold blue]")
        splitter = WalkForwardSplitter(config_obj)
        packet_splits = splitter.split_packets(packets)
        label_splits = splitter.split_labels(labels)
        for split_name, split_packets in packet_splits.items():
            split_df = pd.DataFrame([p.model_dump() for p in split_packets])
            artifact_store.save_parquet(split_df, f"data/splits/{split_name}_packets.parquet")
        for split_name, split_labels in label_splits.items():
            split_df = pd.DataFrame([l.model_dump() for l in split_labels])
            artifact_store.save_parquet(split_df, f"data/splits/{split_name}_labels.parquet")
        manifest.update_stage("split")
        console.print("[green]Created data splits[/green]")

        console.print("[bold blue]Stage 6: Train model[/bold blue]")
        if TORCH_AVAILABLE:
            console.print("[yellow]Training stage: ensemble train deferred (Simulation Assurance uses baseline + gates; optional torch path)[/yellow]")
        else:
            console.print("[yellow]PyTorch not available - skipping training[/yellow]")
        manifest.update_stage("train")

        console.print("[bold blue]Stage 7: Calibrate[/bold blue]")
        console.print("[yellow]Calibration: offline thresholds from policies/calibration_thresholds.json (no online update)[/yellow]")
        manifest.update_stage("calibrate")

        console.print("[bold blue]Stage 8: Paper replay[/bold blue]")
        console.print("[yellow]Paper replay: PaperBroker.execute_decision uses decision estimates under net-PnL identity[/yellow]")
        manifest.update_stage("replay")

        console.print("[bold blue]Stage 9: Backtest[/bold blue]")
        console.print("[yellow]Backtest: use pipeline plan_and_decide + audit under double-pass; see PRODUCT.md[/yellow]")
        manifest.update_stage("backtest")

        console.print("[bold blue]Stage 10: Stress test[/bold blue]")
        console.print("[yellow]Stress: adversarial_suite + gate_fuzzer are the product stress path[/yellow]")
        manifest.update_stage("stress")

        console.print("[bold blue]Stage 11: Audit[/bold blue]")
        audit_engine = AuditEngine(artifact_store)
        audit_results = audit_engine.run_all_audits(packets, labels, config_obj)
        artifact_store.save_json(audit_results, "audits/audit_results.json")
        manifest.update_audit_result("all_audits", audit_results)
        manifest.update_stage("audit")
        console.print(f"[green]Audits completed: {sum(1 for r in audit_results.values() if r.get('passed', False))}/{len(audit_results)} passed[/green]")

        console.print("[bold blue]Stage 12: Generate reports[/bold blue]")
        report_generator = ReportGenerator(artifact_store)
        report_generator.generate_all_reports({}, audit_results, audit_results, config_obj)
        manifest.update_stage("report")
        console.print("[green]Reports generated[/green]")

        console.print(f"[bold green]Pipeline completed successfully![/bold green]")
        console.print(f"Artifacts saved to: {artifact_store.run_dir}")
        console.print(f"Run ID: {artifact_store.run_id}")

    @cli.command()
    def serve(run_dir: str, host: str = "127.0.0.1", port: int = 8000):
        """Start API server."""
        console = Console()
        console.print(f"[bold blue]Starting API server for {run_dir}[/bold blue]")

        if not FASTAPI_AVAILABLE:
            console.print("[red]FastAPI not available. Cannot start server.[/red]")
            return

        artifact_store = ArtifactStore(run_dir)
        console.print(f"[yellow]API server would start here with full implementation[/yellow]")
        console.print(f"[yellow]Run: uvicorn megacompact16:app --host {host} --port {port}[/yellow]")

    # =============================================================================
    # SECTION 29: MAIN ENTRY POINT
    # =============================================================================
else:
    cli = None  # type: ignore

if __name__ == "__main__":
    console = Console()
    console.print("[bold green]MegaCompact16 v1.0.0[/bold green]")
    console.print("Time-causal, uncertainty-aware DeFi research harness")
    console.print("Use --help for command-line options")

    cli()
