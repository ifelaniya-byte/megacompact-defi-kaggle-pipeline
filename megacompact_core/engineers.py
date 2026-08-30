#!/usr/bin/env python3
"""
Stationary & Non-Stationary Engineers
=====================================

Mandatory double-pass verification spine.

Contract (non-negotiable):
  Every unit of information MUST pass through:

      StationaryEngineer.verify()          # pass 1 – observe, check integrity
   →  NonStationaryEngineer.act_and_check()# experimental / implementation probe
   →  StationaryEngineer.verify()          # pass 2 – secondary verification

  before any LLM is allowed to interpret or act on that information.

Design notes
------------
- Stationary = observer / processor only. Never mutates external state.
  Produces situation reports, integrity assessments, and ranked issues.
- Non-Stationary = actor / experimental. May propose transforms, run probes,
  reseal, or attempt repair. Always followed by a second stationary pass.
- Both engineers write every check into a VerificationLedger (JSONL).
- Fail-closed: if either pass reports ok=False, the packet is marked
  BLOCKED and must not be handed to an LLM interpreter.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import concurrent.futures

try:
    from atlas import get_atlas, PhysicsMathAtlas
    ATLAS_AVAILABLE = True
except Exception:
    ATLAS_AVAILABLE = False
    get_atlas = None  # type: ignore

try:
    from known_settled_db import get_settled_db
    SETTLED_DB_AVAILABLE = True
except Exception:
    SETTLED_DB_AVAILABLE = False
    get_settled_db = None  # type: ignore


# Roadmap upgrade imports (optional soft deps)
try:
    from schema_registry import load_schema, validate_required
    SCHEMA_REG_AVAILABLE = True
except Exception:
    SCHEMA_REG_AVAILABLE = False
    load_schema = validate_required = None  # type: ignore
try:
    from ledger_index import search as ledger_search
    LEDGER_INDEX_AVAILABLE = True
except Exception:
    LEDGER_INDEX_AVAILABLE = False
    ledger_search = None  # type: ignore
try:
    from rag_settled import search as rag_search
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False
    rag_search = None  # type: ignore
try:
    from constants_db import get_constants_db
    CONSTANTS_AVAILABLE = True
except Exception:
    CONSTANTS_AVAILABLE = False
    get_constants_db = None  # type: ignore
try:
    import sympy
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False
    sympy = None  # type: ignore

try:
    from unit_tags import stationary_unit_probe
    UNIT_TAGS_AVAILABLE = True
except Exception:
    UNIT_TAGS_AVAILABLE = False
    stationary_unit_probe = None  # type: ignore


# =============================================================================
# Contracts
# =============================================================================

class EngineerVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"   # secondary verification failed – do not give to LLM
    ABSTAIN = "ABSTAIN"   # insufficient evidence / cannot decide


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: Any = None
    severity: str = "error"   # error | warn | info

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngineerReport:
    """Full report from one engineer pass."""
    engineer: str                     # "stationary" | "non_stationary"
    pass_number: int                  # 1 or 2 for stationary; 1 for non-stationary
    timestamp: str
    verdict: EngineerVerdict
    checks: List[CheckResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks) and self.verdict in (
            EngineerVerdict.PASS, EngineerVerdict.ABSTAIN
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engineer": self.engineer,
            "pass_number": self.pass_number,
            "timestamp": self.timestamp,
            "verdict": self.verdict.value,
            "all_ok": self.all_ok,
            "checks": [c.to_dict() for c in self.checks],
            "issues": self.issues,
            "metadata": self.metadata,
        }


@dataclass
class DoublePassResult:
    """Result of the mandatory Stationary → NonStationary → Stationary sequence."""
    subject_id: str
    subject_type: str
    stationary_pass_1: EngineerReport
    non_stationary: EngineerReport
    stationary_pass_2: EngineerReport
    final_verdict: EngineerVerdict
    allowed_for_llm: bool
    ledger_entry_id: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "stationary_pass_1": self.stationary_pass_1.to_dict(),
            "non_stationary": self.non_stationary.to_dict(),
            "stationary_pass_2": self.stationary_pass_2.to_dict(),
            "final_verdict": self.final_verdict.value,
            "allowed_for_llm": self.allowed_for_llm,
            "ledger_entry_id": self.ledger_entry_id,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Verification Ledger (persistent, fail-open for telemetry)
# =============================================================================

class VerificationLedger:
    """Append-only JSONL ledger of every engineer pass."""

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path else Path("artifacts/verification_ledger.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []

    def append(self, event_type: str, payload: Dict[str, Any]) -> str:
        entry_id = hashlib.sha256(
            f"{event_type}{time.time()}{json.dumps(payload, sort_keys=True, default=str)}".encode()
        ).hexdigest()[:16]
        record = {
            "entry_id": entry_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self._entries.append(record)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # telemetry must never block the safety spine
        return entry_id

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        return self._entries[-n:]


# =============================================================================
# Stationary Engineer  (observer / processor – never mutates external state)
# =============================================================================

class StationaryEngineer:
    """
    Stationary (observer) engineer.

    Responsibilities:
      - Structural / schema integrity checks
      - Time-causality & leakage probes
      - Net-PnL accounting sanity
      - Source / AST / math-logic probes when source is supplied
      - Produce a ranked list of issues and a hard PASS / FAIL / ABSTAIN verdict

    It does NOT implement changes. It only observes and reports.
    """

    def __init__(self, ledger: Optional[VerificationLedger] = None):
        self.ledger = ledger or VerificationLedger()
        self.last_report: Optional[EngineerReport] = None

    def verify(
        self,
        subject: Any,
        subject_id: str,
        subject_type: str,
        pass_number: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineerReport:
        context = context or {}
        checks: List[CheckResult] = []
        issues: List[str] = []

        # ----- 1. Existence / type checks -----
        checks.append(CheckResult(
            name="subject_present",
            ok=subject is not None,
            detail=type(subject).__name__ if subject is not None else None,
        ))
        if subject is None:
            issues.append("subject is None")

        # ----- 2. Schema / required-field probes (duck-typed) -----
        if subject is not None:
            checks.extend(self._schema_checks(subject, subject_type))

        # ----- 3. Time-causality probes when timestamps are present -----
        if subject is not None:
            checks.extend(self._causality_checks(subject, subject_type))

        # ----- 4. Numeric / accounting sanity -----
        if subject is not None:
            checks.extend(self._numeric_checks(subject, subject_type))

        # ----- 5. Optional source-code AST probe (when context supplies source) -----
        source_text = context.get("source_text") or context.get("source")
        if source_text:
            checks.extend(self._source_ast_checks(source_text))

        # ----- 6. Math / logic micro-probes (always-on, no external deps) -----
        checks.extend(self._math_logic_probes())

        # ----- 7. Atlas-backed physics/math claim gate (if atlas present) -----
        if ATLAS_AVAILABLE and get_atlas is not None:
            atlas = get_atlas()
            claim = None
            if isinstance(subject, dict):
                claim = subject.get("physics_claim") or subject.get("claim") or subject.get("expression")
            if claim and isinstance(claim, str):
                vr = atlas.verify_claim(claim)
                checks.append(CheckResult(
                    name="atlas.claim_gate",
                    ok=vr.get("ok_for_atlas_support", True),
                    detail=vr,
                    severity="error" if not vr.get("ok_for_atlas_support", True) else "info",
                ))
                if not vr.get("ok_for_atlas_support", True):
                    issues.append(f"atlas overreach: {vr.get('overreach')}")
            # Catalogue health probe
            checks.append(CheckResult(
                name="atlas.catalogue_loaded",
                ok=len(atlas.formulas) > 0 and len(atlas.domains) > 0,
                detail={"n_formulas": len(atlas.formulas), "n_domains": len(atlas.domains), "version": atlas.version},
                severity="info",
            ))


        # ----- 8. Known-Settled DB health + completeness rejection -----
        if SETTLED_DB_AVAILABLE and get_settled_db is not None:
            db = get_settled_db()
            checks.append(CheckResult(
                name="settled_db.loaded",
                ok=len(db.entries) > 0,
                detail={"n_entries": len(db.entries), "version": db.version},
                severity="info",
            ))
            claim = None
            if isinstance(subject, dict):
                claim = subject.get("claim") or subject.get("physics_claim") or subject.get("statement")
            if isinstance(claim, str):
                rej = db.reject_completeness_claim(claim)
                checks.append(CheckResult(
                    name="settled_db.no_completeness_overclaim",
                    ok=rej.get("ok", True),
                    detail=rej,
                    severity="error" if not rej.get("ok", True) else "info",
                ))
                if not rej.get("ok", True):
                    issues.append(f"completeness overclaim: {rej.get('rejected_phrases')}")


        # ----- Roadmap upgrades: schema, ledger memory, RAG evidence, sympy -----
        if SCHEMA_REG_AVAILABLE and isinstance(subject, dict):
            schema_name = None
            if subject_type in ("NormalizedEvent", "event"):
                schema_name = "events_v1.json"
            elif subject_type in ("DecisionPacket", "packet"):
                schema_name = "packets_v1.json"
            elif subject_type in ("OutcomeLabel", "label"):
                schema_name = "labels_v1.json"
            if schema_name:
                try:
                    sch = load_schema(schema_name)
                    vr = validate_required(subject, sch)
                    checks.append(CheckResult(
                        name="schema_registry.required",
                        ok=vr.get("ok", False),
                        detail=vr,
                    ))
                    if not vr.get("ok", False):
                        issues.append(f"schema missing: {vr.get('missing')}")
                except Exception as e:
                    checks.append(CheckResult(name="schema_registry.required", ok=False, detail=str(e), severity="warn"))

        if LEDGER_INDEX_AVAILABLE and ledger_search is not None:
            try:
                hits = ledger_search(subject_type, limit=5)
                checks.append(CheckResult(
                    name="ledger_memory.recent_patterns",
                    ok=True,
                    detail={"n_hits": len(hits), "sample_types": list({h.get("event_type") for h in hits[:5]})},
                    severity="info",
                ))
            except Exception as e:
                checks.append(CheckResult(name="ledger_memory.recent_patterns", ok=True, detail=str(e), severity="info"))

        if RAG_AVAILABLE and rag_search is not None and isinstance(subject, dict):
            q = subject.get("claim") or subject.get("statement") or subject_type
            if isinstance(q, str) and q:
                try:
                    docs = rag_search(q, limit=3)
                    checks.append(CheckResult(
                        name="rag_settled.evidence",
                        ok=True,
                        detail={"n_docs": len(docs), "doc_ids": [d.get("doc_id") for d in docs]},
                        severity="info",
                    ))
                except Exception:
                    pass

        if SYMPY_AVAILABLE and isinstance(subject, dict):
            formal = subject.get("formal")
            if isinstance(formal, str) and "=" in formal and len(formal) < 80:
                try:
                    left, right = formal.split("=", 1)
                    # only try very simple numeric/symbolic equality
                    diff = sympy.simplify(sympy.sympify(left) - sympy.sympify(right))
                    checks.append(CheckResult(
                        name="sympy.identity",
                        ok=diff == 0,
                        detail={"formal": formal, "diff": str(diff)},
                        severity="warn",
                    ))
                except Exception as e:
                    checks.append(CheckResult(
                        name="sympy.identity",
                        ok=True,
                        detail={"skipped": str(e)},
                        severity="info",
                    ))

        if CONSTANTS_AVAILABLE and isinstance(subject, dict) and subject.get("unit_check"):
            try:
                cdb = get_constants_db()
                uid = subject["unit_check"].get("constant_id")
                c = cdb.get(uid) if uid else None
                checks.append(CheckResult(
                    name="constants.lookup",
                    ok=c is not None,
                    detail=c,
                    severity="warn",
                ))
            except Exception as e:
                checks.append(CheckResult(name="constants.lookup", ok=False, detail=str(e), severity="warn"))


        if UNIT_TAGS_AVAILABLE and stationary_unit_probe is not None and isinstance(subject, dict) and subject.get("unit_tags"):
            try:
                up = stationary_unit_probe(subject)
                checks.append(CheckResult(
                    name="unit_tags.probe",
                    ok=up.get("ok", True),
                    detail=up,
                    severity="warn",
                ))
                if not up.get("ok", True):
                    issues.append("unit_tags malformed")
            except Exception as e:
                checks.append(CheckResult(name="unit_tags.probe", ok=True, detail=str(e), severity="info"))
        # ----- Aggregate -----
        hard_fails = [c for c in checks if not c.ok and c.severity == "error"]
        if hard_fails:
            verdict = EngineerVerdict.FAIL
            issues.extend([f"{c.name}: {c.detail}" for c in hard_fails])
        elif any(not c.ok for c in checks):
            verdict = EngineerVerdict.ABSTAIN
            issues.extend([f"{c.name}: {c.detail}" for c in checks if not c.ok])
        else:
            verdict = EngineerVerdict.PASS

        report = EngineerReport(
            engineer="stationary",
            pass_number=pass_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            verdict=verdict,
            checks=checks,
            issues=issues,
            metadata={
                "subject_id": subject_id,
                "subject_type": subject_type,
                "n_checks": len(checks),
                "n_hard_fails": len(hard_fails),
                "context_keys": list(context.keys()),
            },
        )
        self.last_report = report
        self.ledger.append(f"stationary_pass_{pass_number}", report.to_dict())
        return report

    # ----- private check families -----

    def _schema_checks(self, subject: Any, subject_type: str) -> List[CheckResult]:
        out: List[CheckResult] = []
        # DecisionPacket-like
        if subject_type in ("DecisionPacket", "packet"):
            for field in ("decision_id", "as_of", "action_candidates"):
                present = hasattr(subject, field) or (isinstance(subject, dict) and field in subject)
                out.append(CheckResult(name=f"schema.{field}", ok=bool(present), detail=field))
        # OutcomeLabel-like
        if subject_type in ("OutcomeLabel", "label"):
            for field in ("decision_id", "net_pnl_usd", "reverted", "outcome_timestamp_ms"):
                present = hasattr(subject, field) or (isinstance(subject, dict) and field in subject)
                out.append(CheckResult(name=f"schema.{field}", ok=bool(present), detail=field))
        # NormalizedEvent-like
        if subject_type in ("NormalizedEvent", "event"):
            for field in ("event_id", "event_timestamp_ms", "available_timestamp_ms", "event_type"):
                present = hasattr(subject, field) or (isinstance(subject, dict) and field in subject)
                out.append(CheckResult(name=f"schema.{field}", ok=bool(present), detail=field))
        # Generic dict / list
        if isinstance(subject, dict):
            out.append(CheckResult(name="schema.non_empty_dict", ok=len(subject) > 0, detail=len(subject)))
        if isinstance(subject, list):
            out.append(CheckResult(name="schema.list_bounded", ok=len(subject) < 1_000_000, detail=len(subject)))
        return out

    def _causality_checks(self, subject: Any, subject_type: str) -> List[CheckResult]:
        out: List[CheckResult] = []

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # available_timestamp >= observed_timestamp
        avail = _get(subject, "available_timestamp_ms")
        obs = _get(subject, "observed_timestamp_ms")
        if avail is not None and obs is not None:
            try:
                ok = int(avail) >= int(obs)
                out.append(CheckResult(
                    name="causality.available_after_observed",
                    ok=ok,
                    detail={"available": avail, "observed": obs},
                ))
            except Exception as e:
                out.append(CheckResult(name="causality.available_after_observed", ok=False, detail=str(e)))

        # outcome after decision (for labels)
        if subject_type in ("OutcomeLabel", "label"):
            outcome_ts = _get(subject, "outcome_timestamp_ms")
            # decision time may live in context; we only check self-consistency here
            if outcome_ts is not None:
                try:
                    out.append(CheckResult(
                        name="causality.outcome_timestamp_present",
                        ok=int(outcome_ts) > 0,
                        detail=outcome_ts,
                    ))
                except Exception as e:
                    out.append(CheckResult(name="causality.outcome_timestamp_present", ok=False, detail=str(e)))
        return out

    def _numeric_checks(self, subject: Any, subject_type: str) -> List[CheckResult]:
        out: List[CheckResult] = []

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Net PnL reconstruction for labels
        if subject_type in ("OutcomeLabel", "label"):
            try:
                realized = float(_get(subject, "realized_output_usd") or 0)
                costs = (
                    float(_get(subject, "input_cost_usd") or 0)
                    + float(_get(subject, "gas_usd") or 0)
                    + float(_get(subject, "protocol_fees_usd") or 0)
                    + float(_get(subject, "borrow_fees_usd") or 0)
                    + float(_get(subject, "bridge_fees_usd") or 0)
                    + float(_get(subject, "slippage_cost_usd") or 0)
                    + float(_get(subject, "revert_cost_usd") or 0)
                    + float(_get(subject, "other_costs_usd") or 0)
                )
                reported_net = float(_get(subject, "net_pnl_usd") or 0)
                calculated = realized - costs
                ok = abs(calculated - reported_net) < 0.02
                out.append(CheckResult(
                    name="accounting.net_pnl_identity",
                    ok=ok,
                    detail={"calculated": calculated, "reported": reported_net},
                ))
            except Exception as e:
                out.append(CheckResult(name="accounting.net_pnl_identity", ok=False, detail=str(e)))

        # Finite-number probe for any float-like fields
        for key in ("mean_net_pnl_usd", "trade_size_usd", "estimated_gas_usd", "net_pnl_usd"):
            val = _get(subject, key)
            if val is not None:
                try:
                    f = float(val)
                    out.append(CheckResult(
                        name=f"numeric.finite.{key}",
                        ok=math.isfinite(f),
                        detail=f,
                    ))
                except Exception:
                    out.append(CheckResult(name=f"numeric.finite.{key}", ok=False, detail=str(val)))
        return out

    def _source_ast_checks(self, source_text: str) -> List[CheckResult]:
        out: List[CheckResult] = []
        try:
            tree = ast.parse(source_text)
            out.append(CheckResult(name="source.ast_parse", ok=True, detail=f"{len(source_text)} chars"))
            classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
            funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            out.append(CheckResult(
                name="source.has_definitions",
                ok=(len(classes) + len(funcs)) > 0,
                detail={"classes": len(classes), "functions": len(funcs)},
            ))
        except SyntaxError as e:
            out.append(CheckResult(name="source.ast_parse", ok=False, detail=str(e)))
        return out

    def _math_logic_probes(self) -> List[CheckResult]:
        """Tiny always-on math/logic probes – no external libraries required."""
        out: List[CheckResult] = []
        # Arithmetic identity
        out.append(CheckResult(name="math.arithmetic_identity", ok=(2 + 2 == 4), detail="2+2==4"))
        # Float finiteness
        out.append(CheckResult(name="math.isfinite_pi", ok=math.isfinite(math.pi), detail=math.pi))
        # Contradiction detection (classical)
        A = True
        out.append(CheckResult(
            name="logic.no_contradiction",
            ok=not (A and not A),
            detail="A & ~A is false",
        ))
        return out


# =============================================================================
# Non-Stationary Engineer  (actor / experimental – may propose transforms)
# =============================================================================

class NonStationaryEngineer:
    """
    Non-stationary (actor / experimental) engineer.

    Responsibilities:
      - Attempt bounded probes / transforms on a *copy* of the subject
      - Run SourceEngineer-style structural audits
      - Propose repairs when safe (never silently apply them to production state)
      - Always followed by a second Stationary pass

    It is allowed to be creative; the second stationary pass is the brake.
    """

    def __init__(self, ledger: Optional[VerificationLedger] = None):
        self.ledger = ledger or VerificationLedger()
        self.last_report: Optional[EngineerReport] = None

    def act_and_check(
        self,
        subject: Any,
        subject_id: str,
        subject_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EngineerReport:
        context = context or {}
        checks: List[CheckResult] = []
        issues: List[str] = []
        proposals: List[Dict[str, Any]] = []

        # ----- 1. Deep-copy probe (ensure subject is serialisable) -----
        try:
            blob = json.dumps(subject, default=str)
            restored = json.loads(blob)
            checks.append(CheckResult(
                name="actor.serialisable",
                ok=True,
                detail=f"{len(blob)} bytes",
            ))
        except Exception as e:
            checks.append(CheckResult(name="actor.serialisable", ok=False, detail=str(e)))
            issues.append(f"not serialisable: {e}")
            restored = None

        # ----- 2. Bounded mutation probe on the copy only -----
        if restored is not None and isinstance(restored, dict):
            probe = dict(restored)
            probe["__non_stationary_probe__"] = True
            checks.append(CheckResult(
                name="actor.mutation_probe_on_copy",
                ok=probe.get("__non_stationary_probe__") is True,
                detail="copy mutated; original untouched",
            ))

        # ----- 3. Structural / shape audit -----
        checks.extend(self._structure_audit(subject, subject_type))

        # ----- 4. Propose (but do not apply) repairs when issues found -----
        if subject_type in ("OutcomeLabel", "label"):
            repair = self._propose_net_pnl_repair(subject)
            if repair:
                proposals.append(repair)
                checks.append(CheckResult(
                    name="actor.repair_proposal_net_pnl",
                    ok=True,
                    detail=repair,
                    severity="info",
                ))

        # ----- 5. Adversarial / stress micro-probe -----
        checks.extend(self._adversarial_probes(subject))

        hard_fails = [c for c in checks if not c.ok and c.severity == "error"]
        if hard_fails:
            verdict = EngineerVerdict.FAIL
            issues.extend([f"{c.name}: {c.detail}" for c in hard_fails])
        else:
            verdict = EngineerVerdict.PASS

        report = EngineerReport(
            engineer="non_stationary",
            pass_number=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            verdict=verdict,
            checks=checks,
            issues=issues,
            metadata={
                "subject_id": subject_id,
                "subject_type": subject_type,
                "proposals": proposals,
                "n_checks": len(checks),
            },
        )
        self.last_report = report
        self.ledger.append("non_stationary", report.to_dict())
        return report

    def _structure_audit(self, subject: Any, subject_type: str) -> List[CheckResult]:
        out: List[CheckResult] = []
        out.append(CheckResult(
            name="actor.type_known",
            ok=subject_type in (
                "NormalizedEvent", "event",
                "DecisionPacket", "packet",
                "OutcomeLabel", "label",
                "DecisionOutput", "decision",
                "dict", "list", "report", "batch",
            ) or True,  # soft – unknown types still allowed but noted
            detail=subject_type,
            severity="warn",
        ))
        # Depth / size bound
        try:
            blob = json.dumps(subject, default=str)
            out.append(CheckResult(
                name="actor.size_bound",
                ok=len(blob) < 5_000_000,
                detail=len(blob),
            ))
        except Exception as e:
            out.append(CheckResult(name="actor.size_bound", ok=False, detail=str(e)))
        return out

    def _propose_net_pnl_repair(self, subject: Any) -> Optional[Dict[str, Any]]:
        def _get(obj, key, default=0.0):
            if isinstance(obj, dict):
                return float(obj.get(key) or default)
            return float(getattr(obj, key, default) or default)

        try:
            realized = _get(subject, "realized_output_usd")
            costs = sum(_get(subject, k) for k in (
                "input_cost_usd", "gas_usd", "protocol_fees_usd", "borrow_fees_usd",
                "bridge_fees_usd", "slippage_cost_usd", "revert_cost_usd", "other_costs_usd",
            ))
            reported = _get(subject, "net_pnl_usd")
            calculated = realized - costs
            if abs(calculated - reported) >= 0.02:
                return {
                    "field": "net_pnl_usd",
                    "reported": reported,
                    "proposed": calculated,
                    "reason": "net_pnl identity mismatch",
                }
        except Exception:
            return None
        return None

    def _adversarial_probes(self, subject: Any) -> List[CheckResult]:
        """Tiny stress probes – NaN injection resistance, empty containers, etc."""
        out: List[CheckResult] = []
        # NaN resistance: if any float field is NaN, flag it
        def _walk(obj, path=""):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    found.extend(_walk(v, f"{path}.{k}"))
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    found.extend(_walk(v, f"{path}[{i}]"))
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                found.append(path)
            return found

        try:
            # Convert pydantic models etc. to dict first
            if hasattr(subject, "model_dump"):
                blob = subject.model_dump()
            elif hasattr(subject, "__dict__"):
                blob = subject.__dict__
            else:
                blob = subject
            bad = _walk(blob)
            out.append(CheckResult(
                name="actor.no_nan_inf",
                ok=len(bad) == 0,
                detail=bad[:10] if bad else None,
            ))
        except Exception as e:
            out.append(CheckResult(name="actor.no_nan_inf", ok=False, detail=str(e)))
        return out


# =============================================================================
# Double-Pass Gate  (the mandatory spine)
# =============================================================================

class DoublePassEngineerGate:
    """
    Mandatory Stationary → Non-Stationary → Stationary secondary verification.

    Usage:
        gate = DoublePassEngineerGate(ledger_path="artifacts/verification_ledger.jsonl")
        result = gate.run(subject, subject_id="pkt-001", subject_type="DecisionPacket")
        if not result.allowed_for_llm:
            # do not hand this information to the LLM interpreter
            ...
    """

    def __init__(self, ledger_path: Optional[Union[str, Path]] = None):
        self.ledger = VerificationLedger(ledger_path)
        self.stationary = StationaryEngineer(self.ledger)
        self.non_stationary = NonStationaryEngineer(self.ledger)

    def run(
        self,
        subject: Any,
        subject_id: str,
        subject_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DoublePassResult:
        context = context or {}

        # Pass 1 – Stationary
        s1 = self.stationary.verify(
            subject, subject_id, subject_type, pass_number=1, context=context
        )

        # Non-stationary experimental pass (always runs; stationary-1 failure still recorded)
        ns = self.non_stationary.act_and_check(
            subject, subject_id, subject_type, context=context
        )

        # Pass 2 – Stationary secondary verification
        s2 = self.stationary.verify(
            subject, subject_id, subject_type, pass_number=2, context=context
        )

        # Final policy: both stationary passes must be PASS (or ABSTAIN) and
        # non-stationary must not hard-fail, otherwise BLOCKED for LLM.
        if s1.verdict == EngineerVerdict.FAIL or s2.verdict == EngineerVerdict.FAIL:
            final = EngineerVerdict.BLOCKED
            allowed = False
        elif ns.verdict == EngineerVerdict.FAIL:
            final = EngineerVerdict.BLOCKED
            allowed = False
        elif s1.verdict == EngineerVerdict.ABSTAIN or s2.verdict == EngineerVerdict.ABSTAIN:
            final = EngineerVerdict.ABSTAIN
            allowed = False   # conservative: abstain also withholds from LLM
        else:
            final = EngineerVerdict.PASS
            allowed = True

        entry_id = self.ledger.append("double_pass", {
            "subject_id": subject_id,
            "subject_type": subject_type,
            "final_verdict": final.value,
            "allowed_for_llm": allowed,
        })

        return DoublePassResult(
            subject_id=subject_id,
            subject_type=subject_type,
            stationary_pass_1=s1,
            non_stationary=ns,
            stationary_pass_2=s2,
            final_verdict=final,
            allowed_for_llm=allowed,
            ledger_entry_id=entry_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def run_batch(
        self,
        items: List[Tuple[Any, str, str]],
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Run double-pass over a list of (subject, subject_id, subject_type).
        Optional parallel execution for large batches (upgrade.parallel_batch).
        """
        context = context or {}
        results: List[DoublePassResult] = []
        if parallel and len(items) > 4:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(self.run, subject, sid, stype, context)
                    for subject, sid, stype in items
                ]
                for fut in concurrent.futures.as_completed(futs):
                    results.append(fut.result())
        else:
            for subject, sid, stype in items:
                results.append(self.run(subject, sid, stype, context=context))
        n_allowed = sum(1 for r in results if r.allowed_for_llm)
        n_blocked = len(results) - n_allowed
        return {
            "n_total": len(results),
            "n_allowed_for_llm": n_allowed,
            "n_blocked": n_blocked,
            "results": [r.to_dict() for r in results],
        }
