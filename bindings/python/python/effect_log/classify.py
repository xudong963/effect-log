"""Auto-classification of tool effect kinds from function metadata.

Classifies callables into EffectKind categories using heuristic analysis
of function names, docstrings, parameter names, and source AST patterns.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from effect_log.effect_log_native import EffectKind, ToolDef

logger = logging.getLogger("effect_log.classify")

# ── Name prefix → EffectKind maps (longest match wins) ──────────────────────

_PREFIX_MAP: list[tuple[str, EffectKind]] = [
    # ReadOnly
    ("read_", EffectKind.ReadOnly),
    ("fetch_", EffectKind.ReadOnly),
    ("get_", EffectKind.ReadOnly),
    ("search_", EffectKind.ReadOnly),
    ("query_", EffectKind.ReadOnly),
    ("list_", EffectKind.ReadOnly),
    ("check_", EffectKind.ReadOnly),
    ("validate_", EffectKind.ReadOnly),
    ("describe_", EffectKind.ReadOnly),
    ("count_", EffectKind.ReadOnly),
    ("find_", EffectKind.ReadOnly),
    ("lookup_", EffectKind.ReadOnly),
    ("browse_", EffectKind.ReadOnly),
    ("view_", EffectKind.ReadOnly),
    ("show_", EffectKind.ReadOnly),
    ("inspect_", EffectKind.ReadOnly),
    ("parse_", EffectKind.ReadOnly),
    ("transform_", EffectKind.ReadOnly),
    ("format_", EffectKind.ReadOnly),
    ("log_", EffectKind.ReadOnly),
    ("trace_", EffectKind.ReadOnly),
    # IrreversibleWrite
    ("send_", EffectKind.IrreversibleWrite),
    ("email_", EffectKind.IrreversibleWrite),
    ("notify_", EffectKind.IrreversibleWrite),
    ("broadcast_", EffectKind.IrreversibleWrite),
    ("publish_", EffectKind.IrreversibleWrite),
    ("deploy_", EffectKind.IrreversibleWrite),
    ("delete_", EffectKind.IrreversibleWrite),
    ("remove_", EffectKind.IrreversibleWrite),
    ("destroy_", EffectKind.IrreversibleWrite),
    ("drop_", EffectKind.IrreversibleWrite),
    ("purge_", EffectKind.IrreversibleWrite),
    ("revoke_", EffectKind.IrreversibleWrite),
    ("terminate_", EffectKind.IrreversibleWrite),
    ("kill_", EffectKind.IrreversibleWrite),
    ("post_to_", EffectKind.IrreversibleWrite),
    ("tweet_", EffectKind.IrreversibleWrite),
    ("sms_", EffectKind.IrreversibleWrite),
    # IdempotentWrite
    ("create_", EffectKind.IdempotentWrite),
    ("add_", EffectKind.IdempotentWrite),
    ("upsert_", EffectKind.IdempotentWrite),
    ("update_", EffectKind.IdempotentWrite),
    ("put_", EffectKind.IdempotentWrite),
    ("save_", EffectKind.IdempotentWrite),
    ("insert_", EffectKind.IdempotentWrite),
    ("set_", EffectKind.IdempotentWrite),
    ("write_", EffectKind.IdempotentWrite),
    ("store_", EffectKind.IdempotentWrite),
    ("upload_", EffectKind.IdempotentWrite),
    ("register_", EffectKind.IdempotentWrite),
    ("configure_", EffectKind.IdempotentWrite),
    ("enable_", EffectKind.IdempotentWrite),
    ("disable_", EffectKind.IdempotentWrite),
    ("assign_", EffectKind.IdempotentWrite),
    ("tag_", EffectKind.IdempotentWrite),
    # Compensatable
    ("reserve_", EffectKind.Compensatable),
    ("lock_", EffectKind.Compensatable),
    ("allocate_", EffectKind.Compensatable),
    ("book_", EffectKind.Compensatable),
    ("hold_", EffectKind.Compensatable),
    ("checkout_", EffectKind.Compensatable),
    ("claim_", EffectKind.Compensatable),
    # ReadThenWrite
    ("transfer_", EffectKind.ReadThenWrite),
    ("swap_", EffectKind.ReadThenWrite),
    ("exchange_", EffectKind.ReadThenWrite),
    ("move_", EffectKind.ReadThenWrite),
    ("migrate_", EffectKind.ReadThenWrite),
    ("sync_", EffectKind.ReadThenWrite),
    ("reconcile_", EffectKind.ReadThenWrite),
]

# Sort by prefix length descending so longest match wins
_PREFIX_MAP.sort(key=lambda x: len(x[0]), reverse=True)

# Exact name matches for single-word function names
_EXACT_NAME_MAP: dict[str, EffectKind] = {
    "search": EffectKind.ReadOnly,
    "query": EffectKind.ReadOnly,
    "fetch": EffectKind.ReadOnly,
    "get": EffectKind.ReadOnly,
    "read": EffectKind.ReadOnly,
    "list": EffectKind.ReadOnly,
    "find": EffectKind.ReadOnly,
    "lookup": EffectKind.ReadOnly,
    "check": EffectKind.ReadOnly,
    "validate": EffectKind.ReadOnly,
    "count": EffectKind.ReadOnly,
    "parse": EffectKind.ReadOnly,
    "transform": EffectKind.ReadOnly,
    "format": EffectKind.ReadOnly,
    "log": EffectKind.ReadOnly,
    "send": EffectKind.IrreversibleWrite,
    "email": EffectKind.IrreversibleWrite,
    "notify": EffectKind.IrreversibleWrite,
    "publish": EffectKind.IrreversibleWrite,
    "deploy": EffectKind.IrreversibleWrite,
    "delete": EffectKind.IrreversibleWrite,
    "remove": EffectKind.IrreversibleWrite,
    "destroy": EffectKind.IrreversibleWrite,
    "purge": EffectKind.IrreversibleWrite,
    "create": EffectKind.IdempotentWrite,
    "upsert": EffectKind.IdempotentWrite,
    "update": EffectKind.IdempotentWrite,
    "save": EffectKind.IdempotentWrite,
    "insert": EffectKind.IdempotentWrite,
    "write": EffectKind.IdempotentWrite,
    "store": EffectKind.IdempotentWrite,
    "upload": EffectKind.IdempotentWrite,
    "register": EffectKind.IdempotentWrite,
    "reserve": EffectKind.Compensatable,
    "lock": EffectKind.Compensatable,
    "book": EffectKind.Compensatable,
    "transfer": EffectKind.ReadThenWrite,
    "swap": EffectKind.ReadThenWrite,
    "migrate": EffectKind.ReadThenWrite,
    "sync": EffectKind.ReadThenWrite,
}

# ── Docstring keyword families ───────────────────────────────────────────────

_DOCSTRING_KEYWORDS: list[tuple[list[str], EffectKind]] = [
    # ReadOnly
    (
        [
            "read-only",
            "readonly",
            "no side effect",
            "pure function",
            "retrieves",
            "fetches",
            "queries",
            "looks up",
            "returns data",
            "does not modify",
            "non-destructive",
            "safe to retry",
        ],
        EffectKind.ReadOnly,
    ),
    # IrreversibleWrite
    (
        [
            "irreversible",
            "cannot be undone",
            "sends email",
            "sends notification",
            "permanently delete",
            "destructive",
            "cannot undo",
            "broadcasts",
            "deploys",
            "publishes",
            "posts to",
            "external service",
            "side effect",
            "not idempotent",
        ],
        EffectKind.IrreversibleWrite,
    ),
    # IdempotentWrite
    (
        [
            "idempotent",
            "upsert",
            "safe to retry",
            "creates or updates",
            "overwrites",
            "replaces",
            "puts",
            "stores",
            "saves",
            "can be retried",
            "same result if repeated",
        ],
        EffectKind.IdempotentWrite,
    ),
    # Compensatable
    (
        [
            "compensat",
            "can be reversed",
            "can be undone",
            "rollback",
            "reservation",
            "temporary hold",
            "can cancel",
        ],
        EffectKind.Compensatable,
    ),
    # ReadThenWrite
    (
        [
            "read then write",
            "reads and writes",
            "transfers",
            "moves data",
            "migrates",
            "synchronizes",
            "reconciles",
        ],
        EffectKind.ReadThenWrite,
    ),
]

# ── Parameter name signals ───────────────────────────────────────────────────

_PARAM_SIGNALS: list[tuple[list[str], EffectKind]] = [
    # IrreversibleWrite: messaging/notification params
    (
        ["to", "recipient", "recipients", "email", "phone", "channel"],
        EffectKind.IrreversibleWrite,
    ),
    # IdempotentWrite: identity/key params
    (
        ["id", "key", "record_id", "entity_id", "pk", "primary_key"],
        EffectKind.IdempotentWrite,
    ),
    # ReadOnly: query/filter params
    (["query", "search", "filter", "q", "term", "keyword"], EffectKind.ReadOnly),
]

# ── AST patterns ─────────────────────────────────────────────────────────────

_AST_PATTERNS: list[tuple[str, EffectKind]] = [
    # HTTP method patterns
    (r"\.get\(", EffectKind.ReadOnly),
    (r"\.fetch\(", EffectKind.ReadOnly),
    (r"requests\.get\(", EffectKind.ReadOnly),
    (r"\.post\(", EffectKind.IrreversibleWrite),
    (r"requests\.post\(", EffectKind.IrreversibleWrite),
    (r"\.delete\(", EffectKind.IrreversibleWrite),
    (r"requests\.delete\(", EffectKind.IrreversibleWrite),
    (r"\.put\(", EffectKind.IdempotentWrite),
    (r"requests\.put\(", EffectKind.IdempotentWrite),
    (r"\.patch\(", EffectKind.IdempotentWrite),
    (r"requests\.patch\(", EffectKind.IdempotentWrite),
    # SDK patterns
    (r"smtp|sendmail|send_mail", EffectKind.IrreversibleWrite),
    (r"\.send\(", EffectKind.IrreversibleWrite),
    (r"\.publish\(", EffectKind.IrreversibleWrite),
    (r"\.select\(|\.query\(|\.find\(", EffectKind.ReadOnly),
    (r"\.insert\(|\.upsert\(|\.update\(|\.save\(", EffectKind.IdempotentWrite),
    (r"\.drop\(|\.truncate\(", EffectKind.IrreversibleWrite),
]

# ── Layer weights ────────────────────────────────────────────────────────────

_WEIGHT_NAME = 0.50
_WEIGHT_DOCSTRING = 0.25
_WEIGHT_PARAMS = 0.15
_WEIGHT_AST = 0.10


# ── Core types ───────────────────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """Result of classifying a single tool's effect kind."""

    effect_kind: EffectKind
    confidence: float  # 0.0 – 1.0
    reason: str  # human-readable explanation
    source: str  # "heuristic" | "llm" | "default"


@dataclass
class ClassificationReport:
    """Batch classification results with printable table and .apply()."""

    results: dict[str, ClassificationResult] = field(default_factory=dict)
    _funcs: dict[str, Callable] = field(default_factory=dict, repr=False)

    def apply(self, overrides: dict[str, EffectKind] | None = None) -> list[ToolDef]:
        """Convert results to ToolDef list, applying overrides.

        Args:
            overrides: Optional dict mapping function name -> EffectKind
                       to override the auto-classified result.

        Returns:
            List of ToolDef instances ready for EffectLog construction.
        """
        overrides = overrides or {}
        defs = []
        for name, result in self.results.items():
            kind = overrides.get(name, result.effect_kind)
            fn = self._funcs[name]

            def adapted(args, _fn=fn):
                return _fn(**args)

            defs.append(ToolDef(name, kind, adapted))
        return defs

    def __str__(self) -> str:
        lines = []
        max_name = max((len(n) for n in self.results), default=0)
        for name, r in self.results.items():
            kind_str = _kind_name(r.effect_kind)
            conf_str = f"{r.confidence:.2f}"
            warn = "" if r.confidence >= 0.6 else " !!!"
            lines.append(
                f"{name:<{max_name}}  -> {kind_str:<20} ({conf_str})  {r.reason}{warn}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ClassificationReport({len(self.results)} tools)"


# ── Scoring helpers ──────────────────────────────────────────────────────────
# Note: PyO3 EffectKind is not hashable, so we use int(kind) as dict keys
# and _kind_from_int() to convert back.

_ALL_KINDS = [
    EffectKind.ReadOnly,
    EffectKind.IdempotentWrite,
    EffectKind.Compensatable,
    EffectKind.IrreversibleWrite,
    EffectKind.ReadThenWrite,
]
_KIND_BY_INT = {int(k): k for k in _ALL_KINDS}


def _kind_from_int(i: int) -> EffectKind:
    return _KIND_BY_INT[i]


def _ki(kind: EffectKind) -> int:
    """Shorthand: EffectKind -> int key for score dicts."""
    return int(kind)


_KIND_NAMES = {
    int(EffectKind.ReadOnly): "ReadOnly",
    int(EffectKind.IdempotentWrite): "IdempotentWrite",
    int(EffectKind.Compensatable): "Compensatable",
    int(EffectKind.IrreversibleWrite): "IrreversibleWrite",
    int(EffectKind.ReadThenWrite): "ReadThenWrite",
}


def _kind_name(kind: EffectKind) -> str:
    """Get the string name of an EffectKind."""
    return _KIND_NAMES.get(int(kind), str(kind))


def _score_name(name: str) -> dict[int, float]:
    """Layer 1: Score by function name prefix or exact match."""
    scores: dict[int, float] = {}

    # Exact match first
    if name in _EXACT_NAME_MAP:
        kind = _EXACT_NAME_MAP[name]
        scores[_ki(kind)] = 1.0
        return scores

    # Prefix match (longest wins)
    for prefix, kind in _PREFIX_MAP:
        if name.startswith(prefix):
            scores[_ki(kind)] = 1.0
            return scores

    return scores


def _score_docstring(func: Callable) -> dict[int, float]:
    """Layer 2: Score by docstring keyword analysis."""
    scores: dict[int, float] = {}
    doc = inspect.getdoc(func)
    if not doc:
        return scores

    doc_lower = doc.lower()
    for keywords, kind in _DOCSTRING_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in doc_lower)
        if hits > 0:
            # Normalize: more keyword hits = higher confidence, capped at 1.0
            scores[_ki(kind)] = min(hits / 3.0, 1.0)

    return scores


def _score_params(func: Callable) -> dict[int, float]:
    """Layer 3: Score by parameter name signals."""
    scores: dict[int, float] = {}
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return scores

    param_names = [p.lower() for p in sig.parameters]
    for signal_names, kind in _PARAM_SIGNALS:
        hits = sum(1 for p in param_names if p in signal_names)
        if hits > 0:
            scores[_ki(kind)] = min(hits / 2.0, 1.0)

    return scores


def _score_ast(func: Callable) -> dict[int, float]:
    """Layer 4: Score by source code AST patterns."""
    scores: dict[int, float] = {}
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return scores

    for pattern, kind in _AST_PATTERNS:
        ki = _ki(kind)
        if re.search(pattern, source):
            scores[ki] = scores.get(ki, 0.0) + 0.5
            scores[ki] = min(scores[ki], 1.0)

    return scores


def _combine_scores(
    name_scores: dict[int, float],
    doc_scores: dict[int, float],
    param_scores: dict[int, float],
    ast_scores: dict[int, float],
) -> tuple[EffectKind, float, str]:
    """Combine weighted scores from all layers.

    Returns (effect_kind, confidence, reason).
    """
    all_kinds = set(name_scores) | set(doc_scores) | set(param_scores) | set(ast_scores)

    if not all_kinds:
        return EffectKind.IrreversibleWrite, 0.0, "default (no signals)"

    best_kind_int = _ki(EffectKind.IrreversibleWrite)
    best_score = 0.0
    reasons = []

    for ki in all_kinds:
        score = (
            name_scores.get(ki, 0.0) * _WEIGHT_NAME
            + doc_scores.get(ki, 0.0) * _WEIGHT_DOCSTRING
            + param_scores.get(ki, 0.0) * _WEIGHT_PARAMS
            + ast_scores.get(ki, 0.0) * _WEIGHT_AST
        )
        if score > best_score:
            best_score = score
            best_kind_int = ki

    # Build reason string from contributing layers
    if best_kind_int in name_scores:
        reasons.append("name")
    if best_kind_int in doc_scores:
        reasons.append("docstring")
    if best_kind_int in param_scores:
        reasons.append("params")
    if best_kind_int in ast_scores:
        reasons.append("ast")

    reason = " + ".join(reasons) if reasons else "default"
    # Clamp confidence to [0, 1]
    confidence = min(best_score, 1.0)

    return _kind_from_int(best_kind_int), confidence, reason


# ── Public API ───────────────────────────────────────────────────────────────


def classify_effect_kind(
    func: Callable, name: str | None = None
) -> ClassificationResult:
    """Classify a callable's effect kind using heuristic analysis.

    Uses 4 weighted layers: name prefix (0.50), docstring keywords (0.25),
    parameter names (0.15), and source AST patterns (0.10).

    Args:
        func: The callable to classify.
        name: Optional override for the function name (uses func.__name__ if None).

    Returns:
        ClassificationResult with effect_kind, confidence, reason, and source.
    """
    fname = name or getattr(func, "__name__", "")

    name_scores = _score_name(fname)
    doc_scores = _score_docstring(func)
    param_scores = _score_params(func)
    ast_scores = _score_ast(func)

    kind, confidence, reason = _combine_scores(
        name_scores, doc_scores, param_scores, ast_scores
    )

    # Safety: Compensatable requires a compensation function we can't detect,
    # so downgrade to IrreversibleWrite with a hint
    if int(kind) == int(EffectKind.Compensatable):
        reason += " (auto-downgraded from Compensatable — provide compensate= to use Compensatable)"
        kind = EffectKind.IrreversibleWrite

    # Log the classification
    if confidence >= 0.6:
        logger.info("%s -> %s (%.2f, %s)", fname, _kind_name(kind), confidence, reason)
    else:
        logger.warning(
            "%s -> %s (%.2f, %s — consider specifying explicitly)",
            fname,
            _kind_name(kind),
            confidence,
            reason,
        )

    return ClassificationResult(
        effect_kind=kind,
        confidence=confidence,
        reason=reason,
        source="heuristic",
    )


def classify_from_name(name: str) -> ClassificationResult:
    """Classify effect kind from a function name string only.

    Uses only the name prefix/exact match layer. Useful for middleware
    wrapping where the original callable isn't available.

    Args:
        name: The function/tool name to classify.

    Returns:
        ClassificationResult based on name analysis only.
    """
    name_scores = _score_name(name)

    if name_scores:
        best_ki = max(name_scores, key=name_scores.get)
        kind = _kind_from_int(best_ki)
        confidence = name_scores[best_ki] * _WEIGHT_NAME
        reason = "name"
    else:
        kind = EffectKind.IrreversibleWrite
        confidence = 0.0
        reason = "default (no name match)"

    if int(kind) == int(EffectKind.Compensatable):
        reason += " (auto-downgraded from Compensatable)"
        kind = EffectKind.IrreversibleWrite

    if confidence >= 0.6:
        logger.info("%s -> %s (%.2f, %s)", name, _kind_name(kind), confidence, reason)
    else:
        logger.warning(
            "%s -> %s (%.2f, %s — consider specifying explicitly)",
            name,
            _kind_name(kind),
            confidence,
            reason,
        )

    return ClassificationResult(
        effect_kind=kind,
        confidence=confidence,
        reason=reason,
        source="heuristic",
    )


def classify_tools(funcs: Sequence[Callable]) -> ClassificationReport:
    """Batch classify a list of callables.

    Args:
        funcs: Sequence of callables to classify.

    Returns:
        ClassificationReport with results for each function.
        Use report.apply(overrides=) to convert to ToolDef list.
    """
    report = ClassificationReport()
    for func in funcs:
        name = getattr(func, "__name__", str(func))
        result = classify_effect_kind(func, name)
        report.results[name] = result
        report._funcs[name] = func
    return report


def classify_with_llm(func: Callable) -> ClassificationResult:
    """Classify effect kind using an LLM as fallback.

    Requires EFFECT_LOG_LLM_CLASSIFY=1 environment variable and either
    the anthropic or openai SDK installed. Results are cached.

    Args:
        func: The callable to classify.

    Returns:
        ClassificationResult from LLM analysis.
    """
    import os

    if not os.environ.get("EFFECT_LOG_LLM_CLASSIFY"):
        raise RuntimeError(
            "LLM classification requires EFFECT_LOG_LLM_CLASSIFY=1 environment variable"
        )

    fname = getattr(func, "__name__", "unknown")
    doc = inspect.getdoc(func) or ""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = ""

    prompt = (
        f"Classify this Python function's side-effect kind.\n"
        f"Function name: {fname}\n"
        f"Docstring: {doc}\n"
        f"Source:\n{source}\n\n"
        f"Classify as exactly one of: ReadOnly, IdempotentWrite, "
        f"IrreversibleWrite, ReadThenWrite\n"
        f"Respond with just the classification name."
    )

    # Try anthropic first, then openai
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip()
    except (ImportError, Exception):
        try:
            import openai

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()
        except (ImportError, Exception) as e:
            logger.warning("LLM classification failed for %s: %s", fname, e)
            return ClassificationResult(
                effect_kind=EffectKind.IrreversibleWrite,
                confidence=0.0,
                reason="llm failed",
                source="llm",
            )

    kind_map = {
        "ReadOnly": EffectKind.ReadOnly,
        "IdempotentWrite": EffectKind.IdempotentWrite,
        "IrreversibleWrite": EffectKind.IrreversibleWrite,
        "ReadThenWrite": EffectKind.ReadThenWrite,
        "Compensatable": EffectKind.IrreversibleWrite,  # safety downgrade
    }

    kind = kind_map.get(answer, EffectKind.IrreversibleWrite)
    confidence = 0.80 if answer in kind_map else 0.0

    logger.info("%s -> %s (%.2f, llm)", fname, _kind_name(kind), confidence)

    return ClassificationResult(
        effect_kind=kind,
        confidence=confidence,
        reason=f"llm: {answer}",
        source="llm",
    )
