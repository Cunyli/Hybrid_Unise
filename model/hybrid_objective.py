from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


LM_OBJECTIVE_DEFAULTS: dict[str, bool | float] = {
    "history_embedding_dropout_prob": 0.0,
    "history_corruption_replacement_fraction": 0.0,
    "transition_stall_history_corruption": False,
    "prefix_only_aux_weight": 0.0,
    "transition_ce_multiplier": 1.0,
    "normalize_transition_weights_per_sample": False,
    "transition_predecessor_margin": 1.0,
    "transition_predecessor_margin_weight": 0.0,
}
LM_OBJECTIVE_ALLOWED_KEYS = frozenset(LM_OBJECTIVE_DEFAULTS)
_LM_OBJECTIVE_BOOL_KEYS = frozenset(
    {
        "transition_stall_history_corruption",
        "normalize_transition_weights_per_sample",
    }
)


def canonical_lm_objective(value: Mapping[str, Any] | None) -> dict[str, bool | float]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("lm_objective must be a mapping")

    unknown_keys = sorted(set(value) - LM_OBJECTIVE_ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(
            "lm_objective contains unsupported keys "
            f"{unknown_keys}; expected only {sorted(LM_OBJECTIVE_ALLOWED_KEYS)}"
        )

    canonical: dict[str, bool | float] = {}
    for key, default in LM_OBJECTIVE_DEFAULTS.items():
        raw_value = value.get(key, default)
        if key in _LM_OBJECTIVE_BOOL_KEYS:
            if not isinstance(raw_value, bool):
                raise ValueError(f"lm_objective.{key} must be a bool")
            canonical[key] = raw_value
            continue

        if isinstance(raw_value, bool):
            raise ValueError(
                f"lm_objective.{key} must be a finite number, not a bool"
            )
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"lm_objective.{key} must be a finite number"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"lm_objective.{key} must be a finite number")
        canonical[key] = number
    return canonical


def lm_objective_identity(
    value: Mapping[str, Any] | None,
) -> tuple[dict[str, bool | float], str, str]:
    canonical = canonical_lm_objective(value)
    canonical_json = json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return canonical, canonical_json, sha256


def validate_checkpoint_lm_objective_identity(
    checkpoint: Mapping[str, Any],
    expected_json: str,
    expected_sha256: str,
) -> bool:
    checkpoint_json = checkpoint.get("hybrid_lm_objective_json")
    checkpoint_sha256 = checkpoint.get("hybrid_lm_objective_sha256")
    if checkpoint_json is None and checkpoint_sha256 is None:
        raise ValueError(
            "Checkpoint Hybrid-UniSE LM objective metadata is missing; "
            "legacy checkpoints may be used only for explicit stage initialization"
        )
    if not isinstance(checkpoint_json, str) or not isinstance(
        checkpoint_sha256,
        str,
    ):
        raise ValueError(
            "Checkpoint Hybrid-UniSE LM objective metadata is incomplete"
        )

    try:
        decoded = json.loads(checkpoint_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Checkpoint Hybrid-UniSE LM objective JSON is invalid"
        ) from exc
    _, canonical_json, canonical_sha256 = lm_objective_identity(decoded)
    if checkpoint_json != canonical_json or checkpoint_sha256 != canonical_sha256:
        raise ValueError(
            "Checkpoint Hybrid-UniSE LM objective JSON/SHA256 metadata is inconsistent"
        )
    if (
        checkpoint_json != expected_json
        or checkpoint_sha256 != expected_sha256
    ):
        raise ValueError(
            "Checkpoint Hybrid-UniSE LM objective does not match current config"
        )
    return True
