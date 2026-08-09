from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from model.hybrid_objective import (
    LM_OBJECTIVE_DEFAULTS,
    lm_objective_identity,
    validate_checkpoint_lm_objective_identity,
)
from scripts.validate_hybrid_config import validate_config


SCHEMA_VERSION = "transition_margin_pair/v1"
TOPOLOGY_SCHEMA_VERSION = "transition_margin_topology/v1"
ARM_NAMES = ("control", "treatment")
DEFAULT_SPEC_PATH = Path(
    "conf/experiments/transition_margin_probe_pair_v1.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_RETURN_CODE = 124
START_FAILURE_RETURN_CODE = 125
TERMINATION_GRACE_SECONDS = 10


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_with_payload_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["receipt_payload_sha256"] = sha256_text(canonical_json(payload))
    return receipt


def load_valid_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError(f"Receipt must be a mapping: {path}")
    payload = dict(receipt)
    recorded_sha256 = payload.pop("receipt_payload_sha256", None)
    actual_sha256 = sha256_text(canonical_json(payload))
    if recorded_sha256 != actual_sha256:
        raise ValueError(f"Receipt payload SHA256 mismatch: {path}")
    return receipt


def require_receipt_fields(
    receipt: dict[str, Any],
    expected: dict[str, Any],
    context: str,
) -> None:
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"{context} receipt binding mismatch for {field}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_pair_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError("Transition-margin pair spec must be a mapping")
    return spec


def require_relative_output_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a repository-relative path")
    return path


def source_checkpoint_artifact(spec: dict[str, Any]) -> dict[str, Any]:
    matches = [
        artifact
        for artifact in spec.get("artifacts", [])
        if artifact.get("role") == "source_checkpoint"
    ]
    if len(matches) != 1:
        raise ValueError("Pair spec must declare exactly one source_checkpoint artifact")
    return matches[0]


def topology_receipt_template(spec: dict[str, Any]) -> dict[str, Any]:
    return receipt_with_payload_sha256(
        {
            "schema_version": TOPOLOGY_SCHEMA_VERSION,
            "receipt_type": "queue_topology",
            "pair_id": spec["pair_id"],
            "status": "not_checked",
            "checked_at": None,
            "requested_gpus": 1,
            "selected_gpu_count": None,
            "queue_snapshot_sha256": None,
            "topology_snapshot_sha256": None,
        }
    )


def _parse_utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def load_verified_topology_receipt(
    path: Path,
    spec: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    receipt = load_valid_receipt(path.expanduser().resolve())
    expected_keys = {
        "schema_version",
        "receipt_type",
        "pair_id",
        "status",
        "checked_at",
        "requested_gpus",
        "selected_gpu_count",
        "queue_snapshot_sha256",
        "topology_snapshot_sha256",
        "receipt_payload_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("Queue/topology receipt fields do not match the contract")
    require_receipt_fields(
        receipt,
        {
            "schema_version": TOPOLOGY_SCHEMA_VERSION,
            "receipt_type": "queue_topology",
            "pair_id": spec["pair_id"],
            "status": "verified",
            "requested_gpus": 1,
            "selected_gpu_count": 1,
        },
        "queue/topology",
    )
    for field in ("queue_snapshot_sha256", "topology_snapshot_sha256"):
        value = receipt[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Queue/topology receipt {field} must be a SHA256")

    checked_at = _parse_utc_timestamp(receipt["checked_at"], "checked_at")
    current_time = now or datetime.now(timezone.utc)
    age_seconds = (current_time - checked_at).total_seconds()
    max_age_seconds = spec["execution"][
        "queue_topology_receipt_max_age_seconds"
    ]
    if age_seconds < 0 or age_seconds > max_age_seconds:
        raise ValueError("Queue/topology receipt is not fresh")
    return receipt


def validate_pair_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    pair_id = spec.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be a non-empty string")

    required_ancestor_sha = spec.get("required_ancestor_sha")
    if (
        not isinstance(required_ancestor_sha, str)
        or len(required_ancestor_sha) != 40
    ):
        raise ValueError("required_ancestor_sha must be a full Git SHA")

    experiment = spec.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a mapping")
    updates_per_arm = experiment.get("updates_per_arm")
    if isinstance(updates_per_arm, bool) or not isinstance(updates_per_arm, int):
        raise ValueError("experiment.updates_per_arm must be an integer")
    if not 50 <= updates_per_arm <= 100:
        raise ValueError("experiment.updates_per_arm must be in [50, 100]")
    trainer_seed = experiment.get("trainer_seed")
    if isinstance(trainer_seed, bool) or not isinstance(trainer_seed, int):
        raise ValueError("experiment.trainer_seed must be an integer")
    if experiment.get("arm_order") != list(ARM_NAMES):
        raise ValueError(f"experiment.arm_order must be {list(ARM_NAMES)}")
    arms = experiment.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
        raise ValueError(f"experiment.arms must contain exactly {list(ARM_NAMES)}")
    control_weight = arms["control"].get(
        "transition_predecessor_margin_weight"
    )
    treatment_weight = arms["treatment"].get(
        "transition_predecessor_margin_weight"
    )
    if isinstance(control_weight, bool) or float(control_weight) != 0.0:
        raise ValueError("control margin weight must be exactly 0.0")
    if isinstance(treatment_weight, bool) or float(treatment_weight) != 0.25:
        raise ValueError("treatment margin weight must be preregistered as 0.25")

    execution = spec.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("execution must be a mapping")
    for field in ("run_root", "log_root", "checkpoint_root"):
        require_relative_output_path(str(execution.get(field, "")), f"execution.{field}")
    expected_execution_limits = {
        "max_seconds_per_arm": 1800,
        "max_pair_gpu_seconds": 3600,
        "queue_topology_receipt_max_age_seconds": 3600,
    }
    for field, expected_value in expected_execution_limits.items():
        value = execution.get(field)
        if isinstance(value, bool) or value != expected_value:
            raise ValueError(
                f"execution.{field} must be preregistered as {expected_value}"
            )
    if (
        len(ARM_NAMES) * execution["max_seconds_per_arm"]
        > execution["max_pair_gpu_seconds"]
    ):
        raise ValueError("Per-arm timeouts exceed the pair GPU budget")

    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifacts must be a non-empty list")
    artifact_names = []
    artifacts_by_name = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Each artifact must be a mapping")
        name = artifact.get("name")
        path = artifact.get("path")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(name, str) or not name:
            raise ValueError("Each artifact needs a non-empty name")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"Artifact {name!r} path must be absolute")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"Artifact {name!r} needs a lowercase SHA256")
        size_bytes = artifact.get("size_bytes")
        if size_bytes is not None and (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(f"Artifact {name!r} size_bytes must be positive")
        artifact_names.append(name)
        artifacts_by_name[name] = artifact
    if len(artifact_names) != len(set(artifact_names)):
        raise ValueError("Artifact names must be unique")
    required_artifact_names = {
        "source_checkpoint_step200",
        "semantic_vq_codebook_k128",
        "wavlm_config",
        "wavlm_preprocessor_config",
        "wavlm_pytorch_model",
        "simulation_config",
        "train_clean_shards",
        "train_noise_shards",
        "train_rir_shards",
        "validation_recipes",
        "test_recipes",
    }
    missing_artifacts = sorted(required_artifact_names - set(artifact_names))
    if missing_artifacts:
        raise ValueError(f"Pair spec is missing required artifacts: {missing_artifacts}")

    evaluation = spec.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    panel_rows = evaluation.get("panel_rows")
    if not isinstance(panel_rows, list) or len(panel_rows) != 4:
        raise ValueError("evaluation.panel_rows must contain exactly four rows")
    panel_uids = []
    panel_tasks = []
    for row in panel_rows:
        if not isinstance(row, dict):
            raise ValueError("Each evaluation panel row must be a mapping")
        uid = row.get("uid")
        task = row.get("task")
        tensor_artifact = row.get("tensor_artifact")
        first_error_position = row.get("step200_first_error_position")
        positional_agreement = row.get("step200_positional_agreement")
        predecessor_token = row.get(
            "step200_first_error_predecessor_token"
        )
        if not isinstance(uid, str) or not uid:
            raise ValueError("Each evaluation row needs a non-empty uid")
        if task not in {"cs", "sv"}:
            raise ValueError("Each evaluation row task must be cs or sv")
        if tensor_artifact not in artifact_names:
            raise ValueError(
                f"Evaluation tensor artifact is not declared: {tensor_artifact!r}"
            )
        if artifacts_by_name[tensor_artifact].get("role") != "evaluation_tensor":
            raise ValueError(
                f"Evaluation row artifact is not an evaluation_tensor: "
                f"{tensor_artifact!r}"
            )
        if (
            isinstance(first_error_position, bool)
            or not isinstance(first_error_position, int)
            or first_error_position < 0
        ):
            raise ValueError(
                "Each evaluation row needs a non-negative step200 first error"
            )
        if (
            isinstance(positional_agreement, bool)
            or not isinstance(positional_agreement, (int, float))
            or not 0.0 <= positional_agreement <= 1.0
        ):
            raise ValueError(
                "Each evaluation row needs a step200 positional agreement in [0, 1]"
            )
        if (
            isinstance(predecessor_token, bool)
            or not isinstance(predecessor_token, int)
            or predecessor_token != 5
        ):
            raise ValueError(
                "Each frozen step200 first-error predecessor token must be 5"
            )
        panel_uids.append(uid)
        panel_tasks.append(task)
    if len(panel_uids) != len(set(panel_uids)):
        raise ValueError("Evaluation panel UIDs must be unique")
    if sorted(panel_tasks) != ["cs", "cs", "sv", "sv"]:
        raise ValueError("Evaluation panel must contain two CS and two SV rows")
    for field in ("panel_manifest_artifact", "baseline_rows_artifact"):
        if evaluation.get(field) not in artifact_names:
            raise ValueError(f"evaluation.{field} must name a declared artifact")
    expected_evaluation_roles = {
        "panel_manifest_artifact": "evaluation_manifest",
        "baseline_rows_artifact": "evaluation_baseline",
    }
    for field, expected_role in expected_evaluation_roles.items():
        artifact = artifacts_by_name[evaluation[field]]
        if artifact.get("role") != expected_role:
            raise ValueError(
                f"evaluation.{field} artifact role must be {expected_role}"
            )
    if evaluation.get("oracle_target_field") != "token_targets":
        raise ValueError("evaluation.oracle_target_field must be token_targets")
    expected_listening = {
        "listener_id": "project_owner_01",
        "mapping": "sha256_lexicographic",
        "mapping_message": "pair_id\\0uid\\0arm",
        "references": ["noisy", "clean"],
        "rating_categories": [
            "attenuation",
            "burst",
            "instability",
            "intelligibility",
        ],
        "rating_values": ["a_worse", "tie", "b_worse"],
        "max_plays_per_clip": 2,
        "fixed_device_and_volume": True,
        "loudness_normalization": False,
        "unseal_after_rating_receipt": True,
        "failure_rule": "treatment_worse_in_any_category_on_any_row",
    }
    if evaluation.get("blind_listening") != expected_listening:
        raise ValueError("evaluation.blind_listening does not match the contract")
    expected_decision_gates = {
        "positional_agreement": {
            "overall_absolute_pp_min": 5.0,
            "each_cs_sv_absolute_pp_min": 2.0,
        },
        "first_error_position": {
            "median_token_delta_min": 2,
            "rows_later_min": 3,
        },
        "transition_predecessor_rate": {
            "denominator": "all_valid_transition_positions",
            "step200_historical": {
                "overall": {
                    "predecessor_copies": 129,
                    "valid_transitions": 296,
                },
                "cs": {
                    "predecessor_copies": 127,
                    "valid_transitions": 275,
                },
                "sv": {
                    "predecessor_copies": 2,
                    "valid_transitions": 21,
                },
            },
            "overall_relative_reduction_min": 0.25,
            "overall_absolute_pp_reduction_min": 5.0,
            "each_cs_sv_worsening_pp_max": 5.0,
        },
        "teacher_accuracy": {
            "each_cs_sv_drop_pp_max": 2.0,
        },
        "teacher_nll": {
            "each_cs_sv_increase_max": 0.10,
        },
    }
    if evaluation.get("decision_gates") != expected_decision_gates:
        raise ValueError("evaluation.decision_gates does not match the contract")

    base_config = spec.get("base_config")
    if not isinstance(base_config, dict):
        raise ValueError("base_config must be a mapping")
    if base_config.get("model_type") != "hybrid_unise":
        raise ValueError("base_config.model_type must be hybrid_unise")
    if base_config.get("stage") != "gen":
        raise ValueError("base_config.stage must be gen")
    if base_config.get("resume") is not None:
        raise ValueError("Paired probe forbids resume; use explicit stage initialization")
    if base_config.get("devices") != [0]:
        raise ValueError("Paired probe is preregistered for exactly one GPU")
    if base_config.get("max_steps") != updates_per_arm:
        raise ValueError("base_config.max_steps must equal updates_per_arm")
    if base_config.get("max_epochs") != 1:
        raise ValueError("base_config.max_epochs must be exactly 1")
    if base_config.get("seed") != experiment.get("trainer_seed"):
        raise ValueError("Trainer seed does not match the experiment contract")

    source_checkpoint = source_checkpoint_artifact(spec)
    if source_checkpoint.get("name") != "source_checkpoint_step200":
        raise ValueError(
            "source_checkpoint role must use source_checkpoint_step200"
        )
    if base_config.get("stage_init_checkpoint") != source_checkpoint.get("path"):
        raise ValueError("base_config.stage_init_checkpoint must match the source artifact")

    dataset_config = base_config.get("dataset_config", {})
    train_kwargs = dataset_config.get("train_kwargs", {})
    val_kwargs = dataset_config.get("val_kwargs", {})
    test_kwargs = dataset_config.get("test_kwargs", {})
    if train_kwargs.get("shard_shuffle_seed") != experiment.get("trainer_seed"):
        raise ValueError("Data-order seed does not match the trainer seed")
    if train_kwargs.get("skip_bad_samples") is not False:
        raise ValueError("Paired probe must fail instead of skipping a bad sample")

    artifact_paths = {
        name: Path(artifact["path"])
        for name, artifact in artifacts_by_name.items()
    }
    expected_path_bindings = {
        "xcodec.codebook_path": (
            Path(base_config.get("xcodec", {}).get("codebook_path", "")),
            artifact_paths["semantic_vq_codebook_k128"],
        ),
        "train.simulation_config": (
            Path(train_kwargs.get("simulation_config", "")),
            artifact_paths["simulation_config"],
        ),
        "validation.simulation_config": (
            Path(val_kwargs.get("simulation_config", "")),
            artifact_paths["simulation_config"],
        ),
        "test.simulation_config": (
            Path(test_kwargs.get("simulation_config", "")),
            artifact_paths["simulation_config"],
        ),
        "validation.recipe_manifest": (
            Path(val_kwargs.get("recipe_manifest", "")),
            artifact_paths["validation_recipes"],
        ),
        "test.recipe_manifest": (
            Path(test_kwargs.get("recipe_manifest", "")),
            artifact_paths["test_recipes"],
        ),
    }
    split_root = Path(train_kwargs.get("split_root", ""))
    expected_path_bindings.update(
        {
            "train.clean_shards": (
                split_root / "train/clean_shards.jsonl",
                artifact_paths["train_clean_shards"],
            ),
            "train.noise_shards": (
                split_root / "train/noise_shards.jsonl",
                artifact_paths["train_noise_shards"],
            ),
            "train.rir_shards": (
                split_root / "train/rir_shards.jsonl",
                artifact_paths["train_rir_shards"],
            ),
        }
    )
    for field, (configured_path, artifact_path) in expected_path_bindings.items():
        if configured_path != artifact_path:
            raise ValueError(
                f"{field} does not match its immutable artifact path"
            )

    wavlm_root = Path(
        base_config.get("wavlm", {}).get("pretrained_name_or_path", "")
    )
    xcodec_wavlm_root = Path(
        base_config.get("xcodec", {}).get("wavlm_model_path", "")
    )
    if wavlm_root != xcodec_wavlm_root:
        raise ValueError("wavlm and xcodec must use the same WavLM asset root")
    expected_wavlm_artifacts = {
        "wavlm_config": wavlm_root / "config.json",
        "wavlm_preprocessor_config": wavlm_root / "preprocessor_config.json",
        "wavlm_pytorch_model": wavlm_root / "pytorch_model.bin",
    }
    for name, expected_path in expected_wavlm_artifacts.items():
        if artifact_paths[name] != expected_path:
            raise ValueError(f"{name} does not bind the configured WavLM root")

    base_objective = base_config.get("lm_objective")
    if not isinstance(base_objective, dict):
        raise ValueError("base_config.lm_objective must be a mapping")
    if "transition_predecessor_margin_weight" in base_objective:
        raise ValueError("Arm weight must be injected from experiment.arms")
    canonical_objective, _, _ = lm_objective_identity(base_objective)
    expected_objective = dict(LM_OBJECTIVE_DEFAULTS)
    expected_objective["transition_predecessor_margin"] = float(
        experiment.get("transition_predecessor_margin")
    )
    if canonical_objective != expected_objective:
        raise ValueError(
            "No lm_objective change other than the arm margin weight is allowed"
        )


def git_identity(repo_root: Path, required_ancestor_sha: str) -> dict[str, Any]:
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    ancestor_result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_ancestor_sha, head_sha],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor_result.returncode not in {0, 1}:
        raise RuntimeError("Could not verify required Git ancestor")
    return {
        "head_sha": head_sha,
        "required_ancestor_sha": required_ancestor_sha,
        "required_ancestor_present": ancestor_result.returncode == 0,
        "worktree_clean": not status_lines,
        "status_lines": status_lines,
    }


def verify_artifact_records(
    artifacts: list[dict[str, Any]],
    *,
    verify: bool,
) -> tuple[list[dict[str, Any]], bool]:
    records = []
    all_verified = verify
    for artifact in artifacts:
        record = {
            "name": artifact["name"],
            "role": artifact["role"],
            "path": artifact["path"],
            "expected_sha256": artifact["sha256"],
            "expected_size_bytes": artifact.get("size_bytes"),
        }
        if not verify:
            record["status"] = "not_checked"
            records.append(record)
            continue

        path = Path(artifact["path"])
        if not path.is_file():
            record["status"] = "missing"
            all_verified = False
            records.append(record)
            continue
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        record["actual_size_bytes"] = actual_size
        record["actual_sha256"] = actual_sha256
        size_matches = (
            artifact.get("size_bytes") is None
            or actual_size == artifact["size_bytes"]
        )
        sha_matches = actual_sha256 == artifact["sha256"]
        record["status"] = "verified" if size_matches and sha_matches else "mismatch"
        all_verified = all_verified and size_matches and sha_matches
        records.append(record)
    return records, all_verified


def build_arm_config(
    spec: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    config = copy.deepcopy(spec["base_config"])
    weight = float(
        spec["experiment"]["arms"][arm][
            "transition_predecessor_margin_weight"
        ]
    )
    config["lm_objective"]["transition_predecessor_margin_weight"] = weight
    config["log_dir"] = str(Path(spec["execution"]["log_root"]) / arm)
    config["checkpoint_dir"] = str(
        Path(spec["execution"]["checkpoint_root"]) / arm
    )
    _, objective_json, objective_sha256 = lm_objective_identity(
        config["lm_objective"]
    )
    config["transition_margin_probe"] = {
        "pair_id": spec["pair_id"],
        "arm": arm,
        "lm_objective_json": objective_json,
        "lm_objective_sha256": objective_sha256,
        "source_checkpoint_sha256": source_checkpoint_artifact(spec)["sha256"],
    }
    return config


def differing_leaf_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(differing_leaf_paths(left[key], right[key], path))
        return paths
    if left != right:
        return {prefix}
    return set()


def common_scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    common = copy.deepcopy(config)
    common.pop("log_dir")
    common.pop("checkpoint_dir")
    common.pop("transition_margin_probe")
    common["lm_objective"].pop("transition_predecessor_margin_weight")
    return common


def prepare_pair(
    spec_path: Path,
    *,
    output_dir: Path | None = None,
    verify_artifacts: bool = False,
    queue_topology_receipt_path: Path | None = None,
    now: datetime | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    spec = load_pair_spec(spec_path)
    validate_pair_spec(spec)
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (repo_root / spec["execution"]["run_root"]).resolve()
    )
    existing_runtime_receipts: dict[str, dict[str, dict[str, Any] | None]] = {}
    for arm in ARM_NAMES:
        launch_path = output_dir / arm / "launch_receipt.json"
        result_path = output_dir / arm / "result_receipt.json"
        existing_runtime_receipts[arm] = {
            "launch": (
                load_valid_receipt(launch_path) if launch_path.exists() else None
            ),
            "result": (
                load_valid_receipt(result_path) if result_path.exists() else None
            ),
        }
    any_runtime_started = any(
        receipt is not None and receipt.get("status") != "not_run"
        for receipts in existing_runtime_receipts.values()
        for receipt in receipts.values()
    )
    topology_template = topology_receipt_template(spec)
    queue_topology_receipt = topology_template
    if queue_topology_receipt_path is not None:
        queue_topology_receipt = load_verified_topology_receipt(
            queue_topology_receipt_path,
            spec,
            now=now,
        )

    configs = {arm: build_arm_config(spec, arm) for arm in ARM_NAMES}
    allowed_differences = {
        "checkpoint_dir",
        "lm_objective.transition_predecessor_margin_weight",
        "log_dir",
        "transition_margin_probe.arm",
        "transition_margin_probe.lm_objective_json",
        "transition_margin_probe.lm_objective_sha256",
    }
    actual_differences = differing_leaf_paths(
        configs["control"],
        configs["treatment"],
    )
    if actual_differences != allowed_differences:
        raise ValueError(
            "Control/treatment config differences do not match the allowlist: "
            f"{sorted(actual_differences)}"
        )

    scientific_configs = {
        arm: common_scientific_config(config)
        for arm, config in configs.items()
    }
    if scientific_configs["control"] != scientific_configs["treatment"]:
        raise ValueError("Control/treatment scientific configs are not identical")
    common_config_json = canonical_json(scientific_configs["control"])
    common_config_sha256 = sha256_text(common_config_json)

    arm_receipt_data: dict[str, Any] = {}
    for arm, config in configs.items():
        config_path = output_dir / arm / "config.yaml"
        errors = validate_config(
            config,
            config_path,
            check_external_paths=verify_artifacts,
        )
        if errors:
            raise ValueError(f"{arm} config validation failed: {errors}")
        _, objective_json, objective_sha256 = lm_objective_identity(
            config["lm_objective"]
        )
        arm_receipt_data[arm] = {
            "arm": arm,
            "config_path": str(config_path),
            "config_sha256": sha256_text(canonical_json(config)),
            "lm_objective_json": objective_json,
            "lm_objective_sha256": objective_sha256,
            "transition_predecessor_margin_weight": config["lm_objective"][
                "transition_predecessor_margin_weight"
            ],
        }

    pair_objective_json = canonical_json(
        {
            arm: {
                "lm_objective_json": data["lm_objective_json"],
                "lm_objective_sha256": data["lm_objective_sha256"],
            }
            for arm, data in arm_receipt_data.items()
        }
    )
    pair_objective_sha256 = sha256_text(pair_objective_json)
    evaluation_json = canonical_json(spec["evaluation"])
    evaluation_sha256 = sha256_text(evaluation_json)
    artifact_records, artifacts_verified = verify_artifact_records(
        spec["artifacts"],
        verify=verify_artifacts,
    )
    code = git_identity(repo_root, spec["required_ancestor_sha"])
    ready_to_launch = (
        verify_artifacts
        and artifacts_verified
        and code["required_ancestor_present"]
        and code["worktree_clean"]
        and queue_topology_receipt["status"] == "verified"
    )

    preflight_payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "preflight",
        "pair_id": spec["pair_id"],
        "status": "verified" if ready_to_launch else "static_only",
        "ready_to_launch": ready_to_launch,
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "common_scientific_config_sha256": common_config_sha256,
        "pair_objective_json": pair_objective_json,
        "pair_objective_sha256": pair_objective_sha256,
        "evaluation_json": evaluation_json,
        "evaluation_sha256": evaluation_sha256,
        "arm_order": spec["experiment"]["arm_order"],
        "updates_per_arm": spec["experiment"]["updates_per_arm"],
        "trainer_seed": spec["experiment"]["trainer_seed"],
        "max_seconds_per_arm": spec["execution"]["max_seconds_per_arm"],
        "max_pair_gpu_seconds": spec["execution"]["max_pair_gpu_seconds"],
        "queue_topology_receipt": queue_topology_receipt,
        "code": code,
        "artifacts": artifact_records,
        "arms": arm_receipt_data,
    }
    preflight_receipt = receipt_with_payload_sha256(preflight_payload)
    preflight_path = output_dir / "preflight_receipt.json"
    if any_runtime_started:
        if not preflight_path.exists():
            raise ValueError(
                "Runtime evidence exists without its preflight receipt"
            )
        existing_preflight = load_valid_receipt(preflight_path)
        if (
            existing_preflight["receipt_payload_sha256"]
            != preflight_receipt["receipt_payload_sha256"]
        ):
            raise ValueError(
                "Runtime evidence binds a different preflight; refusing overwrite"
            )
        preflight_receipt = existing_preflight
        for arm, config in configs.items():
            config_path = Path(arm_receipt_data[arm]["config_path"])
            if not config_path.is_file():
                raise ValueError(
                    f"{arm} runtime evidence is missing its generated config"
                )
            existing_config = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
            if existing_config != config:
                raise ValueError(
                    f"{arm} runtime evidence binds a different generated config"
                )
    else:
        write_json(
            output_dir / "queue_topology_receipt.template.json",
            topology_template,
        )
        for arm, config in configs.items():
            config_path = Path(arm_receipt_data[arm]["config_path"])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
        write_json(preflight_path, preflight_receipt)

    for arm, arm_data in arm_receipt_data.items():
        command = [
            sys.executable,
            "-m",
            "scripts.train_hybrid",
            "--config",
            arm_data["config_path"],
        ]
        launch_payload = {
            "schema_version": SCHEMA_VERSION,
            "receipt_type": "launch",
            "pair_id": spec["pair_id"],
            "arm": arm,
            "status": "not_run",
            "preflight_payload_sha256": preflight_receipt[
                "receipt_payload_sha256"
            ],
            "pair_objective_sha256": pair_objective_sha256,
            "evaluation_sha256": evaluation_sha256,
            "lm_objective_json": arm_data["lm_objective_json"],
            "lm_objective_sha256": arm_data["lm_objective_sha256"],
            "source_checkpoint_sha256": source_checkpoint_artifact(spec)["sha256"],
            "queue_topology_receipt_payload_sha256": queue_topology_receipt[
                "receipt_payload_sha256"
            ],
            "max_seconds_per_arm": spec["execution"]["max_seconds_per_arm"],
            "effective_timeout_seconds": None,
            "pair_deadline_at": None,
            "command": command,
        }
        launch_receipt = receipt_with_payload_sha256(launch_payload)
        launch_path = output_dir / arm / "launch_receipt.json"

        result_payload = {
            "schema_version": SCHEMA_VERSION,
            "receipt_type": "result",
            "pair_id": spec["pair_id"],
            "arm": arm,
            "status": "not_run",
            "preflight_payload_sha256": preflight_receipt[
                "receipt_payload_sha256"
            ],
            "launch_payload_sha256": launch_receipt[
                "receipt_payload_sha256"
            ],
            "pair_objective_sha256": pair_objective_sha256,
            "evaluation_sha256": evaluation_sha256,
            "lm_objective_json": arm_data["lm_objective_json"],
            "lm_objective_sha256": arm_data["lm_objective_sha256"],
            "source_checkpoint_sha256": source_checkpoint_artifact(spec)["sha256"],
            "queue_topology_receipt_payload_sha256": queue_topology_receipt[
                "receipt_payload_sha256"
            ],
            "max_seconds_per_arm": spec["execution"]["max_seconds_per_arm"],
            "effective_timeout_seconds": None,
            "pair_deadline_at": None,
            "elapsed_seconds": None,
            "timed_out": False,
            "execution_error": None,
            "trainer_return_code": None,
            "return_code": None,
            "completion_validation_errors": [],
            "output_checkpoints": [],
        }
        result_path = output_dir / arm / "result_receipt.json"
        result_receipt = receipt_with_payload_sha256(result_payload)
        existing_launch = (
            load_valid_receipt(launch_path) if launch_path.exists() else None
        )
        existing_result = (
            load_valid_receipt(result_path) if result_path.exists() else None
        )
        if any_runtime_started:
            if (
                existing_launch is None
                or existing_result is None
            ):
                raise ValueError(
                    f"{arm} runtime evidence is incomplete; refusing to overwrite it"
                )
            launch_started = existing_launch.get("status") != "not_run"
            result_started = existing_result.get("status") != "not_run"
            if launch_started != result_started:
                raise ValueError(
                    f"{arm} has incomplete runtime evidence; refusing to overwrite it"
                )
            common_binding = {
                "pair_id": spec["pair_id"],
                "arm": arm,
                "preflight_payload_sha256": preflight_receipt[
                    "receipt_payload_sha256"
                ],
                "pair_objective_sha256": pair_objective_sha256,
                "evaluation_sha256": evaluation_sha256,
                "lm_objective_json": arm_data["lm_objective_json"],
                "lm_objective_sha256": arm_data["lm_objective_sha256"],
                "source_checkpoint_sha256": source_checkpoint_artifact(spec)[
                    "sha256"
                ],
                "queue_topology_receipt_payload_sha256": (
                    queue_topology_receipt["receipt_payload_sha256"]
                ),
                "max_seconds_per_arm": spec["execution"][
                    "max_seconds_per_arm"
                ],
            }
            require_receipt_fields(
                existing_launch,
                common_binding,
                f"{arm} launch",
            )
            require_receipt_fields(
                existing_result,
                {
                    **common_binding,
                    "launch_payload_sha256": existing_launch[
                        "receipt_payload_sha256"
                    ],
                },
                f"{arm} result",
            )
            continue
        write_json(launch_path, launch_receipt)
        write_json(result_path, result_receipt)

    return {
        "spec": spec,
        "output_dir": output_dir,
        "configs": configs,
        "preflight_receipt": preflight_receipt,
        "pair_objective_sha256": pair_objective_sha256,
        "queue_topology_receipt": queue_topology_receipt,
        "arms": arm_receipt_data,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_output_checkpoints(
    repo_root: Path,
    config: dict[str, Any],
    *,
    expected_objective_json: str,
    expected_objective_sha256: str,
) -> list[dict[str, Any]]:
    # Pure Phase A preparation must not require the training-only torch stack.
    import torch

    checkpoint_root = repo_root / config["checkpoint_dir"]
    records = []
    for path in sorted(checkpoint_root.glob("version_*/*.ckpt")):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        global_step = checkpoint.get("global_step")
        objective_error = None
        try:
            validate_checkpoint_lm_objective_identity(
                checkpoint,
                expected_objective_json,
                expected_objective_sha256,
            )
            objective_matches = True
        except ValueError as exc:
            objective_matches = False
            objective_error = str(exc)
        records.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "global_step": global_step,
                "hybrid_stage": checkpoint.get("hybrid_stage"),
                "lm_objective_json": checkpoint.get(
                    "hybrid_lm_objective_json"
                ),
                "lm_objective_sha256": checkpoint.get(
                    "hybrid_lm_objective_sha256"
                ),
                "objective_matches": objective_matches,
                "objective_error": objective_error,
            }
        )
    return records


def acquire_run_lock(prepared: dict[str, Any], arm: str) -> Path:
    lock_path = prepared["output_dir"] / arm / "run.lock"
    lock_payload = {
        "pair_id": prepared["spec"]["pair_id"],
        "arm": arm,
        "preflight_payload_sha256": prepared["preflight_receipt"][
            "receipt_payload_sha256"
        ],
    }
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise ValueError(f"{arm} run lock already exists: {lock_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical_json(lock_payload) + "\n")
    return lock_path


def require_receipt_binding(
    receipt: dict[str, Any],
    *,
    prepared: dict[str, Any],
    arm: str,
) -> None:
    expected = {
        "pair_id": prepared["spec"]["pair_id"],
        "arm": arm,
        "preflight_payload_sha256": prepared["preflight_receipt"][
            "receipt_payload_sha256"
        ],
        "pair_objective_sha256": prepared["pair_objective_sha256"],
        "evaluation_sha256": prepared["preflight_receipt"]["evaluation_sha256"],
        "lm_objective_json": prepared["arms"][arm]["lm_objective_json"],
        "lm_objective_sha256": prepared["arms"][arm]["lm_objective_sha256"],
        "source_checkpoint_sha256": source_checkpoint_artifact(
            prepared["spec"]
        )["sha256"],
        "queue_topology_receipt_payload_sha256": prepared[
            "queue_topology_receipt"
        ]["receipt_payload_sha256"],
        "max_seconds_per_arm": prepared["spec"]["execution"][
            "max_seconds_per_arm"
        ],
    }
    require_receipt_fields(receipt, expected, arm)


def require_fresh_arm_runtime(
    prepared: dict[str, Any],
    arm: str,
    repo_root: Path,
) -> None:
    launch_path = prepared["output_dir"] / arm / "launch_receipt.json"
    result_path = prepared["output_dir"] / arm / "result_receipt.json"
    launch_receipt = load_valid_receipt(launch_path)
    result_receipt = load_valid_receipt(result_path)
    require_receipt_binding(launch_receipt, prepared=prepared, arm=arm)
    require_receipt_binding(result_receipt, prepared=prepared, arm=arm)
    if launch_receipt.get("status") != "not_run":
        raise ValueError(f"{arm} launch receipt is not fresh")
    if result_receipt.get("status") != "not_run":
        raise ValueError(f"{arm} result receipt is not fresh")
    if (
        result_receipt.get("launch_payload_sha256")
        != launch_receipt["receipt_payload_sha256"]
    ):
        raise ValueError(f"{arm} result receipt does not bind its launch receipt")

    config = prepared["configs"][arm]
    for field in ("log_dir", "checkpoint_dir"):
        path = (repo_root / config[field]).resolve()
        if path.exists():
            raise ValueError(f"{arm} requires an absent fresh {field}: {path}")


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def run_command_with_timeout(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "return_code": START_FAILURE_RETURN_CODE,
            "timed_out": False,
            "elapsed_seconds": time.monotonic() - started,
            "execution_error": f"{type(exc).__name__}: {exc}",
        }

    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return {
            "return_code": TIMEOUT_RETURN_CODE,
            "timed_out": True,
            "elapsed_seconds": time.monotonic() - started,
            "execution_error": (
                f"training exceeded the {timeout_seconds}-second arm budget"
            ),
        }
    return {
        "return_code": return_code,
        "timed_out": False,
        "elapsed_seconds": time.monotonic() - started,
        "execution_error": None,
    }


def effective_arm_runtime_budget(
    prepared: dict[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[int, str]:
    checked_at = _parse_utc_timestamp(
        prepared["queue_topology_receipt"]["checked_at"],
        "checked_at",
    )
    pair_deadline = checked_at + timedelta(
        seconds=prepared["spec"]["execution"]["max_pair_gpu_seconds"]
    )
    current_time = now or datetime.now(timezone.utc)
    remaining_seconds = (pair_deadline - current_time).total_seconds()
    usable_seconds = int(remaining_seconds - TERMINATION_GRACE_SECONDS)
    timeout_seconds = min(
        prepared["spec"]["execution"]["max_seconds_per_arm"]
        - TERMINATION_GRACE_SECONDS,
        usable_seconds,
    )
    if timeout_seconds <= 0:
        raise ValueError("Pair GPU wall-clock budget is exhausted")
    return timeout_seconds, pair_deadline.isoformat()


def run_arm(
    spec_path: Path,
    arm: str,
    *,
    output_dir: Path | None,
    confirm_phase_b: bool,
    queue_topology_receipt_path: Path | None,
    repo_root: Path = REPO_ROOT,
) -> int:
    if not confirm_phase_b:
        raise ValueError(
            "run-arm requires --confirm-phase-b after fresh user authorization"
        )
    if queue_topology_receipt_path is None:
        raise ValueError(
            "run-arm requires a fresh --queue-topology-receipt"
        )
    prepared = prepare_pair(
        spec_path,
        output_dir=output_dir,
        verify_artifacts=True,
        queue_topology_receipt_path=queue_topology_receipt_path,
        repo_root=repo_root,
    )
    preflight = prepared["preflight_receipt"]
    if not preflight["ready_to_launch"]:
        raise ValueError("Verified preflight is not ready to launch")
    effective_timeout_seconds, pair_deadline_at = effective_arm_runtime_budget(
        prepared
    )

    arm_order = prepared["spec"]["experiment"]["arm_order"]
    arm_index = arm_order.index(arm)
    if arm_index > 0:
        previous_arm = arm_order[arm_index - 1]
        previous_launch_path = (
            prepared["output_dir"] / previous_arm / "launch_receipt.json"
        )
        previous_result_path = (
            prepared["output_dir"] / previous_arm / "result_receipt.json"
        )
        previous_launch = load_valid_receipt(previous_launch_path)
        previous_result = load_valid_receipt(previous_result_path)
        require_receipt_binding(
            previous_launch,
            prepared=prepared,
            arm=previous_arm,
        )
        require_receipt_binding(
            previous_result,
            prepared=prepared,
            arm=previous_arm,
        )
        if previous_result.get("status") != "completed":
            raise ValueError(
                f"{arm} cannot run before {previous_arm} completes"
            )
        if (
            previous_result.get("launch_payload_sha256")
            != previous_launch["receipt_payload_sha256"]
        ):
            raise ValueError(
                f"{previous_arm} result does not bind its launch receipt"
            )

    require_fresh_arm_runtime(prepared, arm, repo_root)
    run_lock_path = acquire_run_lock(prepared, arm)
    arm_data = prepared["arms"][arm]
    command = [
        sys.executable,
        "-m",
        "scripts.train_hybrid",
        "--config",
        arm_data["config_path"],
    ]
    launch_payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "launch",
        "pair_id": prepared["spec"]["pair_id"],
        "arm": arm,
        "status": "running",
        "started_at": utc_now(),
        "preflight_payload_sha256": preflight["receipt_payload_sha256"],
        "pair_objective_sha256": prepared["pair_objective_sha256"],
        "evaluation_sha256": preflight["evaluation_sha256"],
        "lm_objective_json": arm_data["lm_objective_json"],
        "lm_objective_sha256": arm_data["lm_objective_sha256"],
        "source_checkpoint_sha256": source_checkpoint_artifact(
            prepared["spec"]
        )["sha256"],
        "queue_topology_receipt_payload_sha256": prepared[
            "queue_topology_receipt"
        ]["receipt_payload_sha256"],
        "max_seconds_per_arm": prepared["spec"]["execution"][
            "max_seconds_per_arm"
        ],
        "effective_timeout_seconds": effective_timeout_seconds,
        "pair_deadline_at": pair_deadline_at,
        "run_lock_path": str(run_lock_path),
        "command": command,
    }
    launch_receipt = receipt_with_payload_sha256(launch_payload)
    write_json(
        prepared["output_dir"] / arm / "launch_receipt.json",
        launch_receipt,
    )

    execution_outcome = run_command_with_timeout(
        command,
        cwd=repo_root,
        timeout_seconds=effective_timeout_seconds,
    )
    output_checkpoints = []
    expected_updates = prepared["spec"]["experiment"]["updates_per_arm"]
    completion_validation_errors = []
    if execution_outcome["execution_error"] is not None:
        completion_validation_errors.append(
            execution_outcome["execution_error"]
        )
    try:
        output_checkpoints = collect_output_checkpoints(
            repo_root,
            prepared["configs"][arm],
            expected_objective_json=arm_data["lm_objective_json"],
            expected_objective_sha256=arm_data["lm_objective_sha256"],
        )
    except Exception as exc:
        # Preserve a terminal result receipt even when checkpoint parsing fails.
        completion_validation_errors.append(
            f"checkpoint collection failed: {type(exc).__name__}: {exc}"
        )
    if not output_checkpoints:
        completion_validation_errors.append("no output checkpoint was written")
    if any(
        not checkpoint["objective_matches"]
        for checkpoint in output_checkpoints
    ):
        completion_validation_errors.append(
            "an output checkpoint has mismatched objective metadata"
        )
    if not any(
        checkpoint["global_step"] == expected_updates
        and checkpoint["hybrid_stage"] == "gen"
        and checkpoint["objective_matches"]
        for checkpoint in output_checkpoints
    ):
        completion_validation_errors.append(
            f"no bound gen checkpoint reached global_step={expected_updates}"
        )
    status = (
        "completed"
        if execution_outcome["return_code"] == 0
        and not completion_validation_errors
        else "failed"
    )
    return_code = (
        execution_outcome["return_code"]
        if execution_outcome["return_code"] != 0
        else (0 if status == "completed" else 2)
    )
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "result",
        "pair_id": prepared["spec"]["pair_id"],
        "arm": arm,
        "status": status,
        "completed_at": utc_now(),
        "preflight_payload_sha256": preflight["receipt_payload_sha256"],
        "launch_payload_sha256": launch_receipt["receipt_payload_sha256"],
        "pair_objective_sha256": prepared["pair_objective_sha256"],
        "evaluation_sha256": preflight["evaluation_sha256"],
        "lm_objective_json": arm_data["lm_objective_json"],
        "lm_objective_sha256": arm_data["lm_objective_sha256"],
        "source_checkpoint_sha256": source_checkpoint_artifact(
            prepared["spec"]
        )["sha256"],
        "queue_topology_receipt_payload_sha256": prepared[
            "queue_topology_receipt"
        ]["receipt_payload_sha256"],
        "max_seconds_per_arm": prepared["spec"]["execution"][
            "max_seconds_per_arm"
        ],
        "effective_timeout_seconds": effective_timeout_seconds,
        "pair_deadline_at": pair_deadline_at,
        "elapsed_seconds": execution_outcome["elapsed_seconds"],
        "timed_out": execution_outcome["timed_out"],
        "execution_error": execution_outcome["execution_error"],
        "trainer_return_code": execution_outcome["return_code"],
        "return_code": return_code,
        "completion_validation_errors": completion_validation_errors,
        "output_checkpoints": output_checkpoints,
    }
    write_json(
        prepared["output_dir"] / arm / "result_receipt.json",
        receipt_with_payload_sha256(result_payload),
    )
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or explicitly run the bounded transition-margin pair"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Materialize configs and static/verified receipts without training",
    )
    prepare_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    prepare_parser.add_argument("--output-dir", type=Path)
    prepare_parser.add_argument("--verify-artifacts", action="store_true")
    prepare_parser.add_argument("--queue-topology-receipt", type=Path)

    run_parser = subparsers.add_parser(
        "run-arm",
        help="Run one preregistered arm after a verified preflight",
    )
    run_parser.add_argument("arm", choices=ARM_NAMES)
    run_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--confirm-phase-b", action="store_true")
    run_parser.add_argument("--queue-topology-receipt", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepared = prepare_pair(
            args.spec,
            output_dir=args.output_dir,
            verify_artifacts=args.verify_artifacts,
            queue_topology_receipt_path=args.queue_topology_receipt,
        )
        summary = {
            "output_dir": str(prepared["output_dir"]),
            "pair_id": prepared["spec"]["pair_id"],
            "pair_objective_sha256": prepared["pair_objective_sha256"],
            "ready_to_launch": prepared["preflight_receipt"]["ready_to_launch"],
            "status": prepared["preflight_receipt"]["status"],
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    return run_arm(
        args.spec,
        args.arm,
        output_dir=args.output_dir,
        confirm_phase_b=args.confirm_phase_b,
        queue_topology_receipt_path=args.queue_topology_receipt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
