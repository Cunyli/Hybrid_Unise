import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

import scripts.transition_margin_probe as transition_margin_probe
from model.hybrid_objective import lm_objective_identity
from scripts.transition_margin_probe import (
    TIMEOUT_RETURN_CODE,
    acquire_run_lock,
    canonical_json,
    collect_output_checkpoints,
    effective_arm_runtime_budget,
    load_valid_receipt,
    load_pair_spec,
    load_verified_topology_receipt,
    prepare_pair,
    receipt_with_payload_sha256,
    require_receipt_binding,
    require_fresh_arm_runtime,
    run_arm,
    run_command_with_timeout,
    sha256_file,
    validate_pair_spec,
    verify_artifact_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "conf/transition_margin_probe_pair_v1.yaml"


def independent_leaf_differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(
                    independent_leaf_differences(left[key], right[key], path)
                )
        return paths
    return {prefix} if left != right else set()


def test_lm_objective_identity_is_explicit_and_stable():
    objective = {
        "transition_predecessor_margin_weight": 0.25,
        "transition_predecessor_margin": 1,
    }

    canonical, canonical_objective_json, objective_sha256 = (
        lm_objective_identity(objective)
    )

    assert canonical["transition_predecessor_margin"] == 1.0
    assert canonical["transition_predecessor_margin_weight"] == 0.25
    assert canonical_objective_json == canonical_json(canonical)
    assert objective_sha256 == (
        "5f374abd6de6c38faa094baa9c70b85f221f935a1cfe0653e60e94c99495fdf2"
    )


def test_prepare_pair_writes_static_bound_receipts(tmp_path):
    prepared = prepare_pair(
        SPEC_PATH,
        output_dir=tmp_path / "prepared",
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )

    preflight = prepared["preflight_receipt"]
    assert preflight["status"] == "static_only"
    assert preflight["ready_to_launch"] is False
    assert preflight["max_seconds_per_arm"] == 1800
    assert preflight["max_pair_gpu_seconds"] == 3600
    assert preflight["queue_topology_receipt"]["status"] == "not_checked"
    assert all(
        artifact["status"] == "not_checked"
        for artifact in preflight["artifacts"]
    )
    assert preflight["code"]["required_ancestor_present"] is True

    control = prepared["configs"]["control"]
    treatment = prepared["configs"]["treatment"]
    assert (
        control["lm_objective"]["transition_predecessor_margin_weight"]
        == 0.0
    )
    assert (
        treatment["lm_objective"]["transition_predecessor_margin_weight"]
        == 0.25
    )
    assert control["resume"] is None
    assert treatment["resume"] is None
    assert control["stage_init_checkpoint"] == treatment["stage_init_checkpoint"]

    for arm in ("control", "treatment"):
        launch = json.loads(
            (prepared["output_dir"] / arm / "launch_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        result = json.loads(
            (prepared["output_dir"] / arm / "result_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        arm_data = prepared["arms"][arm]
        assert launch["status"] == "not_run"
        assert result["status"] == "not_run"
        assert (
            launch["preflight_payload_sha256"]
            == preflight["receipt_payload_sha256"]
        )
        assert (
            result["preflight_payload_sha256"]
            == preflight["receipt_payload_sha256"]
        )
        assert (
            result["launch_payload_sha256"]
            == launch["receipt_payload_sha256"]
        )
        assert launch["lm_objective_json"] == arm_data["lm_objective_json"]
        assert (
            result["lm_objective_sha256"]
            == arm_data["lm_objective_sha256"]
        )
        assert (
            launch["evaluation_sha256"]
            == preflight["evaluation_sha256"]
        )
        assert (
            result["evaluation_sha256"]
            == preflight["evaluation_sha256"]
        )
        assert launch["max_seconds_per_arm"] == 1800
        assert result["max_seconds_per_arm"] == 1800
        assert launch["effective_timeout_seconds"] is None
        assert result["pair_deadline_at"] is None


def test_generated_pair_has_exactly_six_independently_checked_differences(
    tmp_path,
):
    prepared = prepare_pair(
        SPEC_PATH,
        output_dir=tmp_path / "prepared",
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )
    configs = {}
    for arm in ("control", "treatment"):
        config_path = Path(prepared["arms"][arm]["config_path"])
        configs[arm] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert independent_leaf_differences(
        configs["control"],
        configs["treatment"],
    ) == {
        "checkpoint_dir",
        "lm_objective.transition_predecessor_margin_weight",
        "log_dir",
        "transition_margin_probe.arm",
        "transition_margin_probe.lm_objective_json",
        "transition_margin_probe.lm_objective_sha256",
    }
    for arm in ("control", "treatment"):
        _, objective_json, objective_sha256 = lm_objective_identity(
            configs[arm]["lm_objective"]
        )
        assert objective_json == prepared["arms"][arm]["lm_objective_json"]
        assert objective_sha256 == prepared["arms"][arm]["lm_objective_sha256"]


def test_pair_spec_rejects_any_second_objective_change():
    spec = copy.deepcopy(load_pair_spec(SPEC_PATH))
    spec["base_config"]["lm_objective"]["transition_ce_multiplier"] = 2.0

    with pytest.raises(ValueError, match="No lm_objective change"):
        validate_pair_spec(spec)


def test_pair_spec_rejects_relaxed_time_or_topology_limits():
    for field, value in (
        ("max_seconds_per_arm", 1801),
        ("max_pair_gpu_seconds", 3601),
        ("queue_topology_receipt_max_age_seconds", 3601),
    ):
        spec = copy.deepcopy(load_pair_spec(SPEC_PATH))
        spec["execution"][field] = value
        with pytest.raises(ValueError, match=field):
            validate_pair_spec(spec)


def test_pair_spec_rejects_unbound_data_path_or_changed_decision_denominator():
    spec = copy.deepcopy(load_pair_spec(SPEC_PATH))
    spec["base_config"]["dataset_config"]["val_kwargs"][
        "recipe_manifest"
    ] = "/tmp/not-the-frozen-recipes.jsonl"
    with pytest.raises(ValueError, match="immutable artifact path"):
        validate_pair_spec(spec)

    spec = copy.deepcopy(load_pair_spec(SPEC_PATH))
    spec["evaluation"]["decision_gates"]["transition_predecessor_rate"][
        "denominator"
    ] = "transition_errors"
    with pytest.raises(ValueError, match="decision_gates"):
        validate_pair_spec(spec)


def test_artifact_verification_checks_hash_and_size(tmp_path):
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"bounded-probe")
    artifact = {
        "name": "fixture",
        "role": "data_contract",
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }

    records, verified = verify_artifact_records([artifact], verify=True)
    assert verified is True
    assert records[0]["status"] == "verified"

    mismatched = dict(artifact, sha256="0" * 64)
    records, verified = verify_artifact_records([mismatched], verify=True)
    assert verified is False
    assert records[0]["status"] == "mismatch"


def test_receipt_payload_hash_excludes_only_its_own_field():
    payload = {"receipt_type": "preflight", "status": "static_only"}
    receipt = receipt_with_payload_sha256(payload)
    payload_copy = dict(receipt)
    payload_sha256 = payload_copy.pop("receipt_payload_sha256")

    assert receipt_with_payload_sha256(payload_copy)[
        "receipt_payload_sha256"
    ] == payload_sha256


def test_fresh_one_gpu_topology_receipt_is_required_and_self_hashed(
    tmp_path,
):
    spec = load_pair_spec(SPEC_PATH)
    checked_at = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "topology.json"
    receipt = receipt_with_payload_sha256(
        {
            "schema_version": "transition_margin_topology/v1",
            "receipt_type": "queue_topology",
            "pair_id": spec["pair_id"],
            "status": "verified",
            "checked_at": checked_at.isoformat(),
            "requested_gpus": 1,
            "selected_gpu_count": 1,
            "queue_snapshot_sha256": "a" * 64,
            "topology_snapshot_sha256": "b" * 64,
        }
    )
    path.write_text(json.dumps(receipt), encoding="utf-8")

    loaded = load_verified_topology_receipt(
        path,
        spec,
        now=checked_at + timedelta(seconds=3599),
    )
    assert loaded == receipt
    with pytest.raises(ValueError, match="not fresh"):
        load_verified_topology_receipt(
            path,
            spec,
            now=checked_at + timedelta(seconds=3601),
        )


def test_effective_arm_timeout_uses_the_shared_pair_deadline(tmp_path):
    spec = load_pair_spec(SPEC_PATH)
    checked_at = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
    prepared = {
        "spec": spec,
        "queue_topology_receipt": {"checked_at": checked_at.isoformat()},
    }

    initial_timeout, _ = effective_arm_runtime_budget(
        prepared,
        now=checked_at,
    )
    assert initial_timeout == 1790
    timeout_seconds, deadline = effective_arm_runtime_budget(
        prepared,
        now=checked_at + timedelta(seconds=2000),
    )
    assert timeout_seconds == 1590
    assert deadline == (checked_at + timedelta(seconds=3600)).isoformat()
    with pytest.raises(ValueError, match="budget is exhausted"):
        effective_arm_runtime_budget(
            prepared,
            now=checked_at + timedelta(seconds=3590),
        )


def test_receipt_binding_rejects_semantic_objective_json_tamper(tmp_path):
    prepared = prepare_pair(
        SPEC_PATH,
        output_dir=tmp_path / "prepared",
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )
    launch_path = prepared["output_dir"] / "control/launch_receipt.json"
    receipt = load_valid_receipt(launch_path)
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256")
    payload["lm_objective_json"] = '{"tampered":true}'
    tampered = receipt_with_payload_sha256(payload)

    with pytest.raises(ValueError, match="lm_objective_json"):
        require_receipt_binding(
            tampered,
            prepared=prepared,
            arm="control",
        )


def test_receipt_tamper_fresh_outputs_and_duplicate_run_lock_fail_closed(
    tmp_path,
):
    prepared = prepare_pair(
        SPEC_PATH,
        output_dir=tmp_path / "prepared",
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )
    require_fresh_arm_runtime(prepared, "control", tmp_path)

    acquire_run_lock(prepared, "control")
    with pytest.raises(ValueError, match="run lock already exists"):
        acquire_run_lock(prepared, "control")

    launch_path = prepared["output_dir"] / "treatment/launch_receipt.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["status"] = "running"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        load_valid_receipt(launch_path)

    log_dir = tmp_path / prepared["configs"]["control"]["log_dir"]
    log_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="absent fresh log_dir"):
        require_fresh_arm_runtime(prepared, "control", tmp_path)


def test_prepare_preserves_started_runtime_receipts(tmp_path):
    output_dir = tmp_path / "prepared"
    prepared = prepare_pair(
        SPEC_PATH,
        output_dir=output_dir,
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )
    launch_path = output_dir / "control/launch_receipt.json"
    result_path = output_dir / "control/result_receipt.json"

    launch = load_valid_receipt(launch_path)
    launch_payload = dict(launch)
    launch_payload.pop("receipt_payload_sha256")
    launch_payload["status"] = "completed"
    launch = receipt_with_payload_sha256(launch_payload)
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    result = load_valid_receipt(result_path)
    result_payload = dict(result)
    result_payload.pop("receipt_payload_sha256")
    result_payload["status"] = "completed"
    result_payload["launch_payload_sha256"] = launch[
        "receipt_payload_sha256"
    ]
    result = receipt_with_payload_sha256(result_payload)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    prepare_pair(
        SPEC_PATH,
        output_dir=output_dir,
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )

    assert load_valid_receipt(launch_path) == launch
    assert load_valid_receipt(result_path) == result
    assert prepared["preflight_receipt"] == load_valid_receipt(
        output_dir / "preflight_receipt.json"
    )


def test_prepare_never_invokes_training_command(tmp_path, monkeypatch):
    real_run = transition_margin_probe.subprocess.run
    observed_commands = []

    def guarded_run(command, *args, **kwargs):
        observed_commands.append(command)
        assert "scripts.train_hybrid" not in command
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(
        transition_margin_probe.subprocess,
        "run",
        guarded_run,
    )
    prepare_pair(
        SPEC_PATH,
        output_dir=tmp_path / "prepared",
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )

    assert observed_commands
    assert all(command[0] == "git" for command in observed_commands)


def test_command_timeout_kills_the_arm_process_group(tmp_path):
    outcome = run_command_with_timeout(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        timeout_seconds=0.01,
    )

    assert outcome["return_code"] == TIMEOUT_RETURN_CODE
    assert outcome["timed_out"] is True
    assert "arm budget" in outcome["execution_error"]


def test_checkpoint_collection_binds_step_stage_and_objective(tmp_path):
    torch = pytest.importorskip("torch")
    _, objective_json, objective_sha256 = lm_objective_identity(
        {"transition_predecessor_margin_weight": 0.25}
    )
    checkpoint_dir = tmp_path / "checkpoints/arm/version_0"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "latest_epoch=00-step=000100.ckpt"
    torch.save(
        {
            "global_step": 100,
            "hybrid_stage": "gen",
            "hybrid_lm_objective_json": objective_json,
            "hybrid_lm_objective_sha256": objective_sha256,
        },
        checkpoint_path,
    )

    records = collect_output_checkpoints(
        tmp_path,
        {"checkpoint_dir": "checkpoints/arm"},
        expected_objective_json=objective_json,
        expected_objective_sha256=objective_sha256,
    )

    assert len(records) == 1
    assert records[0]["global_step"] == 100
    assert records[0]["hybrid_stage"] == "gen"
    assert records[0]["objective_matches"] is True


def test_run_arm_requires_explicit_phase_b_confirmation(tmp_path):
    with pytest.raises(ValueError, match="fresh user authorization"):
        run_arm(
            SPEC_PATH,
            "control",
            output_dir=tmp_path / "never_written",
            confirm_phase_b=False,
            queue_topology_receipt_path=None,
            repo_root=REPO_ROOT,
        )


def test_run_arm_requires_topology_receipt_before_any_preflight_write(
    tmp_path,
):
    output_dir = tmp_path / "never_written"
    with pytest.raises(ValueError, match="queue-topology-receipt"):
        run_arm(
            SPEC_PATH,
            "control",
            output_dir=output_dir,
            confirm_phase_b=True,
            queue_topology_receipt_path=None,
            repo_root=REPO_ROOT,
        )
    assert not output_dir.exists()


def test_run_arm_writes_failed_terminal_receipt_on_checkpoint_error(
    tmp_path,
    monkeypatch,
):
    prepared = prepare_pair(
        SPEC_PATH,
        output_dir=tmp_path / "prepared",
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )
    prepared["preflight_receipt"]["ready_to_launch"] = True
    prepared["preflight_receipt"]["status"] = "verified"

    monkeypatch.setattr(
        transition_margin_probe,
        "prepare_pair",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        transition_margin_probe,
        "require_fresh_arm_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        transition_margin_probe,
        "acquire_run_lock",
        lambda *args, **kwargs: tmp_path / "run.lock",
    )
    monkeypatch.setattr(
        transition_margin_probe,
        "run_command_with_timeout",
        lambda *args, **kwargs: {
            "return_code": 0,
            "timed_out": False,
            "elapsed_seconds": 1.0,
            "execution_error": None,
        },
    )
    monkeypatch.setattr(
        transition_margin_probe,
        "effective_arm_runtime_budget",
        lambda *args, **kwargs: (1800, "2026-08-09T13:00:00+00:00"),
    )

    def fail_checkpoint_collection(*args, **kwargs):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(
        transition_margin_probe,
        "collect_output_checkpoints",
        fail_checkpoint_collection,
    )

    return_code = run_arm(
        SPEC_PATH,
        "control",
        output_dir=prepared["output_dir"],
        confirm_phase_b=True,
        queue_topology_receipt_path=tmp_path / "topology.json",
        repo_root=tmp_path,
    )

    assert return_code == 2
    result = load_valid_receipt(
        prepared["output_dir"] / "control/result_receipt.json"
    )
    assert result["status"] == "failed"
    assert result["trainer_return_code"] == 0
    assert any(
        "checkpoint collection failed" in error
        for error in result["completion_validation_errors"]
    )


def test_generated_config_yaml_round_trips(tmp_path):
    prepared = prepare_pair(
        SPEC_PATH,
        output_dir=tmp_path / "roundtrip",
        verify_artifacts=False,
        repo_root=REPO_ROOT,
    )

    for arm in ("control", "treatment"):
        config_path = Path(prepared["arms"][arm]["config_path"])
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert loaded == prepared["configs"][arm]
