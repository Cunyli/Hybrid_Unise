import inspect
import sys

import pytest
import torch
import yaml

from model import hybrid_inference
from model.xcodec_backends import TransformersXCodecFirstRVQ
from scripts import infer_hybrid_directory


def test_programmatic_inference_requires_explicit_checkpoint():
    for function in (
        hybrid_inference.load_hybrid_model,
        hybrid_inference.enhance,
        hybrid_inference.enhance_file,
    ):
        parameter = inspect.signature(function).parameters["checkpoint"]
        assert parameter.default is inspect.Parameter.empty

    with pytest.raises(ValueError, match="explicit trusted checkpoint"):
        hybrid_inference.load_hybrid_model(
            "does-not-need-to-exist.yaml",
            checkpoint=None,
            device="cpu",
        )


def test_programmatic_inference_loads_checkpoint_strictly(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_type": "hybrid_unise",
                "stage": "disc",
            }
        )
    )
    checkpoint_data = {"state_dict": {"weight": torch.tensor(1.0)}}
    calls = {}

    class FakeModel:
        def __init__(self, config):
            calls["config"] = config
            self.stage = config["stage"]
            self.architecture_config = {"stage": self.stage}

        def to(self, device):
            calls["device"] = device
            return self

        def load_state_dict(self, state_dict, strict):
            calls["state_dict"] = state_dict
            calls["strict"] = strict

        def eval(self):
            calls["evaluated"] = True
            return self

    monkeypatch.setattr(
        hybrid_inference,
        "HybridUniSELightning",
        FakeModel,
    )
    monkeypatch.setattr(
        hybrid_inference,
        "load_hybrid_checkpoint",
        lambda path, map_location: checkpoint_data,
    )
    monkeypatch.setattr(
        hybrid_inference,
        "validate_hybrid_checkpoint_metadata",
        lambda checkpoint, stage: None,
    )
    monkeypatch.setattr(
        hybrid_inference,
        "validate_hybrid_architecture_metadata",
        lambda checkpoint, architecture: None,
    )

    model = hybrid_inference.load_hybrid_model(
        config_path,
        checkpoint="trusted.ckpt",
        device="cpu",
    )

    assert isinstance(model, FakeModel)
    assert calls["config"]["ckpt_path"] == "trusted.ckpt"
    assert calls["state_dict"] is checkpoint_data["state_dict"]
    assert calls["strict"] is True
    assert calls["evaluated"] is True


def test_directory_inference_cli_requires_checkpoint(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_hybrid_directory.py",
            "--input-root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        infer_hybrid_directory.main()

    assert exc_info.value.code == 2


def test_directory_inference_checkpoint_load_is_strict(monkeypatch):
    checkpoint_data = {"state_dict": {"weight": torch.tensor(1.0)}}
    calls = {}

    class FakeModel:
        stage = "fusion"
        architecture_config = {"stage": "fusion"}

        def load_state_dict(self, state_dict, strict):
            calls["state_dict"] = state_dict
            calls["strict"] = strict

    monkeypatch.setattr(
        infer_hybrid_directory,
        "load_hybrid_checkpoint",
        lambda path, map_location: checkpoint_data,
    )
    monkeypatch.setattr(
        infer_hybrid_directory,
        "validate_hybrid_checkpoint_metadata",
        lambda checkpoint, stage: None,
    )
    monkeypatch.setattr(
        infer_hybrid_directory,
        "validate_hybrid_architecture_metadata",
        lambda checkpoint, architecture: None,
    )

    infer_hybrid_directory.load_checkpoint(
        FakeModel(),
        "trusted.ckpt",
        torch.device("cpu"),
    )

    assert calls["state_dict"] is checkpoint_data["state_dict"]
    assert calls["strict"] is True


def test_xcodec_remote_code_is_opt_in_and_revision_is_supported():
    parameters = inspect.signature(
        TransformersXCodecFirstRVQ
    ).parameters

    assert parameters["trust_remote_code"].default is False
    assert "revision" in parameters
