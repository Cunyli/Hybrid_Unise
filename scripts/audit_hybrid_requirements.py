from pathlib import Path

import yaml


CHECKS = [
    (
        "SFI STFT with supported rates",
        "model/audio/stft.py",
        ["window_ms", "hop_ms", "supported_sample_rates", "sfi_stft", "sfi_istft"],
    ),
    (
        "Discriminative TF-GridNet-style branch",
        "model/hybrid_discriminative.py",
        ["DiscriminativeBranch", "num_blocks", "lstm_hidden"],
    ),
    (
        "X-Codec first RVQ interface",
        "model/hybrid_xcodec.py",
        ["XCodecFirstRVQTokenizer", "XCodecTokenBatch", "encode_first_rvq_batch", "rvq_axis", "module:callable", "backend_kwargs"],
    ),
    (
        "Optional X-Codec pretrained backend",
        "model/xcodec_backends.py",
        ["TransformersXCodecFirstRVQ", "XcodecModel", "audio_codes", "token_attr", "trust_remote_code"],
    ),
    (
        "X-Codec backend check script",
        "scripts/check_xcodec_backend.py",
        ["XcodecModel", "--model-path", "encode_first_rvq_batch"],
    ),
    (
        "LM returns token hidden states and masks",
        "model/hybrid_lm.py",
        ["hidden_states", "hidden_mask", "target_mask", "ignore_index", "logits", "targets"],
    ),
    (
        "DPRNN refinement consumes LM hidden states",
        "model/hybrid_refinement.py",
        ["GenerativeRefinementBranch", "lm_hidden", "lm_hidden_mask", "cross_attn"],
    ),
    (
        "Sigmoid fusion mask in [0, 1]",
        "model/hybrid_fusion.py",
        ["torch.sigmoid", "blend_spectra"],
    ),
    (
        "Hybrid structured output",
        "model/hybrid_types.py",
        ["HybridOutput", "final_wav", "disc_wav", "gen_wav", "fusion_mask", "token_logits"],
    ),
    (
        "Stage training and checkpoint metadata",
        "model/hybrid_model.py",
        ["stage", "_apply_stage_freezing", "stage_init_checkpoint", "on_save_checkpoint", "hybrid_stage", "_require_finite", "load_hybrid_checkpoint"],
    ),
    (
        "External PMSQE/SQA loss is explicit",
        "model/hybrid_losses.py",
        ["ExternalLossAdapter", "build_external_loss", "module:callable"],
    ),
    (
        "Verified PMSQE adapter is optional",
        "model/quality_losses.py",
        ["AsteroidPMSQELoss", "SingleSrcPMSQE", "sample_rate"],
    ),
    (
        "Rolling cache supports archive members",
        "dataloader/rolling_cache.py",
        ["ArchiveAudioSource", "zipfile", "tarfile", "UseSimulationRollingCacheDataLoadIter", "batch_format"],
    ),
    (
        "Directory inference writes scp files",
        "scripts/infer_hybrid_directory.py",
        ["inf.scp", "ref.scp", "wav", "--reference-root", "--stage", "find_reference_path"],
    ),
    (
        "Staged training CLI overrides",
        "scripts/train_hybrid.py",
        ["--stage", "--stage_init_checkpoint", "_config_dir", "NamedTemporaryFile"],
    ),
    (
        "Hybrid dict batches are SR/USE only",
        "dataloader/data_module.py",
        ["modes", "make_sr_batch", "sample_rates"],
    ),
]


def main() -> int:
    errors: list[str] = []
    for label, file_name, needles in CHECKS:
        path = Path(file_name)
        if not path.is_file():
            errors.append(f"{label}: missing {file_name}")
            continue
        text = path.read_text()
        for needle in needles:
            if needle not in text:
                errors.append(f"{label}: {file_name} missing {needle!r}")

    config_path = Path("conf/hybrid_unise_urgent2026.yaml")
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text())
        expected = {
            "model_type": "hybrid_unise",
            "stage": "disc",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                errors.append(f"{config_path}: expected {key}={value!r}")
        if config.get("xcodec", {}).get("backend") == "deterministic_stub":
            doc_text = Path("docs/hybrid_unise_reproduction.md").read_text()
            if "not a verified X-Codec model" not in doc_text:
                errors.append("deterministic_stub is enabled but docs do not warn that it is not verified X-Codec")
        if config.get("external_losses", {}).get("pmsqe", {}).get("enabled"):
            errors.append("PMSQE should not be enabled without verified local backend")
        if config.get("external_losses", {}).get("sqa", {}).get("enabled"):
            errors.append("SQA should not be enabled without verified local backend")
        rolling_config_path = Path("conf/hybrid_unise_rolling_cache_example.yaml")
        if rolling_config_path.is_file():
            rolling_config = yaml.safe_load(rolling_config_path.read_text())
            dataset_type = rolling_config.get("dataset_config", {}).get("train_kwargs", {}).get("dataset_type")
            if dataset_type != "use_simulation_rolling_cache":
                errors.append(f"{rolling_config_path}: train dataset must use rolling cache")
        native_config_path = Path("conf/hybrid_unise_native_multisr_example.yaml")
        if native_config_path.is_file():
            native_config = yaml.safe_load(native_config_path.read_text())
            for split in ("train_kwargs", "val_kwargs"):
                modes = native_config.get("dataset_config", {}).get(split, {}).get("modes")
                if modes != ["se"]:
                    errors.append(f"{native_config_path}: dataset_config.{split}.modes must be [se]")
    else:
        errors.append(f"missing {config_path}")

    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print("Hybrid-UniSE requirements audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
