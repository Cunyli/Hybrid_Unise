import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import DataModule
from model import Model


def scalar_value(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Evaluate validation token metrics for a UniSE checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    config["devices"] = [int(device) for device in str(args.devices).split(",") if str(device).strip()]
    config.setdefault("wandb", {})["use_wandb"] = False

    if args.ckpt_path is not None:
        config["ckpt_path"] = str(args.ckpt_path)
    if args.pair_manifest is not None:
        config["dataset_config"]["val_kwargs"]["pair_manifest"] = str(args.pair_manifest)
    config["dataset_config"]["val_kwargs"]["batch_size"] = int(args.batch_size)
    config["dataset_config"]["val_kwargs"]["num_workers"] = min(4, int(args.batch_size))
    config["dataset_config"]["val_kwargs"]["prefetch"] = min(4, int(args.batch_size))
    if args.samples_per_epoch is not None:
        config["dataset_config"]["val_kwargs"]["samples_per_epoch"] = int(args.samples_per_epoch)

    model = Model(config=config)
    data_module = DataModule(**config["dataset_config"])
    data_module.setup("fit")
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        logger=False,
        enable_checkpointing=False,
    )
    metrics = trainer.validate(
        model,
        dataloaders=data_module.val_dataloader(),
        ckpt_path=config["ckpt_path"],
        verbose=False,
        weights_only=False,
    )
    result = {key: scalar_value(value) for key, value in metrics[0].items()} if metrics else {}
    result.update(
        {
            "config": str(args.config.resolve()),
            "ckpt_path": str(Path(config["ckpt_path"]).resolve()),
            "pair_manifest": str(Path(config["dataset_config"]["val_kwargs"]["pair_manifest"]).resolve()),
            "samples_per_epoch": int(config["dataset_config"]["val_kwargs"]["samples_per_epoch"]),
            "batch_size": int(config["dataset_config"]["val_kwargs"]["batch_size"]),
        }
    )

    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
