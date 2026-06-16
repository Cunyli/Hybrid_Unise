import argparse
import sys
import yaml
import pytorch_lightning as pl
import torch
from pathlib import Path

from model import Model
from dataloader import DataModule


def load_model_weights(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model weights from {ckpt_path}")
    if result is None:
        return
    if result.missing_keys:
        print(f"Missing keys: {len(result.missing_keys)}")
        for key in result.missing_keys[:20]:
            print(f"  missing: {key}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {len(result.unexpected_keys)}")
        for key in result.unexpected_keys[:20]:
            print(f"  unexpected: {key}")


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config.setdefault('_config_dir', str(Path(args.config).expanduser().resolve().parent))
    
    if args.save_enhanced is not None:
        config['save_enhanced'] = args.save_enhanced
        Path(args.save_enhanced).mkdir(parents=True, exist_ok=True)
    if args.ckpt_path is not None:
        config['ckpt_path'] = args.ckpt_path
    if args.stage is not None:
        config['stage'] = args.stage
    config['stage_init_checkpoint'] = None
    
    model = Model(config=config)
    ckpt_path = config.get('ckpt_path')
    if ckpt_path is not None:
        load_model_weights(model, ckpt_path)

    # import ipdb; ipdb.set_trace()
    
    data_module = DataModule(**config['dataset_config'])
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        logger=False,
    )

    trainer.test(model, datamodule=data_module, ckpt_path=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test model')
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    parser.add_argument('--save_enhanced', type=str, default=None, help='The dir path to save enhanced wavs.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path for inference.')
    parser.add_argument('--stage', type=str, choices=('disc', 'gen', 'fusion', 'joint'), default=None, help='Override hybrid stage for checkpoint evaluation.')

    args = parser.parse_args()
    sys.exit(main(args))
