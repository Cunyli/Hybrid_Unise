import pytorch_lightning as pl
import yaml
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import re
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from model import Model
from dataloader import DataModule

WANDB_STANDARD_TAG = 'wandb_standard_v1'


class ChartsWandbLogger(WandbLogger):
    def log_metrics(self, metrics, step=None):
        is_lightning_validation = (
            'charts/global_step' not in metrics
            and any(key.startswith('val/') for key in metrics)
        )
        history_step = metrics.get('trainer/global_step', step)
        metrics = {
            ('charts/global_step' if key == 'trainer/global_step' else 'charts/epoch' if key == 'epoch' else key): value
            for key, value in metrics.items()
        }
        metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith(('charts/', 'train/', 'val/', 'val_avqi/'))
        }
        if history_step is not None:
            history_step = int(history_step) + int(is_lightning_validation)
            metrics = dict(metrics, **{'charts/global_step': history_step})
        self.experiment.log(metrics, step=history_step)


def name_token(value, default='na'):
    text = str(value if value not in (None, '') else default).strip().lower()
    text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return text or default


def build_wandb_identity(repo_name, model_name, dataset_type, experiment, change, timestamp):
    run_name = '__'.join(
        [
            name_token(timestamp),
            name_token(repo_name),
            name_token(model_name),
            name_token(dataset_type),
            name_token(change or experiment),
        ]
    )
    group = '__'.join(
        [
            name_token(repo_name),
            name_token(model_name),
            name_token(dataset_type),
            name_token(experiment),
        ]
    )
    return run_name, group


class PruneLatestCheckpoints(Callback):
    def __init__(self, checkpoint_dir, keep=3, pattern='latest_*.ckpt'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep = keep
        self.pattern = pattern

    def _prune(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob(self.pattern),
            key=lambda path: path.stat().st_mtime,
        )
        for checkpoint_path in checkpoints[:-self.keep]:
            checkpoint_path.unlink(missing_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._prune()

    def on_validation_end(self, trainer, pl_module):
        self._prune()


def resolve_resume_path(resume, checkpoint_dir):
    if resume != "auto":
        return resume

    ckpt_root = Path(checkpoint_dir)
    for pattern in ("version_*/latest_*.ckpt", "version_*/*last.ckpt", "version_*/best_*.ckpt", "version_*/*.ckpt"):
        candidates = sorted(ckpt_root.glob(pattern), key=lambda path: path.stat().st_mtime)
        if candidates:
            return str(candidates[-1])
    return None


def infer_dataset_type(config, split):
    return config.get('dataset_config', {}).get(f'{split}_kwargs', {}).get('dataset_type', 'native')


def build_wandb_logger(config, config_path):
    wandb_cfg = config.get('wandb', {})
    repo_name = wandb_cfg.get('repo_name') or Path.cwd().name
    experiment = wandb_cfg.get('experiment') or Path(config['log_dir']).name
    model_name = wandb_cfg.get('model_name') or config.get('model_name') or 'unise'
    dataset_type = infer_dataset_type(config, 'train')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    change = wandb_cfg.get('change') or config.get('wandb_change') or config.get('change') or experiment
    default_run_name, default_group = build_wandb_identity(
        repo_name,
        model_name,
        dataset_type,
        experiment,
        change,
        timestamp,
    )
    run_name = wandb_cfg.get('name') or default_run_name
    run_group = wandb_cfg.get('group') or default_group

    default_tags = [
        repo_name,
        experiment,
        dataset_type,
        model_name,
    ]
    tags = list(dict.fromkeys(default_tags + list(wandb_cfg.get('tags') or []) + [WANDB_STANDARD_TAG]))

    logger = ChartsWandbLogger(
        project=wandb_cfg.get('project', 'unise'),
        entity=wandb_cfg.get('entity'),
        name=run_name,
        group=run_group,
        tags=tags,
        save_dir=config['log_dir'],
        config={
            'repo_name': repo_name,
            'model_name': model_name,
            'experiment': experiment,
            'wandb_change': change,
            'wandb_group': run_group,
            'config_path': str(config_path),
            'log_dir': config.get('log_dir'),
            'checkpoint_dir': config.get('checkpoint_dir'),
            'train_dataset_type': dataset_type,
            'val_dataset_type': infer_dataset_type(config, 'val'),
            'test_dataset_type': infer_dataset_type(config, 'test'),
            'max_epochs': config.get('max_epochs'),
            'wandb_log_interval_steps': config.get('wandb_log_interval_steps', config.get('log_every_n_steps')),
            'validation_interval_steps': config.get('validation_interval_steps', config.get('val_check_interval')),
            'checkpoint_interval_steps': config.get('checkpoint_interval_steps'),
            'devices': config.get('devices'),
        },
    )
    experiment_obj = logger.experiment
    experiment_obj.define_metric('charts/global_step', overwrite=True)
    experiment_obj.define_metric('*', step_metric='charts/global_step', step_sync=True, overwrite=True)
    experiment_obj.define_metric('trainer/global_step', hidden=True, overwrite=True)
    experiment_obj.define_metric('charts/epoch', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('train/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('val/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('val_avqi/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('charts/*', step_metric='charts/global_step', overwrite=True)
    return logger


def main(args):
    pl.seed_everything(3407)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    tb_logger = TensorBoardLogger(save_dir=config['log_dir'], name='tensorboard')
    logger = tb_logger
    if config.get('wandb', {}).get('use_wandb', False):
        logger = [tb_logger, build_wandb_logger(config, args.config)]
    checkpoint_dir = Path(config.get('checkpoint_dir', Path('checkpoints') / Path(config['log_dir']).name))
    validation_interval_steps = int(config.get('validation_interval_steps', config.get('val_check_interval', 1)))
    checkpoint_interval_steps = int(config.get('checkpoint_interval_steps', validation_interval_steps))
    wandb_log_interval_steps = int(config.get('wandb_log_interval_steps', config.get('log_every_n_steps', 1)))
    ckpt_dir = checkpoint_dir / f'version_{tb_logger.version}' #change your folder, where to save files
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config['ckpt_dir'] = ckpt_dir
    resume_path = resolve_resume_path(config.get('resume'), checkpoint_dir)
    model = Model(config=config)
    data_module = DataModule(**config['dataset_config'])
    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='latest_{epoch:02d}-{step:06d}',
        save_top_k=-1,
        every_n_train_steps=checkpoint_interval_steps,
        save_last=False,
    )
    
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        max_epochs=config['max_epochs'],
        val_check_interval=validation_interval_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=[
            latest_checkpoint_callback,
            PruneLatestCheckpoints(ckpt_dir, keep=3),
        ],
        logger=logger,
        strategy="auto" if len(config['devices']) == 1 else 'ddp_find_unused_parameters_true',
        log_every_n_steps=wandb_log_interval_steps,
    )

    trainer.fit(model, data_module, ckpt_path=resume_path, weights_only=False)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    args = parser.parse_args()
    main(args)
