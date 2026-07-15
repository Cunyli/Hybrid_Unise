import pytorch_lightning as pl
import torch
import yaml
import hashlib
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from model import Model
from dataloader import DataModule

WANDB_STANDARD_TAG = 'wandb_standard_v1'
WANDB_MAX_RUN_NAME_LEN = 256
WANDB_MAX_GROUP_LEN = 120


class ChartsWandbLogger(WandbLogger):
    def log_metrics(self, metrics, step=None):
        is_lightning_validation = (
            'charts/global_step' not in metrics
            and any(key.startswith('val/') for key in metrics)
        )
        is_lightning_training = (
            'charts/global_step' not in metrics
            and any(key.startswith('train/') for key in metrics)
        )
        history_step = metrics.get('trainer/global_step', step)
        metrics = {
            ('charts/global_step' if key == 'trainer/global_step' else 'charts/epoch' if key == 'epoch' else key): value
            for key, value in metrics.items()
        }
        metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith(
                (
                    'charts/',
                    'train/',
                    'val/',
                    'val_tf/',
                    'val_free/',
                    'val_avqi/',
                    'val_avqi_pathology/',
                    'val_avqi_health/',
                )
            )
        }
        if history_step is not None:
            history_step = int(history_step) + int(is_lightning_validation or is_lightning_training)
            metrics = dict(metrics, **{'charts/global_step': history_step})
        self.experiment.log(metrics, step=history_step)


def name_token(value, default='na'):
    text = str(value if value not in (None, '') else default).strip().lower()
    text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return text or default


def sanitize_wandb_tag(value, max_len=64):
    text = str(value if value not in (None, '') else 'na').strip()
    if 1 <= len(text) <= max_len:
        return text
    token = name_token(text)
    if len(token) <= max_len:
        return token
    digest = hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]
    return f"{token[:max_len - 9].rstrip('-')}-{digest}"


def shorten_wandb_identifier(value, max_len):
    text = str(value if value not in (None, '') else 'na').strip()
    if 1 <= len(text) <= max_len:
        return text
    token = name_token(text)
    if len(token) <= max_len:
        return token
    digest = hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]
    return f"{token[:max_len - 9].rstrip('-')}-{digest}"


def build_wandb_identity(repo_name, model_name, dataset_type, experiment, change, timestamp):
    run_name = shorten_wandb_identifier(
        '__'.join(
            [
                name_token(timestamp),
                name_token(repo_name),
                name_token(model_name),
                name_token(dataset_type),
                name_token(change or experiment),
            ]
        ),
        WANDB_MAX_RUN_NAME_LEN,
    )
    group = shorten_wandb_identifier(
        '__'.join(
            [
                name_token(repo_name),
                name_token(model_name),
                name_token(dataset_type),
                name_token(experiment),
            ]
        ),
        WANDB_MAX_GROUP_LEN,
    )
    return run_name, group


class PruneLatestCheckpoints(Callback):
    def __init__(self, checkpoint_dir, keep=3, pattern='latest_*.ckpt'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep = keep
        self.pattern = pattern

    @staticmethod
    def _checkpoint_mtime(path):
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return 0

    def _prune(self):
        checkpoints = sorted(
            [path for path in self.checkpoint_dir.glob(self.pattern) if path.exists()],
            key=self._checkpoint_mtime,
        )
        for checkpoint_path in checkpoints[:-self.keep]:
            checkpoint_path.unlink(missing_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero:
            self._prune()

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
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


_CHECKPOINT_STEP_RE = re.compile(r"step[=-](\d+)")


def checkpoint_sort_key(path):
    match = _CHECKPOINT_STEP_RE.search(path.name)
    step = int(match.group(1)) if match else -1
    return (step, path.stat().st_mtime)


def resolve_auto_stage_init_checkpoint(config):
    if config.get("stage_init_checkpoint") != "auto":
        return

    checkpoint_dir = config.get("stage_init_checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError("stage_init_checkpoint_dir is required when stage_init_checkpoint is 'auto'.")

    root = Path(str(checkpoint_dir)).expanduser()
    if not root.is_absolute():
        root = Path(str(config.get("_config_dir", "."))).expanduser() / root
    patterns = config.get("stage_init_checkpoint_patterns") or ["version_*/latest_*.ckpt"]
    candidates = []
    for pattern in patterns:
        candidates.extend(root.glob(str(pattern)))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No stage-init checkpoint candidates found under {root} with patterns {patterns!r}."
        )

    checkpoint_path = max(candidates, key=checkpoint_sort_key)
    config["stage_init_checkpoint"] = str(checkpoint_path)
    print(f"Resolved stage_init_checkpoint=auto to {checkpoint_path}")


def default_checkpoint_monitor(config):
    monitor = config.get('checkpoint_monitor')
    if monitor is not None:
        return monitor
    return 'val/nll' if str(config.get('stage', '')).lower() == 'gen' else 'val/loss'


def build_best_checkpoint_callback(ckpt_dir, config):
    keep = int(config.get('best_checkpoints_to_keep', 3))
    if keep <= 0:
        return None
    monitor = default_checkpoint_monitor(config)
    if not monitor:
        return None
    return ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best_{epoch:02d}-{step:06d}',
        monitor=monitor,
        mode=str(config.get('checkpoint_mode', 'min')),
        save_top_k=keep,
        auto_insert_metric_name=False,
    )


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
    tags = [
        tag
        for tag in dict.fromkeys(
            sanitize_wandb_tag(tag)
            for tag in default_tags + list(wandb_cfg.get('tags') or []) + [WANDB_STANDARD_TAG]
        )
    ]

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
            'avqi_validation_interval_steps': config.get('avqi_validation_interval_steps'),
            'avqi_validation_script': config.get('avqi_validation_script'),
            'avqi_validation_pair_csv': config.get('avqi_validation_pair_csv'),
            'avqi_validation_output_root': config.get('avqi_validation_output_root'),
            'avqi_validation_clean_cache': config.get('avqi_validation_clean_cache'),
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
    experiment_obj.define_metric('val_tf/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('val_free/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('val_avqi/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('val_avqi_pathology/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('val_avqi_health/*', step_metric='charts/global_step', overwrite=True)
    experiment_obj.define_metric('charts/*', step_metric='charts/global_step', overwrite=True)
    return logger


def main(args):
    pl.seed_everything(3407)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config.setdefault('_config_dir', str(Path(args.config).expanduser().resolve().parent))
    probe_run_nonce = os.environ.get('HYBRID_PROBE_RUN_NONCE')
    if probe_run_nonce is not None:
        config['_probe_run_nonce'] = probe_run_nonce
        config['_probe_job_id'] = os.environ.get('SLURM_JOB_ID', 'manual')
    resolve_auto_stage_init_checkpoint(config)
    if config.get('float32_matmul_precision'):
        torch.set_float32_matmul_precision(config['float32_matmul_precision'])
    
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
    callbacks = [latest_checkpoint_callback]
    best_checkpoint_callback = build_best_checkpoint_callback(ckpt_dir, config)
    if best_checkpoint_callback is not None:
        callbacks.append(best_checkpoint_callback)
    callbacks.append(
        PruneLatestCheckpoints(
            ckpt_dir,
            keep=int(config.get('latest_checkpoints_to_keep', 3)),
        )
    )
    
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        max_epochs=config['max_epochs'],
        max_steps=int(config.get('max_steps', -1)),
        val_check_interval=validation_interval_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=callbacks,
        logger=logger,
        strategy="auto" if len(config['devices']) == 1 else 'ddp_find_unused_parameters_true',
        log_every_n_steps=wandb_log_interval_steps,
        precision=config.get('precision', '32-true'),
        accumulate_grad_batches=int(config.get('accumulate_grad_batches', 1)),
    )

    fit_kwargs = {"ckpt_path": resume_path}
    if resume_path is not None:
        fit_kwargs["weights_only"] = False
    trainer.fit(model, datamodule=data_module, **fit_kwargs)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    args = parser.parse_args()
    main(args)
