import importlib.util
import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks
from pathlib import Path
import soundfile as sf
import time
import math
import random

from .bicodec import BiCodecTokenizer
from .llm import LLM_SFT


from transformers import AutoModel


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.stft_conf = config['stft_config']
        
        self.tokenizer = BiCodecTokenizer(model_dir=config['codec_ckpt_dir'])
        self.tokenizer.eval()
        self.tokenizer.requires_grad_(False)
        self.dnn = LLM_SFT(**config['llm_config'])

        self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").eval()
        self.semantic_model.requires_grad_(False)

        self.current_traning_step = -1
        self._train_metric_sums = {}
        self._train_metric_count = 0
        self._train_window_elapsed_sec = 0.0
        self._samples_seen = 0
        self._train_window_samples = 0
        self._train_batch_start_time = None
        self._latest_train_metrics = {}
        self._latest_grad_norm = None
        self.best_avqi_gap_to_clean = float('inf')
        self.latest_avqi_gap_to_clean = None
        self.best_guarded_val_loss = float('inf')

        # self.automatic_optimization = False

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        self.semantic_model.eval()
        return self
    
    @torch.no_grad()
    def extract_semantic_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        # wavs: (b,t)
        wavs = F.pad(wavs, (160, 160))
        
        feats = self.semantic_model(wavs, output_hidden_states=True)
        feats_mix = torch.stack(feats.hidden_states, dim=1).mean(1)

        # 指数压缩
        # symbol = (feats_mix > 0).float() * 2 - 1
        # magnitude = feats_mix.abs() ** 0.3
        # feats_mix = symbol * magnitude
        
        return feats_mix.detach()
    
    def stft_logmel(self, x):
        # x:(B,T)
        assert x.ndim == 2
        hop_length = self.stft_conf['hop_length']
        win_length = self.stft_conf['win_length']
        n_fft = self.stft_conf['n_fft']
        n_mels = self.stft_conf['n_mels']

        pad_length = math.ceil(x.size(-1) / hop_length) * hop_length - x.size(-1)
        x = torch.nn.functional.pad(x, ((win_length - hop_length) // 2, pad_length + (win_length - hop_length) // 2))
        spec = torch.stft(
            x,
            n_fft,
            hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),
            onesided=True,
            center=False,
            return_complex=True,
        ).transpose(1, 2)  # (B,T,F)
        if not hasattr(self, 'fb'):
            fb = melscale_fbanks(n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=8000.0, n_mels=n_mels, sample_rate=16000)
            setattr(self, 'fb', fb.to(x.device))
        mag = spec.abs()  # (b,t,f)
        mel = mag @ self.fb  # (B,T,M)
        mel = torch.log(mel + 1e-10)
        return mel
    
    # 重写 state_dict: 排除 tokenizer semantic_model
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if key.startswith('tokenizer.') or key.startswith('semantic_model.'):
                del state[key]
        return state

    # 重写 load_state_dict: 排除 tokenizer
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=False)
    
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        mode, enroll, mix, speech, interf, fs, lengths, names = batch

        if mode == 'rtse':
            global_tokens, semantic_tokens = self.tokenizer.tokenize(interf)  # (b, 1, 32): int32, (b, T): int64
        else:
            global_tokens, semantic_tokens = self.tokenizer.tokenize(speech)  # (b, 1, 32): int32, (b, T): int64
        
        mix_mel = self.stft_logmel(mix)
        mix_feats = self.extract_semantic_features(mix)
        if enroll is not None:
            enroll_mel = self.stft_logmel(enroll)
            enroll_feats = self.extract_semantic_features(enroll)
        else:
            enroll_mel, enroll_feats = None, None

        loss, metrics = self.dnn(
            task_name = mode,
            enroll_mel = enroll_mel,
            enroll_feats = enroll_feats,
            mix_mel = mix_mel,
            mix_feats = mix_feats,
            global_ids = global_tokens.squeeze(1), 
            semantic_ids = semantic_tokens,
        )
        
        log_metrics = {
            'train/loss': metrics['loss'],
            'train/acc': metrics['acc'],
            'train/loss_global': metrics['global_loss'],
            'train/loss_semantic': metrics['semantic_loss'],
            'train/acc_global': metrics['global_acc'],
            'train/acc_semantic': metrics['semantic_acc'],
        }
        self._latest_train_metrics = log_metrics
        self.current_traning_step += 1
        return loss
        # No need to return loss under manual optimization mode.

    def on_train_batch_start(self, batch, batch_idx):
        self._train_batch_start_time = time.perf_counter()

    def _log_metrics_direct(self, metrics):
        step = int(self.global_step)
        metrics = {**metrics, 'charts/global_step': step}
        for logger in self.trainer.loggers:
            logger.log_metrics(metrics, step=step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        elapsed = time.perf_counter() - self._train_batch_start_time
        batch_size = batch[2].size(0)
        world_size = self.trainer.world_size
        global_batch_size = batch_size * world_size
        for name, value in self._latest_train_metrics.items():
            self._train_metric_sums[name] = self._train_metric_sums.get(name, 0.0) + value.detach()
        self._train_metric_count += 1
        self._train_window_elapsed_sec += elapsed
        self._samples_seen += global_batch_size
        self._train_window_samples += global_batch_size
        interval = int(self.config.get('wandb_log_interval_steps', 50))
        if self._train_metric_count >= interval:
            metrics = {name: value / self._train_metric_count for name, value in self._train_metric_sums.items()}
            metrics.update({
                'charts/epoch': float(self.current_epoch),
                'charts/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                'charts/samples_per_sec': self._train_window_samples / self._train_window_elapsed_sec,
            })
            if self._latest_grad_norm is not None:
                metrics['charts/grad_norm'] = self._latest_grad_norm
            self._log_metrics_direct(metrics)
            self._train_metric_sums = {}
            self._train_metric_count = 0
            self._train_window_elapsed_sec = 0.0
            self._train_window_samples = 0
        avqi_interval = int(self.config.get('avqi_validation_interval_steps', 1000))
        if (
            avqi_interval > 0
            and self.trainer.is_global_zero
            and self.global_step > 0
            and self.global_step % avqi_interval == 0
        ):
            avqi_metrics = self._validation_avqi_metrics()
            gap = avqi_metrics['avqi_gap_to_clean']
            self.latest_avqi_gap_to_clean = gap
            self._log_metrics_direct({
                f'val_avqi/{name}': value
                for name, value in avqi_metrics.items()
            })
            self._save_best_avqi_gap_checkpoint(gap)

    def _save_best_avqi_gap_checkpoint(self, gap):
        if gap < 0 or gap >= self.best_avqi_gap_to_clean:
            return
        self.best_avqi_gap_to_clean = gap
        checkpoint_path = Path(self.config['ckpt_dir']) / 'best_avqi_gap.ckpt'
        self.trainer.save_checkpoint(str(checkpoint_path))

    def _save_best_guarded_checkpoint(self, val_loss):
        if (
            self.latest_avqi_gap_to_clean is None
            or self.latest_avqi_gap_to_clean < 0
            or val_loss >= self.best_guarded_val_loss
        ):
            return
        self.best_guarded_val_loss = val_loss
        checkpoint_path = Path(self.config['ckpt_dir']) / 'best_guarded.ckpt'
        self.trainer.save_checkpoint(str(checkpoint_path))

    def on_save_checkpoint(self, checkpoint):
        checkpoint['best_avqi_gap_to_clean'] = self.best_avqi_gap_to_clean
        checkpoint['latest_avqi_gap_to_clean'] = self.latest_avqi_gap_to_clean
        checkpoint['best_guarded_val_loss'] = self.best_guarded_val_loss
        checkpoint['current_traning_step'] = self.current_traning_step

    def on_load_checkpoint(self, checkpoint):
        self.best_avqi_gap_to_clean = float(checkpoint.get('best_avqi_gap_to_clean', float('inf')))
        latest_gap = checkpoint.get('latest_avqi_gap_to_clean')
        self.latest_avqi_gap_to_clean = float(latest_gap) if latest_gap is not None else None
        self.best_guarded_val_loss = float(checkpoint.get('best_guarded_val_loss', float('inf')))
        self.current_traning_step = checkpoint.get('current_traning_step', self.current_traning_step)

    def _load_avqi_runner(self):
        script_path = self.config.get('avqi_validation_script')
        if not script_path:
            raise ValueError('avqi_validation_script must be set when avqi_validation_interval_steps is enabled')
        script_path = Path(script_path)
        spec = importlib.util.spec_from_file_location('validation_avqi_gap', script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not load AVQI validation script: {script_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_validation_avqi_metrics

    def _validation_avqi_metrics(self):
        self.eval()

        @torch.inference_mode()
        def enhance_one(path):
            src, _ = librosa.load(path, sr=16000)
            src = torch.as_tensor(src, dtype=torch.float32, device=self.device).unsqueeze(0)
            seg_len = 5 * 16000
            pad_len = math.ceil(src.size(-1) / seg_len) * seg_len - src.size(-1)
            seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], 'wrap')
            seg_src = torch.from_numpy(seg_src).to(src.device).reshape(-1, seg_len)
            seg_src = seg_src / src.abs().max(dim=-1, keepdim=True)[0].clamp_min(1e-9)
            mix_mel = self.stft_logmel(seg_src)
            mix_feats = self.extract_semantic_features(seg_src)
            global_ids, semantic_ids = self.dnn.generate(
                task_name='se',
                enroll_mel=None,
                enroll_feats=None,
                mix_mel=mix_mel,
                mix_feats=mix_feats,
                do_sample=False,
            )
            est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)
            return est.reshape(-1)[:src.size(-1)].cpu().numpy(), 16000

        run_validation_avqi_metrics = self._load_avqi_runner()
        metrics = run_validation_avqi_metrics("unise", self.global_step, enhance_one)
        self.train()
        return metrics

    def on_train_epoch_end(self):
        # Manually update scheduler under manual optimization mode.
        # g_sch, d_sch = self.lr_schedulers()
        # g_sch.step()
        # d_sch.step()
        pass

    def validation_step(self, batch, batch_idx):
        mode, enroll, mix, speech, interf, fs, lengths, names = batch

        if mode == 'rtse':
            global_tokens, semantic_tokens = self.tokenizer.tokenize(interf)  # (b, 1, 32): int32, (b, T): int64
        else:
            global_tokens, semantic_tokens = self.tokenizer.tokenize(speech)  # (b, 1, 32): int32, (b, T): int64
        
        mix_mel = self.stft_logmel(mix)
        mix_feats = self.extract_semantic_features(mix)
        if enroll is not None:
            enroll_mel = self.stft_logmel(enroll)
            enroll_feats = self.extract_semantic_features(enroll)
        else:
            enroll_mel, enroll_feats = None, None

        loss, metrics = self.dnn(
            task_name = mode,
            enroll_mel = enroll_mel,
            enroll_feats = enroll_feats,
            mix_mel = mix_mel,
            mix_feats = mix_feats,
            global_ids = global_tokens.squeeze(1), 
            semantic_ids = semantic_tokens,
        )
        
        self.log('charts/epoch', float(self.current_epoch), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        log_metrics = {
            'val/loss': metrics['loss'],
            'val/acc': metrics['acc'],
            'val/loss_global': metrics['global_loss'],
            'val/loss_semantic': metrics['semantic_loss'],
            'val/acc_global': metrics['global_acc'],
            'val/acc_semantic': metrics['semantic_acc'],
            'val/ppl': torch.exp(metrics['loss'].detach().clamp(max=20)),
        }
        self.log_dict(log_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_loss', metrics['loss'], on_step=False, on_epoch=True, logger=False, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        grad_norm_sq = torch.zeros((), device=self.device)
        for parameter in self.dnn.parameters():
            if parameter.grad is not None:
                grad_norm_sq += parameter.grad.detach().norm(2).pow(2)
        grad_norm = grad_norm_sq.sqrt()
        self._latest_grad_norm = float(grad_norm.detach().cpu())

    def on_validation_epoch_end(self,):
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        val_loss = self.trainer.callback_metrics.get('valid_loss')
        if val_loss is None:
            val_loss = self.trainer.callback_metrics.get('val/loss')
        if val_loss is None:
            return
        if torch.is_tensor(val_loss):
            val_loss = float(val_loss.detach().cpu())
        self._save_best_guarded_checkpoint(float(val_loss))
    
    def test_step(self, batch, batch_idx):
        mode, enroll, src, tgt, fs, lengths, names = batch

        do_sample = False
        if mode == 'se':
            seg_len = 5 * 16000
            pad_len = math.ceil(src.size(-1) / seg_len) * seg_len - src.size(-1)
            seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], 'wrap')
            seg_src = torch.from_numpy(seg_src).to(src.device)
            # seg_src = torch.nn.functional.pad(src, (0, pad_len))
            seg_src = seg_src.reshape(-1, seg_len)
            seg_src = seg_src / src.abs().max(dim=-1, keepdim=True)[0]

            mix_mel = self.stft_logmel(seg_src)
            mix_feats = self.extract_semantic_features(seg_src)
            global_ids, semantic_ids = self.dnn.generate(
                task_name='se',
                enroll_mel=None,
                enroll_feats=None,
                mix_mel=mix_mel,
                mix_feats=mix_feats,
                do_sample=do_sample,
            )
            est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)  # (B,t)
            est = est.reshape(-1)[:src.size(-1)]
            est = est.cpu().numpy()

            if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                sf.write(Path(self.config['save_enhanced']) / f'{names[0]}.wav', est, samplerate=int(fs[0]))
        elif mode == 'tse':
            seg_len = 5 * 16000
            pad_len = math.ceil(src.size(-1) / seg_len) * seg_len - src.size(-1)
            # seg_src = torch.nn.functional.pad(src, (0, pad_len))
            seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], 'wrap')
            seg_src = torch.from_numpy(seg_src).to(src.device)
            seg_src = seg_src.reshape(-1, seg_len)

            enroll_mel = self.stft_logmel(enroll)
            enroll_feats = self.extract_semantic_features(enroll)
            enroll_mel = torch.cat([enroll_mel for _ in range(seg_src.size(0))], dim=0)
            enroll_feats = torch.cat([enroll_feats for _ in range(seg_src.size(0))], dim=0)

            mix_mel = self.stft_logmel(seg_src)
            mix_feats = self.extract_semantic_features(seg_src)
            global_ids, semantic_ids = self.dnn.generate(
                task_name='tse',
                enroll_mel=enroll_mel,
                enroll_feats=enroll_feats,
                mix_mel=mix_mel,
                mix_feats=mix_feats,
                do_sample=do_sample,
            )

            est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)  # (B,t)
            est = est.reshape(-1)[:src.size(-1)]
            est = est.cpu().numpy()

            if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                sf.write(Path(self.config['save_enhanced']) / f'{names[0]}.wav', est, samplerate=int(fs[0]))
        elif mode == 'ss':  # 先se，再tse，最后rtse
            seg_len = 5 * 16000
            if src.size(-1) > seg_len:
                seg_src = src[:, :seg_len]
            else:
                # seg_src = torch.nn.functional.pad(src, (0, seg_len - src.size(-1)), 'circular')
                seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, seg_len - src.size(-1))], 'wrap')
                seg_src = torch.from_numpy(seg_src).to(src.device)
            
            mix_mel = self.stft_logmel(seg_src)
            mix_feats = self.extract_semantic_features(seg_src)
            global_ids, semantic_ids = self.dnn.generate(
                task_name='se',
                enroll_mel=None,
                enroll_feats=None,
                mix_mel=mix_mel,
                mix_feats=mix_feats,
                do_sample=do_sample,
            )
            enroll = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)  # (1,t)
            enroll = enroll[:, :seg_len]
            enroll = enroll / (torch.max(torch.abs(enroll)) + 1e-5) * 0.99
            # enroll = self.stft_logmel(enroll)
            enroll_mel = self.stft_logmel(enroll)
            enroll_feats = self.extract_semantic_features(enroll)
            

            pad_len = math.ceil(src.size(-1) / seg_len) * seg_len - src.size(-1)
            seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], 'wrap')
            seg_src = torch.from_numpy(seg_src).to(src.device)
            seg_src = seg_src.reshape(-1, seg_len)
            # enroll = torch.cat([enroll for _ in range(seg_src.size(0))], dim=0)
            enroll_mel = torch.cat([enroll_mel for _ in range(seg_src.size(0))], dim=0)
            enroll_feats = torch.cat([enroll_feats for _ in range(seg_src.size(0))], dim=0)
            mix_mel = self.stft_logmel(seg_src)
            mix_feats = self.extract_semantic_features(seg_src)
            global_ids, semantic_ids = self.dnn.generate(
                task_name='tse',
                enroll_mel=enroll_mel,
                enroll_feats=enroll_feats,
                mix_mel=mix_mel,
                mix_feats=mix_feats,
                do_sample=do_sample,
            )
            est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)  # (B,t)
            est = est.reshape(-1)[:src.size(-1)].cpu().numpy()
            if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                sf.write(Path(self.config['save_enhanced']) / f'{names[0]}_s1.wav', est, samplerate=int(fs[0]))
            

            global_ids, semantic_ids = self.dnn.generate(
                task_name='rtse',
                enroll_mel=enroll_mel,
                enroll_feats=enroll_feats,
                mix_mel=mix_mel,
                mix_feats=mix_feats,
                do_sample=do_sample,
            )
            est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)  # (B,t)
            est = est.reshape(-1)[:src.size(-1)].cpu().numpy()
            if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                sf.write(Path(self.config['save_enhanced']) / f'{names[0]}_s2.wav', est, samplerate=int(fs[0]))


    
    def on_test_epoch_end(self,):
        # print(f'PESQ: {self.trainer.callback_metrics["test_pesq"]:.2f}')
        # print(f'STOI: {self.trainer.callback_metrics["test_stoi"]:.3f}')
        # print("******")
        # print('time:', sum(self.time_list), 's')
        pass

    @torch.inference_mode()
    def generate(self, task_name, enroll, mixture):
        # cond: (B, T)
        if enroll is not None:
            enroll = self.stft_logmel(enroll)
        length = mixture.size(-1)
        mixture = self.stft_logmel(mixture)

        global_ids, semantic_ids = self.dnn.generate(
            task_name=task_name,
            enroll=enroll,
            mixture=mixture,
            do_sample=True,
        )
        wav_rec = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids)[...,:length]  # (1,1,t)

        import soundfile as sf
        sf.write('test.wav', wav_rec.squeeze().cpu().numpy(), 16000)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.dnn.parameters(), **self.config['opt'])

        # Actually, the keys 'interval' and 'frequency' will be ignored under manual optimization mode.
        # We just reserve them for the automatic optimization alternative.
        # sch = {
        #     'scheduler': torch.optim.lr_scheduler.StepLR(opt, **self.config['sch']), 
        #     'interval': 'epoch',
        #     'frequency': 1,
        # }

        def warmup_lambda(step):
            warmup_steps = self.config['sch']['warmup_steps']
            step_decay = self.config['sch']['step_decay']
            if step < warmup_steps:
                # 余弦预热
                return 0.5 * (1 + math.cos(math.pi * (1 - step / warmup_steps)))
            else:
                return max(step_decay ** (step - warmup_steps), self.config['sch']['min_factor'])
            
        sch = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, warmup_lambda), 
            'interval': 'step',
            'frequency': 1,
        }

        return [opt], [sch]
