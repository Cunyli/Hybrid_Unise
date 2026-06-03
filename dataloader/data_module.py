import torch
import soundfile as sf
from typing import Union, List
from pathlib import Path
import numpy as np
import random
import pytorch_lightning as pl
import torch.utils
import torch.utils.data
from copy import deepcopy
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import queue
import threading
import librosa
import yaml
import time
import collections
import sys

from .simulation import simulate_data

import warnings
warnings.filterwarnings("ignore")


class WaveInfo:
    def __init__(self, line: str, type: str):
        split_list = line.strip().split(' ')
        assert type in ['speech', 'noise', 'rir'], type
        if type == 'rir':
            self.utt, self.path = split_list
            self.spk = 'unknown'
            self.fs = None
            self.offset = 0
            self.duration = None
        elif type == 'speech':
            self.utt, self.spk, self.path = split_list
            self.fs = None
            self.offset = 0
            self.duration = None
        elif type == 'noise':
            self.utt, self.fs, start, frames, self.path = split_list
            self.spk = 'unknown'
            self.fs = eval(self.fs)
            self.offset = eval(start) / self.fs
            self.duration = eval(frames) / self.fs


class TrainDataLoadIter:
    def __init__(
        self,
        simulation_config: Union[str, Path],
        speech_scp_path: Union[str, Path, List], 
        noise_scp_path: Union[str, Path, List], 
        rir_scp_path: Union[str, Path, List], 
        speech_scp_base_dir: Union[str, Path] = '',
        batch_size: int = 1, 
        cut_duration: Union[float, List[float]] = 3.0, 
        enroll_duration: float = 5.0,
        num_workers: int = 1, 
        prefetch: int = 0,
        samples_per_epoch: int = 10000,
    ):
        self.is_train = True
        self.batch_size = batch_size
        self.cut_duration = cut_duration
        self.enroll_duration = enroll_duration
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch

        with open(simulation_config, "r") as f:
            self.simulation_config = yaml.safe_load(f)
        
        self.speech_scp_base_dir = Path(speech_scp_base_dir)
        self.speech_list = self.load_scp_to_list(speech_scp_path, 'speech')
        # 按说话人分类
        self.spk2speech = collections.defaultdict(list)
        for speech_info in self.speech_list:
            speech_info.path = self.speech_scp_base_dir / speech_info.path
            self.spk2speech[speech_info.spk].append(speech_info)
        self.spk_list = list(self.spk2speech.keys())
        for spk in self.spk_list:
            assert len(self.spk2speech[spk]) > 1

        self.noise_list = self.load_scp_to_list(noise_scp_path, 'noise')
        self.rir_list = self.load_scp_to_list(rir_scp_path, 'rir')
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    
    def load_scp_to_list(self, scp_path, type):
        path_list = []
        if not isinstance(scp_path, List):
            scp_path = [scp_path]
        for p in scp_path:
            with open(p, 'r') as f:
                for line in f:
                    path_list.append(WaveInfo(line, type))
        return path_list
    
    def pad_or_cut_wav(self, wav, length, offset=None):
        # wav: [1, T]
        if wav.shape[-1] < length: # pad
            wav = np.pad(wav, [(0, 0), (0, length - wav.shape[-1])], mode='wrap')
            return wav, None
        else: # cut
            if offset is None:
                offset = random.randint(0, wav.shape[-1] - length)
            wav = wav[..., offset: offset + length]
            return wav, offset
    
    def normalize_src_tgt(self, src, tgt, low=0.1, high=0.99):
        max_tgt_value = np.max(np.abs(tgt)) + 1e-5
        max_src_value = np.max(np.abs(src)) + 1e-5
        max_value = max(max_tgt_value, max_src_value)
        threshold = high / max_value  # 防止削波

        target_value = random.uniform(low, high)
        factor = min(target_value / max_tgt_value, threshold)
        src = src * factor
        tgt = tgt * factor

        return src, tgt
    
    def normalize_mix_speech_inferf(self, mix, speech, interf, low=0.1, high=0.99):
        a, b, c = np.max(np.abs(mix)), np.max(np.abs(speech)), np.max(np.abs(interf))
        max_value = max(a, b, c) + 1e-5
        min_value = min(a, b, c)

        factor = high / max_value
        if min_value * factor <= low:
            return mix * factor, speech * factor, interf * factor
        else:
            factor = random.uniform(low / (min_value * factor), 1) * factor
            return mix * factor, speech * factor, interf * factor


    def load_wav(self, info: WaveInfo, fs=None):
        wav, fs_ = librosa.load(info.path, dtype=np.float32, sr=fs, mono=False, offset=info.offset, duration=info.duration)
        if wav.ndim == 1:
            wav = wav[None]  # (1, T)
        else:
            wav = wav[:1, :]  # 取第0通道
        return wav, fs_
    
    def load_wav_queue(self, info, fs, q):
        try:
            wav, fs = self.load_wav(info, fs)
            q.put((wav, fs))
        except Exception as e:
            q.put(e)
    
    def load_wav_with_timeout(self, info, fs=None, timeout=1.0):
        result_queue = queue.Queue()
        thread = threading.Thread(target=self.load_wav_queue, args=(info, fs, result_queue))
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"读取音频文件超时：{info.path}")
        
        result = result_queue.get()
        if isinstance(result, Exception):
            raise Exception('load error')
        return result
    
    def process_one_sample(self, fs, cut_duration, mode):
        spk1, spk2 = random.sample(self.spk_list, 2)

        speech_info, enroll_info = random.sample(self.spk2speech[spk1], 2)
        interf_info = random.choice(self.spk2speech[spk2])
        if mode == 'tse' or mode == 'rtse':  # 启用TSE/rTSE模式
            try:
                speech, _ = self.load_wav_with_timeout(speech_info, fs, timeout=2.0)
                enroll, _ = self.load_wav_with_timeout(enroll_info, fs, timeout=2.0)
                interf, _ = self.load_wav_with_timeout(interf_info, fs, timeout=2.0)
            except Exception as e:
                print(e)
                return self.process_one_sample(fs, cut_duration, mode)
        elif mode == 'se' and random.random() < self.simulation_config['se_interference']['prob']:  # SE模式，启用干扰说话人
            try:
                speech, _ = self.load_wav_with_timeout(speech_info, fs, timeout=2.0)
                enroll = None
                interf, _ = self.load_wav_with_timeout(interf_info, fs, timeout=2.0)
            except Exception as e:
                print(e)
                return self.process_one_sample(fs, cut_duration, mode)
        else:  # SE模式，不启用干扰说话人
            try:
                speech, _ = self.load_wav_with_timeout(speech_info, fs, timeout=2.0)
                enroll = None
                interf = None
            except Exception as e:
                print(e)
                return self.process_one_sample(fs, cut_duration, mode)
        
        noise_info = random.choice(self.noise_list)
        noise, _ = self.load_wav(noise_info, fs)
        
        rir_info = random.choice(self.rir_list)
        rir, _ = self.load_wav(rir_info, fs)

        mix, speech, interf = simulate_data(
            mode=mode,
            speech=speech,
            interf=interf,
            noise=noise,
            rir=rir,
            fs=fs,
            config=self.simulation_config,
        )

        if cut_duration is not None:
            length = int(cut_duration * fs)
            mix, offset = self.pad_or_cut_wav(mix, length, offset=None)
            speech, _ = self.pad_or_cut_wav(speech, length, offset)
            if interf is not None:
                interf, _ = self.pad_or_cut_wav(interf, length, offset)
        else:
            length = speech.shape[-1]
        
        if interf is None:
            mix, speech = self.normalize_src_tgt(mix, speech)
        else:
            mix, speech, interf = self.normalize_mix_speech_inferf(mix, speech, interf)

        if enroll is not None:
            enroll, _ = self.pad_or_cut_wav(enroll, int(self.enroll_duration * fs), offset=None)
            enroll = enroll / (np.max(np.abs(enroll)) + 1e-5) * 0.99

        return enroll, mix, speech, interf, fs, length, speech_info.utt
    

    def data_iter_fn(self, q, event):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for _ in range(len(self)): # for each batch
            fs = 16000 # sample a fs
            cut_duration = self.cut_duration if not isinstance(self.cut_duration, list) else random.uniform(*self.cut_duration)  # sample cut_duration
            mode = random.choice(['se', 'tse', 'rtse'])
            batch_enroll = []
            batch_mix = []
            batch_speech = []
            batch_interf = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, [fs] * self.batch_size, [cut_duration] * self.batch_size, [mode] * self.batch_size):
                enroll, mix, speech, interf, fs, length, name = result
                batch_enroll.append(enroll)
                batch_mix.append(mix)
                batch_speech.append(speech)
                batch_interf.append(interf)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_enroll = torch.from_numpy(np.concatenate(batch_enroll, axis=0)).float() if mode != 'se' else None
            batch_mix = torch.from_numpy(np.concatenate(batch_mix, axis=0)).float()
            batch_speech = torch.from_numpy(np.concatenate(batch_speech, axis=0)).float()
            batch_interf = torch.from_numpy(np.concatenate(batch_interf, axis=0)).float() if mode != 'se' else None
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((mode, batch_enroll, batch_mix, batch_speech, batch_interf, batch_fs, lengths, names))
        event.set()
    
    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch + 1)
        event = threading.Event()
        worker = threading.Thread(target=self.data_iter_fn, args=(q, event))
        worker.start()
        while not event.is_set() or not q.empty():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                continue

    def __len__(self):
        """
        :return: number of batches in dataset
        """
        num_batches = int(self.samples_per_epoch // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        else:
            if self.rank < self.samples_per_epoch // self.batch_size - num_batches * self.world_size:
                return num_batches + 1
            else:
                return num_batches


class UseSimulationOnTheFlyDataLoadIter:
    def __init__(
        self,
        use_simulation_root: Union[str, Path],
        clean_json: Union[str, Path],
        noise_json: Union[str, Path],
        rir_json: Union[str, Path],
        simulation_config: Union[str, Path],
        batch_size: int = 1,
        cut_duration: Union[float, List[float]] = 5.0,
        num_workers: int = 1,
        prefetch: int = 0,
        samples_per_epoch: int = 1000,
        mode: str = "train",
        seed: int = 3407,
    ):
        self.is_train = mode == "train"
        self.use_simulation_root = Path(use_simulation_root).expanduser()
        self.clean_paths = self.load_json_list(clean_json)
        self.noise_paths = self.load_json_list(noise_json)
        self.rir_paths = self.load_json_list(rir_json)
        self.batch_size = batch_size
        self.cut_duration = cut_duration
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        self.seed = int(seed)

        with open(Path(simulation_config).expanduser(), "r") as f:
            self.simulation_config = yaml.safe_load(f)
        self.simulation_config.setdefault("stft_cfg", {})["sampling_rate"] = 16000

        self.random_select_and_order, self.apply_degradation_with_wind = self.import_use_simulation()

        if not self.clean_paths:
            raise ValueError(f"No clean paths found in {clean_json}")
        if not self.noise_paths:
            raise ValueError(f"No noise paths found in {noise_json}")
        if not self.rir_paths:
            raise ValueError(f"No RIR paths found in {rir_json}")

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    @staticmethod
    def load_json_list(path):
        with open(Path(path).expanduser(), "r") as f:
            values = json.load(f)
        if not isinstance(values, list):
            raise ValueError(f"{path} must contain a JSON list")
        return [str(Path(x).expanduser()) for x in values]

    def import_use_simulation(self):
        if not self.use_simulation_root.is_dir():
            raise FileNotFoundError(f"USE_simulation repo not found: {self.use_simulation_root}")
        root = str(self.use_simulation_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            simulate_degradation = importlib.import_module("simulate_degradation")
            random_select_and_order = getattr(simulate_degradation, "random_select_and_order")
            apply_degradation = getattr(simulate_degradation, "apply_degradation", None)
            apply_degradation_with_wind = getattr(
                simulate_degradation, "apply_degradation_with_wind", None
            )
        except Exception as exc:
            raise ImportError(
                "Failed to import USE_simulation degradation functions. "
                "Check use_simulation_root and environment dependencies."
            ) from exc
        if apply_degradation_with_wind is None and apply_degradation is None:
            raise ImportError("USE_simulation must define apply_degradation_with_wind or apply_degradation")
        if apply_degradation_with_wind is not None:
            return random_select_and_order, apply_degradation_with_wind

        def apply_degradation_wrapper(cfg, speech, noise, rir, wind_noise, degrad_cfgs, selected_degrads, seed=None):
            return apply_degradation(cfg, speech, noise, rir, degrad_cfgs, selected_degrads, seed=seed)

        return random_select_and_order, apply_degradation_wrapper

    def stable_seed(self, index):
        text = f"{self.seed}|{index}|{self.clean_paths[index % len(self.clean_paths)]}"
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)

    def load_wav(self, path, fs=16000):
        wav, _ = librosa.load(path, dtype=np.float32, sr=fs, mono=False)
        if wav.ndim == 1:
            wav = wav[None]
        else:
            wav = wav[:1, :]
        return wav

    def pad_or_cut_wav(self, wav, length, rng, offset=None):
        if wav.shape[-1] < length:
            wav = np.pad(wav, [(0, 0), (0, length - wav.shape[-1])], mode='wrap')
            return wav, None
        if offset is None:
            offset = int(rng.integers(0, wav.shape[-1] - length + 1))
        return wav[..., offset: offset + length], offset

    def normalize_src_tgt(self, src, tgt, low=0.1, high=0.99):
        max_tgt_value = np.max(np.abs(tgt)) + 1e-5
        max_src_value = np.max(np.abs(src)) + 1e-5
        max_value = max(max_tgt_value, max_src_value)
        threshold = high / max_value

        target_value = random.uniform(low, high)
        factor = min(target_value / max_tgt_value, threshold)
        return src * factor, tgt * factor

    def process_one_sample(self, sample_index):
        fs = 16000
        if self.is_train:
            item_seed = random.randint(0, 2**32 - 1)
            clean_path = random.choice(self.clean_paths)
        else:
            item_seed = self.stable_seed(sample_index)
            clean_path = self.clean_paths[sample_index % len(self.clean_paths)]

        rng = np.random.default_rng(item_seed)
        py_rng = random.Random(item_seed)
        noise_path = self.noise_paths[py_rng.randrange(len(self.noise_paths))]
        rir_path = self.rir_paths[py_rng.randrange(len(self.rir_paths))]

        speech = self.load_wav(clean_path, fs)
        noise = self.load_wav(noise_path, fs)
        rir = self.load_wav(rir_path, fs)

        cut_duration = self.cut_duration if not isinstance(self.cut_duration, list) else random.uniform(*self.cut_duration)
        length = int(cut_duration * fs)
        speech, _ = self.pad_or_cut_wav(speech, length, rng)

        degrad_cfgs, selected_degrads = self.random_select_and_order(self.simulation_config, seed=item_seed)
        clean, mix = self.apply_degradation_with_wind(
            self.simulation_config,
            speech,
            noise,
            rir,
            None,
            degrad_cfgs,
            selected_degrads,
            seed=item_seed,
        )

        if mix.shape[-1] > length:
            mix = mix[..., :length]
        elif mix.shape[-1] < length:
            mix = np.pad(mix, [(0, 0), (0, length - mix.shape[-1])], mode='wrap')

        if clean.shape[-1] > length:
            clean = clean[..., :length]
        elif clean.shape[-1] < length:
            clean = np.pad(clean, [(0, 0), (0, length - clean.shape[-1])], mode='wrap')

        mix, clean = self.normalize_src_tgt(mix, clean)
        name = Path(clean_path).stem
        return None, mix.astype(np.float32), clean.astype(np.float32), None, fs, length, name

    def make_batch(self, batch_mix, batch_speech, batch_fs, lengths, names):
        mix = torch.from_numpy(np.concatenate(batch_mix, axis=0)).float()
        speech = torch.from_numpy(np.concatenate(batch_speech, axis=0)).float()
        fs = torch.LongTensor(batch_fs)
        lengths = torch.LongTensor(lengths)
        if self.mode == "test":
            return ("se", None, mix, speech, fs, lengths, names)
        return ("se", None, mix, speech, None, fs, lengths, names)

    def data_iter_fn(self, q, event):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for batch_idx in range(len(self)):
            start_idx = (batch_idx * self.world_size + self.rank) * self.batch_size
            sample_indices = list(range(start_idx, start_idx + self.batch_size))
            batch_mix = []
            batch_speech = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, sample_indices):
                _, mix, speech, _, fs, length, name = result
                batch_mix.append(mix)
                batch_speech.append(speech)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            q.put(self.make_batch(batch_mix, batch_speech, batch_fs, lengths, names))
        event.set()

    def __iter__(self):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for batch_idx in range(len(self)):
            start_idx = (batch_idx * self.world_size + self.rank) * self.batch_size
            sample_indices = list(range(start_idx, start_idx + self.batch_size))
            batch_mix = []
            batch_speech = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, sample_indices):
                _, mix, speech, _, fs, length, name = result
                batch_mix.append(mix)
                batch_speech.append(speech)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            yield self.make_batch(batch_mix, batch_speech, batch_fs, lengths, names)

    def __len__(self):
        num_batches = int(self.samples_per_epoch // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        if self.rank < self.samples_per_epoch // self.batch_size - num_batches * self.world_size:
            return num_batches + 1
        return num_batches



class UseSimulationFixedPairDataLoadIter:
    def __init__(
        self,
        use_simulation_root: Union[str, Path],
        pair_manifest: Union[str, Path],
        batch_size: int = 1,
        cut_duration: Union[float, List[float], None] = None,
        num_workers: int = 1,
        prefetch: int = 0,
        samples_per_epoch: Union[int, None] = None,
        mode: str = "train",
        seed: int = 3407,
    ):
        self.is_train = mode == "train"
        self.use_simulation_root = Path(use_simulation_root).expanduser()
        self.pair_manifest = Path(pair_manifest).expanduser()
        self.batch_size = int(batch_size)
        self.cut_duration = cut_duration
        self.num_workers = int(num_workers)
        self.prefetch = int(prefetch)
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        self.seed = int(seed)

        self.dataset = self.import_fixed_pair_dataset()(
            pair_manifest=self.pair_manifest,
            wav_len=None,
            num_per_epoch=0,
            random_start=False,
            target_sample_rate=16000,
            mode="train" if self.is_train else "validation",
            normalize=True,
            seed=self.seed,
        )
        self.meta = list(getattr(self.dataset, "meta_selected", getattr(self.dataset, "meta", [])))
        if not self.meta:
            raise ValueError(f"No fixed pairs found in {self.pair_manifest}")
        if self.samples_per_epoch is None or int(self.samples_per_epoch) <= 0:
            self.samples_per_epoch = len(self.meta)
        else:
            self.samples_per_epoch = int(self.samples_per_epoch)

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def import_fixed_pair_dataset(self):
        if not self.use_simulation_root.is_dir():
            raise FileNotFoundError(f"USE_simulation repo not found: {self.use_simulation_root}")
        root = str(self.use_simulation_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            module = importlib.import_module("use_simulation_datasets")
            return getattr(module, "FixedPairDataset")
        except Exception as exc:
            raise ImportError(
                "Failed to import USE_simulation FixedPairDataset. "
                "Check use_simulation_root and environment dependencies."
            ) from exc

    def stable_seed(self, sample_index):
        item = self.meta[sample_index % len(self.meta)]
        text = f"{self.seed}|{self.mode}|{sample_index}|{item.get('id', '')}"
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)

    def target_length(self, rng):
        if self.cut_duration is None:
            return None
        if isinstance(self.cut_duration, list):
            duration = random.Random(int(rng.integers(0, 2**32 - 1))).uniform(*self.cut_duration)
        else:
            duration = float(self.cut_duration)
        return int(duration * 16000)

    @staticmethod
    def first_active_start(clean, length, threshold=0.01, min_active_ratio=0.05):
        max_start = clean.shape[-1] - length
        if max_start <= 0:
            return 0
        active = (np.abs(clean[0]) > threshold).astype(np.float32)
        prefix = np.concatenate(([0.0], np.cumsum(active, dtype=np.float64)))
        counts = prefix[length:] - prefix[:-length]
        valid = np.flatnonzero(counts >= length * min_active_ratio)
        return int(valid[0]) if valid.size else 0

    @staticmethod
    def pad_or_cut_pair(mix, clean, length, rng, random_start=True):
        orig_len = min(mix.shape[-1], clean.shape[-1])
        mix = mix[..., :orig_len]
        clean = clean[..., :orig_len]
        if length is None:
            return mix, clean, orig_len
        if orig_len < length:
            mix = np.pad(mix, [(0, 0), (0, length - orig_len)], mode="wrap")
            clean = np.pad(clean, [(0, 0), (0, length - orig_len)], mode="wrap")
            return mix, clean, length
        if random_start and orig_len > length:
            offset = int(rng.integers(0, orig_len - length + 1))
        else:
            offset = UseSimulationFixedPairDataLoadIter.first_active_start(clean, length)
        return mix[..., offset: offset + length], clean[..., offset: offset + length], length

    def process_one_sample(self, sample_index):
        if self.is_train:
            index = random.randrange(len(self.dataset))
            rng = np.random.default_rng()
            random_start = True
        else:
            index = sample_index % len(self.dataset)
            rng = np.random.default_rng(self.stable_seed(sample_index))
            random_start = False

        mix, clean, info = self.dataset[index]
        mix = np.asarray(mix, dtype=np.float32)[None]
        clean = np.asarray(clean, dtype=np.float32)[None]
        length = self.target_length(rng)
        mix, clean, length = self.pad_or_cut_pair(mix, clean, length, rng, random_start=random_start)
        name = info.get("id") or Path(info.get("noisy_path", f"item_{index}")).stem
        return None, mix.astype(np.float32), clean.astype(np.float32), None, 16000, length, name

    def make_batch(self, batch_mix, batch_speech, batch_fs, lengths, names):
        mix = torch.from_numpy(np.concatenate(batch_mix, axis=0)).float()
        speech = torch.from_numpy(np.concatenate(batch_speech, axis=0)).float()
        fs = torch.LongTensor(batch_fs)
        lengths = torch.LongTensor(lengths)
        if self.mode == "test":
            return ("se", None, mix, speech, fs, lengths, names)
        return ("se", None, mix, speech, None, fs, lengths, names)

    def data_iter_fn(self, q, event):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for batch_idx in range(len(self)):
            start_idx = (batch_idx * self.world_size + self.rank) * self.batch_size
            sample_indices = list(range(start_idx, start_idx + self.batch_size))
            batch_mix = []
            batch_speech = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, sample_indices):
                _, mix, speech, _, fs, length, name = result
                batch_mix.append(mix)
                batch_speech.append(speech)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            q.put(self.make_batch(batch_mix, batch_speech, batch_fs, lengths, names))
        event.set()

    def __iter__(self):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for batch_idx in range(len(self)):
            start_idx = (batch_idx * self.world_size + self.rank) * self.batch_size
            sample_indices = list(range(start_idx, start_idx + self.batch_size))
            batch_mix = []
            batch_speech = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, sample_indices):
                _, mix, speech, _, fs, length, name = result
                batch_mix.append(mix)
                batch_speech.append(speech)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            yield self.make_batch(batch_mix, batch_speech, batch_fs, lengths, names)

    def __len__(self):
        num_batches = int(self.samples_per_epoch // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        if self.rank < self.samples_per_epoch // self.batch_size - num_batches * self.world_size:
            return num_batches + 1
        return num_batches


class ValDataLoadIter:
    def __init__(
        self,
        data_enroll_dir: Union[str, Path],
        data_src_dir: Union[str, Path],
        data_tgt_dir: Union[str, Path],
        mode: str,
        enroll_duration: float = 5.0,
        batch_size: int = 1,
        num_workers: int = 1,
        prefetch: int = 0,
    ):
        self.is_train = False
        self.batch_size = batch_size
        self.mode = mode
        self.enroll_duration = enroll_duration

        if data_enroll_dir is not None:
            self.data_enroll_dir = Path(data_enroll_dir)
        else:
            self.data_enroll_dir = None
        self.data_src_dir = Path(data_src_dir)
        self.data_tgt_dir = Path(data_tgt_dir)

        self.wav_names = [p.name for p in self.data_src_dir.glob('*.flac')] + [p.name for p in self.data_src_dir.glob('*.wav')]
        self.num_workers = num_workers
        self.prefetch = prefetch
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    
    def load_wav(self, path, fs=None):
        wav, fs_ = sf.read(path, dtype='float32', always_2d=True)
        wav = wav[:, :1].T
        if fs is not None and fs != fs_:
            wav = librosa.resample(wav, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
            return wav, fs
        return wav, fs_
    
    def process_one_sample(self, name):
        assert self.batch_size == 1
        src, fs1 = self.load_wav(self.data_src_dir / name, fs=16000)
        tgt, fs2 = self.load_wav(self.data_tgt_dir / name, fs=16000)
        if self.data_enroll_dir is not None:
            enroll, fs3 = self.load_wav(self.data_enroll_dir / name, fs=16000)
        else:
            enroll, fs3 = None, None
        
        if enroll is not None:
            length = int(self.enroll_duration * 16000)
            if enroll.shape[-1] < length:
                enroll = np.pad(enroll, [(0, 0), (0, length - enroll.shape[-1])], mode='wrap')
            else:
                enroll = enroll[..., :length]
            enroll = enroll / (np.max(np.abs(enroll)) + 1e-5) * 0.99
        
        length = src.shape[-1]
        return enroll, src, tgt, 16000, length, Path(name).stem

    def data_iter_fn(self, q, event):
        wav_names = deepcopy(self.wav_names)
        assert self.batch_size == 1
        
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for sample_idx in range(self.rank * self.batch_size, len(wav_names), self.world_size * self.batch_size):
            batch_enroll = []
            batch_src = []
            batch_tgt = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, wav_names[sample_idx:sample_idx + self.batch_size]):
                enroll, src, tgt, fs, length, name = result
                batch_enroll.append(enroll)
                batch_src.append(src)
                batch_tgt.append(tgt)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_enroll = torch.from_numpy(np.concatenate(batch_enroll, axis=0)).float() if self.data_enroll_dir else None
            batch_src = torch.from_numpy(np.concatenate(batch_src, axis=0)).float()
            batch_tgt = torch.from_numpy(np.concatenate(batch_tgt, axis=0)).float()
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((self.mode, batch_enroll, batch_src, batch_tgt, batch_fs, lengths, names))
        event.set()

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch + 1)
        event = threading.Event()
        worker = threading.Thread(target=self.data_iter_fn, args=(q, event))
        worker.start()
        while not event.is_set() or not q.empty():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                continue

    def __len__(self):
        """
        :return: number of batches in dataset
        """
        num_batches = int(len(self.wav_names) // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        else:
            if self.rank < len(self.wav_names) // self.batch_size - num_batches * self.world_size:
                return num_batches + 1
            else:
                return num_batches


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_kwargs,
        val_kwargs,
        test_kwargs,
    ):
        super().__init__()
        self.train_kwargs = train_kwargs
        self.val_kwargs = val_kwargs
        self.test_kwargs = test_kwargs

    @staticmethod
    def build_dataset(kwargs, default_cls):
        dataset_type = kwargs.get('dataset_type')
        dataset_kwargs = {key: value for key, value in kwargs.items() if key != 'dataset_type'}
        if dataset_type in ('use_simulation', 'use_simulation_onthefly'):
            return UseSimulationOnTheFlyDataLoadIter(**dataset_kwargs)
        if dataset_type == 'use_simulation_fixed':
            return UseSimulationFixedPairDataLoadIter(**dataset_kwargs)
        return default_cls(**kwargs)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_iter = self.build_dataset(self.train_kwargs, TrainDataLoadIter)
            self.val_iter = self.build_dataset(self.val_kwargs, TrainDataLoadIter)
        if stage == 'test' or stage is None:
            self.test_iter = self.build_dataset(self.test_kwargs, ValDataLoadIter)

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    def test_dataloader(self):
        return self.test_iter
