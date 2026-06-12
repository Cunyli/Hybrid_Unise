import io
import json
import random
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from .simulation import simulate_data


GAP_WEBDATASET_ACTIVE_ROOT = Path(
    "/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active"
)
FORBIDDEN_TEAMWORK_ARCHIVE = Path(
    "/m/teamwork/t412_symptosonic/lil14/gap_pretrain/webdataset_archive"
)


def stable_uint32(*parts):
    text = "|".join(str(part) for part in parts)
    return int(__import__("hashlib").sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def make_sr_batch(
    mode,
    degraded_wav,
    clean_wav,
    sample_rate,
    length,
    utterance_id,
    batch_format="tuple",
    test=False,
    source_path=None,
    clean_path=None,
):
    if batch_format == "dict":
        return {
            "mode": mode,
            "degraded_wav": degraded_wav,
            "clean_wav": clean_wav,
            "sample_rate": sample_rate,
            "length": length,
            "utterance_id": utterance_id,
            "source_path": source_path,
            "clean_path": clean_path,
        }
    if test:
        return (mode, None, degraded_wav, clean_wav, sample_rate, length, utterance_id)
    return (mode, None, degraded_wav, clean_wav, None, sample_rate, length, utterance_id)


@dataclass(frozen=True)
class GapManifestRow:
    key: str
    role: str
    dataset: str
    role_dir: str
    shard: str
    audio_member: str
    json_member: Optional[str]

    @property
    def shard_path(self) -> Path:
        return Path(self.role_dir) / self.shard


def _resolve_active_root(active_root: Union[str, Path]) -> Path:
    root = Path(active_root).expanduser().resolve()
    forbidden = FORBIDDEN_TEAMWORK_ARCHIVE.resolve()
    if root == forbidden or forbidden in root.parents:
        raise ValueError(
            "Gap WebDataset loader must not read the removed Teamwork archive: "
            f"{FORBIDDEN_TEAMWORK_ARCHIVE}"
        )
    if not root.exists():
        raise FileNotFoundError(f"Gap WebDataset active root does not exist: {root}")
    return root


def discover_gap_role_dirs(
    active_root: Union[str, Path] = GAP_WEBDATASET_ACTIVE_ROOT,
    roles: Sequence[str] = ("clean", "noise", "rir"),
) -> Tuple[Dict[str, List[Path]], List[Path]]:
    """Return verified role dirs and unverified MLS_HQ_en role dirs under active_root."""
    root = _resolve_active_root(active_root)
    role_set = set(roles)
    verified = {role: [] for role in roles}
    skipped_mls_hq_en = []

    component_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: (path.name == "v1", path.name),
    )
    for component_dir in component_dirs:
        for role in roles:
            role_dir = component_dir / role
            if not role_dir.is_dir() or role_dir.name not in role_set:
                continue
            if (role_dir / "_verified.ok").is_file():
                manifest = role_dir / "manifest.jsonl"
                shards = role_dir / "shards.json"
                if manifest.is_file() and shards.is_file() and next(role_dir.glob("*.tar"), None) is not None:
                    verified[role].append(role_dir)
                continue
            if component_dir.name.startswith("v1_mls_hq_en_clean_chunk"):
                skipped_mls_hq_en.append(role_dir)

    return verified, skipped_mls_hq_en


def load_gap_manifest_rows(
    active_root: Union[str, Path] = GAP_WEBDATASET_ACTIVE_ROOT,
    roles: Sequence[str] = ("clean", "noise", "rir"),
    role_dirs: Optional[Dict[str, Sequence[Union[str, Path]]]] = None,
    max_rows_per_role: Optional[int] = None,
) -> Tuple[Dict[str, List[GapManifestRow]], List[Path]]:
    if role_dirs is None:
        discovered, skipped_mls_hq_en = discover_gap_role_dirs(active_root, roles=roles)
    else:
        skipped_mls_hq_en = []
        discovered = {
            role: [Path(path).expanduser().resolve() for path in paths]
            for role, paths in role_dirs.items()
        }

    rows_by_role = {role: [] for role in roles}
    for role in roles:
        for role_dir in discovered.get(role, []):
            if max_rows_per_role is not None and len(rows_by_role[role]) >= max_rows_per_role:
                break
            if not (role_dir / "_verified.ok").is_file():
                continue
            manifest_path = role_dir / "manifest.jsonl"
            with manifest_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if max_rows_per_role is not None and len(rows_by_role[role]) >= max_rows_per_role:
                        break
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("status") != "done":
                        continue
                    row_role = data.get("role", role)
                    if row_role != role:
                        continue
                    shard = str(data.get("shard", ""))
                    if not shard or not (role_dir / shard).is_file():
                        continue
                    try:
                        rows_by_role[role].append(
                            GapManifestRow(
                                key=str(data["key"]),
                                role=role,
                                dataset=str(data.get("dataset", role_dir.parent.name)),
                                role_dir=str(role_dir),
                                shard=shard,
                                audio_member=str(data["audio_member"]),
                                json_member=data.get("json_member"),
                            )
                        )
                    except KeyError as exc:
                        raise KeyError(f"Missing {exc} in {manifest_path}:{line_no}") from exc

    return rows_by_role, skipped_mls_hq_en


class WorkerTarLRU:
    def __init__(self, max_open: int = 8):
        self.max_open = int(max_open)
        self._handles: OrderedDict[str, tarfile.TarFile] = OrderedDict()

    def get(self, path: Union[str, Path]) -> tarfile.TarFile:
        key = str(path)
        handle = self._handles.get(key)
        if handle is not None:
            self._handles.move_to_end(key)
            return handle
        handle = tarfile.open(key, "r:")
        self._handles[key] = handle
        self._handles.move_to_end(key)
        while len(self._handles) > self.max_open:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


class GapWebDatasetPool(Dataset):
    def __init__(
        self,
        rows: Sequence[GapManifestRow],
        target_sample_rate: int = 16000,
        tar_cache_size: int = 8,
    ):
        self.rows = list(rows)
        self.target_sample_rate = int(target_sample_rate)
        self.tar_cache_size = int(tar_cache_size)
        self._tar_cache: Optional[WorkerTarLRU] = None
        if not self.rows:
            raise ValueError("GapWebDatasetPool requires at least one manifest row")

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar_cache"] = None
        return state

    def reset_worker_state(self):
        if self._tar_cache is not None:
            self._tar_cache.close()
        self._tar_cache = None

    @property
    def tar_cache(self) -> WorkerTarLRU:
        if self._tar_cache is None:
            self._tar_cache = WorkerTarLRU(self.tar_cache_size)
        return self._tar_cache

    def _read_audio_bytes(self, row: GapManifestRow) -> bytes:
        tar = self.tar_cache.get(row.shard_path)
        member = tar.extractfile(row.audio_member)
        if member is None:
            raise FileNotFoundError(f"{row.audio_member} not found in {row.shard_path}")
        return member.read()

    def _decode_audio(self, payload: bytes) -> Tuple[np.ndarray, int]:
        try:
            wav, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            wav = wav[:, :1].T
        except Exception:
            try:
                import torchaudio
            except Exception as exc:
                raise RuntimeError("soundfile decode failed and torchaudio is unavailable") from exc
            waveform, sample_rate = torchaudio.load(io.BytesIO(payload))
            wav = waveform[:1].numpy().astype(np.float32, copy=False)
        if self.target_sample_rate and sample_rate != self.target_sample_rate:
            wav = librosa.resample(
                wav,
                orig_sr=sample_rate,
                target_sr=self.target_sample_rate,
                res_type="soxr_hq",
            )
            sample_rate = self.target_sample_rate
        return np.asarray(wav, dtype=np.float32), int(sample_rate)

    def __getitem__(self, index: int):
        row = self.rows[index % len(self.rows)]
        wav, sample_rate = self._decode_audio(self._read_audio_bytes(row))
        return {
            "uid": row.key,
            "role": row.role,
            "dataset": row.dataset,
            "wav": wav,
            "sample_rate": sample_rate,
            "row": row,
        }


def _pad_or_crop(wav: np.ndarray, length: int, rng: np.random.Generator, offset: Optional[int] = None):
    if wav.shape[-1] < length:
        if wav.shape[-1] == 0:
            raise ValueError("Cannot pad empty waveform")
        return np.pad(wav, [(0, 0), (0, length - wav.shape[-1])], mode="wrap"), None
    if wav.shape[-1] == length:
        return wav, 0
    if offset is None:
        offset = int(rng.integers(0, wav.shape[-1] - length + 1))
    return wav[..., offset : offset + length], offset


def _normalize_src_tgt(src: np.ndarray, tgt: np.ndarray, py_rng: random.Random, low=0.1, high=0.99):
    max_tgt_value = np.max(np.abs(tgt)) + 1e-5
    max_src_value = np.max(np.abs(src)) + 1e-5
    max_value = max(max_tgt_value, max_src_value)
    threshold = high / max_value
    target_value = py_rng.uniform(low, high)
    factor = min(target_value / max_tgt_value, threshold)
    return src * factor, tgt * factor


class GapWebDatasetDegradationDataset(Dataset):
    def __init__(
        self,
        active_root: Union[str, Path] = GAP_WEBDATASET_ACTIVE_ROOT,
        simulation_config: Optional[Union[str, Path, dict]] = None,
        target_sample_rate: int = 16000,
        cut_duration: Union[float, Sequence[float]] = 5.0,
        samples_per_epoch: int = 100000,
        mode: str = "train",
        seed: int = 3407,
        tar_cache_size: int = 8,
        max_rows_per_role: Optional[int] = None,
        role_dirs: Optional[Dict[str, Sequence[Union[str, Path]]]] = None,
    ):
        import yaml

        self.active_root = _resolve_active_root(active_root)
        self.target_sample_rate = int(target_sample_rate)
        self.cut_duration = cut_duration
        self.samples_per_epoch = int(samples_per_epoch)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0

        rows_by_role, self.skipped_unverified_mls_hq_en = load_gap_manifest_rows(
            self.active_root,
            role_dirs=role_dirs,
            max_rows_per_role=max_rows_per_role,
        )
        self.clean = GapWebDatasetPool(rows_by_role["clean"], target_sample_rate, tar_cache_size)
        self.noise = GapWebDatasetPool(rows_by_role["noise"], target_sample_rate, tar_cache_size)
        self.rir = GapWebDatasetPool(rows_by_role["rir"], target_sample_rate, tar_cache_size)

        if simulation_config is None:
            self.simulation_config = {
                "se_interference": {"sir": [2.0, 20.0]},
                "tse_interference": {"sir": [-5.0, 5.0]},
                "reverberation": {"prob": 0.3},
                "noise": {"prob": 0.8, "snr": [-5.0, 20.0]},
                "bandwidth_limitation": {"prob": 0.0, "fs_new": [self.target_sample_rate], "res_type": "soxr_hq"},
                "clipping": {"prob": 0.0, "min_quantile": [0.0, 0.1], "max_quantile": [0.9, 1.0]},
                "packet_loss": {
                    "prob": 0.0,
                    "packet_duration_ms": 20,
                    "packet_loss_rate": [0.05, 0.25],
                    "max_continuous_packet_loss": 10,
                },
            }
        elif isinstance(simulation_config, dict):
            self.simulation_config = simulation_config
        else:
            with Path(simulation_config).expanduser().open("r", encoding="utf-8") as f:
                self.simulation_config = yaml.safe_load(f)

    def __len__(self):
        return self.samples_per_epoch

    def __getstate__(self):
        state = self.__dict__.copy()
        for pool_name in ("clean", "noise", "rir"):
            state[pool_name] = getattr(self, pool_name)
        return state

    def reset_worker_state(self):
        self.clean.reset_worker_state()
        self.noise.reset_worker_state()
        self.rir.reset_worker_state()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _item_seed(self, index: int) -> int:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        epoch = self.epoch if self.mode == "train" else 0
        return stable_uint32(self.seed, "gap-webdataset", self.mode, epoch, worker_id, index)

    def _duration_samples(self, py_rng: random.Random) -> int:
        if isinstance(self.cut_duration, (list, tuple)):
            duration = py_rng.uniform(float(self.cut_duration[0]), float(self.cut_duration[1]))
        else:
            duration = float(self.cut_duration)
        return int(round(duration * self.target_sample_rate))

    def _select_index(self, pool: GapWebDatasetPool, py_rng: random.Random, index: int) -> int:
        if self.mode == "train":
            return py_rng.randrange(len(pool))
        return index % len(pool)

    def __getitem__(self, index: int):
        item_seed = self._item_seed(index)
        py_rng = random.Random(item_seed)
        rng = np.random.default_rng(item_seed)

        clean_sample = self.clean[self._select_index(self.clean, py_rng, index)]
        noise_sample = self.noise[self._select_index(self.noise, py_rng, index)]
        rir_sample = self.rir[self._select_index(self.rir, py_rng, index)]

        length = self._duration_samples(py_rng)
        clean_wav, _ = _pad_or_crop(clean_sample["wav"], length, rng)
        noisy, clean_target, _ = simulate_data(
            mode="se",
            speech=clean_wav,
            interf=None,
            noise=noise_sample["wav"],
            rir=rir_sample["wav"],
            fs=self.target_sample_rate,
            config=self.simulation_config,
            py_rng=py_rng,
            rng=rng,
        )
        noisy, _ = _pad_or_crop(noisy, length, rng, offset=0)
        clean_target, _ = _pad_or_crop(clean_target, length, rng, offset=0)
        noisy, clean_target = _normalize_src_tgt(noisy, clean_target, py_rng)

        info = {
            "uid": clean_sample["uid"],
            "clean_uid": clean_sample["uid"],
            "noise_uid": noise_sample["uid"],
            "rir_uid": rir_sample["uid"],
            "clean_dataset": clean_sample["dataset"],
            "noise_dataset": noise_sample["dataset"],
            "rir_dataset": rir_sample["dataset"],
            "sample_rate": self.target_sample_rate,
            "seed": item_seed,
        }
        return {
            "noisy": noisy.astype(np.float32),
            "clean": clean_target.astype(np.float32),
            "sample_rate": self.target_sample_rate,
            "length": length,
            "uid": clean_sample["uid"],
            "info": info,
        }


def gap_degradation_collate(batch, batch_format: str = "dict", test: bool = False):
    noisy = torch.from_numpy(np.concatenate([item["noisy"] for item in batch], axis=0)).float()
    clean = torch.from_numpy(np.concatenate([item["clean"] for item in batch], axis=0)).float()
    sample_rate = torch.LongTensor([item["sample_rate"] for item in batch])
    length = torch.LongTensor([item["length"] for item in batch])
    uid = [item["uid"] for item in batch]
    info = [item["info"] for item in batch]

    output = make_sr_batch(
        "se",
        noisy,
        clean,
        sample_rate,
        length,
        uid,
        batch_format=batch_format,
        test=test,
    )
    if isinstance(output, dict):
        output["info"] = info
    return output


def gap_worker_init_fn(_worker_id):
    worker = get_worker_info()
    if worker is not None and hasattr(worker.dataset, "reset_worker_state"):
        worker.dataset.reset_worker_state()


class _DistributedRangeSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return (len(self.dataset) + self.world_size - 1 - self.rank) // self.world_size


class GapWebDatasetDataLoadIter:
    def __init__(
        self,
        active_root: Union[str, Path] = GAP_WEBDATASET_ACTIVE_ROOT,
        simulation_config: Optional[Union[str, Path, dict]] = None,
        batch_size: int = 1,
        cut_duration: Union[float, Sequence[float]] = 5.0,
        num_workers: int = 1,
        prefetch: int = 2,
        samples_per_epoch: int = 100000,
        mode: str = "train",
        seed: int = 3407,
        batch_format: str = "tuple",
        target_sample_rate: int = 16000,
        tar_cache_size: int = 8,
        persistent_workers: bool = False,
        max_rows_per_role: Optional[int] = None,
        role_dirs: Optional[Dict[str, Sequence[Union[str, Path]]]] = None,
    ):
        self.dataset = GapWebDatasetDegradationDataset(
            active_root=active_root,
            simulation_config=simulation_config,
            target_sample_rate=target_sample_rate,
            cut_duration=cut_duration,
            samples_per_epoch=samples_per_epoch,
            mode=mode,
            seed=seed,
            tar_cache_size=tar_cache_size,
            max_rows_per_role=max_rows_per_role,
            role_dirs=role_dirs,
        )
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch = int(prefetch)
        self.mode = mode
        self.batch_format = batch_format
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.epoch = 0
        self.is_train = mode == "train"

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def __iter__(self):
        self.dataset.set_epoch(self.epoch)
        if self.is_train:
            self.epoch += 1
        sampler = None
        if self.world_size > 1:
            sampler = _DistributedRangeSampler(self.dataset, self.rank, self.world_size)
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = max(1, self.prefetch)
            kwargs["persistent_workers"] = self.persistent_workers
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: gap_degradation_collate(
                batch,
                batch_format=self.batch_format,
                test=self.mode == "test",
            ),
            worker_init_fn=gap_worker_init_fn,
            **kwargs,
        )
        return iter(loader)

    def __len__(self):
        sample_count = len(self.dataset)
        if self.world_size > 1:
            sample_count = (sample_count + self.world_size - 1 - self.rank) // self.world_size
        return sample_count // self.batch_size
