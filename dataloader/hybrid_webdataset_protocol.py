import io
import json
import random
import tarfile
from collections import OrderedDict
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Sequence, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from .gap_webdataset import (
    FORBIDDEN_TEAMWORK_ARCHIVE,
    _normalize_src_tgt,
    _pad_or_crop,
    gap_worker_init_fn,
    make_sr_batch,
    stable_uint32,
)
from .simulation import simulate_data


DEFAULT_HYBRID_PROTOCOL_ROOT = Path(
    "/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active"
)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def _check_not_teamwork(path: Union[str, Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    forbidden = FORBIDDEN_TEAMWORK_ARCHIVE.resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise ValueError(f"Hybrid UniSE WebDataset protocol must not read Teamwork archive: {FORBIDDEN_TEAMWORK_ARCHIVE}")
    return resolved


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    path = _check_not_teamwork(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Union[str, Path], rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_item(row: dict) -> dict:
    shard_dir = row.get("_shard_dir") or row.get("role_dir")
    if not shard_dir:
        raise KeyError("manifest row must contain _shard_dir or role_dir")
    return {
        "_shard_dir": str(shard_dir),
        "shard": str(row["shard"]),
        "audio_member": str(row["audio_member"]),
        "json_member": row.get("json_member"),
        "key": str(row["key"]),
        "role": str(row.get("role", "")),
        "dataset": str(row.get("dataset", "")),
    }


def shard_key(row: dict) -> str:
    item = manifest_item(row)
    return f"{item['_shard_dir']}::{item['shard']}"


def group_rows_by_shard(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        if row.get("status", "done") != "done":
            continue
        item = manifest_item(row)
        tar_path = Path(item["_shard_dir"]) / item["shard"]
        if not tar_path.is_file():
            continue
        grouped.setdefault(shard_key(item), []).append(item)
    return grouped


class WorkerTarCache:
    def __init__(self, max_open: int = 8):
        self.max_open = int(max_open)
        self._handles: OrderedDict[str, tarfile.TarFile] = OrderedDict()

    def get(self, shard_dir: Union[str, Path], shard: str) -> tarfile.TarFile:
        path = str(_check_not_teamwork(Path(shard_dir) / shard))
        handle = self._handles.get(path)
        if handle is not None:
            self._handles.move_to_end(path)
            return handle
        handle = tarfile.open(path, "r:")
        self._handles[path] = handle
        self._handles.move_to_end(path)
        while len(self._handles) > self.max_open:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


class WebDatasetAudioReader:
    def __init__(self, target_sample_rate: int = 16000, tar_cache_size: int = 8):
        self.target_sample_rate = int(target_sample_rate)
        self.tar_cache_size = int(tar_cache_size)
        self._cache: Optional[WorkerTarCache] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def reset_worker_state(self):
        if self._cache is not None:
            self._cache.close()
        self._cache = None

    @property
    def cache(self) -> WorkerTarCache:
        if self._cache is None:
            self._cache = WorkerTarCache(self.tar_cache_size)
        return self._cache

    def decode(self, payload: bytes) -> np.ndarray:
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
        if sample_rate != self.target_sample_rate:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=self.target_sample_rate, res_type="soxr_hq")
        return np.asarray(wav, dtype=np.float32)

    def read(self, item: dict) -> np.ndarray:
        if "_tar_offset_data" in item and "_tar_size" in item:
            tar_path = _check_not_teamwork(Path(item["_shard_dir"]) / item["shard"])
            with tar_path.open("rb") as handle:
                handle.seek(int(item["_tar_offset_data"]))
                return self.decode(handle.read(int(item["_tar_size"])))
        tar = self.cache.get(item["_shard_dir"], item["shard"])
        member = tar.extractfile(item["audio_member"])
        if member is None:
            raise FileNotFoundError(f"{item['audio_member']} not found in {item['_shard_dir']}/{item['shard']}")
        return self.decode(member.read())


def load_simulation_config(simulation_config: Union[str, Path, dict]) -> dict:
    if isinstance(simulation_config, dict):
        return simulation_config
    import yaml

    with Path(simulation_config).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _duration_samples(cut_duration: Union[float, Sequence[float]], target_sample_rate: int, rng: random.Random) -> int:
    if isinstance(cut_duration, (list, tuple)):
        duration = rng.uniform(float(cut_duration[0]), float(cut_duration[1]))
    else:
        duration = float(cut_duration)
    return int(round(duration * int(target_sample_rate)))


def apply_recipe_degradation(
    clean_wav: np.ndarray,
    noise_wav: np.ndarray,
    rir_wav: np.ndarray,
    simulation_config: dict,
    seed: int,
    target_sample_rate: int,
    cut_duration: Union[float, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, int]:
    py_rng = random.Random(int(seed))
    rng = np.random.default_rng(int(seed))
    length = _duration_samples(cut_duration, target_sample_rate, py_rng)
    clean_crop, _ = _pad_or_crop(clean_wav, length, rng)
    noisy, clean_target, _ = simulate_data(
        mode="se",
        speech=clean_crop,
        interf=None,
        noise=noise_wav,
        rir=rir_wav,
        fs=int(target_sample_rate),
        config=simulation_config,
        py_rng=py_rng,
        rng=rng,
    )
    noisy, _ = _pad_or_crop(noisy, length, rng, offset=0)
    clean_target, _ = _pad_or_crop(clean_target, length, rng, offset=0)
    noisy, clean_target = _normalize_src_tgt(noisy, clean_target, py_rng)
    return noisy.astype(np.float32), clean_target.astype(np.float32), length


class HybridUniSEWebDatasetFixedRecipeDataset(Dataset):
    def __init__(
        self,
        recipe_manifest: Union[str, Path],
        simulation_config: Union[str, Path, dict],
        target_sample_rate: int = 16000,
        cut_duration: Union[float, Sequence[float]] = 5.0,
        tar_cache_size: int = 8,
        max_recipes: Optional[int] = None,
    ):
        self.recipe_manifest = _check_not_teamwork(recipe_manifest)
        rows = load_jsonl(self.recipe_manifest)
        self.recipes = rows[: int(max_recipes)] if max_recipes is not None else rows
        if not self.recipes:
            raise ValueError(f"No fixed recipes found in {self.recipe_manifest}")
        self.simulation_config = load_simulation_config(simulation_config)
        self.target_sample_rate = int(target_sample_rate)
        self.cut_duration = cut_duration
        self.reader = WebDatasetAudioReader(target_sample_rate=target_sample_rate, tar_cache_size=tar_cache_size)

    def __len__(self):
        return len(self.recipes)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["reader"] = self.reader
        return state

    def reset_worker_state(self):
        self.reader.reset_worker_state()

    def __getitem__(self, index: int) -> dict:
        recipe = self.recipes[index]
        seed = int(recipe["seed"])
        target_sample_rate = int(recipe.get("target_sample_rate", self.target_sample_rate))
        cut_duration = recipe.get("cut_duration", self.cut_duration)
        clean_wav = self.reader.read(recipe["clean"])
        noise_wav = self.reader.read(recipe["noise"])
        rir_wav = self.reader.read(recipe["rir"])
        noisy, clean, length = apply_recipe_degradation(
            clean_wav,
            noise_wav,
            rir_wav,
            self.simulation_config,
            seed,
            target_sample_rate,
            cut_duration,
        )
        return {
            "noisy": noisy,
            "clean": clean,
            "sample_rate": target_sample_rate,
            "length": length,
            "uid": recipe["uid"],
            "info": recipe,
        }


def hybrid_recipe_collate(batch: Sequence[dict], batch_format: str = "tuple", test: bool = False):
    mix = torch.from_numpy(np.concatenate([item["noisy"] for item in batch], axis=0)).float()
    clean = torch.from_numpy(np.concatenate([item["clean"] for item in batch], axis=0)).float()
    fs = torch.LongTensor([int(item["sample_rate"]) for item in batch])
    lengths = torch.LongTensor([int(item["length"]) for item in batch])
    names = [str(item["uid"]) for item in batch]
    output = make_sr_batch("se", mix, clean, fs, lengths, names, batch_format=batch_format, test=test)
    if isinstance(output, dict):
        output["info"] = [item["info"] for item in batch]
    return output


class HybridUniSEWebDatasetFixedRecipeDataLoadIter:
    def __init__(
        self,
        recipe_manifest: Union[str, Path],
        simulation_config: Union[str, Path, dict],
        batch_size: int = 1,
        cut_duration: Union[float, Sequence[float]] = 5.0,
        target_sample_rate: int = 16000,
        num_workers: int = 1,
        prefetch: int = 2,
        batch_format: str = "tuple",
        tar_cache_size: int = 8,
        max_recipes: Optional[int] = None,
        mode: str = "validation",
    ):
        self.dataset = HybridUniSEWebDatasetFixedRecipeDataset(
            recipe_manifest=recipe_manifest,
            simulation_config=simulation_config,
            target_sample_rate=target_sample_rate,
            cut_duration=cut_duration,
            tar_cache_size=tar_cache_size,
            max_recipes=max_recipes,
        )
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch = int(prefetch)
        self.batch_format = batch_format
        self.mode = mode
        self.is_train = False

    def __iter__(self):
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = max(1, self.prefetch)
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: hybrid_recipe_collate(
                batch,
                batch_format=self.batch_format,
                test=self.mode == "test",
            ),
            worker_init_fn=gap_worker_init_fn,
            **kwargs,
        )
        return iter(loader)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class HybridUniSEWebDatasetStreamDataset(Dataset):
    """Compatibility train fallback backed by the protocol shard lists.

    This is intentionally bounded and deterministic. It uses the protocol train
    shard rows as pools and samples from them by seed; a future pass can replace
    this with true sequential shard streaming without changing the dataset_type.
    """

    def __init__(
        self,
        split_root: Union[str, Path],
        simulation_config: Union[str, Path, dict],
        target_sample_rate: int = 16000,
        cut_duration: Union[float, Sequence[float]] = 5.0,
        samples_per_epoch: int = 100000,
        shard_shuffle_seed: int = 3407,
        tar_cache_size: int = 8,
        max_rows_per_role: Optional[int] = None,
        **_unused,
    ):
        self.split_root = _check_not_teamwork(split_root)
        self.simulation_config = load_simulation_config(simulation_config)
        self.target_sample_rate = int(target_sample_rate)
        self.cut_duration = cut_duration
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(shard_shuffle_seed)
        self.epoch = 0
        self.reader = WebDatasetAudioReader(target_sample_rate=target_sample_rate, tar_cache_size=tar_cache_size)
        self.clean = self._load_rows("train/clean_shards.jsonl", max_rows_per_role)
        self.noise = self._load_rows("train/noise_shards.jsonl", max_rows_per_role)
        self.rir = self._load_rows("train/rir_shards.jsonl", max_rows_per_role)
        for role, rows in (("clean", self.clean), ("noise", self.noise), ("rir", self.rir)):
            if not rows:
                raise ValueError(f"No {role} rows found under {self.split_root}")

    def _load_rows(self, rel_path: str, max_rows: Optional[int]) -> List[dict]:
        shard_records = load_jsonl(self.split_root / rel_path)
        rows: List[dict] = []
        rng = random.Random(self.seed)
        rng.shuffle(shard_records)
        for record in shard_records:
            rows.extend(record.get("samples", []))
            if max_rows is not None and len(rows) >= int(max_rows):
                return rows[: int(max_rows)]
        return rows

    def __len__(self):
        return self.samples_per_epoch

    def reset_worker_state(self):
        self.reader.reset_worker_state()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        seed = stable_uint32(self.seed, self.epoch, worker_id, index)
        py_rng = random.Random(seed)
        clean_item = self.clean[py_rng.randrange(len(self.clean))]
        noise_item = self.noise[py_rng.randrange(len(self.noise))]
        rir_item = self.rir[py_rng.randrange(len(self.rir))]
        noisy, clean, length = apply_recipe_degradation(
            self.reader.read(clean_item),
            self.reader.read(noise_item),
            self.reader.read(rir_item),
            self.simulation_config,
            seed,
            self.target_sample_rate,
            self.cut_duration,
        )
        uid = str(clean_item["key"])
        return {
            "noisy": noisy,
            "clean": clean,
            "sample_rate": self.target_sample_rate,
            "length": length,
            "uid": uid,
            "info": {"uid": uid, "seed": seed, "clean": clean_item, "noise": noise_item, "rir": rir_item},
        }


class HybridUniSEWebDatasetSequentialStreamDataset(IterableDataset):
    """Sequential shard stream with bounded clean/noise/RIR shuffle buffers."""

    def __init__(
        self,
        split_root: Union[str, Path],
        simulation_config: Union[str, Path, dict],
        target_sample_rate: int = 16000,
        cut_duration: Union[float, Sequence[float]] = 5.0,
        samples_per_epoch: int = 100000,
        clean_shuffle_buffer: int = 4096,
        noise_buffer_size: int = 2048,
        rir_buffer_size: int = 2048,
        shard_shuffle_seed: int = 3407,
        tar_cache_size: int = 8,
        max_shards_per_role: Optional[int] = None,
        max_rows_per_role: Optional[int] = None,
        **_unused,
    ):
        self.split_root = _check_not_teamwork(split_root)
        self.simulation_config = load_simulation_config(simulation_config)
        self.target_sample_rate = int(target_sample_rate)
        self.cut_duration = cut_duration
        self.samples_per_epoch = int(samples_per_epoch)
        self.clean_shuffle_buffer = max(1, int(clean_shuffle_buffer))
        self.noise_buffer_size = max(1, int(noise_buffer_size))
        self.rir_buffer_size = max(1, int(rir_buffer_size))
        self.seed = int(shard_shuffle_seed)
        self.epoch = 0
        self.max_rows_per_role = int(max_rows_per_role) if max_rows_per_role is not None else None
        self.reader = WebDatasetAudioReader(target_sample_rate=target_sample_rate, tar_cache_size=tar_cache_size)
        self.clean_shards = self._load_shard_records("train/clean_shards.jsonl", max_shards_per_role)
        self.noise_shards = self._load_shard_records("train/noise_shards.jsonl", max_shards_per_role)
        self.rir_shards = self._load_shard_records("train/rir_shards.jsonl", max_shards_per_role)
        for role, shards in (("clean", self.clean_shards), ("noise", self.noise_shards), ("rir", self.rir_shards)):
            if not shards:
                raise ValueError(f"No {role} shard records found under {self.split_root}")

    def _load_shard_records(self, rel_path: str, max_shards: Optional[int]) -> List[dict]:
        records = load_jsonl(self.split_root / rel_path)
        if max_shards is not None:
            records = records[: int(max_shards)]
        return records

    def __len__(self):
        return self.samples_per_epoch

    def reset_worker_state(self):
        self.reader.reset_worker_state()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _worker_context(self):
        worker = get_worker_info()
        local_worker_id = int(worker.id) if worker is not None else 0
        local_num_workers = int(worker.num_workers) if worker is not None else 1
        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
        else:
            rank = 0
            world_size = 1
        global_worker_id = rank * local_num_workers + local_worker_id
        global_num_workers = world_size * local_num_workers
        return global_worker_id, global_num_workers

    def _ordered_shards(self, shards: Sequence[dict], role: str, worker_id: int, num_workers: int) -> List[dict]:
        ordered = list(shards)
        random.Random(stable_uint32(self.seed, self.epoch, role)).shuffle(ordered)
        return ordered[worker_id::num_workers] or ordered

    def _iter_tar_items(self, shard: dict):
        tar_path = _check_not_teamwork(Path(shard["_shard_dir"]) / shard["shard"])
        with tarfile.open(tar_path, "r:") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                member_path = PurePosixPath(member.name)
                if member_path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                yield {
                    "_shard_dir": str(shard["_shard_dir"]),
                    "shard": str(shard["shard"]),
                    "audio_member": member.name,
                    "json_member": str(member_path.with_suffix(".json")),
                    "key": member_path.stem,
                    "role": str(shard.get("role", "")),
                    "dataset": str(shard.get("dataset", "")),
                    "_tar_offset_data": int(member.offset_data),
                    "_tar_size": int(member.size),
                }

    def _iter_items(self, shards: Sequence[dict], role: str, worker_id: int, num_workers: int):
        emitted = 0
        for shard in self._ordered_shards(shards, role, worker_id, num_workers):
            samples = shard.get("samples")
            item_iter = iter(samples) if samples else self._iter_tar_items(shard)
            for item in item_iter:
                yield item
                emitted += 1
                if self.max_rows_per_role is not None and emitted >= self.max_rows_per_role:
                    return

    def _cycled_items(self, shards: Sequence[dict], role: str, worker_id: int, num_workers: int):
        while True:
            yielded = False
            for item in self._iter_items(shards, role, worker_id, num_workers):
                yielded = True
                yield item
            if not yielded:
                raise ValueError(f"No {role} samples available for worker {worker_id}/{num_workers}")

    def _fill_item_buffer(self, buffer: list, item_iter, target_size: int) -> None:
        while len(buffer) < target_size:
            buffer.append(next(item_iter))

    def _pop_random_item(self, buffer: list, item_iter, target_size: int, rng: random.Random):
        self._fill_item_buffer(buffer, item_iter, target_size)
        index = rng.randrange(len(buffer))
        item = buffer.pop(index)
        self._fill_item_buffer(buffer, item_iter, target_size)
        return item

    def __iter__(self):
        worker_id, num_workers = self._worker_context()
        base_quota = self.samples_per_epoch // num_workers
        worker_quota = base_quota + (1 if worker_id < self.samples_per_epoch % num_workers else 0)
        rng = random.Random(stable_uint32(self.seed, self.epoch, worker_id, "sequential-stream"))
        clean_iter = self._cycled_items(self.clean_shards, "clean", worker_id, num_workers)
        noise_iter = self._cycled_items(self.noise_shards, "noise", worker_id, num_workers)
        rir_iter = self._cycled_items(self.rir_shards, "rir", worker_id, num_workers)
        clean_buffer = []
        noise_buffer = []
        rir_buffer = []
        self._fill_item_buffer(clean_buffer, clean_iter, self.clean_shuffle_buffer)
        emitted = 0
        while emitted < worker_quota:
            clean_item = self._pop_random_item(
                clean_buffer,
                clean_iter,
                self.clean_shuffle_buffer,
                rng,
            )
            noise_item = self._pop_random_item(
                noise_buffer,
                noise_iter,
                self.noise_buffer_size,
                rng,
            )
            rir_item = self._pop_random_item(
                rir_buffer,
                rir_iter,
                self.rir_buffer_size,
                rng,
            )
            seed = stable_uint32(self.seed, self.epoch, worker_id, emitted, clean_item["key"], noise_item["key"], rir_item["key"])
            noisy, clean, length = apply_recipe_degradation(
                self.reader.read(clean_item),
                self.reader.read(noise_item),
                self.reader.read(rir_item),
                self.simulation_config,
                seed,
                self.target_sample_rate,
                self.cut_duration,
            )
            uid = str(clean_item["key"])
            emitted += 1
            yield {
                "noisy": noisy,
                "clean": clean,
                "sample_rate": self.target_sample_rate,
                "length": length,
                "uid": uid,
                "info": {"uid": uid, "seed": seed, "clean": clean_item, "noise": noise_item, "rir": rir_item},
            }


class HybridUniSEWebDatasetStreamDataLoadIter:
    def __init__(
        self,
        split_root: Union[str, Path],
        simulation_config: Union[str, Path, dict],
        batch_size: int = 1,
        cut_duration: Union[float, Sequence[float]] = 5.0,
        samples_per_epoch: int = 100000,
        target_sample_rate: int = 16000,
        num_workers: int = 1,
        prefetch: int = 2,
        batch_format: str = "tuple",
        shard_shuffle_seed: int = 3407,
        tar_cache_size: int = 8,
        persistent_workers: bool = False,
        max_rows_per_role: Optional[int] = None,
        max_shards_per_role: Optional[int] = None,
        streaming_mode: str = "sequential",
        **buffer_knobs,
    ):
        if streaming_mode not in {"sequential", "indexed"}:
            raise ValueError("streaming_mode must be 'sequential' or 'indexed'")
        dataset_cls = (
            HybridUniSEWebDatasetSequentialStreamDataset
            if streaming_mode == "sequential"
            else HybridUniSEWebDatasetStreamDataset
        )
        dataset_kwargs = dict(
            split_root=split_root,
            simulation_config=simulation_config,
            target_sample_rate=target_sample_rate,
            cut_duration=cut_duration,
            samples_per_epoch=samples_per_epoch,
            shard_shuffle_seed=shard_shuffle_seed,
            tar_cache_size=tar_cache_size,
            max_rows_per_role=max_rows_per_role,
            **buffer_knobs,
        )
        if streaming_mode == "sequential":
            dataset_kwargs["max_shards_per_role"] = max_shards_per_role
        self.dataset = dataset_cls(**dataset_kwargs)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch = int(prefetch)
        self.batch_format = batch_format
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.epoch = 0
        self.is_train = True
        self.streaming_mode = streaming_mode

    def __iter__(self):
        self.dataset.set_epoch(self.epoch)
        self.epoch += 1
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = max(1, self.prefetch)
            kwargs["persistent_workers"] = self.persistent_workers
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=lambda batch: hybrid_recipe_collate(batch, batch_format=self.batch_format, test=False),
            worker_init_fn=gap_worker_init_fn,
            **kwargs,
        )
        return iter(loader)

    def __len__(self):
        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
        else:
            world_size = 1
        return len(self.dataset) // (world_size * self.batch_size)
