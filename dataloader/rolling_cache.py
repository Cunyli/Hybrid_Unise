import io
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import threading
import time
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import importlib
import librosa
import numpy as np
import soundfile as sf
import yaml


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
DEFAULT_CACHE_WAIT_SECONDS = 3600
DEFAULT_CLEAN_STATUSES = {
    "available_existing",
    "downloaded_archive",
    "downloaded_archive_verified",
    "downloaded_archive_verified_cv25_replacement",
}
DEFAULT_NOISE_STATUSES = {
    "available_existing",
    "downloaded_archive",
}
DEFAULT_RIR_STATUSES = {
    "available_existing",
    "available_existing_partial_against_official_archive",
    "available_downloaded_archive",
}


_GLOBAL_RNG_LOCK = threading.Lock()


def stable_uint32(*parts) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


@contextmanager
def preserve_global_rng():
    with _GLOBAL_RNG_LOCK:
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def get_distributed_state() -> Tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 1, 0
    try:
        dist = importlib.import_module("torch.distributed")
    except ModuleNotFoundError:
        return 1, 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return world_size, int(os.environ.get("RANK", "0"))


def distributed_barrier_if_initialized() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    try:
        dist = importlib.import_module("torch.distributed")
    except ModuleNotFoundError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@dataclass(frozen=True)
class PoolEntry:
    dataset: str
    role: str
    status: str
    path: Path


@dataclass(frozen=True)
class ArchiveAudioSource:
    dataset: str
    role: str
    status: str
    archive_path: Path
    member_name: str
    member_size: int


def is_audio_path(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def archive_kind(path: Union[str, Path]) -> Optional[str]:
    path = Path(path)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.suffix.lower() == ".zip":
        return "zip"
    if suffixes[-2:] in ([".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]):
        return "tar"
    if path.suffix.lower() in {".tar", ".tgz", ".tbz2", ".txz"}:
        return "tar"
    return None


def parse_pool_manifest(path: Union[str, Path]) -> List[PoolEntry]:
    entries = []
    manifest_path = Path(path).expanduser()
    with manifest_path.open("r") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line_number == 1 and line.lower().startswith("dataset\trole\tstatus\tpath"):
                continue
            parts = line.split("\t", 3)
            if len(parts) != 4:
                raise ValueError(f"Invalid dry pool manifest line in {manifest_path}:{line_number}: {line}")
            dataset, role, status, item_path = parts
            entries.append(
                PoolEntry(
                    dataset=dataset,
                    role=role,
                    status=status,
                    path=Path(item_path).expanduser(),
                )
            )
    return entries


def select_existing_audio_entries(
    entries: Iterable[PoolEntry],
    allowed_statuses: Iterable[str],
) -> Tuple[List[PoolEntry], Dict[str, int]]:
    allowed = set(allowed_statuses)
    selected = []
    stats = {
        "total": 0,
        "selected_audio": 0,
        "skipped_status": 0,
        "skipped_non_audio": 0,
        "skipped_missing": 0,
    }
    for entry in entries:
        stats["total"] += 1
        if entry.status not in allowed:
            stats["skipped_status"] += 1
            continue
        if not is_audio_path(entry.path):
            stats["skipped_non_audio"] += 1
            continue
        selected.append(entry)
        stats["selected_audio"] += 1
    return selected, stats


def iter_archive_audio_sources(
    entry: PoolEntry,
    max_members: Optional[int] = None,
) -> Iterable[ArchiveAudioSource]:
    kind = archive_kind(entry.path)
    if kind == "zip":
        with zipfile.ZipFile(entry.path) as archive:
            yielded = 0
            for info in archive.infolist():
                if info.is_dir() or not is_audio_path(info.filename):
                    continue
                yield ArchiveAudioSource(
                    dataset=entry.dataset,
                    role=entry.role,
                    status=entry.status,
                    archive_path=entry.path,
                    member_name=info.filename,
                    member_size=int(info.file_size),
                )
                yielded += 1
                if max_members is not None and yielded >= max_members:
                    break
        return

    if kind == "tar":
        with tarfile.open(entry.path, "r:*") as archive:
            yielded = 0
            for member in archive:
                if not member.isfile() or not is_audio_path(member.name):
                    continue
                yield ArchiveAudioSource(
                    dataset=entry.dataset,
                    role=entry.role,
                    status=entry.status,
                    archive_path=entry.path,
                    member_name=member.name,
                    member_size=int(member.size),
                )
                yielded += 1
                if max_members is not None and yielded >= max_members:
                    break


def select_audio_sources(
    entries: Iterable[PoolEntry],
    allowed_statuses: Iterable[str],
    include_archives: bool = True,
    max_archives_to_index: int = 16,
    max_archive_members_per_archive: int = 8,
) -> Tuple[List[Union[PoolEntry, ArchiveAudioSource]], Dict[str, int]]:
    entries = list(entries)
    selected, stats = select_existing_audio_entries(entries, allowed_statuses)
    stats.update(
        {
            "selected_archive_audio": 0,
            "indexed_archives": 0,
            "skipped_unsupported_archive": 0,
            "skipped_archive_error": 0,
        }
    )
    if not include_archives:
        return selected, stats

    allowed = set(allowed_statuses)
    archive_entries = [
        entry
        for entry in entries
        if entry.status in allowed
        and archive_kind(entry.path) is not None
        and entry.path.is_file()
    ]
    archive_entries = archive_entries[: max(0, int(max_archives_to_index))]
    for entry in archive_entries:
        try:
            archive_sources = list(
                iter_archive_audio_sources(
                    entry,
                    max_members=max_archive_members_per_archive,
                )
            )
        except (OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile, zipfile.LargeZipFile):
            stats["skipped_archive_error"] += 1
            continue
        if not archive_sources:
            stats["skipped_unsupported_archive"] += 1
            continue
        selected.extend(archive_sources)
        stats["indexed_archives"] += 1
        stats["selected_archive_audio"] += len(archive_sources)
    return selected, stats


def count_archive_or_shard_entries(entries: Iterable[PoolEntry]) -> int:
    count = 0
    for entry in entries:
        status = entry.status.lower()
        if "archive" in status or "shard" in status or "git_lfs" in status:
            count += 1
    return count


def default_manifest_paths(dry_data_root: Union[str, Path]) -> Dict[str, Path]:
    root = Path(dry_data_root).expanduser()
    manifest_dir = root / "manifests"
    return {
        "clean": manifest_dir / "speech_clean_pool.list",
        "noise": manifest_dir / "noise_pool.list",
        "rir": manifest_dir / "rir_pool.list",
    }


def stable_key(index: int, source_path: Path) -> str:
    stem = source_path.stem.replace("/", "_").replace(" ", "_")
    return f"{index:09d}_{stem}"


def source_key_stem(source: Union[PoolEntry, ArchiveAudioSource]) -> str:
    if isinstance(source, ArchiveAudioSource):
        return f"{source.archive_path.stem}_{Path(source.member_name).stem}"
    return source.path.stem


def source_dataset(source: Union[PoolEntry, ArchiveAudioSource]) -> str:
    return source.dataset


def source_status(source: Union[PoolEntry, ArchiveAudioSource]) -> str:
    return source.status


def source_role(source: Union[PoolEntry, ArchiveAudioSource]) -> str:
    return source.role


def source_path_text(source: Union[PoolEntry, ArchiveAudioSource]) -> str:
    if isinstance(source, ArchiveAudioSource):
        return f"{source.archive_path}::{source.member_name}"
    return str(source.path)


def source_size(source: Union[PoolEntry, ArchiveAudioSource]) -> int:
    if isinstance(source, ArchiveAudioSource):
        return int(source.member_size)
    return int(source.path.stat().st_size)


def read_archive_member_bytes(source: ArchiveAudioSource) -> bytes:
    kind = archive_kind(source.archive_path)
    if kind == "zip":
        with zipfile.ZipFile(source.archive_path) as archive:
            return archive.read(source.member_name)
    if kind == "tar":
        with tarfile.open(source.archive_path, "r:*") as archive:
            extracted = archive.extractfile(source.member_name)
            if extracted is None:
                raise FileNotFoundError(f"Missing archive member: {source_path_text(source)}")
            with extracted:
                return extracted.read()
    raise ValueError(f"Unsupported archive type: {source.archive_path}")


def audio_data_from_source(source: Union[PoolEntry, ArchiveAudioSource]) -> Tuple[np.ndarray, int]:
    if isinstance(source, ArchiveAudioSource):
        data, sample_rate = sf.read(io.BytesIO(read_archive_member_bytes(source)), dtype="float32", always_2d=True)
        return data, int(sample_rate)
    data, sample_rate = sf.read(source.path, dtype="float32", always_2d=True)
    return data, int(sample_rate)


def wav_bytes_from_audio_source(source: Union[PoolEntry, ArchiveAudioSource]) -> Tuple[bytes, int, float]:
    data, sample_rate = audio_data_from_source(source)
    if data.shape[1] > 1:
        data = data[:, :1]
    duration = float(data.shape[0]) / float(sample_rate)
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), int(sample_rate), duration


def add_bytes_to_tar(tar: tarfile.TarFile, member_name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(member_name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def materialize_rolling_cache(
    dry_data_root: Union[str, Path],
    cache_dir: Union[str, Path],
    run_id: Optional[str] = None,
    cache_size_gb: float = 20.0,
    shard_size_mb: int = 1024,
    seed: int = 3407,
    clean_manifest: Optional[Union[str, Path]] = None,
    cleanup_policy: str = "keep_current",
    include_archives: bool = True,
    max_archives_to_index: int = 128,
    max_archive_members_per_archive: int = 32,
    clean_statuses: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    if cache_size_gb <= 0:
        raise ValueError("cache_size_gb must be positive")
    if shard_size_mb <= 0:
        raise ValueError("shard_size_mb must be positive")
    if cleanup_policy not in {"keep_current", "refresh"}:
        raise ValueError("cleanup_policy must be 'keep_current' or 'refresh'")

    run_id = run_id or os.environ.get("SLURM_JOB_ID") or f"rolling_seed{int(seed)}"
    cache_root = Path(cache_dir).expanduser()
    run_dir = cache_root / str(run_id)
    manifest_path = run_dir / "manifest.jsonl"
    stats_path = run_dir / "stats.json"
    if cleanup_policy == "keep_current" and manifest_path.is_file() and stats_path.is_file():
        return json.loads(stats_path.read_text())
    if cleanup_policy == "refresh" and run_dir.exists():
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    clean_manifest_path = Path(clean_manifest).expanduser() if clean_manifest else default_manifest_paths(dry_data_root)["clean"]
    parsed_clean_entries = parse_pool_manifest(clean_manifest_path)
    clean_entries, clean_stats = select_audio_sources(
        parsed_clean_entries,
        allowed_statuses=clean_statuses or DEFAULT_CLEAN_STATUSES,
        include_archives=include_archives,
        max_archives_to_index=max_archives_to_index,
        max_archive_members_per_archive=max_archive_members_per_archive,
    )
    if not clean_entries:
        raise ValueError(f"No available clean wav/flac source found in {clean_manifest_path}")

    rng = random.Random(int(seed))
    candidates = clean_entries[:]
    rng.shuffle(candidates)
    target_bytes = int(float(cache_size_gb) * 1024**3)
    shard_limit = int(shard_size_mb) * 1024**2
    manifest_tmp = run_dir / "manifest.jsonl.tmp"
    stats_tmp = run_dir / "stats.json.tmp"

    sampled = 0
    skipped_read_error = 0
    total_cache_wav_bytes = 0
    total_source_bytes = 0
    shard_index = 0
    shard_payload_bytes = 0
    tar = None
    shard_name = None

    try:
        with manifest_tmp.open("w") as manifest_file:
            for entry in candidates:
                if total_cache_wav_bytes >= target_bytes:
                    break
                try:
                    wav_payload, sample_rate, duration = wav_bytes_from_audio_source(entry)
                    input_size = source_size(entry)
                except (OSError, RuntimeError, ValueError, sf.LibsndfileError):
                    skipped_read_error += 1
                    continue

                if tar is None or shard_payload_bytes >= shard_limit:
                    if tar is not None:
                        tar.close()
                    shard_name = f"clean-{shard_index:06d}.tar"
                    tar = tarfile.open(run_dir / shard_name, "w")
                    shard_index += 1
                    shard_payload_bytes = 0

                key = stable_key(sampled, Path(source_key_stem(entry)))
                wav_member = f"{key}.wav"
                json_member = f"{key}.json"
                metadata = {
                    "key": key,
                    "dataset": source_dataset(entry),
                    "role": source_role(entry),
                    "status": source_status(entry),
                    "source_path": source_path_text(entry),
                    "source_size": input_size,
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "cache_wav_bytes": len(wav_payload),
                    "source_type": "archive_member" if isinstance(entry, ArchiveAudioSource) else "file",
                }
                metadata_payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
                add_bytes_to_tar(tar, wav_member, wav_payload)
                add_bytes_to_tar(tar, json_member, metadata_payload)

                record = {
                    "key": key,
                    "shard": shard_name,
                    "wav_member": wav_member,
                    "json_member": json_member,
                    "source_path": source_path_text(entry),
                    "dataset": source_dataset(entry),
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "cache_wav_bytes": len(wav_payload),
                    "source_type": "archive_member" if isinstance(entry, ArchiveAudioSource) else "file",
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                sampled += 1
                total_cache_wav_bytes += len(wav_payload)
                total_source_bytes += input_size
                shard_payload_bytes += len(wav_payload) + len(metadata_payload)
    finally:
        if tar is not None:
            tar.close()

    if sampled == 0:
        raise RuntimeError(f"Failed to materialize any clean samples from {clean_manifest_path}")

    stats = {
        "run_id": str(run_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "clean_manifest": str(clean_manifest_path),
        "seed": int(seed),
        "cache_size_gb_requested": float(cache_size_gb),
        "shard_size_mb": int(shard_size_mb),
        "sampled": sampled,
        "num_shards": shard_index,
        "total_cache_wav_bytes": total_cache_wav_bytes,
        "total_source_bytes": total_source_bytes,
        "skipped_read_error": skipped_read_error,
        "skipped_archive_or_shard": count_archive_or_shard_entries(parsed_clean_entries),
        "clean_pool_stats": clean_stats,
        "include_archives": bool(include_archives),
        "max_archives_to_index": int(max_archives_to_index),
        "max_archive_members_per_archive": int(max_archive_members_per_archive),
    }
    stats_tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    stats_tmp.replace(stats_path)
    manifest_tmp.replace(manifest_path)
    return stats


def wait_for_cache_manifest(manifest_path: Union[str, Path], timeout_seconds: int = DEFAULT_CACHE_WAIT_SECONDS) -> None:
    path = Path(manifest_path)
    deadline = time.time() + int(timeout_seconds)
    while time.time() < deadline:
        if path.is_file():
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for rolling cache manifest: {path}")


def install_torchaudio_io_import_shim() -> None:
    try:
        importlib.import_module("torchaudio.io")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "torchaudio.io":
            raise

    try:
        torchaudio = importlib.import_module("torchaudio")
    except ModuleNotFoundError:
        return

    module = types.ModuleType("torchaudio.io")

    class CodecConfig:
        def __init__(self, qscale=None):
            self.qscale = qscale

    class AudioEffector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Codec degradation requires torchaudio.io.AudioEffector. "
                "Install a torchaudio build that provides torchaudio.io, "
                "or keep codec_prob: 0.0 in the simulation config."
            )

    module.CodecConfig = CodecConfig
    module.AudioEffector = AudioEffector
    sys.modules["torchaudio.io"] = module
    setattr(torchaudio, "io", module)


class RollingCacheTarReader:
    def __init__(self, cache_dir: Union[str, Path], manifest_path: Union[str, Path]):
        self.cache_dir = Path(cache_dir).expanduser()
        self.manifest_path = Path(manifest_path).expanduser()
        self.records = []
        with self.manifest_path.open("r") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"Rolling cache manifest is empty: {self.manifest_path}")

    def read_wav(self, index: int, fs: int = 16000) -> Tuple[np.ndarray, Dict[str, object]]:
        record = self.records[index % len(self.records)]
        with tarfile.open(self.cache_dir / record["shard"], "r") as tar:
            extracted = tar.extractfile(record["wav_member"])
            if extracted is None:
                raise FileNotFoundError(f"Missing {record['wav_member']} in {record['shard']}")
            with extracted:
                payload = extracted.read()
        wav, source_fs = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        wav = wav[:, :1].T
        if fs is not None and source_fs != fs:
            wav = librosa.resample(wav, orig_sr=source_fs, target_sr=fs, res_type="soxr_hq")
        return wav.astype(np.float32), record

    def close(self) -> None:
        return None

    def __del__(self):
        self.close()


class UseSimulationRollingCacheDataLoadIter:
    def __init__(
        self,
        dry_data_root: Union[str, Path],
        use_simulation_root: Union[str, Path],
        simulation_config: Union[str, Path],
        cache_dir: Union[str, Path] = "tmp/rolling_cache",
        run_id: Optional[str] = None,
        cache_size_gb: float = 20.0,
        shard_size_mb: int = 1024,
        cleanup_policy: str = "keep_current",
        clean_manifest: Optional[Union[str, Path]] = None,
        noise_manifest: Optional[Union[str, Path]] = None,
        rir_manifest: Optional[Union[str, Path]] = None,
        wind_manifest: Optional[Union[str, Path]] = None,
        include_archives: bool = True,
        max_archives_to_index: int = 16,
        max_archive_members_per_archive: int = 8,
        clean_statuses: Optional[Iterable[str]] = None,
        batch_size: int = 1,
        cut_duration: Union[float, List[float]] = 5.0,
        num_workers: int = 1,
        prefetch: int = 0,
        samples_per_epoch: int = 1000,
        mode: str = "train",
        seed: int = 3407,
        cache_wait_seconds: int = DEFAULT_CACHE_WAIT_SECONDS,
        include_clean_archives: Optional[bool] = None,
        include_noise_archives: bool = False,
        include_rir_archives: Optional[bool] = None,
        include_wind_archives: bool = False,
        batch_format: str = "tuple",
    ):
        self.is_train = mode == "train"
        self.dry_data_root = Path(dry_data_root).expanduser()
        self.use_simulation_root = Path(use_simulation_root).expanduser()
        self.simulation_config_path = Path(simulation_config).expanduser()
        self.run_id = run_id or os.environ.get("SLURM_JOB_ID") or f"rolling_seed{int(seed)}"
        self.cache_dir_root = Path(cache_dir).expanduser()
        self.cache_run_dir = self.cache_dir_root / str(self.run_id)
        self.manifest_path = self.cache_run_dir / "manifest.jsonl"
        self.stats_path = self.cache_run_dir / "stats.json"
        self.batch_size = int(batch_size)
        self.cut_duration = cut_duration
        self.num_workers = int(num_workers)
        self.prefetch = int(prefetch)
        self.samples_per_epoch = int(samples_per_epoch)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0
        self.batch_format = batch_format

        with self.simulation_config_path.open("r") as f:
            self.simulation_config = yaml.safe_load(f)
        self.simulation_config.setdefault("stft_cfg", {})["sampling_rate"] = 16000

        self.world_size, self.rank = get_distributed_state()

        if self.rank == 0:
            materialize_rolling_cache(
                dry_data_root=self.dry_data_root,
                cache_dir=self.cache_dir_root,
                run_id=self.run_id,
                cache_size_gb=cache_size_gb,
                shard_size_mb=shard_size_mb,
                seed=self.seed,
                clean_manifest=clean_manifest,
                cleanup_policy=cleanup_policy,
                include_archives=include_archives if include_clean_archives is None else include_clean_archives,
                max_archives_to_index=max_archives_to_index,
                max_archive_members_per_archive=max_archive_members_per_archive,
                clean_statuses=clean_statuses,
            )
        distributed_barrier_if_initialized()
        wait_for_cache_manifest(self.manifest_path, timeout_seconds=cache_wait_seconds)

        manifest_paths = default_manifest_paths(self.dry_data_root)
        noise_manifest_path = Path(noise_manifest).expanduser() if noise_manifest else manifest_paths["noise"]
        rir_manifest_path = Path(rir_manifest).expanduser() if rir_manifest else manifest_paths["rir"]
        wind_manifest_path = Path(wind_manifest).expanduser() if wind_manifest else noise_manifest_path
        self.noise_paths, self.noise_stats = select_audio_sources(
            parse_pool_manifest(noise_manifest_path),
            allowed_statuses=DEFAULT_NOISE_STATUSES,
            include_archives=include_noise_archives,
            max_archives_to_index=max_archives_to_index,
            max_archive_members_per_archive=max_archive_members_per_archive,
        )
        self.rir_paths, self.rir_stats = select_audio_sources(
            parse_pool_manifest(rir_manifest_path),
            allowed_statuses=DEFAULT_RIR_STATUSES,
            include_archives=include_archives if include_rir_archives is None else include_rir_archives,
            max_archives_to_index=max_archives_to_index,
            max_archive_members_per_archive=max_archive_members_per_archive,
        )
        self.wind_paths, self.wind_stats = select_audio_sources(
            parse_pool_manifest(wind_manifest_path),
            allowed_statuses=DEFAULT_NOISE_STATUSES,
            include_archives=include_wind_archives,
            max_archives_to_index=max_archives_to_index,
            max_archive_members_per_archive=max_archive_members_per_archive,
        )
        if not self.noise_paths:
            raise ValueError(f"No existing noise wav/flac found in {noise_manifest_path}")
        if not self.rir_paths:
            raise ValueError(f"No existing RIR wav/flac found in {rir_manifest_path}")
        if not self.wind_paths:
            raise ValueError(f"No existing wind wav/flac found in {wind_manifest_path}")

        self.clean_reader = RollingCacheTarReader(self.cache_run_dir, self.manifest_path)
        self.random_select_and_order, self.apply_degradation_with_wind = self.import_use_simulation()

    def import_use_simulation(self):
        if not self.use_simulation_root.is_dir():
            raise FileNotFoundError(f"USE_simulation repo not found: {self.use_simulation_root}")
        root = str(self.use_simulation_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        install_torchaudio_io_import_shim()
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
        epoch = self.epoch if self.is_train else 0
        return stable_uint32(self.seed, self.mode, self.rank, epoch, index, len(self.clean_reader.records))

    @staticmethod
    def load_wav(source: Union[str, Path, PoolEntry, ArchiveAudioSource], fs=16000):
        if isinstance(source, (PoolEntry, ArchiveAudioSource)):
            data, source_fs = audio_data_from_source(source)
            wav = data[:, :1].T
            if fs is not None and source_fs != fs:
                wav = librosa.resample(wav, orig_sr=source_fs, target_sr=fs, res_type="soxr_hq")
            return wav.astype(np.float32)
        wav, _ = librosa.load(source, dtype=np.float32, sr=fs, mono=False)
        if wav.ndim == 1:
            wav = wav[None]
        else:
            wav = wav[:1, :]
        return wav

    @staticmethod
    def pad_or_cut_wav(wav, length, rng, offset=None):
        if wav.shape[-1] < length:
            wav = np.pad(wav, [(0, 0), (0, length - wav.shape[-1])], mode="wrap")
            return wav, None
        if offset is None:
            offset = int(rng.integers(0, wav.shape[-1] - length + 1))
        return wav[..., offset: offset + length], offset

    @staticmethod
    def normalize_src_tgt(src, tgt, py_rng, low=0.1, high=0.99):
        max_tgt_value = np.max(np.abs(tgt)) + 1e-5
        max_src_value = np.max(np.abs(src)) + 1e-5
        max_value = max(max_tgt_value, max_src_value)
        threshold = high / max_value
        target_value = py_rng.uniform(low, high)
        factor = min(target_value / max_tgt_value, threshold)
        return src * factor, tgt * factor

    def process_one_sample(self, sample_index):
        fs = 16000
        item_seed = self.stable_seed(sample_index)
        clean_index = sample_index % len(self.clean_reader.records)

        rng = np.random.default_rng(item_seed)
        py_rng = random.Random(item_seed)
        noise_entry = self.noise_paths[py_rng.randrange(len(self.noise_paths))]
        rir_entry = self.rir_paths[py_rng.randrange(len(self.rir_paths))]
        wind_entry = self.wind_paths[py_rng.randrange(len(self.wind_paths))]

        speech, clean_record = self.clean_reader.read_wav(clean_index, fs=fs)
        noise = self.load_wav(noise_entry, fs)
        rir = self.load_wav(rir_entry, fs)
        wind_noise = self.load_wav(wind_entry, fs)

        if self.cut_duration is None:
            length = speech.shape[-1]
        else:
            cut_duration = self.cut_duration
            if isinstance(self.cut_duration, list):
                cut_duration = py_rng.uniform(*self.cut_duration)
            length = int(float(cut_duration) * fs)
            speech, _ = self.pad_or_cut_wav(speech, length, rng)

        with preserve_global_rng():
            degrad_cfgs, selected_degrads = self.random_select_and_order(self.simulation_config, seed=item_seed)
            clean, mix = self.apply_degradation_with_wind(
                self.simulation_config,
                speech,
                noise,
                rir,
                wind_noise,
                degrad_cfgs,
                selected_degrads,
                seed=item_seed,
            )

        if mix.shape[-1] > length:
            mix = mix[..., :length]
        elif mix.shape[-1] < length:
            mix = np.pad(mix, [(0, 0), (0, length - mix.shape[-1])], mode="wrap")

        if clean.shape[-1] > length:
            clean = clean[..., :length]
        elif clean.shape[-1] < length:
            clean = np.pad(clean, [(0, 0), (0, length - clean.shape[-1])], mode="wrap")

        mix, clean = self.normalize_src_tgt(mix, clean, py_rng)
        name = clean_record.get("key", f"rolling_{sample_index}")
        return None, mix.astype(np.float32), clean.astype(np.float32), None, fs, length, name

    def make_batch(self, batch_mix, batch_speech, batch_fs, lengths, names):
        torch = importlib.import_module("torch")
        mix = torch.from_numpy(np.concatenate(batch_mix, axis=0)).float()
        speech = torch.from_numpy(np.concatenate(batch_speech, axis=0)).float()
        fs = torch.LongTensor(batch_fs)
        lengths = torch.LongTensor(lengths)
        if self.batch_format == "dict":
            return {
                "mode": "se",
                "degraded_wav": mix,
                "clean_wav": speech,
                "sample_rate": fs,
                "length": lengths,
                "utterance_id": names,
                "source_path": None,
                "clean_path": None,
            }
        if self.mode == "test":
            return ("se", None, mix, speech, fs, lengths, names)
        return ("se", None, mix, speech, None, fs, lengths, names)

    def __iter__(self):
        if self.is_train:
            self.epoch += 1
        for batch_idx in range(len(self)):
            start_idx = (batch_idx * self.world_size + self.rank) * self.batch_size
            sample_indices = list(range(start_idx, start_idx + self.batch_size))
            batch_mix = []
            batch_speech = []
            batch_fs = []
            lengths = []
            names = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
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
