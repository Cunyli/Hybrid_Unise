import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from dataloader.rolling_cache import (
    RollingCacheTarReader,
    UseSimulationRollingCacheDataLoadIter,
    materialize_rolling_cache,
    parse_pool_manifest,
    select_existing_audio_entries,
)


def write_wav(path, value=0.1, sample_rate=16000, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.full((int(sample_rate * seconds), 1), value, dtype=np.float32)
    sf.write(path, wav, sample_rate)


def write_tar_with_wav(path, member_name, value=0.1):
    source = path.parent / f"{Path(member_name).stem}.wav"
    write_wav(source, value=value)
    with tarfile.open(path, "w:gz") as archive:
        archive.add(source, arcname=member_name)
    source.unlink()


def write_zip_with_wav(path, member_name, value=0.1):
    source = path.parent / f"{Path(member_name).stem}.wav"
    write_wav(source, value=value)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(source, arcname=member_name)
    source.unlink()


class RollingCacheTest(unittest.TestCase):
    def test_manifest_parser_preserves_paths_with_spaces_and_filters_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "dir with spaces" / "a.wav"
            write_wav(audio_path)
            manifest = root / "pool.list"
            manifest.write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\tspeech_clean\tavailable_existing\t{audio_path}\n"
                f"archive\tspeech_clean\tdownloaded_archive\t{root / 'archive.tar.gz'}\n"
            )

            entries = parse_pool_manifest(manifest)
            selected, stats = select_existing_audio_entries(entries, {"available_existing"})

            self.assertEqual(len(entries), 2)
            self.assertEqual(selected[0].path, audio_path)
            self.assertEqual(stats["selected_audio"], 1)
            self.assertEqual(stats["skipped_status"], 1)

    def test_materialize_and_read_tar_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dry_root = root / "dry"
            manifest_dir = dry_root / "manifests"
            manifest_dir.mkdir(parents=True)
            wav_path = root / "source clean.wav"
            write_wav(wav_path, value=0.2)
            (manifest_dir / "speech_clean_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\tspeech_clean\tavailable_existing\t{wav_path}\n"
                f"demo_archive\tspeech_clean\tdownloaded_archive\t{root / 'demo.tar.gz'}\n"
            )

            stats = materialize_rolling_cache(
                dry_data_root=dry_root,
                cache_dir=root / "cache",
                run_id="unit",
                cache_size_gb=0.000001,
                shard_size_mb=1,
                seed=7,
            )
            manifest_path = Path(stats["manifest_path"])
            records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            reader = RollingCacheTarReader(root / "cache" / "unit", manifest_path)
            wav, record = reader.read_wav(0, fs=16000)
            reader.close()

            self.assertEqual(stats["sampled"], 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(wav.shape[0], 1)
            self.assertEqual(record["key"], records[0]["key"])

    def test_materialize_clean_from_archive_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dry_root = root / "dry"
            manifest_dir = dry_root / "manifests"
            manifest_dir.mkdir(parents=True)
            archive_path = root / "clean_archive.tar.gz"
            write_tar_with_wav(archive_path, "speaker/utt001.wav", value=0.3)
            (manifest_dir / "speech_clean_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo_archive\tspeech_clean\tdownloaded_archive\t{archive_path}\n"
            )

            stats = materialize_rolling_cache(
                dry_data_root=dry_root,
                cache_dir=root / "cache",
                run_id="archive_unit",
                cache_size_gb=0.000001,
                shard_size_mb=1,
                seed=7,
                clean_statuses=["downloaded_archive"],
                max_archives_to_index=1,
                max_archive_members_per_archive=1,
            )
            records = [json.loads(line) for line in Path(stats["manifest_path"]).read_text().splitlines()]

            self.assertEqual(stats["sampled"], 1)
            self.assertEqual(stats["clean_pool_stats"]["selected_archive_audio"], 1)
            self.assertEqual(records[0]["source_type"], "archive_member")
            self.assertIn("clean_archive.tar.gz::speaker/utt001.wav", records[0]["source_path"])

    def test_dataloader_smoke_with_fake_use_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dry_root = root / "dry"
            manifest_dir = dry_root / "manifests"
            manifest_dir.mkdir(parents=True)

            clean = root / "clean.wav"
            noise = root / "noise.wav"
            rir = root / "rir.wav"
            write_wav(clean, value=0.2)
            write_wav(noise, value=0.01)
            write_wav(rir, value=0.001, seconds=0.1)

            (manifest_dir / "speech_clean_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\tspeech_clean\tavailable_existing\t{clean}\n"
            )
            (manifest_dir / "noise_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\tnoise\tavailable_existing\t{noise}\n"
            )
            (manifest_dir / "rir_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\trir\tavailable_existing\t{rir}\n"
            )

            use_sim_root = root / "USE_simulation"
            use_sim_root.mkdir()
            (use_sim_root / "simulate_degradation.py").write_text(
                "def random_select_and_order(cfg, seed=None):\n"
                "    return {}, ['noise']\n"
                "\n"
                "def apply_degradation_with_wind(cfg, speech, noise, rir, wind_noise, degrad_cfgs, selected_degrads, seed=None):\n"
                "    return speech, speech + 0.01 * noise[..., :speech.shape[-1]]\n"
            )

            sim_config = root / "sim.yaml"
            sim_config.write_text(
                "stft_cfg:\n"
                "  sampling_rate: 16000\n"
                "degradation_cfg:\n"
                "  noise_prob: 1.0\n"
            )

            dataset = UseSimulationRollingCacheDataLoadIter(
                dry_data_root=dry_root,
                use_simulation_root=use_sim_root,
                simulation_config=sim_config,
                cache_dir=root / "cache",
                run_id="unit",
                cache_size_gb=0.000001,
                shard_size_mb=1,
                cleanup_policy="refresh",
                batch_size=1,
                cut_duration=[0.25, 0.25],
                num_workers=1,
                samples_per_epoch=2,
                mode="train",
                seed=7,
                batch_format="dict",
            )
            batch = next(iter(dataset))

            self.assertEqual(batch["mode"], "se")
            self.assertEqual(batch["degraded_wav"].shape, (1, 4000))
            self.assertEqual(batch["clean_wav"].shape, (1, 4000))
            self.assertEqual(batch["sample_rate"].tolist(), [16000])
            self.assertEqual(batch["length"].tolist(), [4000])

    def test_dataloader_smoke_with_archive_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dry_root = root / "dry"
            manifest_dir = dry_root / "manifests"
            manifest_dir.mkdir(parents=True)

            clean_archive = root / "clean_archive.zip"
            noise_archive = root / "noise_archive.tar.gz"
            rir_archive = root / "rir_archive.tar.gz"
            write_zip_with_wav(clean_archive, "clean/utt001.wav", value=0.2)
            write_tar_with_wav(noise_archive, "noise/noise001.wav", value=0.01)
            write_tar_with_wav(rir_archive, "rir/rir001.wav", value=0.001)

            (manifest_dir / "speech_clean_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\tspeech_clean\tdownloaded_archive\t{clean_archive}\n"
            )
            (manifest_dir / "noise_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\tnoise\tdownloaded_archive\t{noise_archive}\n"
            )
            (manifest_dir / "rir_pool.list").write_text(
                "dataset\trole\tstatus\tpath\n"
                f"demo\trir\tavailable_downloaded_archive\t{rir_archive}\n"
            )

            use_sim_root = root / "USE_simulation"
            use_sim_root.mkdir()
            (use_sim_root / "simulate_degradation.py").write_text(
                "def random_select_and_order(cfg, seed=None):\n"
                "    return {}, ['noise']\n"
                "\n"
                "def apply_degradation_with_wind(cfg, speech, noise, rir, wind_noise, degrad_cfgs, selected_degrads, seed=None):\n"
                "    return speech, speech + 0.01 * noise[..., :speech.shape[-1]]\n"
            )
            sim_config = root / "sim.yaml"
            sim_config.write_text(
                "stft_cfg:\n"
                "  sampling_rate: 16000\n"
                "degradation_cfg:\n"
                "  noise_prob: 1.0\n"
            )

            dataset = UseSimulationRollingCacheDataLoadIter(
                dry_data_root=dry_root,
                use_simulation_root=use_sim_root,
                simulation_config=sim_config,
                cache_dir=root / "cache",
                run_id="archive_unit",
                cache_size_gb=0.000001,
                shard_size_mb=1,
                cleanup_policy="refresh",
                clean_statuses=["downloaded_archive"],
                include_noise_archives=True,
                include_rir_archives=True,
                include_wind_archives=True,
                max_archives_to_index=1,
                max_archive_members_per_archive=1,
                batch_size=1,
                cut_duration=[0.25, 0.25],
                num_workers=1,
                samples_per_epoch=2,
                mode="train",
                seed=7,
            )
            batch = next(iter(dataset))

            self.assertEqual(batch[0], "se")
            self.assertEqual(batch[2].shape, (1, 4000))
            self.assertEqual(batch[3].shape, (1, 4000))
            self.assertEqual(dataset.noise_stats["selected_archive_audio"], 1)
            self.assertEqual(dataset.rir_stats["selected_archive_audio"], 1)


if __name__ == "__main__":
    unittest.main()
