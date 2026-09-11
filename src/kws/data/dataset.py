"""torch Dataset for the small-keyword-set KWS task: target keywords + unknown + silence."""
import random
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from kws.data.audio_io import load_waveform
from kws.data.augment import SpecAugmenter, WaveformAugmenter
from kws.data.features import FeatureExtractor
from kws.data.silence import SilenceSampler
from kws.data.splits import TEST, TRAIN, VAL, build_split_index
from kws.data.unknown import unknown_sample_cap, unknown_words

UNKNOWN_LABEL_NAME = "_unknown_"
SILENCE_LABEL_NAME = "_silence_"


@dataclass
class Entry:
    label: int
    rel_path: str | None  # None => synthesized silence sample


def build_label_map(target_keywords: list[str]) -> dict[str, int]:
    label_map = {word: i for i, word in enumerate(target_keywords)}
    label_map[UNKNOWN_LABEL_NAME] = len(target_keywords)
    label_map[SILENCE_LABEL_NAME] = len(target_keywords) + 1
    return label_map


class SpeechCommandsKWSDataset(Dataset):
    def __init__(self, entries: list[Entry], dataset_root: Path, feature_extractor: FeatureExtractor,
                 sample_rate: int, clip_len: int, silence_sampler: SilenceSampler,
                 waveform_augmenter: WaveformAugmenter | None = None,
                 spec_augmenter: SpecAugmenter | None = None):
        self.entries = entries
        self.dataset_root = dataset_root
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.clip_len = clip_len
        self.silence_sampler = silence_sampler
        self.waveform_augmenter = waveform_augmenter
        self.spec_augmenter = spec_augmenter

    def __len__(self) -> int:
        return len(self.entries)

    def _load_waveform(self, entry: Entry) -> torch.Tensor:
        if entry.rel_path is None:
            return self.silence_sampler.sample()
        waveform = load_waveform(self.dataset_root / entry.rel_path, self.sample_rate)
        if waveform.shape[-1] < self.clip_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_len - waveform.shape[-1]))
        elif waveform.shape[-1] > self.clip_len:
            waveform = waveform[:, :self.clip_len]
        return waveform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[index]
        waveform = self._load_waveform(entry)
        if self.waveform_augmenter is not None:
            waveform = self.waveform_augmenter(waveform)
        features = self.feature_extractor(waveform)
        if self.spec_augmenter is not None:
            features = self.spec_augmenter(features)
        return features, entry.label


def _build_entries(split_index: dict[str, list[tuple[str, str]]], split: str,
                    label_map: dict[str, int], unknown_pool: Collection[str],
                    unknown_ratio_to_avg_keyword_count: float,
                    silence_ratio_to_avg_keyword_count: float,
                    rng: random.Random) -> list[Entry]:
    by_word: dict[str, list[str]] = {}
    for word, rel_path in split_index[split]:
        by_word.setdefault(word, []).append(rel_path)

    entries: list[Entry] = []
    keyword_counts = []
    for word, label in label_map.items():
        if word in (UNKNOWN_LABEL_NAME, SILENCE_LABEL_NAME):
            continue
        paths = by_word.get(word, [])
        keyword_counts.append(len(paths))
        entries.extend(Entry(label=label, rel_path=p) for p in paths)

    avg_keyword_count = sum(keyword_counts) / max(len(keyword_counts), 1)
    unknown_count = unknown_sample_cap(
        avg_keyword_count, unknown_ratio_to_avg_keyword_count,
    )
    # Sets have process-dependent iteration order because of Python hash
    # randomization. Sort the pool before shuffling so a fixed ``rng`` seed
    # selects the same unknown examples in every process.
    unknown_paths = [p for w in sorted(unknown_pool) for p in by_word.get(w, [])]
    rng.shuffle(unknown_paths)
    if len(unknown_paths) < unknown_count:
        raise ValueError(
            f"split {split!r} has only {len(unknown_paths)} unknown samples, "
            f"but needs {unknown_count} to satisfy the configured class balance"
        )
    unknown_paths = unknown_paths[:unknown_count]
    unknown_label = label_map[UNKNOWN_LABEL_NAME]
    entries.extend(Entry(label=unknown_label, rel_path=p) for p in unknown_paths)

    num_silence = int(round(avg_keyword_count * silence_ratio_to_avg_keyword_count))
    silence_label = label_map[SILENCE_LABEL_NAME]
    entries.extend(Entry(label=silence_label, rel_path=None) for _ in range(num_silence))

    rng.shuffle(entries)
    return entries


def build_datasets(data_cfg: dict, augment: bool, seed: int = 0):
    dataset_cfg = data_cfg["dataset"]
    dataset_root = Path(dataset_cfg["root"])
    target_keywords = data_cfg["target_keywords"]
    unknown_cfg = data_cfg["unknown"]
    silence_cfg = data_cfg["silence"]

    label_map = build_label_map(target_keywords)
    unknown_pool = unknown_words(target_keywords)
    all_word_dirs = target_keywords + unknown_pool

    split_index = build_split_index(
        dataset_root, all_word_dirs,
        data_cfg["splits"]["testing_list"], data_cfg["splits"]["validation_list"],
    )

    sample_rate = dataset_cfg["sample_rate"]
    clip_len = int(sample_rate * dataset_cfg["clip_seconds"])
    feature_extractor = FeatureExtractor(
        sample_rate=sample_rate, n_mels=data_cfg["features"]["n_mels"],
        win_length_ms=data_cfg["features"]["win_length_ms"],
        hop_length_ms=data_cfg["features"]["hop_length_ms"],
        feature_type=data_cfg["features"]["type"],
    )
    noise_dir = dataset_root / silence_cfg["background_noise_dir"]
    silence_sampler = SilenceSampler(
        noise_dir, sample_rate, dataset_cfg["clip_seconds"],
        silence_cfg["min_gain"], silence_cfg["max_gain"],
    )

    datasets = {}
    rng = random.Random(seed)
    for split in (TRAIN, VAL, TEST):
        entries = _build_entries(
            split_index, split, label_map, unknown_pool,
            unknown_cfg["target_ratio_to_avg_keyword_count"],
            silence_cfg["target_ratio_to_avg_keyword_count"], rng,
        )
        is_train = split == TRAIN and augment
        waveform_augmenter = WaveformAugmenter(noise_dir, sample_rate) if is_train else None
        spec_augmenter = SpecAugmenter() if is_train else None
        datasets[split] = SpeechCommandsKWSDataset(
            entries, dataset_root, feature_extractor, sample_rate, clip_len,
            silence_sampler, waveform_augmenter, spec_augmenter,
        )

    return datasets, label_map
