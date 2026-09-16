"""torch Dataset for the small-keyword-set KWS task: target keywords + unknown + silence."""
import math
import random
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import Dataset

from kws.data.audio_io import load_waveform
from kws.data.augment import SpecAugmenter, WaveformAugmenter, build_augmenters
from kws.data.features import FeatureExtractor
from kws.data.silence import SilenceSampler
from kws.data.splits import TEST, TRAIN, VAL, build_split_index
from kws.data.unknown import unknown_sample_cap, unknown_words
from kws.utils.logging import get_logger

UNKNOWN_LABEL_NAME = "_unknown_"
SILENCE_LABEL_NAME = "_silence_"

logger = get_logger(__name__)


@dataclass
class Entry:
    label: int
    rel_path: str | None  # None => synthesized silence sample
    # Position in a materialized silence set; None when silence is redrawn on
    # every access (the deployment-oriented default).
    silence_index: int | None = None


def build_label_map(target_keywords: list[str]) -> dict[str, int]:
    label_map = {word: i for i, word in enumerate(target_keywords)}
    label_map[UNKNOWN_LABEL_NAME] = len(target_keywords)
    label_map[SILENCE_LABEL_NAME] = len(target_keywords) + 1
    return label_map


class SpeechCommandsKWSDataset(Dataset):
    def __init__(self, entries: list[Entry], dataset_root: Path, feature_extractor: FeatureExtractor,
                 sample_rate: int, clip_len: int, silence_sampler: SilenceSampler,
                 waveform_augmenter: WaveformAugmenter | None = None,
                 spec_augmenter: SpecAugmenter | None = None,
                 cache_features: bool = False,
                 silence_clips: list[torch.Tensor] | None = None):
        self.entries = entries
        self.dataset_root = dataset_root
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.clip_len = clip_len
        self.silence_sampler = silence_sampler
        # When present, silence was drawn once at build time (NeMo writes its
        # silence clips to disk), so evaluation sees the same waveforms on
        # every epoch and every run with the same seed.
        self.silence_clips = silence_clips
        self.waveform_augmenter = waveform_augmenter
        self.spec_augmenter = spec_augmenter
        # A list keeps the index lookup cheap.  When requested, this cache also
        # stores synthesized silence and augmented examples: the augmentation
        # recipe is sampled once during ``precompute_features`` and reused for
        # every epoch.
        self._feature_cache: list[torch.Tensor | None] | torch.Tensor | None = (
            [None] * len(entries)
            if cache_features
            else None
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _load_waveform(self, entry: Entry) -> torch.Tensor:
        if entry.rel_path is None:
            if self.silence_clips is not None and entry.silence_index is not None:
                return self.silence_clips[entry.silence_index]
            return self.silence_sampler.sample()
        waveform = load_waveform(self.dataset_root / entry.rel_path, self.sample_rate)
        if waveform.shape[-1] < self.clip_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_len - waveform.shape[-1]))
        elif waveform.shape[-1] > self.clip_len:
            waveform = waveform[:, :self.clip_len]
        return waveform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[index]
        if self._feature_cache is not None:
            if isinstance(self._feature_cache, torch.Tensor):
                return self._feature_cache[index], entry.label
            cached = self._feature_cache[index]
            if cached is not None:
                return cached, entry.label
        features = self._extract_features(entry)
        if self._feature_cache is not None:
            # A tensor cache has already been finalized and returned above;
            # this assignment only occurs while a lazy list is being filled.
            assert isinstance(self._feature_cache, list)
            self._feature_cache[index] = features.detach()
        return features, entry.label

    def _extract_features(self, entry: Entry) -> torch.Tensor:
        waveform = self._load_waveform(entry)
        if self.waveform_augmenter is not None:
            waveform = self.waveform_augmenter(waveform)
        features = self.feature_extractor(waveform)
        if self.spec_augmenter is not None:
            features = self.spec_augmenter(features)
        return features

    def precompute_features(self) -> int:
        """Populate the feature cache and return the number of new entries.

        The cache intentionally includes silence entries and augmented
        waveforms when requested.  This means the configured random recipe is
        sampled once at run startup and then held constant across epochs.
        """
        if self._feature_cache is None or isinstance(self._feature_cache, torch.Tensor):
            return 0
        started = time.monotonic()
        cached_count = 0
        with torch.no_grad():
            for index, entry in enumerate(self.entries):
                if self._feature_cache[index] is not None:
                    continue
                self._feature_cache[index] = self._extract_features(entry).detach()
                cached_count += 1
        if cached_count:
            # DataLoader workers on macOS use ``spawn`` and otherwise pickle a
            # separate copy of every cached tensor.  A shared contiguous tensor
            # keeps one CPU cache visible to all workers.  Restricted hosts may
            # reject shared memory; retaining the list is still correct and the
            # resilient loader can fall back to one process in that case.
            # A shape mismatch is a feature-extraction defect and must remain
            # visible; only the host shared-memory operation is optional.
            assert all(feature is not None for feature in self._feature_cache)
            stacked = torch.stack(cast(list[torch.Tensor], self._feature_cache), dim=0)
            try:
                stacked.share_memory_()
            except RuntimeError as error:
                logger.warning(
                    "Could not place feature cache in shared memory (%s); "
                    "keeping a private cache",
                    error,
                )
            else:
                self._feature_cache = stacked
            logger.info(
                "Precomputed %d/%d feature tensors in %.1fs",
                cached_count,
                len(self.entries),
                time.monotonic() - started,
            )
        return cached_count


def _build_entries(split_index: dict[str, list[tuple[str, str]]], split: str,
                    label_map: dict[str, int], unknown_pool: Collection[str],
                    unknown_ratio_to_avg_keyword_count: float,
                    silence_ratio_to_avg_keyword_count: float,
                    rng: random.Random, *, unknown_count: int | None = None,
                    silence_count: int | None = None) -> list[Entry]:
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
    if unknown_count is None:
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

    num_silence = (
        int(round(avg_keyword_count * silence_ratio_to_avg_keyword_count))
        if silence_count is None else silence_count
    )
    silence_label = label_map[SILENCE_LABEL_NAME]
    entries.extend(
        Entry(label=silence_label, rel_path=None, silence_index=index)
        for index in range(num_silence)
    )

    rng.shuffle(entries)
    return entries


CLASS_COUNT_ROUNDING = ("round", "ceil")


def _class_count(class_cfg: Mapping, avg_keyword_count: float, *, class_name: str) -> int:
    """Number of unknown/silence examples to draw for one split.

    ``round`` is this project's default.  ``ceil`` reproduces NeMo's
    ``process_speech_commands_data.py``, which sizes both balanced classes as
    ``int(np.ceil(0.1 * num_total_samples))`` over the ten keyword classes --
    i.e. the ceiling of the keyword-class mean.  That is where the SparkNet
    manifests' 3077/371/408 come from, so deriving them keeps the recipe
    correct for any keyword set rather than hard-coding one corpus's counts.
    """
    rounding = str(class_cfg.get("rounding", "round"))
    if rounding not in CLASS_COUNT_ROUNDING:
        raise ValueError(
            f"{class_name}.rounding must be one of {CLASS_COUNT_ROUNDING}, got {rounding!r}"
        )
    exact = avg_keyword_count * class_cfg["target_ratio_to_avg_keyword_count"]
    return math.ceil(exact) if rounding == "ceil" else int(round(exact))


def build_datasets(
    data_cfg: dict,
    augment: bool,
    seed: int = 0,
    *,
    cache_features: bool = False,
    cache_train_features: bool = False,
    augmentation: Mapping | None = None,
    splits: Collection[str] | None = None,
):
    """Build the requested train, validation, and test datasets.

    ``augmentation`` is the train config's optional block of augmentation
    options (see ``kws.data.augment.build_augmenters``); it only matters when
    ``augment`` is true. ``splits`` defaults to all three splits to preserve
    existing callers; callers that only evaluate validation can avoid building
    and precomputing the test dataset by requesting ``(TRAIN, VAL)``.
    """
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
        log_mels=data_cfg["features"].get("log_mels", False),
    )
    noise_dir = dataset_root / silence_cfg["background_noise_dir"]
    silence_sampler = SilenceSampler(
        noise_dir, sample_rate, dataset_cfg["clip_seconds"],
        silence_cfg["min_gain"], silence_cfg["max_gain"],
    )
    materialize_silence = silence_cfg.get("materialize", False)
    if not isinstance(materialize_silence, bool):
        raise ValueError("silence.materialize must be a boolean")

    requested_splits = (TRAIN, VAL, TEST) if splits is None else tuple(splits)
    invalid_splits = set(requested_splits) - {TRAIN, VAL, TEST}
    if invalid_splits:
        raise ValueError(f"unsupported dataset splits: {sorted(invalid_splits)!r}")

    datasets = {}
    rng = random.Random(seed)
    for split in requested_splits:
        split_index_entries = split_index[split]
        keyword_counts = [
            sum(1 for word, _ in split_index_entries if word == keyword)
            for keyword in target_keywords
        ]
        average_keyword_count = sum(keyword_counts) / max(len(keyword_counts), 1)
        silence_count = _class_count(
            silence_cfg, average_keyword_count, class_name="silence",
        )
        entries = _build_entries(
            split_index, split, label_map, unknown_pool,
            unknown_cfg["target_ratio_to_avg_keyword_count"],
            silence_cfg["target_ratio_to_avg_keyword_count"], rng,
            unknown_count=_class_count(
                unknown_cfg, average_keyword_count, class_name="unknown",
            ),
            silence_count=silence_count,
        )
        is_train = split == TRAIN and augment
        waveform_augmenter, spec_augmenter = (
            build_augmenters(augmentation, noise_dir, sample_rate) if is_train else (None, None)
        )
        # Validation/test are deterministic and use the general cache flag.
        # A non-augmented training split is deterministic too, so cache it
        # without requiring the augmented-training opt-in.  Augmented training
        # needs the explicit opt-in because caching its tensors trades
        # per-epoch diversity for much lower input and feature overhead.
        cache_this_split = cache_features and (
            split != TRAIN or cache_train_features or not is_train
        )
        # A string seed is hashed with SHA-512 by ``random.seed``, so unlike a
        # tuple it is stable across processes (PYTHONHASHSEED randomizes str
        # hashing).  Naming the split gives each one a disjoint silence draw.
        silence_clips = (
            silence_sampler.materialize_clips(
                silence_count, random.Random(f"{seed}:silence:{split}"),
            )
            if materialize_silence
            else None
        )
        datasets[split] = SpeechCommandsKWSDataset(
            entries, dataset_root, feature_extractor, sample_rate, clip_len,
            silence_sampler, waveform_augmenter, spec_augmenter,
            # With cache_train_features enabled, this includes one fixed
            # augmented view of every training entry (including silence).
            cache_features=cache_this_split,
            silence_clips=silence_clips,
        )

        if cache_this_split:
            datasets[split].precompute_features()

    return datasets, label_map
