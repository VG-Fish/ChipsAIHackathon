import torch
from typing import Any, cast

from kws.data.dataset import Entry, SpeechCommandsKWSDataset, build_datasets
from kws.data.splits import TEST, TRAIN, VAL


class _Extractor:
    def __init__(self):
        self.calls = 0

    def __call__(self, waveform):
        self.calls += 1
        return waveform.unsqueeze(0)


class _Silence:
    def __init__(self):
        self.calls = 0

    def sample(self):
        self.calls += 1
        return torch.ones(1, 2)


def test_precomputed_features_skip_repeated_deterministic_extraction(monkeypatch, tmp_path):
    extractor = _Extractor()
    silence = _Silence()
    dataset = SpeechCommandsKWSDataset(
        [Entry(0, "yes/example.wav"), Entry(1, None)],
        tmp_path,
        cast(Any, extractor),
        sample_rate=16_000,
        clip_len=2,
        silence_sampler=cast(Any, silence),
        cache_features=True,
    )
    waveform = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(
        "kws.data.dataset.load_waveform", lambda *_args: waveform.clone()
    )

    assert dataset.precompute_features() == 2
    assert extractor.calls == 2
    first, _ = dataset[0]
    second, _ = dataset[0]
    assert torch.equal(first, second)
    assert extractor.calls == 2

    # Synthesized silence is cached too when this dataset opts into fixed
    # augmented views, so repeated accesses do not ask the sampler again.
    dataset[1]
    dataset[1]
    assert silence.calls == 1


def test_feature_cache_reuses_one_augmented_view_including_silence(monkeypatch, tmp_path):
    extractor = _Extractor()
    silence = _Silence()

    class _Augmenter:
        calls = 0

        def __call__(self, waveform):
            self.calls += 1
            return waveform + self.calls

    augmenter = _Augmenter()
    dataset = SpeechCommandsKWSDataset(
        [Entry(0, "yes/example.wav"), Entry(1, None)],
        tmp_path,
        cast(Any, extractor),
        sample_rate=16_000,
        clip_len=2,
        silence_sampler=cast(Any, silence),
        waveform_augmenter=cast(Any, augmenter),
        cache_features=True,
    )
    waveform = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(
        "kws.data.dataset.load_waveform", lambda *_args: waveform.clone()
    )

    assert dataset.precompute_features() == 2
    assert extractor.calls == 2
    assert augmenter.calls == 2
    assert silence.calls == 1
    first = dataset[0][0].clone()
    second = dataset[1][0].clone()
    assert torch.equal(first, dataset[0][0])
    assert torch.equal(second, dataset[1][0])
    assert extractor.calls == 2
    assert augmenter.calls == 2
    assert silence.calls == 1


def test_build_datasets_caches_non_augmented_training_split(monkeypatch, tmp_path):
    """The general cache flag also covers deterministic training data."""
    seen_cache_flags = []

    class _Dataset:
        def __init__(self, *_args, cache_features=False, **_kwargs):
            seen_cache_flags.append(cache_features)

        def precompute_features(self):
            return 1

    monkeypatch.setattr("kws.data.dataset.SpeechCommandsKWSDataset", _Dataset)
    monkeypatch.setattr(
        "kws.data.dataset.FeatureExtractor", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        "kws.data.dataset.SilenceSampler", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "kws.data.dataset.build_split_index",
        lambda *_args, **_kwargs: {
            TRAIN: [("yes", "yes/example.wav")],
            VAL: [("yes", "yes/example.wav")],
            TEST: [("yes", "yes/example.wav")],
        },
    )

    config = {
        "dataset": {
            "root": str(tmp_path),
            "sample_rate": 16_000,
            "clip_seconds": 1.0,
        },
        "target_keywords": ["yes"],
        "unknown": {"target_ratio_to_avg_keyword_count": 0.0},
        "silence": {
            "background_noise_dir": "_background_noise_",
            "target_ratio_to_avg_keyword_count": 0.0,
            "min_gain": 0.1,
            "max_gain": 1.0,
        },
        "features": {
            "type": "logmel",
            "n_mels": 40,
            "win_length_ms": 30,
            "hop_length_ms": 10,
        },
        "splits": {"testing_list": "testing.txt", "validation_list": "validation.txt"},
    }

    build_datasets(config, augment=False, cache_features=True)

    assert seen_cache_flags == [True, True, True]
