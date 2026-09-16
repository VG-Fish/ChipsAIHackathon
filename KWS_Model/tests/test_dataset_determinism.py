import random

import pytest

from kws.data.dataset import _build_entries, build_label_map
from kws.data.splits import TEST, TRAIN, VAL


def test_unknown_sampling_is_independent_of_pool_order():
    label_map = build_label_map(["yes", "no"])
    split_index = {
        TRAIN: [],
        VAL: [],
        TEST: [
            ("yes", "yes/0.wav"),
            ("no", "no/0.wav"),
            ("alpha", "alpha/0.wav"),
            ("alpha", "alpha/1.wav"),
            ("beta", "beta/0.wav"),
            ("beta", "beta/1.wav"),
        ],
    }

    def paths_for(pool):
        entries = _build_entries(
            split_index,
            TEST,
            label_map,
            pool,
            unknown_ratio_to_avg_keyword_count=2.0,
            silence_ratio_to_avg_keyword_count=0.0,
            rng=random.Random(7),
        )
        return [(entry.label, entry.rel_path) for entry in entries]

    assert paths_for(["alpha", "beta"]) == paths_for(["beta", "alpha"])


def test_unknown_and_silence_are_twice_the_keyword_mean():
    keywords = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    label_map = build_label_map(keywords)
    split_index = {TRAIN: [], VAL: [], TEST: []}
    for keyword in keywords:
        split_index[TEST].extend((keyword, f"{keyword}/{i}.wav") for i in range(2))
    split_index[TEST].extend(("other", f"other/{i}.wav") for i in range(4))

    entries = _build_entries(
        split_index,
        TEST,
        label_map,
        ["other"],
        unknown_ratio_to_avg_keyword_count=2.0,
        silence_ratio_to_avg_keyword_count=2.0,
        rng=random.Random(7),
    )

    counts = {label: 0 for label in label_map.values()}
    for entry in entries:
        counts[entry.label] += 1

    keyword_count = 2
    assert len(label_map) == 12
    assert counts[label_map["_unknown_"]] == 4
    assert counts[label_map["_silence_"]] == 4
    assert all(counts[label_map[keyword]] == keyword_count for keyword in keywords)


def test_ceil_rounding_reproduces_the_nemo_balanced_manifest_counts():
    """NeMo sizes _unknown_/_silence_ as ceil of the keyword-class mean.

    The v0.02 keyword means are 3076.9 / 370.3 / 407.4 for train / val / test,
    which ceil to the 3077 / 371 / 408 the SparkNet manifests contain; plain
    rounding would give 3077 / 370 / 407.
    """
    from kws.data.dataset import _class_count

    ceil_cfg = {"target_ratio_to_avg_keyword_count": 1.0, "rounding": "ceil"}
    round_cfg = {"target_ratio_to_avg_keyword_count": 1.0}

    means = {"training": 3076.9, "validation": 370.3, "testing": 407.4}
    assert [
        _class_count(ceil_cfg, means[split], class_name="unknown")
        for split in ("training", "validation", "testing")
    ] == [3077, 371, 408]
    assert [
        _class_count(round_cfg, means[split], class_name="unknown")
        for split in ("training", "validation", "testing")
    ] == [3077, 370, 407]


def test_class_count_scales_with_the_configured_ratio():
    from kws.data.dataset import _class_count

    cfg = {"target_ratio_to_avg_keyword_count": 2.0, "rounding": "ceil"}
    assert _class_count(cfg, 370.3, class_name="silence") == 741


def test_unknown_class_rejects_an_unsupported_rounding_mode():
    from kws.data.dataset import _class_count

    with pytest.raises(ValueError, match="unknown.rounding"):
        _class_count(
            {"target_ratio_to_avg_keyword_count": 1.0, "rounding": "floor"},
            370.3, class_name="unknown",
        )
