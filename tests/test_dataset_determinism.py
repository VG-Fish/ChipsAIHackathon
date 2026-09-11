import random

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
