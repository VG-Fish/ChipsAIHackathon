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
            max_ratio_to_avg_keyword_count=2.0,
            silence_fraction_of_epoch=0.0,
            rng=random.Random(7),
        )
        return [(entry.label, entry.rel_path) for entry in entries]

    assert paths_for(["alpha", "beta"]) == paths_for(["beta", "alpha"])
