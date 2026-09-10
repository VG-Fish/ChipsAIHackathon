from pathlib import Path

from kws.data.splits import TEST, TRAIN, VAL, build_split_index, which_set


def _make_fake_dataset(tmp_path: Path):
    for word, files in {
        "yes": ["spk1_nohash_0.wav", "spk1_nohash_1.wav", "spk2_nohash_0.wav", "spk3_nohash_0.wav"],
        "no": ["spk1_nohash_0.wav", "spk4_nohash_0.wav"],
    }.items():
        word_dir = tmp_path / word
        word_dir.mkdir()
        for fname in files:
            (word_dir / fname).write_bytes(b"")

    (tmp_path / "testing_list.txt").write_text("yes/spk2_nohash_0.wav\nno/spk4_nohash_0.wav\n")
    (tmp_path / "validation_list.txt").write_text("yes/spk3_nohash_0.wav\n")
    return tmp_path


def test_official_lists_take_priority(tmp_path):
    dataset_root = _make_fake_dataset(tmp_path)
    index = build_split_index(dataset_root, ["yes", "no"], "testing_list.txt", "validation_list.txt")

    test_paths = {p for _, p in index[TEST]}
    val_paths = {p for _, p in index[VAL]}
    assert "yes/spk2_nohash_0.wav" in test_paths
    assert "no/spk4_nohash_0.wav" in test_paths
    assert "yes/spk3_nohash_0.wav" in val_paths


def test_same_speaker_stays_in_one_split(tmp_path):
    dataset_root = _make_fake_dataset(tmp_path)
    index = build_split_index(dataset_root, ["yes", "no"], "testing_list.txt", "validation_list.txt")

    # spk1 has two "yes" utterances not in the official lists -> both fall back
    # to which_set(), which hashes on the speaker id with "_nohash_*" stripped,
    # so they must land in the same split as each other.
    spk1_files = ["yes/spk1_nohash_0.wav", "yes/spk1_nohash_1.wav"]
    splits_seen = set()
    for split_name, entries in index.items():
        paths = {p for _, p in entries}
        for f in spk1_files:
            if f in paths:
                splits_seen.add(split_name)
    assert len(splits_seen) == 1


def test_which_set_is_deterministic():
    a = which_set("yes/deadbeef_nohash_0.wav")
    b = which_set("yes/deadbeef_nohash_1.wav")
    assert a == b  # same speaker id despite different nohash suffix

    c = which_set("yes/deadbeef_nohash_0.wav")
    assert a == c  # deterministic across calls


def test_no_overlap_between_splits(tmp_path):
    dataset_root = _make_fake_dataset(tmp_path)
    index = build_split_index(dataset_root, ["yes", "no"], "testing_list.txt", "validation_list.txt")
    train_paths = {p for _, p in index[TRAIN]}
    val_paths = {p for _, p in index[VAL]}
    test_paths = {p for _, p in index[TEST]}
    assert not (train_paths & val_paths)
    assert not (train_paths & test_paths)
    assert not (val_paths & test_paths)
