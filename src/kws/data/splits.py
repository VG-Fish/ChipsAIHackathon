"""Train/val/test split assignment for Google Speech Commands v2.

Primary source of truth is Google's official testing_list.txt / validation_list.txt
(shipped inside the v0.02 tarball). The classic which_set() hash-bucketing function
is reimplemented only as a fallback for any file not present in those lists, so
speaker/utterance groupings stay stable and results remain comparable to the
literature. Never re-derive a random split anywhere else in this codebase.
"""
import hashlib
import re
from pathlib import Path

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
DEFAULT_VALIDATION_PERCENTAGE = 10
DEFAULT_TESTING_PERCENTAGE = 10

TRAIN = "training"
VAL = "validation"
TEST = "testing"


def which_set(filename: str, validation_percentage: float = DEFAULT_VALIDATION_PERCENTAGE,
              testing_percentage: float = DEFAULT_TESTING_PERCENTAGE) -> str:
    """Google's official hash-bucketing split assignment (fallback path only).

    Hashes on the base speaker/utterance id (the "_nohash_<n>" suffix is stripped)
    so that all recordings from the same speaker/utterance land in the same split.
    """
    base_name = Path(filename).name
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    percentage_hash = (
        (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
        * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    )
    if percentage_hash < validation_percentage:
        return VAL
    elif percentage_hash < (testing_percentage + validation_percentage):
        return TEST
    else:
        return TRAIN


def load_official_lists(dataset_root: Path, testing_list: str, validation_list: str) -> dict[str, str]:
    """Return {relative_path: split} for every file listed in Google's official lists.

    Paths are relative to the dataset root, e.g. "yes/0a7c2a8d_nohash_0.wav".
    """
    assignment: dict[str, str] = {}
    val_path = dataset_root / validation_list
    test_path = dataset_root / testing_list
    for path, split in [(val_path, VAL), (test_path, TEST)]:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    assignment[line] = split
    return assignment


def build_split_index(dataset_root: Path, word_dirs: list[str], testing_list: str,
                       validation_list: str) -> dict[str, list[tuple[str, str]]]:
    """Scan `word_dirs` under `dataset_root`, assign each wav to train/val/test.

    Returns {split_name: [(word, relative_path), ...]}.
    """
    official = load_official_lists(dataset_root, testing_list, validation_list)
    index: dict[str, list[tuple[str, str]]] = {TRAIN: [], VAL: [], TEST: []}
    for word in word_dirs:
        word_dir = dataset_root / word
        if not word_dir.is_dir():
            continue
        for wav_path in sorted(word_dir.glob("*.wav")):
            rel_path = f"{word}/{wav_path.name}"
            split = official.get(rel_path) or which_set(rel_path)
            index[split].append((word, rel_path))
    return index
