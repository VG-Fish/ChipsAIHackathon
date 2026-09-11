"""The pooled "unknown" class: every Speech Commands v2 word that isn't a target keyword.

Because only a handful of the 35 words are used as target-keyword classes, the
remaining words are a natural, in-domain "unknown" pool -- no external corpus
(and its domain-mismatch risk) is needed.
"""

# All 35 words in Google Speech Commands v2.
ALL_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual",
]


def unknown_words(target_keywords: list[str]) -> list[str]:
    target_set = set(target_keywords)
    unknown = [w for w in ALL_WORDS if w not in target_set]
    missing = target_set - set(ALL_WORDS)
    if missing:
        raise ValueError(f"target_keywords contains words not in Speech Commands v2: {missing}")
    return unknown


def unknown_sample_cap(avg_keyword_count: float, target_ratio_to_avg_keyword_count: float) -> int:
    """Number of unknown samples to draw relative to the keyword-class mean.

    The leftover-word pool has far more raw samples than any single keyword, so
    it is deliberately subsampled to the configured target count.
    """
    return int(round(avg_keyword_count * target_ratio_to_avg_keyword_count))
