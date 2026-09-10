"""Download and extract Google Speech Commands v2 (v0.02), with MD5 verification."""
import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

import yaml

CHUNK_SIZE = 1 << 20


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract(url: str, expected_md5: str, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    tar_path = dest_root.parent / Path(url).name

    if not tar_path.exists():
        print(f"Downloading {url} -> {tar_path}")
        urllib.request.urlretrieve(url, tar_path)
    else:
        print(f"Found existing archive at {tar_path}, skipping download")

    actual_md5 = _md5(tar_path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {tar_path}: expected {expected_md5}, got {actual_md5}. "
            "This usually means a corrupted download or the wrong dataset version (v0.01 vs v0.02)."
        )
    print("MD5 verified.")

    print(f"Extracting to {dest_root}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_root, filter="data")
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/speech_commands_v2.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["dataset"]

    download_and_extract(cfg["url"], cfg["md5"], Path(cfg["root"]))


if __name__ == "__main__":
    main()
