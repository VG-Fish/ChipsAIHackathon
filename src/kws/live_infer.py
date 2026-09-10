"""Live microphone demo: record rolling 1-second windows and print predictions.

Closer to the eventual ESP32 use case than evaluate.py's test-split metrics or
infer.py's single-file mode -- lets you just talk and see what the model hears.
Requires `sounddevice` (records through the default input device).
"""
import argparse
import time

import numpy as np
import sounddevice as sd
import torch
import yaml

from kws.data.features import build_feature_extractor
from kws.evaluate import load_model_from_checkpoint
from kws.infer import format_prediction, predict
from kws.utils.device import get_device
from kws.utils.logging import get_logger

logger = get_logger(__name__)


def record_clip(sample_rate: int, clip_seconds: float) -> torch.Tensor:
    num_samples = int(sample_rate * clip_seconds)
    audio = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return torch.from_numpy(audio.T)  # (1, num_samples)


def run_live_loop(checkpoint_path: str, data_cfg: dict, interval_s: float) -> None:
    device = get_device()
    model, ckpt = load_model_from_checkpoint(checkpoint_path, device)
    label_names = [name for name, _ in sorted(ckpt["label_map"].items(), key=lambda kv: kv[1])]
    feature_extractor = build_feature_extractor(data_cfg)

    sample_rate = data_cfg["dataset"]["sample_rate"]
    clip_seconds = data_cfg["dataset"]["clip_seconds"]

    print(f"Listening on the default microphone (Ctrl+C to stop). Say one of: "
          f"{data_cfg['target_keywords']}\n")
    try:
        while True:
            waveform = record_clip(sample_rate, clip_seconds)
            probs = predict(model, waveform, feature_extractor, device)
            print(format_prediction(label_names, probs))
            print("-" * 40)
            time.sleep(max(interval_s - clip_seconds, 0))
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--interval", type=float, default=1.5,
                         help="Seconds between the start of each recording window")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    run_live_loop(args.checkpoint, data_cfg, args.interval)


if __name__ == "__main__":
    main()
