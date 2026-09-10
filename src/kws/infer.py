"""Single-file inference: run a trained checkpoint on an arbitrary WAV clip.

For ad-hoc/manual testing (your own recordings, edge cases) as opposed to the
aggregate test-split metrics produced by evaluate.py.
"""
import argparse

import torch
import yaml

from kws.data.audio_io import load_waveform
from kws.data.augment import crop_or_pad
from kws.data.features import build_feature_extractor
from kws.evaluate import load_model_from_checkpoint
from kws.utils.device import get_device
from kws.utils.logging import get_logger

logger = get_logger(__name__)


def predict(model, waveform: torch.Tensor, feature_extractor, device) -> torch.Tensor:
    """Returns per-class probabilities for a single (1, num_samples) waveform."""
    features = feature_extractor(waveform).unsqueeze(0).to(device)  # (1, 1, n_mels, time)
    with torch.no_grad():
        logits = model(features)
    return torch.softmax(logits, dim=1).squeeze(0).cpu()


def format_prediction(label_names: list[str], probs: torch.Tensor) -> str:
    order = torch.argsort(probs, descending=True)
    lines = [f"  {label_names[i]:>12s}: {probs[i].item()*100:5.1f}%" for i in order]
    predicted = label_names[order[0].item()]
    return f"Predicted: {predicted}\n" + "\n".join(lines)


def predict_wav_file(checkpoint_path: str, wav_path: str, data_cfg: dict) -> str:
    device = get_device()
    model, ckpt = load_model_from_checkpoint(checkpoint_path, device)
    label_names = [name for name, _ in sorted(ckpt["label_map"].items(), key=lambda kv: kv[1])]

    feature_extractor = build_feature_extractor(data_cfg)
    sample_rate = data_cfg["dataset"]["sample_rate"]
    clip_len = int(sample_rate * data_cfg["dataset"]["clip_seconds"])

    waveform = crop_or_pad(load_waveform(wav_path, sample_rate), clip_len)
    probs = predict(model, waveform, feature_extractor, device)
    return format_prediction(label_names, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav", required=True, help="Path to a WAV file to classify")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    print(predict_wav_file(args.checkpoint, args.wav, data_cfg))


if __name__ == "__main__":
    main()
