"""Measure paired teacher views and KD gradients without changing a training run."""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from kws.data.dataset import build_datasets
from kws.data.splits import TRAIN, VAL
from kws.models.registry import build_model
from kws.optimize.kd import FrozenTeacher
from kws.utils.seed import set_seed


def summarize(logits: torch.Tensor, labels: torch.Tensor, label_map: dict[str, int]) -> dict:
    probs = logits.softmax(1)
    correct = probs.argmax(1).eq(labels)
    target = F.one_hot(labels, probs.shape[1]) * 0.9 + 0.1 / probs.shape[1]
    return {
        "samples": len(labels),
        "accuracy": correct.float().mean().item(),
        "mean_confidence": probs.max(1).values.mean().item(),
        "mean_true_class_probability": probs.gather(1, labels[:, None]).mean().item(),
        "mean_entropy_nats": -(probs * probs.clamp_min(1e-30).log()).sum(1).mean().item(),
        "mean_l1_from_smoothed_labels": (probs - target).abs().sum(1).mean().item(),
        "old_kd_mean_target_total_variation": (probs - target).abs().sum(1).mean().item() / 14,
        "classes": {
            name: {
                "samples": int(labels.eq(index).sum()),
                "accuracy": correct[labels.eq(index)].float().mean().item(),
                "mean_probabilities": probs[labels.eq(index)].mean(0).tolist(),
            }
            for name, index in label_map.items() if labels.eq(index).any()
        },
    }


def gradient_summary(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     labels: torch.Tensor, temperature: float, alpha: float,
                     smoothing: float) -> dict:
    target = F.one_hot(labels, student_logits.shape[1]) * (1 - smoothing) + smoothing / student_logits.shape[1]
    ce = student_logits.softmax(1) - target
    kd = temperature * ((student_logits / temperature).softmax(1) - (teacher_logits / temperature).softmax(1))
    ce_weighted = (1 - alpha) * ce
    kd_weighted = alpha * kd
    return {
        "temperature": temperature, "response_weight": alpha, "label_smoothing": smoothing,
        "weighted_kd_to_ce_norm_ratio": (kd_weighted.norm() / ce_weighted.norm().clamp_min(1e-12)).item(),
        "ce_kd_cosine": F.cosine_similarity(ce.flatten(), kd.flatten(), dim=0).item(),
        "fraction_conflicting_samples": F.cosine_similarity(ce, kd, dim=1).lt(0).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", default="models/checkpoints/ds_cnn_l_12class.pt")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/light_kd.yaml")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    torch.set_num_threads(2)
    set_seed(args.seed)
    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    train_cfg = yaml.safe_load(Path(args.train_config).read_text())
    datasets, label_map = build_datasets(data_cfg, augment=True, seed=0, cache_features=False,
                                         augmentation=train_cfg.get("augmentation"))
    teacher = FrozenTeacher(args.teacher_checkpoint, torch.device("cpu"))
    if teacher.label_map != label_map:
        raise ValueError("Teacher class ordering does not match data")
    # Take a single immutable snapshot even when best.pt is replaced mid-audit.
    checkpoint_bytes = Path(args.student_checkpoint).read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    student = build_model(checkpoint["model_cfg"], teacher.input_shape, teacher.num_classes)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    all_logits = {}
    report = {"seed": args.seed, "dataset_seed": 0, "samples_per_split": args.samples,
              "teacher_sha256": teacher.checkpoint_sha256, "student_checkpoint": args.student_checkpoint,
              "student_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
              "student_recorded_val_acc": checkpoint.get("val_acc"), "model_mode": "eval",
              "torch_version": str(torch.__version__), "data_config": data_cfg, "train_config": train_cfg,
              "label_map": label_map, "views": {}}
    with torch.inference_mode():
        for split in (TRAIN, VAL):
            dataset = datasets[split]
            indices = random.Random(args.seed).sample(range(len(dataset)), min(args.samples, len(dataset)))
            clean_logits, augmented_logits, student_logits, labels = [], [], [], []
            for start in range(0, len(indices), 32):
                clean, augmented = [], []
                for index in indices[start:start + 32]:
                    entry = dataset.entries[index]
                    waveform = dataset._load_waveform(entry)
                    clean.append(dataset.feature_extractor(waveform))
                    if dataset.waveform_augmenter is not None:
                        waveform = dataset.waveform_augmenter(waveform)
                    features = dataset.feature_extractor(waveform)
                    if dataset.spec_augmenter is not None:
                        features = dataset.spec_augmenter(features)
                    augmented.append(features)
                    labels.append(entry.label)
                clean_tensor = torch.stack(clean)
                augmented_tensor = torch.stack(augmented)
                clean_logits.append(teacher(clean_tensor)[0])
                augmented_logits.append(teacher(augmented_tensor)[0] if split == TRAIN else clean_logits[-1])
                student_logits.append(student(augmented_tensor))
                if start % 256 == 0:
                    print(f"{split}: {min(start + 32, len(indices))}/{len(indices)}", flush=True)
            clean_t, augmented_t, student_t = map(torch.cat, (clean_logits, augmented_logits, student_logits))
            labels_t = torch.tensor(labels)
            all_logits[split] = {"clean": clean_t, "augmented": augmented_t, "student": student_t,
                                 "labels": labels_t, "indices": indices}
            report["views"][split] = {
                "teacher_clean": summarize(clean_t, labels_t, label_map),
                "teacher_augmented": summarize(augmented_t, labels_t, label_map),
                "student_augmented": summarize(student_t, labels_t, label_map),
                "teacher_prediction_flip_rate": clean_t.argmax(1).ne(augmented_t.argmax(1)).float().mean().item(),
                "gradients": [gradient_summary(student_t, augmented_t, labels_t, temp, alpha, smoothing)
                              for temp, alpha, smoothing in ((1, 1/7, .1), (1, .5, .1), (2, .5, .1), (2, .5, 0), (4, .5, 0))],
            }
            print(json.dumps({split: {name: {k: v for k, v in stats.items() if k != "classes"}
                                           for name, stats in report["views"][split].items()
                                           if isinstance(stats, dict)}}), flush=True)
            print(json.dumps(report["views"][split]["gradients"]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "teacher_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(all_logits, args.output_dir / "paired_logits.pt")


if __name__ == "__main__":
    main()
