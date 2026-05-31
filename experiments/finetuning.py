#!/usr/bin/env python3
"""Fine-tune RoBERTa on text classification datasets with holdout + CV.

Datasets are expected under resources/datasets/<dataset-name> with parallel
files: texts.txt for examples and score.txt for integer labels. Curriculum
learning can additionally use dificulty.txt or difficulty.txt.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from transformers import (
        AutoTokenizer,
        RobertaForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    TRAINING_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    np = None
    torch = None
    accuracy_score = None
    f1_score = None
    precision_score = None
    recall_score = None
    StratifiedKFold = None
    train_test_split = None
    DataLoader = None
    Dataset = object
    tqdm = None
    AutoTokenizer = None
    RobertaForSequenceClassification = None
    get_linear_schedule_with_warmup = None
    TRAINING_IMPORT_ERROR = error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "resources" / "datasets"
DEFAULT_DATASET_NAME = "subj"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DIFFICULTY_ORDER = {
    "easy": 0.0,
    "medium": 1.0,
    "hard": 2.0,
}


@dataclass
class DifficultyValue:
    label: str
    rank: float


def no_grad(func):
    if torch is None:
        return func
    return torch.no_grad()(func)


@dataclass
class EpochMetrics:
    split: str
    fold: int
    epoch: int
    train_loss: float
    eval_loss: float
    train_precision: float
    train_accuracy: float
    train_recall: float
    train_f1: float
    eval_precision: float
    eval_accuracy: float
    eval_recall: float
    eval_f1: float
    epoch_time_seconds: float
    accumulated_time_seconds: float


@dataclass
class BatchMetrics:
    split: str
    fold: int
    epoch: int
    batch: int
    batch_size: int
    batch_time_seconds: float
    epoch_accumulated_time_seconds: float
    total_accumulated_time_seconds: float
    loss: float
    precision: float
    accuracy: float
    recall: float
    f1: float
    learning_rate: float
    difficulty_mean: Optional[float]
    difficulty_min: Optional[float]
    difficulty_max: Optional[float]
    difficulty_labels: Optional[str]


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: AutoTokenizer,
        max_length: int,
        difficulties: Optional[Sequence[DifficultyValue]] = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulties = list(difficulties) if difficulties is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        if self.difficulties is not None:
            item["difficulty_rank"] = torch.tensor(self.difficulties[index].rank, dtype=torch.float)
        return item


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RoBERTa with 80/20 holdout and cross-validation."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset folder name inside resources/datasets. Example: subj, mr, vader_movie_2L.",
    )
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional explicit dataset directory. Overrides --datasets-root/--dataset-name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to outputs/finetuning_<dataset-name>_roberta.",
    )
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--metrics-average",
        default="macro",
        choices=["binary", "micro", "macro", "weighted"],
        help="Averaging strategy for precision, recall, and F1.",
    )
    parser.add_argument(
        "-cl",
        "--curriculum-learning",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Enable curriculum learning. Accepts -cl, --curriculum-learning, or --curriculum-learning=True.",
    )
    parser.add_argument(
        "--difficulty-column",
        default="dificulty",
        help="Difficulty column/file stem used for curriculum learning. Also accepts difficulty as fallback.",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Run cross-validation only; skip final train-on-80-percent/test-on-20-percent stage.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets with texts.txt and score.txt, then exit.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_available_datasets(datasets_root: Path) -> List[str]:
    if not datasets_root.exists():
        return []
    return sorted(
        path.name
        for path in datasets_root.iterdir()
        if path.is_dir() and (path / "texts.txt").is_file() and (path / "score.txt").is_file()
    )


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir
    return args.datasets_root / args.dataset_name


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return DEFAULT_OUTPUT_ROOT / f"finetuning_{args.dataset_name}_roberta"


def find_difficulty_path(dataset_dir: Path, difficulty_column: str) -> Optional[Path]:
    candidates = [
        dataset_dir / f"{difficulty_column}.txt",
        dataset_dir / "dificulty.txt",
        dataset_dir / "difficulty.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_tabular_dataset_path(dataset_dir: Path) -> Optional[Path]:
    for filename in ("dataset.csv", "data.csv", "dataset.tsv", "data.tsv"):
        path = dataset_dir / filename
        if path.is_file():
            return path
    return None


def parse_difficulty(value: str) -> DifficultyValue:
    label = value.strip()
    normalized = label.lower()
    if normalized in DIFFICULTY_ORDER:
        return DifficultyValue(label=normalized, rank=DIFFICULTY_ORDER[normalized])
    try:
        rank = float(normalized)
    except ValueError as exc:
        valid = ", ".join(DIFFICULTY_ORDER)
        raise ValueError(
            f"Unknown difficulty value {value!r}. Expected one of {valid}, or a numeric value."
        ) from exc
    return DifficultyValue(label=label, rank=rank)


def format_difficulty_rank(rank: float) -> str:
    for label, label_rank in DIFFICULTY_ORDER.items():
        if abs(rank - label_rank) < 1e-8:
            return label
    return f"{rank:g}"


def read_tabular_dataset(
    dataset_path: Path,
    difficulty_column: str,
) -> Tuple[List[str], List[int], Optional[List[DifficultyValue]], Dict[int, int]]:
    delimiter = "\t" if dataset_path.suffix == ".tsv" else ","
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {dataset_path}.")

        text_column = "text" if "text" in reader.fieldnames else "texts"
        label_column = "score" if "score" in reader.fieldnames else "label"
        difficulty_names = [difficulty_column, "dificulty", "difficulty"]
        selected_difficulty_column = next(
            (name for name in difficulty_names if name in reader.fieldnames),
            None,
        )
        if text_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(
                f"{dataset_path} must contain text/texts and score/label columns. "
                f"Found: {reader.fieldnames}"
            )

        texts: List[str] = []
        raw_labels: List[int] = []
        raw_difficulties: Optional[List[DifficultyValue]] = [] if selected_difficulty_column else None
        for row in reader:
            texts.append(row[text_column])
            raw_labels.append(int(row[label_column]))
            if raw_difficulties is not None and selected_difficulty_column is not None:
                raw_difficulties.append(parse_difficulty(row[selected_difficulty_column]))

    label_to_id = {label: index for index, label in enumerate(sorted(set(raw_labels)))}
    labels = [label_to_id[label] for label in raw_labels]
    return texts, labels, raw_difficulties, label_to_id


def load_text_classification_dataset(
    dataset_dir: Path,
    difficulty_column: str,
) -> Tuple[List[str], List[int], Optional[List[DifficultyValue]], Dict[int, int]]:
    tabular_dataset_path = find_tabular_dataset_path(dataset_dir)
    if tabular_dataset_path is not None:
        return read_tabular_dataset(tabular_dataset_path, difficulty_column)

    texts_path = dataset_dir / "texts.txt"
    labels_path = dataset_dir / "score.txt"
    difficulty_path = find_difficulty_path(dataset_dir, difficulty_column)

    texts = texts_path.read_text(encoding="utf-8").splitlines()
    raw_labels = [int(line.strip()) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    difficulties = (
        [parse_difficulty(line) for line in difficulty_path.read_text(encoding="utf-8").splitlines()]
        if difficulty_path is not None
        else None
    )

    if len(texts) != len(raw_labels):
        raise ValueError(
            f"Mismatched dataset lengths: {texts_path} has {len(texts)} rows, "
            f"{labels_path} has {len(raw_labels)} rows."
        )
    if difficulties is not None and len(texts) != len(difficulties):
        raise ValueError(
            f"Mismatched dataset lengths: {texts_path} has {len(texts)} rows, "
            f"{difficulty_path} has {len(difficulties)} rows."
        )
    if not texts:
        raise ValueError(f"No examples found in {dataset_dir}.")

    label_to_id = {label: index for index, label in enumerate(sorted(set(raw_labels)))}
    labels = [label_to_id[label] for label in raw_labels]
    return texts, labels, difficulties, label_to_id


def build_loader(
    texts: Sequence[str],
    labels: Sequence[int],
    difficulties: Optional[Sequence[DifficultyValue]],
    indices: Sequence[int],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    curriculum_learning: bool = False,
) -> DataLoader:
    ordered_indices = list(indices)
    if curriculum_learning:
        if difficulties is None:
            raise ValueError(
                "Curriculum learning requires a dificulty/difficulty column or sidecar file."
            )
        ordered_indices = sorted(ordered_indices, key=lambda index: difficulties[index].rank)

    subset_texts = [texts[index] for index in ordered_indices]
    subset_labels = [labels[index] for index in ordered_indices]
    subset_difficulties = (
        [difficulties[index] for index in ordered_indices] if difficulties is not None else None
    )
    dataset = TextClassificationDataset(
        subset_texts,
        subset_labels,
        tokenizer,
        max_length,
        subset_difficulties,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not curriculum_learning,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(model_name: str, num_labels: int, device: torch.device) -> RobertaForSequenceClassification:
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model.to(device)


def classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    average: str,
) -> Tuple[float, float, float, float]:
    return (
        precision_score(targets, predictions, average=average, zero_division=0),
        accuracy_score(targets, predictions),
        recall_score(targets, predictions, average=average, zero_division=0),
        f1_score(targets, predictions, average=average, zero_division=0),
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(
    *,
    model: RobertaForSequenceClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    split_name: str,
    fold: int,
    epoch: int,
    metrics_average: str,
    run_start_time: float,
) -> Tuple[float, float, float, float, float, List[BatchMetrics]]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    predictions: List[int] = []
    targets: List[int] = []
    batch_metrics: List[BatchMetrics] = []
    epoch_start_time = time.perf_counter()

    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        synchronize_device(device)
        batch_start_time = time.perf_counter()
        batch = {key: value.to(device) for key, value in batch.items()}
        difficulty_ranks = batch.pop("difficulty_rank", None)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        batch_predictions = torch.argmax(outputs.logits, dim=-1)
        batch_targets = batch["labels"]
        batch_prediction_list = batch_predictions.detach().cpu().tolist()
        batch_target_list = batch_targets.detach().cpu().tolist()
        batch_precision, batch_accuracy, batch_recall, batch_f1 = classification_metrics(
            batch_target_list,
            batch_prediction_list,
            metrics_average,
        )
        synchronize_device(device)
        batch_time = time.perf_counter() - batch_start_time
        epoch_accumulated_time = time.perf_counter() - epoch_start_time
        total_accumulated_time = time.perf_counter() - run_start_time
        difficulty_mean = difficulty_ranks.mean().item() if difficulty_ranks is not None else None
        difficulty_min = difficulty_ranks.min().item() if difficulty_ranks is not None else None
        difficulty_max = difficulty_ranks.max().item() if difficulty_ranks is not None else None
        difficulty_labels = None
        if difficulty_ranks is not None:
            labels_in_batch = [
                format_difficulty_rank(rank) for rank in difficulty_ranks.detach().cpu().tolist()
            ]
            difficulty_labels = "|".join(sorted(set(labels_in_batch), key=labels_in_batch.index))

        batch_size = batch_targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        predictions.extend(batch_prediction_list)
        targets.extend(batch_target_list)
        batch_metrics.append(
            BatchMetrics(
                split=split_name,
                fold=fold,
                epoch=epoch,
                batch=batch_index,
                batch_size=batch_size,
                batch_time_seconds=batch_time,
                epoch_accumulated_time_seconds=epoch_accumulated_time,
                total_accumulated_time_seconds=total_accumulated_time,
                loss=loss.item(),
                precision=batch_precision,
                accuracy=batch_accuracy,
                recall=batch_recall,
                f1=batch_f1,
                learning_rate=scheduler.get_last_lr()[0],
                difficulty_mean=difficulty_mean,
                difficulty_min=difficulty_min,
                difficulty_max=difficulty_max,
                difficulty_labels=difficulty_labels,
            )
        )

    train_loss = total_loss / max(total_examples, 1)
    train_precision, train_accuracy, train_recall, train_f1 = classification_metrics(
        targets,
        predictions,
        metrics_average,
    )
    return train_loss, train_precision, train_accuracy, train_recall, train_f1, batch_metrics


@no_grad
def evaluate(
    model: RobertaForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    metrics_average: str,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: List[int] = []
    targets: List[int] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        batch.pop("difficulty_rank", None)
        outputs = model(**batch)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1)

        batch_size = batch["labels"].size(0)
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size
        predictions.extend(batch_predictions.cpu().tolist())
        targets.extend(batch["labels"].cpu().tolist())

    eval_loss = total_loss / max(total_examples, 1)
    precision, accuracy, recall, f1 = classification_metrics(
        targets,
        predictions,
        metrics_average,
    )
    return eval_loss, precision, accuracy, recall, f1


def run_training(
    *,
    split_name: str,
    fold: int,
    texts: Sequence[str],
    labels: Sequence[int],
    difficulties: Optional[Sequence[DifficultyValue]],
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    num_labels: int,
    device: torch.device,
    output_dir: Path,
) -> Tuple[List[EpochMetrics], List[BatchMetrics]]:
    train_loader = build_loader(
        texts,
        labels,
        difficulties,
        train_indices,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        curriculum_learning=args.curriculum_learning,
    )
    eval_loader = build_loader(
        texts,
        labels,
        difficulties,
        eval_indices,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_name, num_labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    metrics: List[EpochMetrics] = []
    all_batch_metrics: List[BatchMetrics] = []
    run_start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"\n[{split_name} fold={fold}] epoch {epoch}/{args.epochs}")
        epoch_start_time = time.perf_counter()
        train_loss, train_precision, train_accuracy, train_recall, train_f1, batch_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                split_name=split_name,
                fold=fold,
                epoch=epoch,
                metrics_average=args.metrics_average,
                run_start_time=run_start_time,
            )
        )
        eval_loss, eval_precision, eval_accuracy, eval_recall, eval_f1 = evaluate(
            model,
            eval_loader,
            device,
            args.metrics_average,
        )
        epoch_time = time.perf_counter() - epoch_start_time
        accumulated_time = time.perf_counter() - run_start_time
        row = EpochMetrics(
            split_name,
            fold,
            epoch,
            train_loss,
            eval_loss,
            train_precision,
            train_accuracy,
            train_recall,
            train_f1,
            eval_precision,
            eval_accuracy,
            eval_recall,
            eval_f1,
            epoch_time,
            accumulated_time,
        )
        metrics.append(row)
        all_batch_metrics.extend(batch_metrics)
        append_metrics(output_dir / "metrics.csv", [row])
        append_batch_metrics(output_dir / "batch_metrics.csv", batch_metrics)
        print(
            f"loss train={train_loss:.4f} eval={eval_loss:.4f} "
            f"train_f1={train_f1:.4f} eval_f1={eval_f1:.4f} "
            f"eval_acc={eval_accuracy:.4f}"
        )

    return metrics, all_batch_metrics


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "fold",
                "epoch",
                "train_loss",
                "eval_loss",
                "train_precision",
                "train_accuracy",
                "train_recall",
                "train_f1",
                "eval_precision",
                "eval_accuracy",
                "eval_recall",
                "eval_f1",
                "epoch_time_seconds",
                "accumulated_time_seconds",
            ],
        )
        writer.writeheader()


def append_metrics(path: Path, metrics: Iterable[EpochMetrics]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "fold",
                "epoch",
                "train_loss",
                "eval_loss",
                "train_precision",
                "train_accuracy",
                "train_recall",
                "train_f1",
                "eval_precision",
                "eval_accuracy",
                "eval_recall",
                "eval_f1",
                "epoch_time_seconds",
                "accumulated_time_seconds",
            ],
        )
        for row in metrics:
            writer.writerow(
                {
                    "split": row.split,
                    "fold": row.fold,
                    "epoch": row.epoch,
                    "train_loss": row.train_loss,
                    "eval_loss": row.eval_loss,
                    "train_precision": row.train_precision,
                    "train_accuracy": row.train_accuracy,
                    "train_recall": row.train_recall,
                    "train_f1": row.train_f1,
                    "eval_precision": row.eval_precision,
                    "eval_accuracy": row.eval_accuracy,
                    "eval_recall": row.eval_recall,
                    "eval_f1": row.eval_f1,
                    "epoch_time_seconds": row.epoch_time_seconds,
                    "accumulated_time_seconds": row.accumulated_time_seconds,
                }
            )


def write_batch_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "fold",
                "epoch",
                "batch",
                "batch_size",
                "batch_time_seconds",
                "epoch_accumulated_time_seconds",
                "total_accumulated_time_seconds",
                "loss",
                "precision",
                "accuracy",
                "recall",
                "f1",
                "learning_rate",
                "difficulty_mean",
                "difficulty_min",
                "difficulty_max",
                "difficulty_labels",
            ],
        )
        writer.writeheader()


def append_batch_metrics(path: Path, metrics: Iterable[BatchMetrics]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "fold",
                "epoch",
                "batch",
                "batch_size",
                "batch_time_seconds",
                "epoch_accumulated_time_seconds",
                "total_accumulated_time_seconds",
                "loss",
                "precision",
                "accuracy",
                "recall",
                "f1",
                "learning_rate",
                "difficulty_mean",
                "difficulty_min",
                "difficulty_max",
                "difficulty_labels",
            ],
        )
        for row in metrics:
            writer.writerow(
                {
                    "split": row.split,
                    "fold": row.fold,
                    "epoch": row.epoch,
                    "batch": row.batch,
                    "batch_size": row.batch_size,
                    "batch_time_seconds": row.batch_time_seconds,
                    "epoch_accumulated_time_seconds": row.epoch_accumulated_time_seconds,
                    "total_accumulated_time_seconds": row.total_accumulated_time_seconds,
                    "loss": row.loss,
                    "precision": row.precision,
                    "accuracy": row.accuracy,
                    "recall": row.recall,
                    "f1": row.f1,
                    "learning_rate": row.learning_rate,
                    "difficulty_mean": row.difficulty_mean,
                    "difficulty_min": row.difficulty_min,
                    "difficulty_max": row.difficulty_max,
                    "difficulty_labels": row.difficulty_labels,
                }
            )


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        datasets = list_available_datasets(args.datasets_root)
        print("\n".join(datasets) if datasets else f"No datasets found in {args.datasets_root}")
        return
    if TRAINING_IMPORT_ERROR is not None:
        raise SystemExit(
            f"Missing training dependency: {TRAINING_IMPORT_ERROR.name}. "
            "Install the project requirements before running fine-tuning."
        )

    set_seed(args.seed)

    dataset_dir = resolve_dataset_dir(args)
    output_dir = resolve_output_dir(args)
    texts, labels, difficulties, label_to_id = load_text_classification_dataset(
        dataset_dir,
        args.difficulty_column,
    )
    if args.curriculum_learning and difficulties is None:
        raise ValueError(
            "Curriculum learning is active, but no dificulty/difficulty column was found. "
            "Add dificulty.txt, difficulty.txt, or a tabular dataset with that column."
        )
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {len(texts)} examples from {dataset_dir}")
    print(f"Label mapping: {label_to_id}")
    print(f"Curriculum learning: {args.curriculum_learning}")
    print(f"Using {device} with model {args.model_name}")

    all_indices = np.arange(len(labels))
    train_pool_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    train_pool_labels = np.array(labels)[train_pool_indices]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    all_metrics: List[EpochMetrics] = []
    all_batch_metrics: List[BatchMetrics] = []
    metrics_path = output_dir / "metrics.csv"
    batch_metrics_path = output_dir / "batch_metrics.csv"
    write_metrics_header(metrics_path)
    write_batch_metrics_header(batch_metrics_path)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_position, val_position) in enumerate(
        cv.split(train_pool_indices, train_pool_labels),
        start=1,
    ):
        fold_train_indices = train_pool_indices[train_position].tolist()
        fold_val_indices = train_pool_indices[val_position].tolist()
        epoch_metrics, batch_metrics = run_training(
            split_name="cv",
            fold=fold,
            texts=texts,
            labels=labels,
            difficulties=difficulties,
            train_indices=fold_train_indices,
            eval_indices=fold_val_indices,
            tokenizer=tokenizer,
            args=args,
            num_labels=num_labels,
            device=device,
            output_dir=output_dir,
        )
        all_metrics.extend(epoch_metrics)
        all_batch_metrics.extend(batch_metrics)

    if not args.skip_final_test:
        epoch_metrics, batch_metrics = run_training(
            split_name="test",
            fold=0,
            texts=texts,
            labels=labels,
            difficulties=difficulties,
            train_indices=train_pool_indices.tolist(),
            eval_indices=test_indices.tolist(),
            tokenizer=tokenizer,
            args=args,
            num_labels=num_labels,
            device=device,
            output_dir=output_dir,
        )
        all_metrics.extend(epoch_metrics)
        all_batch_metrics.extend(batch_metrics)

    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved batch metrics to {batch_metrics_path}")


if __name__ == "__main__":
    main()
