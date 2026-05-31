#!/usr/bin/env python3
"""Assign curriculum-learning difficulty labels to text datasets.

The script reads a dataset folder with texts.txt and score.txt, scores each
sample with a selected literature-based readability formula, then writes:

- dificulty.txt: one easy/medium/hard label per sample
- dataset_with_dificulty.csv: text, score, raw difficulty score, and label
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "resources" / "datasets"
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]*", re.UNICODE)
VOWELS = "aeiouy"


@dataclass
class DatasetRows:
    texts: List[str]
    labels: List[str]


@dataclass
class DifficultyRow:
    index: int
    text: str
    label: str
    score: float
    dificulty: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign easy/medium/hard difficulty categories to a text dataset."
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
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
        "--strategy",
        default="flesch-kincaid",
        choices=[
            "flesch-reading-ease",
            "flesch-kincaid",
            "gunning-fog",
            "smog",
            "coleman-liau",
            "automated-readability",
            "roberta-mlm-loss",
        ],
        help="Difficulty scoring strategy.",
    )
    parser.add_argument(
        "--easy-ratio",
        type=float,
        default=1 / 3,
        help="Fraction of samples assigned to easy, after sorting by score.",
    )
    parser.add_argument(
        "--medium-ratio",
        type=float,
        default=1 / 3,
        help="Fraction of samples assigned to medium, after sorting by score.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <dataset-dir>/dataset_with_dificulty.csv.",
    )
    parser.add_argument(
        "--output-difficulty",
        type=Path,
        default=None,
        help="Difficulty sidecar output path. Defaults to <dataset-dir>/dificulty.txt.",
    )
    parser.add_argument(
        "--roberta-model-name",
        default="roberta-base",
        help="Pretrained masked language model used by --strategy roberta-mlm-loss.",
    )
    parser.add_argument(
        "--roberta-batch-size",
        type=int,
        default=16,
        help="Batch size for --strategy roberta-mlm-loss.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max length for --strategy roberta-mlm-loss.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for --strategy roberta-mlm-loss. Defaults to cuda if available, else cpu.",
    )
    return parser.parse_args()


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir
    return args.datasets_root / args.dataset_name


def load_dataset(dataset_dir: Path) -> DatasetRows:
    texts_path = dataset_dir / "texts.txt"
    labels_path = dataset_dir / "score.txt"
    if not texts_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Expected {texts_path} and {labels_path}. "
            "This scorer currently targets the repo's parallel text/score format."
        )

    texts = texts_path.read_text(encoding="utf-8").splitlines()
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    if len(texts) != len(labels):
        raise ValueError(
            f"Mismatched dataset lengths: texts.txt has {len(texts)} rows, "
            f"score.txt has {len(labels)} rows."
        )
    if not texts:
        raise ValueError(f"No examples found in {dataset_dir}.")
    return DatasetRows(texts=texts, labels=labels)


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def split_sentences(text: str) -> List[str]:
    sentences = [match.group(0).strip() for match in SENTENCE_PATTERN.finditer(text)]
    return [sentence for sentence in sentences if tokenize(sentence)] or [text]


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0

    groups = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in VOWELS
        if is_vowel and not previous_was_vowel:
            groups += 1
        previous_was_vowel = is_vowel

    if word.endswith("e") and groups > 1 and not word.endswith(("le", "ye")):
        groups -= 1
    return max(groups, 1)


@dataclass
class ReadabilityStats:
    words: int
    sentences: int
    syllables: int
    complex_words: int
    characters: int

    @property
    def words_per_sentence(self) -> float:
        return self.words / max(self.sentences, 1)

    @property
    def syllables_per_word(self) -> float:
        return self.syllables / max(self.words, 1)

    @property
    def complex_word_percentage(self) -> float:
        return 100 * self.complex_words / max(self.words, 1)

    @property
    def letters_per_100_words(self) -> float:
        return 100 * self.characters / max(self.words, 1)

    @property
    def sentences_per_100_words(self) -> float:
        return 100 * self.sentences / max(self.words, 1)


def readability_stats(text: str) -> ReadabilityStats:
    tokens = tokenize(text)
    sentence_count = len(split_sentences(text))
    syllables = [count_syllables(token) for token in tokens]
    character_count = sum(len(re.sub(r"[^a-z]", "", token.lower())) for token in tokens)
    return ReadabilityStats(
        words=len(tokens),
        sentences=sentence_count,
        syllables=sum(syllables),
        complex_words=sum(1 for count in syllables if count >= 3),
        characters=character_count,
    )


def flesch_reading_ease_score(stats: ReadabilityStats) -> float:
    reading_ease = 206.835 - 1.015 * stats.words_per_sentence - 84.6 * stats.syllables_per_word
    return -reading_ease


def flesch_kincaid_score(stats: ReadabilityStats) -> float:
    return 0.39 * stats.words_per_sentence + 11.8 * stats.syllables_per_word - 15.59


def gunning_fog_score(stats: ReadabilityStats) -> float:
    return 0.4 * (stats.words_per_sentence + stats.complex_word_percentage)


def smog_score(stats: ReadabilityStats) -> float:
    return 1.043 * math.sqrt(stats.complex_words * (30 / max(stats.sentences, 1))) + 3.1291


def coleman_liau_score(stats: ReadabilityStats) -> float:
    return 0.0588 * stats.letters_per_100_words - 0.296 * stats.sentences_per_100_words - 15.8


def automated_readability_score(stats: ReadabilityStats) -> float:
    return 4.71 * (stats.characters / max(stats.words, 1)) + 0.5 * stats.words_per_sentence - 21.43


def roberta_mlm_loss_scores(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device_name: str | None,
) -> List[float]:
    try:
        import torch
        from torch.nn import CrossEntropyLoss
        from tqdm.auto import tqdm
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The roberta-mlm-loss strategy requires torch and transformers. "
            "Install the project requirements before using it."
        ) from exc

    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    loss_function = CrossEntropyLoss(reduction="none")
    scores: List[float] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="roberta-loss", leave=False):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        for special_token_id in tokenizer.all_special_ids:
            special_tokens_mask |= labels.eq(special_token_id)
        labels[special_tokens_mask] = -100

        with torch.no_grad():
            logits = model(**encoded).logits

        token_losses = loss_function(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view(labels.size())
        valid_tokens = labels.ne(-100)
        sample_losses = token_losses.sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)
        scores.extend(sample_losses.detach().cpu().tolist())
    return scores


def score_texts(args: argparse.Namespace, texts: Sequence[str]) -> List[float]:
    strategies: Dict[str, Callable[[Sequence[str]], List[float]]] = {
        "flesch-reading-ease": lambda values: [
            flesch_reading_ease_score(readability_stats(text)) for text in values
        ],
        "flesch-kincaid": lambda values: [
            flesch_kincaid_score(readability_stats(text)) for text in values
        ],
        "gunning-fog": lambda values: [gunning_fog_score(readability_stats(text)) for text in values],
        "smog": lambda values: [smog_score(readability_stats(text)) for text in values],
        "coleman-liau": lambda values: [
            coleman_liau_score(readability_stats(text)) for text in values
        ],
        "automated-readability": lambda values: [
            automated_readability_score(readability_stats(text)) for text in values
        ],
    }
    if args.strategy == "roberta-mlm-loss":
        return roberta_mlm_loss_scores(
            texts,
            args.roberta_model_name,
            args.roberta_batch_size,
            args.max_length,
            args.device,
        )
    return strategies[args.strategy](texts)


def validate_ratios(easy_ratio: float, medium_ratio: float) -> None:
    if easy_ratio < 0 or medium_ratio < 0:
        raise ValueError("Ratios must be non-negative.")
    if easy_ratio + medium_ratio >= 1:
        raise ValueError("--easy-ratio plus --medium-ratio must be less than 1.")


def assign_categories(
    scores: Sequence[float],
    easy_ratio: float,
    medium_ratio: float,
) -> List[str]:
    validate_ratios(easy_ratio, medium_ratio)
    count = len(scores)
    easy_count = int(round(count * easy_ratio))
    medium_count = int(round(count * medium_ratio))
    if count >= 3:
        easy_count = max(1, easy_count)
        medium_count = max(1, medium_count)
    if easy_count + medium_count > count:
        medium_count = max(0, count - easy_count)

    categories = ["hard"] * count
    sorted_indices = sorted(range(count), key=lambda index: (scores[index], index))
    for position, index in enumerate(sorted_indices):
        if position < easy_count:
            categories[index] = "easy"
        elif position < easy_count + medium_count:
            categories[index] = "medium"
    return categories


def build_rows(dataset: DatasetRows, scores: Sequence[float], categories: Sequence[str]) -> List[DifficultyRow]:
    return [
        DifficultyRow(
            index=index,
            text=text,
            label=label,
            score=scores[index],
            dificulty=categories[index],
        )
        for index, (text, label) in enumerate(zip(dataset.texts, dataset.labels))
    ]


def write_difficulty_file(path: Path, rows: Iterable[DifficultyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(row.dificulty for row in rows) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[DifficultyRow], strategy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "text",
                "score",
                "difficulty_score",
                "difficulty_strategy",
                "dificulty",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row.index,
                    "text": row.text,
                    "score": row.label,
                    "difficulty_score": row.score,
                    "difficulty_strategy": strategy,
                    "dificulty": row.dificulty,
                }
            )


def print_summary(rows: Sequence[DifficultyRow], strategy: str, csv_path: Path, difficulty_path: Path) -> None:
    counts = Counter(row.dificulty for row in rows)
    print(f"Assigned difficulty with strategy: {strategy}")
    print(f"easy={counts['easy']} medium={counts['medium']} hard={counts['hard']}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved difficulty labels to {difficulty_path}")


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args)
    dataset = load_dataset(dataset_dir)
    scores = score_texts(args, dataset.texts)
    categories = assign_categories(scores, args.easy_ratio, args.medium_ratio)
    rows = build_rows(dataset, scores, categories)

    csv_path = args.output_csv or (dataset_dir / "dataset_with_dificulty.csv")
    difficulty_path = args.output_difficulty or (dataset_dir / "dificulty.txt")
    write_csv(csv_path, rows, args.strategy)
    write_difficulty_file(difficulty_path, rows)
    print_summary(rows, args.strategy, csv_path, difficulty_path)


if __name__ == "__main__":
    main()
