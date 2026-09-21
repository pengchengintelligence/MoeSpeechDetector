
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import hashlib
import inspect
import json
import os
import platform
import random
import re
import socket
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.dca_moe_cached_dataset import (  # noqa: E402
    DCAMoECachedDataset,
    dca_moe_cached_collate_fn,
)
from models.dca_moe_dirichlet_tcp_fixed_s_full_seoc import (  # noqa: E402
    DCAMoELayeredTCPFixedSSEOCPosEncNoLangModel,
)


LOSS_TYPE = (
    "dirichlet_fixed_s_tcp_full_3d_structural_evidence_odds_consistency_"
    "speech_sinusoidal_chunk_position_encoding_no_language_embedding"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default="/data2/moe/data/manifest_dca_moe_cached_features.csv",
        help="Cached manifest containing cached_feature_path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/data2/moe/outputs/"
            "dca_moe_cv5_tcp_fixed_s_seoc_posenc_nolang"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ADReSS", "ADReSSo", "NCMMSC", "TAUKADIAL"],
    )

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument(
        "--split_seed",
        type=int,
        default=2026,
        help=(
            "Seed used only for the fixed subject-level StratifiedKFold split. "
            "All recordings from the same subject are kept in the same fold."
        ),
    )
    parser.add_argument(
        "--train_seed_base",
        type=int,
        default=43,
        help="Run r uses train_seed_base + r - 1.",
    )
    parser.add_argument(
        "--subject_id_column",
        type=str,
        default="subject_id",
        help=(
            "Optional manifest column containing participant IDs. If this column "
            "is absent, subject IDs are inferred from sample_id using dataset-specific "
            "rules for TAUKADIAL and NCMMSC; other datasets use sample_id itself."
        ),
    )

    parser.add_argument(
        "--lambda_unimodal",
        type=float,
        default=0.10,
        help="Weight of the auxiliary speech/text evidential losses.",
    )
    parser.add_argument(
        "--lambda_tcp",
        type=float,
        default=0.05,
        help="Weight of the TCP regression loss.",
    )
    parser.add_argument(
        "--lambda_seoc",
        type=float,
        default=0.005,
        help=(
            "Weight of Full 3D Structural Evidence-Odds Consistency regularization. "
            "This replaces the former branch-level CU regularizer and is "
            "multiplied by tcp_ratio for gradual enabling."
        ),
    )
    parser.add_argument(
        "--lambda_con",
        type=float,
        default=0.0,
        help=(
            "Weight of the speech-text Jensen-Shannon disagreement "
            "regularizer. Default 0 disables its training effect."
        ),
    )

    parser.add_argument(
        "--tcp_warmup_epochs",
        type=int,
        default=5,
        help=(
            "For these initial epochs the fusion exactly reproduces raw "
            "e_s + e_t while the TCP heads still learn."
        ),
    )
    parser.add_argument(
        "--tcp_ramp_epochs",
        type=int,
        default=10,
        help="Epochs used to ramp the TCP fusion ratio from 0 to 1.",
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Set <=0 to disable early stopping.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--evidence_activation",
        type=str,
        default="softplus",
        choices=["softplus", "relu"],
    )
    parser.add_argument("--tcp_dropout", type=float, default=0.1)
    parser.add_argument(
        "--max_speech_chunks",
        type=int,
        default=256,
        help=(
            "Initial capacity of the fixed sinusoidal speech-chunk position "
            "encoding table. It extends automatically if a batch contains "
            "more chunks."
        ),
    )
    parser.add_argument(
        "--speech_position_dropout",
        type=float,
        default=0.0,
        help=(
            "Dropout applied after adding speech chunk position encoding. "
            "Default 0.0 keeps the position-encoding ablation minimal."
        ),
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Maximum gradient norm; <=0 disables gradient clipping.",
    )

    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload cached features into CPU RAM.",
    )
    parser.add_argument(
        "--split_only",
        action="store_true",
        help=(
            "Only build and audit subject-grouped folds, write fold CSV files, "
            "then exit without training. Recommended before a long experiment."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )

    return parser.parse_args()



def _json_safe(value):
    """Recursively convert common Python/NumPy/PyTorch values to JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            ensure_ascii=False,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_parameters(model: torch.nn.Module) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
    }


def _collect_environment(device: torch.device) -> dict:
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_or_resolved_device": str(device),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        environment.update(
            {
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count_visible": torch.cuda.device_count(),
                "gpu_names_visible": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    return environment


def _label_counts(frame: pd.DataFrame) -> dict:
    counts = frame["label"].astype(int).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _subject_label_counts(frame: pd.DataFrame) -> dict:
    subject_labels = (
        frame[["subject_id", "label"]]
        .drop_duplicates(subset=["subject_id"])
    )
    counts = subject_labels["label"].astype(int).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _sample_stem(sample_id: object) -> str:
    """Return a normalized filename stem from a manifest sample_id value."""
    value = str(sample_id).strip().replace("\\", "/")
    if not value:
        raise ValueError("Encountered an empty sample_id while deriving subject IDs.")
    return Path(value.rsplit("/", 1)[-1]).stem


def infer_subject_id(dataset: str, sample_id: object) -> str:
    """Infer participant ID from dataset-specific recording names.

    TAUKADIAL examples:
        taukadial-002-1 -> taukadial-002
        taukdial-002-2 -> taukdial-002

    NCMMSC examples:
        AD_F_040108_003 -> AD_F_040108
        AD_F_040108_014 -> AD_F_040108

    ADReSS and ADReSSo are treated as one recording per sample_id unless an
    explicit subject_id column is supplied in the manifest.
    """
    dataset_key = str(dataset).strip().upper()
    stem = _sample_stem(sample_id)

    if dataset_key == "TAUKADIAL":
        match = re.fullmatch(
            r"(?i)(tauk(?:a)?dial)[-_](\d+)(?:[-_](\d+))?",
            stem,
        )
        if match is None:
            raise ValueError(
                "Cannot infer TAUKADIAL subject_id from sample_id="
                f"{sample_id!r}. Expected a name such as taukadial-002-1 or "
                "taukdial-002-1. Add a subject_id column to the manifest if "
                "your naming rule is different."
            )
        return f"taukadial-{match.group(2)}"

    if dataset_key == "NCMMSC":
        # Preferred rule: class/sex + six-digit participant code + utterance ID.
        match = re.fullmatch(r"(.+?_\d{6})_\d+", stem)
        if match is None:
            # Conservative fallback for equivalent NCMMSC names whose subject
            # code is not exactly six digits: remove only the final numeric
            # recording/utterance suffix.
            match = re.fullmatch(r"(.+)_\d+", stem)
        if match is None:
            raise ValueError(
                "Cannot infer NCMMSC subject_id from sample_id="
                f"{sample_id!r}. Expected a name such as AD_F_040108_003. "
                "Add a subject_id column to the manifest if your naming rule "
                "is different."
            )
        return match.group(1)

    return stem


def attach_and_validate_subject_ids(
    dataset_df: pd.DataFrame,
    dataset_name: str,
    subject_id_column: str,
    n_splits: int,
) -> Tuple[pd.DataFrame, str]:
    """Attach subject_id and validate participant-level CV assumptions."""
    frame = dataset_df.copy()

    if subject_id_column in frame.columns:
        source = f"manifest_column:{subject_id_column}"
        subject_series = frame[subject_id_column]
        if subject_series.isna().any():
            bad_rows = frame.loc[subject_series.isna(), "sample_id"].head(10).tolist()
            raise ValueError(
                f"Column {subject_id_column!r} contains missing values. "
                f"Example sample_ids: {bad_rows}"
            )
        subject_series = subject_series.astype(str).str.strip()
        if (subject_series == "").any():
            bad_rows = frame.loc[subject_series == "", "sample_id"].head(10).tolist()
            raise ValueError(
                f"Column {subject_id_column!r} contains empty values. "
                f"Example sample_ids: {bad_rows}"
            )
        frame["subject_id"] = subject_series
    else:
        source = "dataset_specific_inference_from_sample_id"
        frame["subject_id"] = [
            infer_subject_id(dataset_name, sample_id)
            for sample_id in frame["sample_id"]
        ]

    frame["subject_id"] = frame["subject_id"].astype(str).str.strip()

    # Every participant must have a single target label. Otherwise a grouped
    # classification split is not well-defined and should not be silently run.
    label_nunique = frame.groupby("subject_id")["label"].nunique()
    inconsistent = label_nunique[label_nunique > 1]
    if not inconsistent.empty:
        examples = inconsistent.index.astype(str).tolist()[:10]
        raise ValueError(
            "Some subject IDs have recordings with conflicting labels: "
            f"{examples}. Please correct the manifest before training."
        )

    subject_table = (
        frame[["subject_id", "label"]]
        .drop_duplicates(subset=["subject_id"])
    )
    subject_class_counts = (
        subject_table["label"].astype(int).value_counts().sort_index()
    )
    too_small = subject_class_counts[subject_class_counts < n_splits]
    if not too_small.empty:
        raise ValueError(
            "Subject-grouped stratified CV requires at least n_splits unique "
            "subjects in every class. Insufficient subject counts: "
            f"{too_small.to_dict()}, n_splits={n_splits}."
        )

    duplicate_sample_ids = frame["sample_id"].astype(str).duplicated(keep=False)
    if duplicate_sample_ids.any():
        examples = (
            frame.loc[duplicate_sample_ids, "sample_id"]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "sample_id must be unique within a dataset. Duplicate examples: "
            f"{examples}"
        )

    return frame, source


def _assert_no_subject_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold_idx: int,
) -> None:
    train_subjects = set(train_df["subject_id"].astype(str))
    val_subjects = set(val_df["subject_id"].astype(str))
    overlap = sorted(train_subjects.intersection(val_subjects))
    if overlap:
        raise RuntimeError(
            f"Subject leakage detected in fold {fold_idx}: {overlap[:20]}"
        )


def _build_fold_records(folds) -> list:
    records = []
    for fold_idx, train_df, val_df in folds:
        train_subjects = train_df["subject_id"].astype(str).nunique()
        val_subjects = val_df["subject_id"].astype(str).nunique()
        overlap_count = len(
            set(train_df["subject_id"].astype(str)).intersection(
                set(val_df["subject_id"].astype(str))
            )
        )
        records.append(
            {
                "fold": int(fold_idx),
                "train_size": int(len(train_df)),
                "validation_size": int(len(val_df)),
                "train_subjects": int(train_subjects),
                "validation_subjects": int(val_subjects),
                "subject_overlap_count": int(overlap_count),
                "train_label_counts": _label_counts(train_df),
                "validation_label_counts": _label_counts(val_df),
                "train_subject_label_counts": _subject_label_counts(train_df),
                "validation_subject_label_counts": _subject_label_counts(val_df),
            }
        )
    return records


def build_subject_fold_assignment(folds) -> pd.DataFrame:
    """Create one auditable row per participant and its validation fold."""
    rows = []
    seen_subjects = set()
    for fold_idx, _, val_df in folds:
        grouped = val_df.groupby("subject_id", sort=True)
        for subject_id, subject_frame in grouped:
            subject_key = str(subject_id)
            if subject_key in seen_subjects:
                raise RuntimeError(
                    f"Subject {subject_key!r} appears in validation more than once."
                )
            seen_subjects.add(subject_key)
            labels = subject_frame["label"].astype(int).unique()
            if len(labels) != 1:
                raise RuntimeError(
                    f"Subject {subject_key!r} has inconsistent labels in fold audit."
                )
            rows.append(
                {
                    "subject_id": subject_key,
                    "label": int(labels[0]),
                    "n_recordings": int(len(subject_frame)),
                    "validation_fold": int(fold_idx),
                    "sample_ids": "|".join(
                        subject_frame["sample_id"].astype(str).tolist()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["validation_fold", "label", "subject_id"]
    ).reset_index(drop=True)


def save_experiment_config(
    args,
    dataset_df: pd.DataFrame,
    folds,
    dataset_output_dir: Path,
    device: torch.device,
    subject_id_source: str,
):
    """Save one complete experiment-level configuration JSON."""
    model_probe = DCAMoELayeredTCPFixedSSEOCPosEncNoLangModel(
        hidden_dim=args.hidden_dim,
        num_classes=2,
        evidence_activation=args.evidence_activation,
        tcp_dropout=args.tcp_dropout,
        max_speech_chunks=args.max_speech_chunks,
        speech_position_dropout=args.speech_position_dropout,
    )

    model_file = Path(
        inspect.getfile(model_probe.__class__)
    ).resolve()
    training_file = Path(__file__).resolve()
    seeds = [args.train_seed_base + index for index in range(args.runs)]

    payload = {
        "experiment_name": (
            "dca_moe_fixed_s_tcp_full_3d_seoc_posenc_no_language_embedding"
        ),
        "dataset": args.dataset,
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(dataset_output_dir.resolve()),
        "cross_validation": {
            "folds": args.folds,
            "runs_per_fold": args.runs,
            "total_training_jobs": args.folds * args.runs,
            "splitter": "subject-level StratifiedKFold(shuffle=True)",
            "grouping_unit": "subject_id",
            "subject_id_source": subject_id_source,
            "subject_leakage_allowed": False,
            "split_seed": args.split_seed,
            "train_seed_base": args.train_seed_base,
            "train_seed_rule": (
                "train_seed=train_seed_base+run_idx-1"
            ),
            "training_seeds": seeds,
            "fold_records": _build_fold_records(folds),
        },
        "data": {
            "dataset_total_size": int(len(dataset_df)),
            "dataset_total_subjects": int(dataset_df["subject_id"].nunique()),
            "dataset_label_counts": _label_counts(dataset_df),
            "dataset_subject_label_counts": _subject_label_counts(dataset_df),
            "recordings_per_subject": {
                "min": int(dataset_df.groupby("subject_id").size().min()),
                "max": int(dataset_df.groupby("subject_id").size().max()),
                "mean": float(dataset_df.groupby("subject_id").size().mean()),
            },
            "original_split_counts": {
                str(key): int(value)
                for key, value in dataset_df["split"].value_counts().items()
            },
            "required_manifest_columns": [
                "sample_id",
                "dataset",
                "split",
                "label",
                "lang_id",
                "text_expert_group",
                "cached_feature_path",
            ],
            "optional_subject_id_column": args.subject_id_column,
            "effective_subject_id_column": "subject_id",
            "lang_id_retained_in_cache": True,
            "language_embedding_used_by_heads": False,
            "language_aware_text_expert_masking": True,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip_max_norm": args.grad_clip,
            "early_stop_monitor": "validation_f1",
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "num_workers": args.num_workers,
            "preload": bool(args.preload),
            "requested_device": args.device,
            "resolved_device": str(device),
        },
        "loss": {
            "loss_type": LOSS_TYPE,
            "lambda_unimodal": args.lambda_unimodal,
            "lambda_tcp": args.lambda_tcp,
            "lambda_seoc": args.lambda_seoc,
            "lambda_con": args.lambda_con,
            "tcp_warmup_epochs": args.tcp_warmup_epochs,
            "tcp_ramp_epochs": args.tcp_ramp_epochs,
            "dirichlet_prediction_loss": (
                "sum((y-p)^2)+sum(p*(1-p)/(S+1))"
            ),
            "tcp_target": (
                "detached_true_class_probability_of_each_branch"
            ),
            "seoc_rule": (
                "mean(||log1p([(1-u_s)/u_s,(1-u_t)/u_t,"
                "(1-u_f)/u_f])-sg(log1p([c_s/(1-c_s),"
                "c_t/(1-c_t),c_s/(1-c_s)+c_t/(1-c_t)]))||^2)"
            ),
            "seoc_geometry": (
                "fixed-S additive closure: r_f^D=r_s^D+r_t^D, "
                "with r=(1-u)/u=E/K"
            ),
            "seoc_tcp_target": (
                "detached TCP odds with r_f^T=r_s^T+r_t^T"
            ),
            "conflict_rule": "Jensen-Shannon divergence in bits",
        },
        "model": {
            **model_probe.get_model_config(),
            **_count_parameters(model_probe),
        },
        "software_and_hardware": _collect_environment(device),
        "source_files": {
            "model_file": str(model_file),
            "model_sha256": (
                _sha256_file(model_file) if model_file.exists() else None
            ),
            "training_file": str(training_file),
            "training_sha256": _sha256_file(training_file),
        },
        "command_line": {
            "argv": list(sys.argv),
            "parsed_arguments": vars(args),
        },
        "statistical_exports": {
            "n5_definition": (
                "five fold-level observations; each fold value is the mean "
                "over all runs within that fold"
            ),
            "n20_definition": (
                "twenty seed-level observations; each seed value is the mean "
                "over five folds for the matching run/seed"
            ),
            "standard_deviation": "sample standard deviation (ddof=1)",
            "paired_test_guidance": (
                "Compare models with matching fold IDs for n=5 or matching "
                "train seeds for n=20 using a paired t-test."
            ),
        },
    }
    config_path = dataset_output_dir / "experiment_config.json"
    _write_json(config_path, payload)
    del model_probe
    return config_path, payload


def save_run_config(
    args,
    run_dir: Path,
    dataset_name: str,
    fold_idx: int,
    run_idx: int,
    train_seed: int,
    train_csv: Path,
    val_csv: Path,
    model: torch.nn.Module,
    device: torch.device,
):
    train_frame = pd.read_csv(train_csv, usecols=["label"])
    val_frame = pd.read_csv(val_csv, usecols=["label"])
    payload = {
        "dataset": dataset_name,
        "fold": fold_idx,
        "run": run_idx,
        "split_seed": args.split_seed,
        "train_seed": train_seed,
        "train_seed_base": args.train_seed_base,
        "train_seed_rule": "train_seed=train_seed_base+run_idx-1",
        "train_size": int(len(train_frame)),
        "validation_size": int(len(val_frame)),
        "train_label_counts": _label_counts(train_frame),
        "validation_label_counts": _label_counts(val_frame),
        "train_csv": str(train_csv.resolve()),
        "validation_csv": str(val_csv.resolve()),
        "model": {
            **model.get_model_config(),
            **_count_parameters(model),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip_max_norm": args.grad_clip,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "device": str(device),
            "num_workers": args.num_workers,
            "preload": bool(args.preload),
        },
        "loss": {
            "loss_type": LOSS_TYPE,
            "lambda_unimodal": args.lambda_unimodal,
            "lambda_tcp": args.lambda_tcp,
            "lambda_seoc": args.lambda_seoc,
            "lambda_con": args.lambda_con,
            "tcp_warmup_epochs": args.tcp_warmup_epochs,
            "tcp_ramp_epochs": args.tcp_ramp_epochs,
        },
        "parsed_arguments": vars(args),
    }
    path = run_dir / "run_config.json"
    _write_json(path, payload)
    return path


TTEST_METRIC_COLUMNS = [
    "val_acc",
    "val_f1",
    "val_precision",
    "val_recall",
    "val_specificity",
    "val_auc",
    "val_brier",
]


def _make_ttest_summary(
    observations: pd.DataFrame,
    metric_columns: list,
    n_definition: str,
) -> pd.DataFrame:
    rows = []
    for column in metric_columns:
        values = pd.to_numeric(observations[column], errors="coerce").dropna()
        rows.append(
            {
                "metric": column.removeprefix("val_"),
                "source_column": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "sem": float(values.sem(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "n_definition": n_definition,
                "std_definition": "sample standard deviation (ddof=1)",
            }
        )
    return pd.DataFrame(rows)


def export_ttest_files(
    all_runs_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
):
    """Create new t-test inputs without changing any existing CSV files.

    n=5: one observation per fold, obtained by averaging the 20 matching
         run results within each fold.
    n=20: one observation per run/seed, obtained by averaging its five fold
          results.
    """
    missing = [
        column
        for column in TTEST_METRIC_COLUMNS
        if column not in all_runs_df.columns
    ]
    if missing:
        raise ValueError(
            f"Cannot export t-test files; missing metrics: {missing}"
        )

    metric_columns = list(TTEST_METRIC_COLUMNS)

    n5 = (
        all_runs_df.groupby("fold", as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values("fold")
        .reset_index(drop=True)
    )
    n5.insert(0, "dataset", dataset_name)
    n5.insert(1, "unit_type", "fold")
    n5.insert(2, "n_target", 5)
    n5.insert(3, "aggregation", "mean_over_runs_within_fold")
    n5_path = output_dir / "ttest_n5_fold_observations.csv"
    n5.to_csv(n5_path, index=False, encoding="utf-8-sig")

    n5_summary = _make_ttest_summary(
        n5,
        metric_columns,
        (
            "n=5 fold-level observations; each value is the mean over all "
            "runs within one fold"
        ),
    )
    n5_summary_path = output_dir / "ttest_n5_fold_summary.csv"
    n5_summary.to_csv(
        n5_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    seed_keys = ["run", "train_seed"]
    n20 = (
        all_runs_df.groupby(seed_keys, as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values(seed_keys)
        .reset_index(drop=True)
    )
    n20.insert(0, "dataset", dataset_name)
    n20.insert(1, "unit_type", "train_seed")
    n20.insert(2, "n_target", 20)
    n20.insert(3, "aggregation", "mean_over_five_folds_for_same_seed")
    n20_path = output_dir / "ttest_n20_seed_observations.csv"
    n20.to_csv(n20_path, index=False, encoding="utf-8-sig")

    n20_summary = _make_ttest_summary(
        n20,
        metric_columns,
        (
            "n=20 seed-level observations; each value is the mean over five "
            "folds for one matching training seed"
        ),
    )
    n20_summary_path = output_dir / "ttest_n20_seed_summary.csv"
    n20_summary.to_csv(
        n20_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "n5_observations": n5_path,
        "n5_summary": n5_summary_path,
        "n20_observations": n20_path,
        "n20_summary": n20_summary_path,
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    out = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }

    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = float("nan")

    try:
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
        out.update(
            {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "specificity": float(tn / (tn + fp + 1e-12)),
            }
        )
    except Exception:
        out.update(
            {
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
                "specificity": float("nan"),
            }
        )

    return out


def make_loader(
    csv_path,
    batch_size,
    shuffle,
    num_workers,
    preload=False,
):
    dataset = DCAMoECachedDataset(str(csv_path), preload=preload)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": dca_moe_cached_collate_fn,
        "pin_memory": True,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


def get_tcp_ratio(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """Return the gradual fixed-S TCP activation ratio for an epoch."""
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(
        min(
            1.0,
            max(0.0, (epoch - warmup_epochs) / ramp_epochs),
        )
    )


def dirichlet_prediction_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
):
    """
    Trusted evidential accuracy loss:

        L_acc = sum_k (y_k - p_k)^2
                + sum_k p_k(1-p_k)/(S+1)
    """
    labels_onehot = F.one_hot(
        labels,
        num_classes=num_classes,
    ).float()

    strength = torch.sum(alpha, dim=-1, keepdim=True)
    prob = alpha / strength

    err = torch.sum((labels_onehot - prob) ** 2, dim=-1)
    var = torch.sum(
        prob * (1.0 - prob) / (strength + 1.0),
        dim=-1,
    )
    return torch.mean(err + var)


def per_sample_js_divergence(
    prob_a: torch.Tensor,
    prob_b: torch.Tensor,
    eps: float = 1e-8,
):
    """Per-sample Jensen-Shannon divergence in bits."""
    prob_a = torch.clamp(prob_a, min=eps, max=1.0)
    prob_b = torch.clamp(prob_b, min=eps, max=1.0)
    prob_m = torch.clamp(0.5 * (prob_a + prob_b), min=eps, max=1.0)

    entropy_a = -torch.sum(prob_a * torch.log2(prob_a), dim=-1)
    entropy_b = -torch.sum(prob_b * torch.log2(prob_b), dim=-1)
    entropy_m = -torch.sum(prob_m * torch.log2(prob_m), dim=-1)

    return entropy_m - 0.5 * entropy_a - 0.5 * entropy_b


def opinion_consistency_loss(
    prob_a: torch.Tensor,
    prob_b: torch.Tensor,
):
    """Mean speech-text JS divergence used as optional conflict regularizer."""
    return torch.mean(per_sample_js_divergence(prob_a, prob_b))


def gather_true_class_probability(
    prob: torch.Tensor,
    labels: torch.Tensor,
):
    return torch.gather(
        prob,
        dim=1,
        index=labels.unsqueeze(1),
    )


def trusted_tcp_evidential_loss(
    out: dict,
    labels: torch.Tensor,
    lambda_unimodal: float,
    lambda_tcp: float,
    lambda_seoc: float,
    lambda_con: float,
    tcp_ratio: float,
    num_classes: int = 2,
):
    """
    Total loss:

        L_total = L_fused
                + lambda_unimodal * L_unimodal
                + lambda_tcp * L_tcp
                + lambda_seoc      * tcp_ratio * L_SEOC
                + lambda_con * L_con
    """
    loss_fused = dirichlet_prediction_loss(
        alpha=out["alpha"],
        labels=labels,
        num_classes=num_classes,
    )

    loss_speech = dirichlet_prediction_loss(
        alpha=out["speech_alpha"],
        labels=labels,
        num_classes=num_classes,
    )
    loss_text = dirichlet_prediction_loss(
        alpha=out["text_alpha"],
        labels=labels,
        num_classes=num_classes,
    )
    loss_unimodal = 0.5 * (loss_speech + loss_text)

    # TCP targets are detached raw-branch true-class probabilities.
    speech_tcp_target = gather_true_class_probability(
        out["speech_prob"].detach(),
        labels,
    )
    text_tcp_target = gather_true_class_probability(
        out["text_prob"].detach(),
        labels,
    )

    loss_tcp_speech = F.mse_loss(
        out["speech_tcp_confidence"],
        speech_tcp_target,
    )
    loss_tcp_text = F.mse_loss(
        out["text_tcp_confidence"],
        text_tcp_target,
    )
    loss_tcp = 0.5 * (loss_tcp_speech + loss_tcp_text)

    dtype_eps = torch.finfo(out["speech_uncertainty"].dtype).eps

    speech_u = out["speech_uncertainty"].clamp(
        min=dtype_eps,
        max=1.0 - dtype_eps,
    )
    text_u = out["text_uncertainty"].clamp(
        min=dtype_eps,
        max=1.0 - dtype_eps,
    )
    fused_u = out["uncertainty"].clamp(
        min=dtype_eps,
        max=1.0 - dtype_eps,
    )

    speech_c = out["speech_tcp_confidence"].detach().clamp(
        min=dtype_eps,
        max=1.0 - dtype_eps,
    )
    text_c = out["text_tcp_confidence"].detach().clamp(
        min=dtype_eps,
        max=1.0 - dtype_eps,
    )

    speech_dir_odds = (1.0 - speech_u) / speech_u
    text_dir_odds = (1.0 - text_u) / text_u
    fused_dir_odds = (1.0 - fused_u) / fused_u

    speech_tcp_odds = speech_c / (1.0 - speech_c)
    text_tcp_odds = text_c / (1.0 - text_c)
    fused_tcp_odds = speech_tcp_odds + text_tcp_odds

    dir_log_reliability = torch.cat(
        [
            torch.log1p(speech_dir_odds),
            torch.log1p(text_dir_odds),
            torch.log1p(fused_dir_odds),
        ],
        dim=-1,
    )
    tcp_log_reliability = torch.cat(
        [
            torch.log1p(speech_tcp_odds),
            torch.log1p(text_tcp_odds),
            torch.log1p(fused_tcp_odds),
        ],
        dim=-1,
    )

    seoc_squared_error = (
        dir_log_reliability - tcp_log_reliability
    ).pow(2)
    loss_seoc_speech = seoc_squared_error[:, 0].mean()
    loss_seoc_text = seoc_squared_error[:, 1].mean()
    loss_seoc_fused = seoc_squared_error[:, 2].mean()
    loss_seoc = seoc_squared_error.mean()

    # This is a diagnostic identity of fixed-S fusion and is not another loss.
    seoc_dirichlet_closure_error = torch.abs(
        fused_dir_odds - (speech_dir_odds + text_dir_odds)
    ).mean()

    effective_lambda_seoc = lambda_seoc * float(tcp_ratio)

    # This term penalizes branch disagreement if lambda_con > 0.
    loss_con = opinion_consistency_loss(
        prob_a=out["speech_prob"],
        prob_b=out["text_prob"],
    )

    loss_total = (
        loss_fused
        + lambda_unimodal * loss_unimodal
        + lambda_tcp * loss_tcp
        + effective_lambda_seoc * loss_seoc
        + lambda_con * loss_con
    )

    loss_tensors = {
        "loss_fused": loss_fused,
        "loss_speech": loss_speech,
        "loss_text": loss_text,
        "loss_unimodal": loss_unimodal,
        "loss_tcp_speech": loss_tcp_speech,
        "loss_tcp_text": loss_tcp_text,
        "loss_tcp": loss_tcp,
        "loss_seoc_speech": loss_seoc_speech,
        "loss_seoc_text": loss_seoc_text,
        "loss_seoc_fused": loss_seoc_fused,
        "loss_seoc": loss_seoc,
        "seoc_dirichlet_closure_error": seoc_dirichlet_closure_error,
        "seoc_dir_speech_log_reliability": (
            dir_log_reliability[:, 0].mean()
        ),
        "seoc_dir_text_log_reliability": (
            dir_log_reliability[:, 1].mean()
        ),
        "seoc_dir_fused_log_reliability": (
            dir_log_reliability[:, 2].mean()
        ),
        "seoc_tcp_speech_log_reliability": (
            tcp_log_reliability[:, 0].mean()
        ),
        "seoc_tcp_text_log_reliability": (
            tcp_log_reliability[:, 1].mean()
        ),
        "seoc_tcp_fused_log_reliability": (
            tcp_log_reliability[:, 2].mean()
        ),
        "effective_lambda_seoc": torch.as_tensor(
            effective_lambda_seoc,
            device=loss_total.device,
            dtype=loss_total.dtype,
        ),
        "loss_con": loss_con,
        "loss_total": loss_total,
    }
    loss_items = {
        name: float(value.detach().cpu().item())
        for name, value in loss_tensors.items()
    }

    return loss_total, loss_items


def _mean_or_nan(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _append_batch_diagnostics(store: Dict[str, List[float]], out: dict):
    store["uncertainty"].extend(
        out["uncertainty"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["strength"].extend(
        out["strength"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["speech_uncertainty"].extend(
        out["speech_uncertainty"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["text_uncertainty"].extend(
        out["text_uncertainty"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["speech_strength"].extend(
        out["speech_strength"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["text_strength"].extend(
        out["text_strength"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["speech_tcp_confidence"].extend(
        out["speech_tcp_confidence"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["text_tcp_confidence"].extend(
        out["text_tcp_confidence"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["speech_tcp_weight"].extend(
        out["tcp_weights"][:, 0].detach().cpu().numpy().tolist()
    )
    store["text_tcp_weight"].extend(
        out["tcp_weights"][:, 1].detach().cpu().numpy().tolist()
    )
    store["speech_used_weight"].extend(
        out["modality_weights"][:, 0].detach().cpu().numpy().tolist()
    )
    store["text_used_weight"].extend(
        out["modality_weights"][:, 1].detach().cpu().numpy().tolist()
    )
    store["speech_fused_strength"].extend(
        out["speech_fused_strength"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["text_fused_strength"].extend(
        out["text_fused_strength"].detach().squeeze(-1).cpu().numpy().tolist()
    )
    store["strength_preservation_error"].extend(
        out["strength_preservation_error"]
        .detach()
        .squeeze(-1)
        .cpu()
        .numpy()
        .tolist()
    )
    store["modality_js_conflict"].extend(
        per_sample_js_divergence(
            out["speech_prob"].detach(),
            out["text_prob"].detach(),
        ).cpu().numpy().tolist()
    )


def _new_diagnostic_store():
    keys = [
        "uncertainty",
        "strength",
        "speech_uncertainty",
        "text_uncertainty",
        "speech_strength",
        "text_strength",
        "speech_tcp_confidence",
        "text_tcp_confidence",
        "speech_tcp_weight",
        "text_tcp_weight",
        "speech_used_weight",
        "text_used_weight",
        "speech_fused_strength",
        "text_fused_strength",
        "strength_preservation_error",
        "modality_js_conflict",
    ]
    return {key: [] for key in keys}


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_unimodal: float,
    lambda_tcp: float,
    lambda_seoc: float,
    lambda_con: float,
    tcp_ratio: float,
    grad_clip: float,
):
    model.train()

    loss_records = []
    y_true, y_prob, y_pred = [], [], []
    diagnostics = _new_diagnostic_store()

    for batch in loader:
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out, _ = model(batch, tcp_ratio=tcp_ratio)

        loss, loss_items = trusted_tcp_evidential_loss(
            out=out,
            labels=labels,
            lambda_unimodal=lambda_unimodal,
            lambda_tcp=lambda_tcp,
            lambda_seoc=lambda_seoc,
            lambda_con=lambda_con,
            tcp_ratio=tcp_ratio,
            num_classes=2,
        )

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )
        optimizer.step()

        loss_records.append(loss_items)

        prob_all = out["prob"].detach()
        prob_class1 = prob_all[:, 1]
        pred = torch.argmax(prob_all, dim=-1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(prob_class1.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        _append_batch_diagnostics(diagnostics, out)

    metrics = compute_metrics(y_true, y_prob, y_pred)
    for key in loss_records[0].keys() if loss_records else []:
        metrics[key] = _mean_or_nan([row[key] for row in loss_records])
    metrics["loss"] = metrics.get("loss_total", float("nan"))
    metrics["tcp_ratio"] = float(tcp_ratio)

    for key, values in diagnostics.items():
        metrics[key] = _mean_or_nan(values)

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    lambda_unimodal: float,
    lambda_tcp: float,
    lambda_seoc: float,
    lambda_con: float,
    tcp_ratio: float,
):
    model.eval()

    loss_records = []
    y_true, y_prob, y_pred = [], [], []
    sample_ids, datasets, original_splits = [], [], []
    diagnostics = _new_diagnostic_store()

    speech_probs, text_probs = [], []
    modality_records = []
    speech_expert_records = []
    text_expert_records = []

    for batch in loader:
        labels = batch["label"].to(device)
        out, aux = model(batch, tcp_ratio=tcp_ratio)

        _, loss_items = trusted_tcp_evidential_loss(
            out=out,
            labels=labels,
            lambda_unimodal=lambda_unimodal,
            lambda_tcp=lambda_tcp,
            lambda_seoc=lambda_seoc,
            lambda_con=lambda_con,
            tcp_ratio=tcp_ratio,
            num_classes=2,
        )
        loss_records.append(loss_items)

        prob_all = out["prob"]
        prob_class1 = prob_all[:, 1]
        pred = torch.argmax(prob_all, dim=-1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(prob_class1.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

        sample_ids.extend(batch["sample_id"])
        datasets.extend(batch["dataset"])
        original_splits.extend(batch["split"])

        speech_probs.extend(out["speech_prob"][:, 1].cpu().numpy().tolist())
        text_probs.extend(out["text_prob"][:, 1].cpu().numpy().tolist())
        _append_batch_diagnostics(diagnostics, out)

        tcp_conf = torch.cat(
            [
                out["speech_tcp_confidence"],
                out["text_tcp_confidence"],
            ],
            dim=-1,
        ).cpu().numpy()
        tcp_weights = out["tcp_weights"].cpu().numpy()
        base_weights = out["base_modality_weights"].cpu().numpy()
        used_weights = out["modality_weights"].cpu().numpy()
        raw_total_e = out["raw_total_evidence_strength"].squeeze(-1).cpu().numpy()
        speech_fused_e = out["speech_fused_strength"].squeeze(-1).cpu().numpy()
        text_fused_e = out["text_fused_strength"].squeeze(-1).cpu().numpy()
        strength_error = out["strength_preservation_error"].squeeze(-1).cpu().numpy()
        js_conflict = per_sample_js_divergence(
            out["speech_prob"],
            out["text_prob"],
        ).cpu().numpy()

        for i, (sid, ds, split_name) in enumerate(
            zip(batch["sample_id"], batch["dataset"], batch["split"])
        ):
            modality_records.append(
                {
                    "sample_id": sid,
                    "dataset": ds,
                    "original_split": split_name,
                    "speech_tcp_confidence": float(tcp_conf[i, 0]),
                    "text_tcp_confidence": float(tcp_conf[i, 1]),
                    "speech_tcp_weight": float(tcp_weights[i, 0]),
                    "text_tcp_weight": float(tcp_weights[i, 1]),
                    "base_speech_weight": float(base_weights[i, 0]),
                    "base_text_weight": float(base_weights[i, 1]),
                    "used_speech_weight": float(used_weights[i, 0]),
                    "used_text_weight": float(used_weights[i, 1]),
                    "raw_total_evidence_strength": float(raw_total_e[i]),
                    "speech_fused_evidence_strength": float(speech_fused_e[i]),
                    "text_fused_evidence_strength": float(text_fused_e[i]),
                    "strength_preservation_error": float(strength_error[i]),
                    "modality_js_conflict": float(js_conflict[i]),
                    "tcp_ratio": float(tcp_ratio),
                }
            )

        if "speech_expert_weights" in aux:
            names = aux["speech_expert_names"]
            weights = aux["speech_expert_weights"].numpy()
            for sid, ds, split_name, row_w in zip(
                batch["sample_id"],
                batch["dataset"],
                batch["split"],
                weights,
            ):
                record = {
                    "sample_id": sid,
                    "dataset": ds,
                    "original_split": split_name,
                }
                record.update({name: float(w) for name, w in zip(names, row_w)})
                speech_expert_records.append(record)

        if "text_expert_weights" in aux:
            names = aux["text_expert_names"]
            weights = aux["text_expert_weights"].numpy()
            for sid, ds, split_name, row_w in zip(
                batch["sample_id"],
                batch["dataset"],
                batch["split"],
                weights,
            ):
                record = {
                    "sample_id": sid,
                    "dataset": ds,
                    "original_split": split_name,
                }
                record.update({name: float(w) for name, w in zip(names, row_w)})
                text_expert_records.append(record)

    metrics = compute_metrics(y_true, y_prob, y_pred)
    for key in loss_records[0].keys() if loss_records else []:
        metrics[key] = _mean_or_nan([row[key] for row in loss_records])
    metrics["loss"] = metrics.get("loss_total", float("nan"))
    metrics["tcp_ratio"] = float(tcp_ratio)
    for key, values in diagnostics.items():
        metrics[key] = _mean_or_nan(values)

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "dataset": datasets,
            "original_split": original_splits,
            "cv_split": ["val"] * len(sample_ids),
            "cv_fold": [-1] * len(sample_ids),
            "label": y_true,
            "prob": y_prob,
            "pred": y_pred,
            "uncertainty": diagnostics["uncertainty"],
            "strength": diagnostics["strength"],
            "speech_prob": speech_probs,
            "text_prob": text_probs,
            "speech_uncertainty": diagnostics["speech_uncertainty"],
            "text_uncertainty": diagnostics["text_uncertainty"],
            "speech_strength": diagnostics["speech_strength"],
            "text_strength": diagnostics["text_strength"],
            "speech_tcp_confidence": diagnostics["speech_tcp_confidence"],
            "text_tcp_confidence": diagnostics["text_tcp_confidence"],
            "speech_tcp_weight": diagnostics["speech_tcp_weight"],
            "text_tcp_weight": diagnostics["text_tcp_weight"],
            "speech_used_weight": diagnostics["speech_used_weight"],
            "text_used_weight": diagnostics["text_used_weight"],
            "speech_fused_evidence_strength": diagnostics["speech_fused_strength"],
            "text_fused_evidence_strength": diagnostics["text_fused_strength"],
            "strength_preservation_error": diagnostics["strength_preservation_error"],
            "modality_js_conflict": diagnostics["modality_js_conflict"],
            "tcp_ratio": [float(tcp_ratio)] * len(sample_ids),
        }
    )

    aux_dfs = {
        "modality_tcp": pd.DataFrame(modality_records),
        "speech_expert": pd.DataFrame(speech_expert_records),
        "text_expert": pd.DataFrame(text_expert_records),
    }
    return metrics, pred_df, aux_dfs


def save_aux_dfs(prefix: str, aux_dfs: dict, run_dir: Path):
    for name, df in aux_dfs.items():
        if df.empty:
            continue
        df.to_csv(
            run_dir / f"{prefix}_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def make_cv_folds(
    dataset_df: pd.DataFrame,
    n_splits: int,
    split_seed: int,
):
    """Build stratified folds at the participant level.

    Stratification is applied to one row per subject, then all recordings of
    each selected subject are expanded back into the sample-level train/val
    DataFrames. This guarantees that a participant can never appear in both
    subsets of the same fold.
    """
    required = {"subject_id", "label"}
    missing = sorted(required - set(dataset_df.columns))
    if missing:
        raise ValueError(
            f"Subject-grouped CV requires columns {sorted(required)}; missing {missing}."
        )

    subject_table = (
        dataset_df[["subject_id", "label"]]
        .drop_duplicates(subset=["subject_id"])
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    subject_labels = subject_table["label"].astype(int).values

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )

    folds = []
    for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(
        skf.split(subject_table["subject_id"], subject_labels),
        start=1,
    ):
        train_subjects = set(
            subject_table.iloc[train_subject_idx]["subject_id"].astype(str)
        )
        val_subjects = set(
            subject_table.iloc[val_subject_idx]["subject_id"].astype(str)
        )

        train_df = dataset_df[
            dataset_df["subject_id"].astype(str).isin(train_subjects)
        ].copy()
        val_df = dataset_df[
            dataset_df["subject_id"].astype(str).isin(val_subjects)
        ].copy()

        _assert_no_subject_leakage(train_df, val_df, fold_idx)

        train_df["cv_split"] = "train"
        val_df["cv_split"] = "val"
        train_df["cv_fold"] = fold_idx
        val_df["cv_fold"] = fold_idx

        # Keep deterministic row order inside each generated manifest.
        train_df = train_df.sort_values(["subject_id", "sample_id"]).reset_index(drop=True)
        val_df = val_df.sort_values(["subject_id", "sample_id"]).reset_index(drop=True)
        folds.append((fold_idx, train_df, val_df))

    # Every subject must occur in validation exactly once across K folds.
    assignment = build_subject_fold_assignment(folds)
    expected_subjects = int(dataset_df["subject_id"].nunique())
    if len(assignment) != expected_subjects:
        raise RuntimeError(
            "Invalid grouped CV assignment: expected "
            f"{expected_subjects} validation subjects across all folds, got "
            f"{len(assignment)}."
        )

    return folds


def safe_metric_value(x, default=-1.0):
    try:
        if pd.isna(x) or np.isnan(float(x)):
            return default
        return float(x)
    except Exception:
        return default


def train_one_run(
    args,
    dataset_name: str,
    fold_idx: int,
    run_idx: int,
    train_seed: int,
    train_csv: Path,
    val_csv: Path,
    run_dir: Path,
    device,
):
    seed_everything(train_seed)
    print(f"[INFO] Fixed training seed: {train_seed}")

    train_loader = make_loader(
        csv_path=train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        preload=args.preload,
    )
    val_loader = make_loader(
        csv_path=val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        preload=args.preload,
    )

    model = DCAMoELayeredTCPFixedSSEOCPosEncNoLangModel(
        hidden_dim=args.hidden_dim,
        num_classes=2,
        evidence_activation=args.evidence_activation,
        tcp_dropout=args.tcp_dropout,
        max_speech_chunks=args.max_speech_chunks,
        speech_position_dropout=args.speech_position_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config_path = save_run_config(
        args=args,
        run_dir=run_dir,
        dataset_name=dataset_name,
        fold_idx=fold_idx,
        run_idx=run_idx,
        train_seed=train_seed,
        train_csv=train_csv,
        val_csv=val_csv,
        model=model,
        device=device,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = run_dir / "model.pt"
    history = []

    no_improve_epochs = 0
    stopped_early = False
    stopped_epoch = None
    stop_reason = "max_epochs_reached"

    for epoch in range(1, args.epochs + 1):
        tcp_ratio = get_tcp_ratio(
            epoch=epoch,
            warmup_epochs=args.tcp_warmup_epochs,
            ramp_epochs=args.tcp_ramp_epochs,
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_unimodal=args.lambda_unimodal,
            lambda_tcp=args.lambda_tcp,
            lambda_seoc=args.lambda_seoc,
            lambda_con=args.lambda_con,
            tcp_ratio=tcp_ratio,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_unimodal=args.lambda_unimodal,
            lambda_tcp=args.lambda_tcp,
            lambda_seoc=args.lambda_seoc,
            lambda_con=args.lambda_con,
            tcp_ratio=tcp_ratio,
        )

        current_val_f1 = safe_metric_value(
            val_metrics.get("f1", -1.0),
            default=-1.0,
        )
        improved = current_val_f1 > (
            best_val_f1 + args.early_stop_min_delta
        )

        if improved:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            no_improve_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "tcp_ratio": tcp_ratio,
                    "val_metrics": val_metrics,
                    "dataset": dataset_name,
                    "fold": fold_idx,
                    "run": run_idx,
                    "split_seed": args.split_seed,
                    "train_seed": train_seed,
                    "train_seed_base": args.train_seed_base,
                    "train_seed_rule": "train_seed = train_seed_base + run_idx - 1",
                    "loss_type": LOSS_TYPE,
                    "lambda_unimodal": args.lambda_unimodal,
                    "lambda_tcp": args.lambda_tcp,
                    "lambda_seoc": args.lambda_seoc,
                    "lambda_con": args.lambda_con,
                    "speech_position_encoding": "fixed_sinusoidal",
                    "uses_language_embedding": False,
                    "max_speech_chunks": args.max_speech_chunks,
                    "speech_position_dropout": args.speech_position_dropout,
                    "tcp_warmup_epochs": args.tcp_warmup_epochs,
                    "tcp_ramp_epochs": args.tcp_ramp_epochs,
                    "grad_clip": args.grad_clip,
                    "run_config_path": str(run_config_path),
                    "args": vars(args),
                },
                best_path,
            )
        else:
            no_improve_epochs += 1

        row = {
            "dataset": dataset_name,
            "fold": fold_idx,
            "run": run_idx,
            "split_seed": args.split_seed,
            "train_seed": train_seed,
            "loss_type": LOSS_TYPE,
            "lambda_unimodal": args.lambda_unimodal,
            "lambda_tcp": args.lambda_tcp,
            "lambda_seoc": args.lambda_seoc,
            "lambda_con": args.lambda_con,
            "speech_position_encoding": "fixed_sinusoidal",
            "uses_language_embedding": False,
            "max_speech_chunks": args.max_speech_chunks,
            "speech_position_dropout": args.speech_position_dropout,
            "tcp_ratio": tcp_ratio,
            "epoch": epoch,
            "best_epoch_so_far": best_epoch,
            "best_val_f1_so_far": best_val_f1,
            "no_improve_epochs": no_improve_epochs,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"[{dataset_name}] Fold {fold_idx}/{args.folds} "
            f"Run {run_idx}/{args.runs} Epoch {epoch:03d} | "
            f"seed={train_seed} tcp_ratio={tcp_ratio:.3f} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"fused={train_metrics['loss_fused']:.4f} "
            f"uni={train_metrics['loss_unimodal']:.4f} "
            f"tcp={train_metrics['loss_tcp']:.4f} "
            f"con={train_metrics['loss_con']:.4f} "
            f"f1={train_metrics['f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"auc={val_metrics['auc']:.4f} "
            f"unc={val_metrics['uncertainty']:.4f} "
            f"tcp_w=({val_metrics['speech_tcp_weight']:.3f},"
            f"{val_metrics['text_tcp_weight']:.3f}) "
            f"S_err={val_metrics['strength_preservation_error']:.2e} | "
            f"best_f1={best_val_f1:.4f} no_improve={no_improve_epochs}"
        )

        if (
            args.early_stop_patience > 0
            and no_improve_epochs >= args.early_stop_patience
        ):
            stopped_early = True
            stopped_epoch = epoch
            stop_reason = (
                f"early_stopping_patience_{args.early_stop_patience}"
            )
            print(
                f"[EARLY STOP] dataset={dataset_name} fold={fold_idx} "
                f"run={run_idx} stopped_epoch={stopped_epoch} "
                f"best_epoch={best_epoch} best_val_f1={best_val_f1:.4f}"
            )
            break

    pd.DataFrame(history).to_csv(
        run_dir / "history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not best_path.exists():
        raise RuntimeError(f"Best checkpoint was not saved: {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    best_tcp_ratio = float(ckpt.get("tcp_ratio", 1.0))

    val_metrics, val_pred, val_aux = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        lambda_unimodal=args.lambda_unimodal,
        lambda_tcp=args.lambda_tcp,
        lambda_seoc=args.lambda_seoc,
        lambda_con=args.lambda_con,
        tcp_ratio=best_tcp_ratio,
    )

    val_pred["cv_fold"] = fold_idx
    val_pred["cv_split"] = "val"
    val_pred.to_csv(
        run_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_aux_dfs("val", val_aux, run_dir)

    result = {
        "dataset": dataset_name,
        "fold": fold_idx,
        "run": run_idx,
        "split_seed": args.split_seed,
        "train_seed": train_seed,
        "train_seed_base": args.train_seed_base,
        "train_seed_rule": "train_seed = train_seed_base + run_idx - 1",
        "loss_type": LOSS_TYPE,
        "lambda_unimodal": args.lambda_unimodal,
        "lambda_tcp": args.lambda_tcp,
        "lambda_seoc": args.lambda_seoc,
        "lambda_con": args.lambda_con,
        "speech_position_encoding": "fixed_sinusoidal",
            "uses_language_embedding": False,
        "max_speech_chunks": args.max_speech_chunks,
        "speech_position_dropout": args.speech_position_dropout,
        "tcp_warmup_epochs": args.tcp_warmup_epochs,
        "tcp_ramp_epochs": args.tcp_ramp_epochs,
        "best_tcp_ratio": best_tcp_ratio,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "best_model_path": str(best_path),
        "run_config_path": str(run_config_path),
        "grad_clip": args.grad_clip,
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(
        f"\n[RUN BEST] dataset={dataset_name} fold={fold_idx} run={run_idx} "
        f"seed={train_seed} best_epoch={best_epoch} "
        f"tcp_ratio={best_tcp_ratio:.3f} val_f1={val_metrics['f1']:.4f} "
        f"val_auc={val_metrics['auc']:.4f} "
        f"val_uncertainty={val_metrics['uncertainty']:.4f} "
        f"S_error={val_metrics['strength_preservation_error']:.2e}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def choose_best_run_for_fold(fold_results):
    sorted_results = sorted(
        fold_results,
        key=lambda item: (
            safe_metric_value(item.get("val_f1", -1.0), default=-1.0),
            safe_metric_value(item.get("val_auc", -1.0), default=-1.0),
            safe_metric_value(item.get("val_acc", -1.0), default=-1.0),
            -safe_metric_value(item.get("val_loss", 1e9), default=1e9),
        ),
        reverse=True,
    )
    return sorted_results[0]


def summarize_best_folds(
    best_fold_results,
    output_dir: Path,
    dataset_name: str,
    args,
):
    df = pd.DataFrame(best_fold_results)
    out_csv = output_dir / "cv5_best_folds_summary.csv"
    out_json = output_dir / "cv5_best_folds_summary.json"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    mean_std = {
        "dataset": dataset_name,
        "folds": args.folds,
        "split_seed": args.split_seed,
        "train_seed_base": args.train_seed_base,
        "train_seed_rule": "train_seed = train_seed_base + run_idx - 1",
        "loss_type": LOSS_TYPE,
        "lambda_unimodal": args.lambda_unimodal,
        "lambda_tcp": args.lambda_tcp,
        "lambda_seoc": args.lambda_seoc,
        "lambda_con": args.lambda_con,
        "speech_position_encoding": "fixed_sinusoidal",
            "uses_language_embedding": False,
        "max_speech_chunks": args.max_speech_chunks,
        "speech_position_dropout": args.speech_position_dropout,
        "tcp_warmup_epochs": args.tcp_warmup_epochs,
        "tcp_ramp_epochs": args.tcp_ramp_epochs,
        "runs_per_fold": args.runs,
        "epochs": args.epochs,
    }

    for column in df.columns:
        if column.startswith("val_") and pd.api.types.is_numeric_dtype(df[column]):
            mean_std[column + "_mean"] = float(df[column].mean())
            mean_std[column + "_std"] = float(df[column].std())

    if "epochs_trained" in df.columns:
        mean_std["epochs_trained_mean"] = float(df["epochs_trained"].mean())
        mean_std["epochs_trained_std"] = float(df["epochs_trained"].std())
    if "stopped_early" in df.columns:
        mean_std["stopped_early_count"] = int(df["stopped_early"].sum())

    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(mean_std, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f"[CV5 FINAL SUMMARY] {dataset_name}")
    print("=" * 100)
    print(df)
    print("\n[MEAN ± STD OVER FIVE FOLD-BEST RESULTS]")
    for key, value in mean_std.items():
        print(f"{key}: {value}")
    print(f"\n[SAVE] {out_csv}")
    print(f"[SAVE] {out_json}")

    return df, mean_std


def main():
    args = parse_args()

    if (
        args.lambda_unimodal < 0
        or args.lambda_tcp < 0
        or args.lambda_seoc < 0
        or args.lambda_con < 0
    ):
        raise ValueError("All loss weights must be non-negative.")
    if args.tcp_warmup_epochs < 0 or args.tcp_ramp_epochs < 0:
        raise ValueError("TCP warm-up/ramp epochs must be non-negative.")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")
    if args.max_speech_chunks <= 0:
        raise ValueError("max_speech_chunks must be positive.")
    if not 0.0 <= args.speech_position_dropout < 1.0:
        raise ValueError(
            "speech_position_dropout must be in [0, 1)."
        )

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    full_df = pd.read_csv(manifest_path)
    required_cols = [
        "sample_id",
        "dataset",
        "split",
        "label",
        "lang_id",
        "text_expert_group",
        "cached_feature_path",
    ]
    missing_cols = [column for column in required_cols if column not in full_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    dataset_df = full_df[full_df["dataset"] == args.dataset].copy()
    if dataset_df.empty:
        raise RuntimeError(f"No samples found for dataset: {args.dataset}")
    dataset_df["label"] = dataset_df["label"].astype(int)
    dataset_df, subject_id_source = attach_and_validate_subject_ids(
        dataset_df=dataset_df,
        dataset_name=args.dataset,
        subject_id_column=args.subject_id_column,
        n_splits=args.folds,
    )

    dataset_output_dir = Path(args.output_dir) / args.dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 100)
    print(
        "[DCA-MoE SPEECH SINUSOIDAL POSITION ENCODING + FIXED-S TCP "
        "+ FULL 3D-SEOC: STRATIFIED FIVE-FOLD CV]"
    )
    print("=" * 100)
    settings = {
        "dataset": args.dataset,
        "manifest": args.manifest,
        "output_dir": str(dataset_output_dir),
        "device": str(device),
        "folds": args.folds,
        "runs_per_fold": args.runs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "speech_position_encoding": "fixed_sinusoidal",
            "uses_language_embedding": False,
        "max_speech_chunks": args.max_speech_chunks,
        "speech_position_dropout": args.speech_position_dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "evidence_activation": args.evidence_activation,
        "tcp_dropout": args.tcp_dropout,
        "training_seeds": [
            args.train_seed_base + index
            for index in range(args.runs)
        ],
        "num_workers": args.num_workers,
        "preload": args.preload,
        "split_only": args.split_only,
        "split_seed": args.split_seed,
        "cv_splitter": "subject-level StratifiedKFold",
        "subject_id_source": subject_id_source,
        "total_subjects": int(dataset_df["subject_id"].nunique()),
        "train_seed_base": args.train_seed_base,
        "loss_type": LOSS_TYPE,
        "lambda_unimodal": args.lambda_unimodal,
        "lambda_tcp": args.lambda_tcp,
        "lambda_seoc": args.lambda_seoc,
        "lambda_con": args.lambda_con,
        "tcp_warmup_epochs": args.tcp_warmup_epochs,
        "tcp_ramp_epochs": args.tcp_ramp_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
    }
    for key, value in settings.items():
        print(f"{key:26s}: {value}")

    print("\n[MERGED DATASET COUNTS]")
    print(dataset_df.groupby(["dataset", "split"]).size())
    print("\n[MERGED LABEL COUNTS: RECORDINGS]")
    print(dataset_df["label"].value_counts().to_dict())
    print("\n[SUBJECT COUNTS]")
    print(f"total subjects: {dataset_df['subject_id'].nunique()}")
    print(f"subject-id source: {subject_id_source}")
    print(f"subjects by label: {_subject_label_counts(dataset_df)}")
    recordings_per_subject = dataset_df.groupby("subject_id").size()
    print(
        "recordings per subject: "
        f"min={recordings_per_subject.min()} "
        f"max={recordings_per_subject.max()} "
        f"mean={recordings_per_subject.mean():.3f}"
    )

    folds = make_cv_folds(
        dataset_df=dataset_df,
        n_splits=args.folds,
        split_seed=args.split_seed,
    )

    subject_assignment = build_subject_fold_assignment(folds)
    subject_assignment_path = (
        dataset_output_dir / "cv_subject_fold_assignment.csv"
    )
    subject_assignment.to_csv(
        subject_assignment_path,
        index=False,
        encoding="utf-8-sig",
    )

    split_audit = {
        "dataset": args.dataset,
        "splitter": "subject-level StratifiedKFold(shuffle=True)",
        "split_seed": args.split_seed,
        "folds": args.folds,
        "subject_id_source": subject_id_source,
        "total_recordings": int(len(dataset_df)),
        "total_subjects": int(dataset_df["subject_id"].nunique()),
        "subjects_by_label": _subject_label_counts(dataset_df),
        "every_subject_in_validation_exactly_once": bool(
            subject_assignment["subject_id"].nunique()
            == dataset_df["subject_id"].nunique()
            == len(subject_assignment)
        ),
        "all_fold_subject_overlap_counts_zero": bool(
            all(
                not set(train_df["subject_id"].astype(str)).intersection(
                    set(val_df["subject_id"].astype(str))
                )
                for _, train_df, val_df in folds
            )
        ),
        "fold_records": _build_fold_records(folds),
        "subject_assignment_csv": str(subject_assignment_path.resolve()),
    }
    split_audit_path = dataset_output_dir / "cv_group_split_audit.json"
    _write_json(split_audit_path, split_audit)

    experiment_config_path, _ = save_experiment_config(
        args=args,
        dataset_df=dataset_df,
        folds=folds,
        dataset_output_dir=dataset_output_dir,
        device=device,
        subject_id_source=subject_id_source,
    )
    print(f"\n[SAVE] subject fold assignment: {subject_assignment_path}")
    print(f"[SAVE] grouped split audit: {split_audit_path}")
    print(f"[SAVE] experiment config: {experiment_config_path}")

    if args.split_only:
        for fold_idx, train_df, val_df in folds:
            fold_dir = dataset_output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(
                fold_dir / "fold_train.csv",
                index=False,
                encoding="utf-8-sig",
            )
            val_df.to_csv(
                fold_dir / "fold_val.csv",
                index=False,
                encoding="utf-8-sig",
            )
        print("[SPLIT ONLY] Fold manifests and audit files were created; training skipped.")
        return

    all_run_results = []
    best_fold_results = []

    for fold_idx, train_df, val_df in folds:
        fold_dir = dataset_output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_csv = fold_dir / "fold_train.csv"
        val_csv = fold_dir / "fold_val.csv"
        train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 100)
        print(f"[FOLD {fold_idx}/{args.folds}]")
        print("=" * 100)
        train_subjects = train_df["subject_id"].nunique()
        val_subjects = val_df["subject_id"].nunique()
        subject_overlap = set(train_df["subject_id"]).intersection(
            set(val_df["subject_id"])
        )
        print(f"train samples : {len(train_df)}")
        print(f"val samples   : {len(val_df)}")
        print(f"train subjects: {train_subjects}")
        print(f"val subjects  : {val_subjects}")
        print(f"subject overlap: {len(subject_overlap)}")
        print(f"train labels (recordings): {train_df['label'].value_counts().to_dict()}")
        print(f"val labels   (recordings): {val_df['label'].value_counts().to_dict()}")
        print(f"train labels (subjects): {_subject_label_counts(train_df)}")
        print(f"val labels   (subjects): {_subject_label_counts(val_df)}")

        fold_results = []
        for run_idx in range(1, args.runs + 1):
            train_seed = args.train_seed_base + run_idx - 1
            run_dir = fold_dir / f"run_{run_idx}_seed{train_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "-" * 100)
            print(
                f"[DATASET {args.dataset}] [FOLD {fold_idx}/{args.folds}] "
                f"[RUN {run_idx}/{args.runs}] seed={train_seed} "
                f"lambda_uni={args.lambda_unimodal} "
                f"lambda_tcp={args.lambda_tcp} "
                f"lambda_seoc={args.lambda_seoc} "
                f"lambda_con={args.lambda_con} "
                f"speech_pos=fixed_sinusoidal"
            )
            print("-" * 100)

            result = train_one_run(
                args=args,
                dataset_name=args.dataset,
                fold_idx=fold_idx,
                run_idx=run_idx,
                train_seed=train_seed,
                train_csv=train_csv,
                val_csv=val_csv,
                run_dir=run_dir,
                device=device,
            )
            fold_results.append(result)
            all_run_results.append(result)

        fold_runs_df = pd.DataFrame(fold_results)
        fold_runs_csv = fold_dir / f"fold_{fold_idx}_runs_summary.csv"
        fold_runs_df.to_csv(
            fold_runs_csv,
            index=False,
            encoding="utf-8-sig",
        )

        best_result = dict(choose_best_run_for_fold(fold_results))
        best_result["fold_best_selected_by"] = "max_val_f1"
        best_result["fold_runs_summary_path"] = str(fold_runs_csv)

        best_json = fold_dir / f"fold_{fold_idx}_best_summary.json"
        with open(best_json, "w", encoding="utf-8") as handle:
            json.dump(best_result, handle, indent=2, ensure_ascii=False)
        best_fold_results.append(best_result)

        print("\n" + "*" * 100)
        print(f"[FOLD {fold_idx} BEST RUN]")
        print(
            f"run={best_result['run']} seed={best_result['train_seed']} "
            f"best_epoch={best_result['best_epoch']} "
            f"tcp_ratio={best_result['best_tcp_ratio']:.3f} "
            f"val_f1={best_result['val_f1']:.4f} "
            f"val_acc={best_result['val_acc']:.4f} "
            f"val_auc={best_result['val_auc']:.4f} "
            f"val_uncertainty={best_result['val_uncertainty']:.4f}"
        )
        print(f"[SAVE] {fold_runs_csv}")
        print(f"[SAVE] {best_json}")
        print("*" * 100)

    all_runs_df = pd.DataFrame(all_run_results)
    all_runs_csv = dataset_output_dir / "cv5_all_runs_summary.csv"
    all_runs_df.to_csv(all_runs_csv, index=False, encoding="utf-8-sig")

    ttest_paths = export_ttest_files(
        all_runs_df=all_runs_df,
        output_dir=dataset_output_dir,
        dataset_name=args.dataset,
    )

    summarize_best_folds(
        best_fold_results=best_fold_results,
        output_dir=dataset_output_dir,
        dataset_name=args.dataset,
        args=args,
    )

    print(f"\n[SAVE] all runs summary: {all_runs_csv}")
    for name, path in ttest_paths.items():
        print(f"[SAVE] {name}: {path}")
    print(
        "[DONE] Speech sinusoidal position encoding + Fixed-S TCP + "
        "branch confidence-uncertainty alignment training finished."
    )


if __name__ == "__main__":
    main()