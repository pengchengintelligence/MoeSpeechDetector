#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build cached DCA-MoE feature files.

This script does NOT re-extract features.
It only merges already extracted text and speech feature .pt files into one cached .pt file per sample.

Input:
    /data1/haoran_moe/data/manifest_dca_moe_layered_features.csv

Original feature files:
    text_layered_feature_path
    speech_layered_feature_path

Output:
    /data1/haoran_moe/features/dca_moe_cached/<dataset>/<sample_id>.pt
    /data1/haoran_moe/data/manifest_dca_moe_cached_features.csv

Each cached .pt contains:
    sample_id
    dataset
    split
    label
    lang_id
    text_expert_group
    text_feats
    speech_feats
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default="/data1/haoran_moe/data/manifest_dca_moe_layered_features.csv",
        help="Manifest containing text_layered_feature_path and speech_layered_feature_path.",
    )

    parser.add_argument(
        "--save_root",
        type=str,
        default="/data1/haoran_moe/features/dca_moe_cached",
        help="Directory to save merged cached features.",
    )

    parser.add_argument(
        "--output_manifest",
        type=str,
        default="/data1/haoran_moe/data/manifest_dca_moe_cached_features.csv",
        help="Output manifest with cached_feature_path column.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached .pt files.",
    )

    return parser.parse_args()


def safe_torch_load(path: Path):
    """
    Use weights_only=True when supported by current PyTorch.
    If the installed PyTorch does not support it, fall back to normal torch.load.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    save_root = Path(args.save_root)
    output_manifest = Path(args.output_manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = [
        "sample_id",
        "dataset",
        "split",
        "label",
        "lang_id",
        "text_expert_group",
        "text_layered_feature_path",
        "speech_layered_feature_path",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in manifest: {col}")

    cached_paths = []
    missing_text = 0
    missing_speech = 0
    saved_count = 0
    skipped_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building cached features"):
        sample_id = str(row["sample_id"]).strip()
        dataset = str(row["dataset"]).strip()

        text_path = Path(str(row["text_layered_feature_path"]))
        speech_path = Path(str(row["speech_layered_feature_path"]))

        out_dir = save_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{sample_id}.pt"
        cached_paths.append(str(out_path))

        if out_path.exists() and not args.overwrite:
            skipped_count += 1
            continue

        if not text_path.exists():
            print(f"[MISSING TEXT] {text_path}")
            missing_text += 1
            continue

        if not speech_path.exists():
            print(f"[MISSING SPEECH] {speech_path}")
            missing_speech += 1
            continue

        text_data = safe_torch_load(text_path)
        speech_data = safe_torch_load(speech_path)

        if "text_feats" not in text_data:
            raise KeyError(f"text_feats not found in {text_path}")

        if "speech_feats" not in speech_data:
            raise KeyError(f"speech_feats not found in {speech_path}")

        cached = {
            "sample_id": sample_id,
            "dataset": dataset,
            "split": str(row["split"]),
            "label": int(row["label"]),
            "lang_id": int(row["lang_id"]),
            "text_expert_group": str(row["text_expert_group"]),
            "text_feats": text_data["text_feats"],
            "speech_feats": speech_data["speech_feats"],
        }

        torch.save(cached, out_path)
        saved_count += 1

    out_df = df.copy()
    out_df["cached_feature_path"] = cached_paths

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_manifest, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[DONE] Cached feature building finished.")
    print("=" * 80)
    print(f"Input manifest   : {manifest_path}")
    print(f"Cache root       : {save_root}")
    print(f"Output manifest  : {output_manifest}")
    print(f"Total samples    : {len(df)}")
    print(f"Saved            : {saved_count}")
    print(f"Skipped existing : {skipped_count}")
    print(f"Missing text     : {missing_text}")
    print(f"Missing speech   : {missing_speech}")


if __name__ == "__main__":
    main()