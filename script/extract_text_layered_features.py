#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract layered text expert features for DCA-MoE.

For each sample and each selected text expert, save:
    all_layer_pooled_feats: [L, D]
    last4_token_feats:      [4, T, D]
    attention_mask:         [T]

where:
    L = number of hidden-state layers, usually embedding layer + transformer layers
    T = token length after padding/truncation
    D = hidden dimension

Output:
    features/text_layered/<dataset>/<sample_id>.pt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


TEXT_MODEL_CONFIG = {
    "english": {
        "bert-large-uncased": "/data1/haoran_moe/pretrained_models/text/English/bert-large-uncased",
        "roberta-large": "/data1/haoran_moe/pretrained_models/text/English/roberta-large",
        "deberta-v3-large": "/data1/haoran_moe/pretrained_models/text/English/deberta-v3-large",
        "bge-large-en-v1.5": "/data1/haoran_moe/pretrained_models/text/English/bge-large-en-v1.5",
    },
    "chinese": {
        "chinese-bert-wwm-ext": "/data1/haoran_moe/pretrained_models/text/chinese/chinese-bert-wwm-ext",
        "chinese-roberta-wwm-ext-large": "/data1/haoran_moe/pretrained_models/text/chinese/chinese-roberta-wwm-ext-large",
        "DeBERTa-v2-Large-Chinese": "/data1/haoran_moe/pretrained_models/text/chinese/DeBERTa-v2-Large-Chinese",
        "bge-large-zh-v1.5": "/data1/haoran_moe/pretrained_models/text/chinese/bge-large-zh-v1.5",
    },
    "multilingual": {
        "bert-base-multilingual-cased": "/data1/haoran_moe/pretrained_models/text/multilingual/bert-base-multilingual-cased",
        "xlm-roberta-large": "/data1/haoran_moe/pretrained_models/text/multilingual/xlm-roberta-large",
        "mdeberta-v3-base": "/data1/haoran_moe/pretrained_models/text/multilingual/mdeberta-v3-base",
        "bge-m3": "/data1/haoran_moe/pretrained_models/text/multilingual/bge-m3",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default="/data1/haoran_moe/data/manifest_whisper_large_v3_all.csv",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/data1/haoran_moe/features/text_layered",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--store_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Use float16 to reduce disk usage.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def to_label(x):
    try:
        return int(x)
    except Exception:
        return x


def make_output_path(save_root: Path, dataset: str, sample_id: str) -> Path:
    out_dir = save_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{sample_id}.pt"


def load_existing_or_init(row: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    if out_path.exists():
        data = torch.load(out_path, map_location="cpu")
        if "text_feats" not in data:
            data["text_feats"] = {}
        return data

    data = {
        "sample_id": str(row["sample_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "label": to_label(row["label"]),
        "lang": str(row["lang"]),
        "lang_id": int(row["lang_id"]) if "lang_id" in row and pd.notna(row["lang_id"]) else None,
        "text_expert_group": str(row["text_expert_group"]),
        "transcript_text": str(row.get("transcript_text", "")),
        "text_feats": {},
    }
    return data


def mask_mean_pool_all_layers(
    hidden_states: List[torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    hidden_states:
        tuple/list of L tensors, each [B, T, D]
    attention_mask:
        [B, T]

    return:
        all_layer_pooled_feats: [B, L, D]
    """
    stacked = torch.stack(list(hidden_states), dim=1)  # [B, L, T, D]
    mask = attention_mask[:, None, :, None].float()    # [B, 1, T, 1]

    summed = (stacked * mask).sum(dim=2)               # [B, L, D]
    denom = mask.sum(dim=2).clamp(min=1e-6)            # [B, 1, 1]

    pooled = summed / denom                            # [B, L, D]
    return pooled


def choose_store_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def load_text_model(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def encode_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    store_dtype: torch.dtype,
):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(
        **encoded,
        output_hidden_states=True,
        return_dict=True,
    )

    if outputs.hidden_states is None:
        raise RuntimeError("outputs.hidden_states is None. Please check model config.")

    hidden_states = outputs.hidden_states
    attention_mask = encoded["attention_mask"]  # [B, T]

    all_layer_pooled = mask_mean_pool_all_layers(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )  # [B, L, D]

    stacked = torch.stack(list(hidden_states), dim=1)  # [B, L, T, D]
    last4_token_feats = stacked[:, -4:, :, :]          # [B, 4, T, D]

    return {
        "all_layer_pooled_feats": all_layer_pooled.detach().cpu().to(store_dtype),
        "last4_token_feats": last4_token_feats.detach().cpu().to(store_dtype),
        "attention_mask": attention_mask.detach().cpu().to(torch.long),
    }


def expert_already_done(data: Dict[str, Any], expert_name: str) -> bool:
    if "text_feats" not in data:
        return False
    if expert_name not in data["text_feats"]:
        return False

    feat = data["text_feats"][expert_name]
    required_keys = ["all_layer_pooled_feats", "last4_token_feats", "attention_mask"]

    return all(k in feat for k in required_keys)


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    save_root = Path(args.save_root)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    store_dtype = choose_store_dtype(args.store_dtype)

    print(f"[INFO] device: {device}")
    print(f"[INFO] manifest: {manifest_path}")
    print(f"[INFO] save_root: {save_root}")
    print(f"[INFO] max_length: {args.max_length}")
    print(f"[INFO] batch_size: {args.batch_size}")
    print(f"[INFO] store_dtype: {store_dtype}")

    df = pd.read_csv(manifest_path)

    required_cols = [
        "sample_id",
        "dataset",
        "split",
        "label",
        "lang",
        "text_expert_group",
        "transcript_text",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in manifest: {col}")

    df["transcript_text"] = df["transcript_text"].fillna("").astype(str)

    for group_name, model_dict in TEXT_MODEL_CONFIG.items():
        group_df = df[df["text_expert_group"] == group_name].copy()

        if len(group_df) == 0:
            print(f"[SKIP] No samples for text expert group: {group_name}")
            continue

        print("\n" + "=" * 80)
        print(f"[TEXT GROUP] {group_name} | samples={len(group_df)}")
        print("=" * 80)

        rows = group_df.to_dict("records")

        for expert_name, model_path in model_dict.items():
            print("\n" + "-" * 80)
            print(f"[LOAD TEXT EXPERT] {expert_name}")
            print(f"[PATH] {model_path}")
            print("-" * 80)

            tokenizer, model = load_text_model(model_path, device)

            for start in tqdm(range(0, len(rows), args.batch_size), desc=f"{group_name}:{expert_name}"):
                batch_rows = rows[start:start + args.batch_size]

                effective_rows = []
                texts = []

                for r in batch_rows:
                    sample_id = str(r["sample_id"])
                    dataset = str(r["dataset"])
                    out_path = make_output_path(save_root, dataset, sample_id)

                    data = load_existing_or_init(r, out_path)

                    if args.resume and expert_already_done(data, expert_name):
                        continue

                    effective_rows.append(r)
                    texts.append(str(r["transcript_text"]))

                if len(effective_rows) == 0:
                    continue

                feats = encode_batch(
                    texts=texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=args.max_length,
                    store_dtype=store_dtype,
                )

                for i, r in enumerate(effective_rows):
                    sample_id = str(r["sample_id"])
                    dataset = str(r["dataset"])
                    out_path = make_output_path(save_root, dataset, sample_id)

                    data = load_existing_or_init(r, out_path)

                    data["text_feats"][expert_name] = {
                        "all_layer_pooled_feats": feats["all_layer_pooled_feats"][i],
                        "last4_token_feats": feats["last4_token_feats"][i],
                        "attention_mask": feats["attention_mask"][i],
                        "model_path": model_path,
                        "max_length": args.max_length,
                    }

                    torch.save(data, out_path)

            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n[DONE] Layered text features extracted.")


if __name__ == "__main__":
    main()