

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def safe_torch_load(path: Path):
    """
    Use weights_only=True if supported by current PyTorch.
    Fall back to normal torch.load if the PyTorch version does not support it.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class DCAMoECachedDataset(Dataset):
    def __init__(self, manifest_csv: str, preload: bool = False):
        self.manifest_csv = manifest_csv
        self.df = pd.read_csv(manifest_csv)
        self.preload = preload
        self.cache = None

        required_cols = [
            "sample_id",
            "dataset",
            "split",
            "label",
            "lang_id",
            "text_expert_group",
            "cached_feature_path",
        ]

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        if self.preload:
            self._preload_to_ram()

    def _preload_to_ram(self):
        """
        Load all samples in current manifest into CPU RAM.

        Important:
            This loads features to CPU RAM, not GPU memory.
            GPU still receives only each current batch during training.
        """
        self.cache = []

        print("=" * 100)
        print("[PRELOAD] Loading cached features into CPU RAM")
        print(f"[PRELOAD] manifest: {self.manifest_csv}")
        print(f"[PRELOAD] samples : {len(self.df)}")
        print("=" * 100)

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preload cached features"):
            cached_path = Path(str(row["cached_feature_path"]))

            if not cached_path.exists():
                raise FileNotFoundError(f"Cached feature not found: {cached_path}")

            data = safe_torch_load(cached_path)
            self.cache.append(data)

        print(f"[PRELOAD DONE] Loaded {len(self.cache)} samples into CPU RAM.")

    def __len__(self):
        return len(self.df)

    def _load_one(self, idx: int) -> Dict[str, Any]:
        if self.preload:
            return self.cache[idx]

        row = self.df.iloc[idx]
        cached_path = Path(str(row["cached_feature_path"]))

        if not cached_path.exists():
            raise FileNotFoundError(f"Cached feature not found: {cached_path}")

        return safe_torch_load(cached_path)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self._load_one(idx)

        text_feats = {}
        for expert, feat in data["text_feats"].items():
            item = {
                "all_layer_pooled_feats": feat["all_layer_pooled_feats"].float(),
            }

            if "attention_mask" in feat:
                item["attention_mask"] = feat["attention_mask"].long()

            text_feats[expert] = item

        speech_feats = {}
        for expert, feat in data["speech_feats"].items():
            speech_feats[expert] = {
                "all_layer_chunk_feats": feat["all_layer_chunk_feats"].float(),
                "chunk_mask": feat["chunk_mask"].bool(),
            }

            if "chunk_spans" in feat:
                speech_feats[expert]["chunk_spans"] = feat["chunk_spans"]

        return {
            "sample_id": str(data["sample_id"]),
            "dataset": str(data["dataset"]),
            "split": str(data["split"]),
            "label": torch.tensor(int(data["label"]), dtype=torch.long),
            "lang_id": torch.tensor(int(data["lang_id"]), dtype=torch.long),
            "text_expert_group": str(data["text_expert_group"]),
            "text_feats": text_feats,
            "speech_feats": speech_feats,
        }


def pad_speech_layered(seqs: List[torch.Tensor], masks: List[torch.Tensor]):
    """
    Pad speech layered features.

    Input:
        seqs:
            list of [L, N_i, D]
        masks:
            list of [N_i]

    Output:
        padded:
            [B, L, max_N, D]
        padded_mask:
            [B, max_N]
    """
    batch_size = len(seqs)
    L = seqs[0].shape[0]
    D = seqs[0].shape[2]
    max_N = max(x.shape[1] for x in seqs)

    padded = torch.zeros(batch_size, L, max_N, D, dtype=torch.float32)
    padded_mask = torch.zeros(batch_size, max_N, dtype=torch.bool)

    for i, (x, m) in enumerate(zip(seqs, masks)):
        n = x.shape[1]
        padded[i, :, :n, :] = x.float()
        padded_mask[i, :n] = m.bool()

    return padded, padded_mask


def dca_moe_cached_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_ids = [x["sample_id"] for x in batch]
    datasets = [x["dataset"] for x in batch]
    splits = [x["split"] for x in batch]
    text_expert_groups = [x["text_expert_group"] for x in batch]

    labels = torch.stack([x["label"] for x in batch], dim=0)
    lang_ids = torch.stack([x["lang_id"] for x in batch], dim=0)

    # Speech experts
    speech_experts = sorted(set(k for item in batch for k in item["speech_feats"].keys()))

    speech_feats = {}
    speech_masks = {}

    for expert in speech_experts:
        seqs = []
        masks = []

        for item in batch:
            if expert not in item["speech_feats"]:
                raise KeyError(f"Missing speech expert {expert} in sample {item['sample_id']}")

            seqs.append(item["speech_feats"][expert]["all_layer_chunk_feats"])
            masks.append(item["speech_feats"][expert]["chunk_mask"])

        padded, padded_mask = pad_speech_layered(seqs, masks)
        speech_feats[expert] = padded
        speech_masks[expert] = padded_mask

    # Text experts.
    # Different language samples can have different text expert sets.
    text_experts = sorted(set(k for item in batch for k in item["text_feats"].keys()))

    text_feats = {}
    text_available = {}

    for expert in text_experts:
        L = None
        D = None

        for item in batch:
            if expert in item["text_feats"]:
                feat = item["text_feats"][expert]["all_layer_pooled_feats"]
                L, D = feat.shape
                break

        if L is None or D is None:
            continue

        feats = torch.zeros(len(batch), L, D, dtype=torch.float32)
        available = torch.zeros(len(batch), dtype=torch.bool)

        for i, item in enumerate(batch):
            if expert in item["text_feats"]:
                feats[i] = item["text_feats"][expert]["all_layer_pooled_feats"].float()
                available[i] = True

        text_feats[expert] = feats
        text_available[expert] = available

    return {
        "sample_id": sample_ids,
        "dataset": datasets,
        "split": splits,
        "label": labels,
        "lang_id": lang_ids,
        "text_expert_group": text_expert_groups,
        "speech_feats": speech_feats,
        "speech_masks": speech_masks,
        "text_feats": text_feats,
        "text_available": text_available,
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    manifest = "/data2/moe/data/manifest_dca_moe_cached_features.csv"

    dataset = DCAMoECachedDataset(manifest, preload=True)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dca_moe_cached_collate_fn,
    )

    batch = next(iter(loader))

    print("sample_id:", batch["sample_id"])
    print("label:", batch["label"])
    print("lang_id:", batch["lang_id"])

    print("\nSpeech:")
    for k, v in batch["speech_feats"].items():
        print(k, v.shape, batch["speech_masks"][k].shape)

    print("\nText:")
    for k, v in batch["text_feats"].items():
        print(k, v.shape, batch["text_available"][k])