#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract layered speech expert chunk features for DCA-MoE.

For each sample and each speech expert, save:
    all_layer_chunk_feats: [L, N, D]
    chunk_mask:            [N]

where:
    L = number of hidden-state layers, usually embedding layer + transformer layers
    N = number of audio chunks
    D = hidden dimension

Output:
    features/speech_layered/<dataset>/<sample_id>.pt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor


SPEECH_MODEL_CONFIG = {
    "wav2vec2_large": "/data1/haoran_moe/pretrained_models/speech/wav2vec2_large",
    "wavlm-large": "/data1/haoran_moe/pretrained_models/speech/wavlm-large",
    "hubert-large-ls960-ft": "/data1/haoran_moe/pretrained_models/speech/hubert-large-ls960-ft",
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
        default="/data1/haoran_moe/features/speech_layered",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--overlap_seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
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


def choose_store_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


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
        if "speech_feats" not in data:
            data["speech_feats"] = {}
        return data

    data = {
        "sample_id": str(row["sample_id"]),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "label": to_label(row["label"]),
        "lang": str(row["lang"]),
        "lang_id": int(row["lang_id"]) if "lang_id" in row and pd.notna(row["lang_id"]) else None,
        "audio_path": str(row["audio_path"]),
        "speech_feats": {},
    }
    return data


def read_audio(audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(audio_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if audio.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    return audio.astype(np.float32), sr


def split_audio_with_overlap(
    audio: np.ndarray,
    sr: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    chunk_len = int(sr * chunk_seconds)
    overlap_len = int(sr * overlap_seconds)

    if chunk_len <= 0:
        raise ValueError(f"Invalid chunk_seconds: {chunk_seconds}")

    if overlap_len < 0:
        raise ValueError(f"Invalid overlap_seconds: {overlap_seconds}")

    if overlap_len >= chunk_len:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds.")

    if len(audio) <= chunk_len:
        return [audio], [(0.0, len(audio) / sr)]

    step = chunk_len - overlap_len
    chunks = []
    spans = []

    start = 0
    while start < len(audio):
        end = min(start + chunk_len, len(audio))
        chunk = audio[start:end]

        if len(chunk) >= int(0.5 * sr):
            chunks.append(chunk)
            spans.append((start / sr, end / sr))

        if end >= len(audio):
            break

        start += step

    return chunks, spans


def load_speech_processor(model_path: str):
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    return processor


def load_speech_model(model_path: str, device: torch.device):
    processor = load_speech_processor(model_path)

    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return processor, model


@torch.no_grad()
def encode_one_chunk_all_layers(
    chunk: np.ndarray,
    sr: int,
    processor,
    model,
    device: torch.device,
    store_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return:
        layer_chunk_feat: [L, D]

    For each hidden-state layer:
        hidden: [1, T, D]
        mean over T -> [D]
    """
    inputs = processor(
        chunk,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

    if outputs.hidden_states is None:
        raise RuntimeError("outputs.hidden_states is None. Please check model config.")

    layer_feats = []

    for h in outputs.hidden_states:
        # h: [1, T, D]
        pooled = h.mean(dim=1).squeeze(0)  # [D]
        layer_feats.append(pooled.detach().cpu().to(store_dtype))

    layer_chunk_feat = torch.stack(layer_feats, dim=0)  # [L, D]
    return layer_chunk_feat


@torch.no_grad()
def encode_chunks_all_layers(
    chunks: List[np.ndarray],
    sr: int,
    processor,
    model,
    device: torch.device,
    store_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return:
        all_layer_chunk_feats: [L, N, D]
    """
    chunk_layer_feats = []

    for chunk in chunks:
        layer_chunk_feat = encode_one_chunk_all_layers(
            chunk=chunk,
            sr=sr,
            processor=processor,
            model=model,
            device=device,
            store_dtype=store_dtype,
        )  # [L, D]

        chunk_layer_feats.append(layer_chunk_feat)

    # [N, L, D]
    stacked = torch.stack(chunk_layer_feats, dim=0)

    # [L, N, D]
    all_layer_chunk_feats = stacked.permute(1, 0, 2).contiguous()

    return all_layer_chunk_feats


def expert_already_done(data: Dict[str, Any], expert_name: str) -> bool:
    if "speech_feats" not in data:
        return False
    if expert_name not in data["speech_feats"]:
        return False

    feat = data["speech_feats"][expert_name]
    required_keys = ["all_layer_chunk_feats", "chunk_mask", "chunk_spans"]

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
    print(f"[INFO] chunk_seconds: {args.chunk_seconds}")
    print(f"[INFO] overlap_seconds: {args.overlap_seconds}")
    print(f"[INFO] target_sr: {args.target_sr}")
    print(f"[INFO] store_dtype: {store_dtype}")

    df = pd.read_csv(manifest_path)

    required_cols = ["sample_id", "dataset", "split", "label", "lang", "audio_path"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in manifest: {col}")

    rows = df.to_dict("records")

    for expert_name, model_path in SPEECH_MODEL_CONFIG.items():
        print("\n" + "=" * 80)
        print(f"[LOAD SPEECH EXPERT] {expert_name}")
        print(f"[PATH] {model_path}")
        print("=" * 80)

        processor, model = load_speech_model(model_path, device)

        for row in tqdm(rows, desc=f"speech:{expert_name}"):
            sample_id = str(row["sample_id"])
            dataset = str(row["dataset"])
            out_path = make_output_path(save_root, dataset, sample_id)

            data = load_existing_or_init(row, out_path)

            if args.resume and expert_already_done(data, expert_name):
                continue

            audio_path = str(row["audio_path"])
            audio, sr = read_audio(audio_path, target_sr=args.target_sr)

            chunks, spans = split_audio_with_overlap(
                audio=audio,
                sr=sr,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
            )

            all_layer_chunk_feats = encode_chunks_all_layers(
                chunks=chunks,
                sr=sr,
                processor=processor,
                model=model,
                device=device,
                store_dtype=store_dtype,
            )  # [L, N, D]

            num_chunks = all_layer_chunk_feats.shape[1]
            chunk_mask = torch.ones(num_chunks, dtype=torch.bool)

            data["speech_feats"][expert_name] = {
                "all_layer_chunk_feats": all_layer_chunk_feats,
                "chunk_mask": chunk_mask,
                "chunk_spans": spans,
                "model_path": model_path,
                "chunk_seconds": args.chunk_seconds,
                "overlap_seconds": args.overlap_seconds,
                "target_sr": args.target_sr,
            }

            torch.save(data, out_path)

        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[DONE] Layered speech features extracted.")


if __name__ == "__main__":
    main()