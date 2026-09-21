


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT_EXPERT_DIMS = {
    "bert-large-uncased": 1024,
    "roberta-large": 1024,
    "deberta-v3-large": 1024,
    "bge-large-en-v1.5": 1024,

    "chinese-bert-wwm-ext": 768,
    "chinese-roberta-wwm-ext-large": 1024,
    "DeBERTa-v2-Large-Chinese": 1024,
    "bge-large-zh-v1.5": 1024,

    "bert-base-multilingual-cased": 768,
    "xlm-roberta-large": 1024,
    "mdeberta-v3-base": 768,
    "bge-m3": 1024,
}


SPEECH_EXPERT_DIMS = {
    "wav2vec2_large": 1024,
    "wavlm-large": 1024,
    "hubert-large-ls960-ft": 1024,
}


def safe_module_name(name: str) -> str:
    """Convert an expert name into a valid ModuleDict key."""
    return (
        name.replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


class LayerWeightedPool(nn.Module):
    def __init__(self, num_layers: int = 25):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                text:   [B, L, D]
                speech: [B, L, N, D]
        """
        num_available_layers = x.shape[1]
        weights = F.softmax(
            self.layer_logits[:num_available_layers],
            dim=0,
        )

        if x.dim() == 3:
            out = torch.einsum("l,bld->bd", weights, x)
        elif x.dim() == 4:
            out = torch.einsum("l,blnd->bnd", weights, x)
        else:
            raise ValueError(f"Unsupported x dim: {x.dim()}")

        return out, weights


class SinusoidalChunkPositionEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding for ordered speech chunks.

    The encoding is added after the expert-specific linear projection and
    before the Transformer Encoder. It introduces no trainable position
    parameters. The internal buffer is automatically extended when a batch
    contains more chunks than the initial ``max_chunks`` value.

    Args:
        hidden_dim:
            Hidden size of the projected speech chunk representation.
        max_chunks:
            Initial number of chunk positions stored in the buffer.
        dropout:
            Optional dropout applied after adding position encoding. The
            default is 0.0 so that the position-encoding ablation changes only
            the presence of positional information, not regularization.

    Input/Output:
        x: [B, N, H]
    """

    def __init__(
        self,
        hidden_dim: int,
        max_chunks: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}."
            )
        if max_chunks <= 0:
            raise ValueError(
                f"max_chunks must be positive, got {max_chunks}."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}."
            )

        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

        pe = self._build_encoding(
            num_positions=int(max_chunks),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("pe", pe, persistent=False)

    def _build_encoding(
        self,
        num_positions: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a [1, num_positions, hidden_dim] sinusoidal table."""
        position = torch.arange(
            num_positions,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)

        even_indices = torch.arange(
            0,
            self.hidden_dim,
            2,
            device=device,
            dtype=torch.float32,
        )
        div_term = torch.exp(
            even_indices
            * (-math.log(10000.0) / float(self.hidden_dim))
        )

        pe = torch.zeros(
            num_positions,
            self.hidden_dim,
            device=device,
            dtype=torch.float32,
        )
        pe[:, 0::2] = torch.sin(position * div_term)

        odd_width = pe[:, 1::2].shape[1]
        if odd_width > 0:
            pe[:, 1::2] = torch.cos(
                position * div_term[:odd_width]
            )

        return pe.unsqueeze(0).to(dtype=dtype)

    def _ensure_capacity(self, num_chunks: int, device: torch.device):
        if num_chunks <= self.pe.shape[1]:
            return

        expanded = self._build_encoding(
            num_positions=num_chunks,
            device=device,
            dtype=torch.float32,
        )
        # Reassigning a registered buffer keeps it a buffer.
        self.pe = expanded

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Expected x with shape [B, N, H], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, "
                f"got {x.shape[-1]}."
            )

        num_chunks = x.shape[1]
        self._ensure_capacity(num_chunks, x.device)

        pe = self.pe[:, :num_chunks].to(
            device=x.device,
            dtype=x.dtype,
        )
        out = x + pe

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    "mask must have shape [B, N], "
                    f"got {tuple(mask.shape)} for x={tuple(x.shape)}."
                )
            out = out.masked_fill(
                ~mask.to(dtype=torch.bool).unsqueeze(-1),
                0.0,
            )

        return self.dropout(out)


class SpeechExpertEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 25,
        max_chunks: int = 256,
        position_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_pool = LayerWeightedPool(num_layers=num_layers)
        self.proj = nn.Linear(input_dim, hidden_dim)

        # Explicit temporal position information for ordered speech chunks.
        self.chunk_position_encoding = SinusoidalChunkPositionEncoding(
            hidden_dim=hidden_dim,
            max_chunks=max_chunks,
            dropout=position_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x:    [B, L, N, D]
            mask: [B, N], True means valid
        """
        mask = mask.to(dtype=torch.bool)

        x, layer_weights = self.layer_pool(x)  # [B, N, D]
        x = self.proj(x)                       # [B, N, H]

        # Padded chunks contain no valid content and receive no position code.
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = self.chunk_position_encoding(x, mask=mask)

        key_padding_mask = ~mask
        x = self.temporal_encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        # A padded query can still acquire a non-zero vector inside a standard
        # Transformer. Clear it again before attention pooling.
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        attn_logits = self.attn(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights * mask.to(attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        return pooled, {
            "layer_weights": layer_weights.detach().cpu(),
            "chunk_attn": attn_weights.detach().cpu(),
            "position_encoding_type": "fixed_sinusoidal",
        }


class TextExpertEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 25):
        super().__init__()
        self.layer_pool = LayerWeightedPool(num_layers=num_layers)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """Args: x: [B, L, D]."""
        x, layer_weights = self.layer_pool(x)
        x = self.proj(x)
        x = self.norm(x)
        x = F.gelu(x)

        return x, {
            "layer_weights": layer_weights.detach().cpu(),
        }


class DCAMoELayeredTCPFixedSSEOCPosEncNoLangModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 2,
        evidence_activation: str = "softplus",
        tcp_dropout: float = 0.1,
        max_speech_chunks: int = 256,
        speech_position_dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for the speech Transformer.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.evidence_activation_name = evidence_activation
        self.tcp_dropout = float(tcp_dropout)
        self.max_speech_chunks = int(max_speech_chunks)
        self.speech_position_dropout = float(speech_position_dropout)
        self.position_encoding_type = "fixed_sinusoidal"
        self.eps = float(eps)

        self.speech_expert_names = list(SPEECH_EXPERT_DIMS.keys())
        self.text_expert_names = list(TEXT_EXPERT_DIMS.keys())

        self.speech_name_to_key = {
            name: safe_module_name(name)
            for name in self.speech_expert_names
        }
        self.text_name_to_key = {
            name: safe_module_name(name)
            for name in self.text_expert_names
        }

        self.speech_key_to_name = {
            value: key for key, value in self.speech_name_to_key.items()
        }
        self.text_key_to_name = {
            value: key for key, value in self.text_name_to_key.items()
        }

        self.speech_encoders = nn.ModuleDict({
            self.speech_name_to_key[name]: SpeechExpertEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                max_chunks=self.max_speech_chunks,
                position_dropout=self.speech_position_dropout,
            )
            for name, dim in SPEECH_EXPERT_DIMS.items()
        })
        self.text_encoders = nn.ModuleDict({
            self.text_name_to_key[name]: TextExpertEncoder(dim, hidden_dim)
            for name, dim in TEXT_EXPERT_DIMS.items()
        })

        # Expert-level MoE gates inside each modality.
        self.speech_gate = nn.Linear(hidden_dim, 1)
        self.text_gate = nn.Linear(hidden_dim, 1)

        # No language embedding in this ablation. Evidence and TCP heads
        # receive the modality representations directly.
        single_modality_input_dim = hidden_dim

        # Raw branch evidence heads. These determine the original two-branch S.
        self.speech_evidence_head = nn.Sequential(
            nn.Linear(single_modality_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        self.text_evidence_head = nn.Sequential(
            nn.Linear(single_modality_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

        # Lightweight, branch-specific TCP confidence predictors.
        # They predict absolute branch confidence before normalization.
        self.speech_tcp_head = nn.Sequential(
            nn.LayerNorm(single_modality_input_dim),
            nn.Dropout(tcp_dropout),
            nn.Linear(single_modality_input_dim, 1),
        )
        self.text_tcp_head = nn.Sequential(
            nn.LayerNorm(single_modality_input_dim),
            nn.Dropout(tcp_dropout),
            nn.Linear(single_modality_input_dim, 1),
        )

        if evidence_activation == "relu":
            self.evidence_activation = nn.ReLU()
        elif evidence_activation == "softplus":
            self.evidence_activation = nn.Softplus()
        else:
            raise ValueError(
                f"Unsupported evidence_activation: {evidence_activation}. "
                "Use 'softplus' or 'relu'."
            )

    def get_model_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable description of the model structure.

        This method is metadata-only and does not change forward computation.
        It is used by the training script to record a reproducible experiment
        configuration alongside the existing CSV/JSON outputs.
        """
        return {
            "model_class": self.__class__.__name__,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "evidence_activation": self.evidence_activation_name,
            "tcp_dropout": self.tcp_dropout,
            "max_speech_chunks": self.max_speech_chunks,
            "speech_position_encoding": "fixed_sinusoidal",
            "speech_position_dropout": self.speech_position_dropout,
            "uses_language_embedding": False,
            "language_aware_text_expert_masking": True,
            "speech_expert_names": list(self.speech_expert_names),
            "text_expert_names": list(self.text_expert_names),
            "speech_expert_dims": dict(SPEECH_EXPERT_DIMS),
            "text_expert_dims": dict(TEXT_EXPERT_DIMS),
            "fusion_rule": (
                "e_f=(E_s+E_t)*(w_s_used*q_s+w_t_used*q_t)"
            ),
            "fixed_strength_rule": "S_f=K+E_s+E_t",
            "tcp_weight_rule": "w_m_tcp=c_m/(c_s+c_t)",
            "base_weight_rule": "w_m_base=E_m/(E_s+E_t)",
            "used_weight_rule": (
                "w_used=(1-rho)*w_base+rho*w_tcp"
            ),
            "regularizer_type": "full_3d_structural_evidence_odds_consistency",
            "regularizer_scope": "speech_text_fused_three_level_reliability_hierarchy",
            "regularizer_target_gradient": "stop_gradient_on_tcp_side",
        }

    def encode_speech(
        self,
        speech_feats: Dict[str, torch.Tensor],
        speech_masks: Dict[str, torch.Tensor],
    ):
        expert_reprs = []
        expert_names = []
        aux = {}

        for original_name in self.speech_expert_names:
            if original_name not in speech_feats:
                continue

            key = self.speech_name_to_key[original_name]
            encoder = self.speech_encoders[key]

            repr_i, aux_i = encoder(
                speech_feats[original_name],
                speech_masks[original_name],
            )

            expert_reprs.append(repr_i)
            expert_names.append(original_name)
            aux[f"speech_{safe_module_name(original_name)}"] = aux_i

        if not expert_reprs:
            raise RuntimeError("No speech expert features found.")

        reps = torch.stack(expert_reprs, dim=1)
        gate_logits = self.speech_gate(reps).squeeze(-1)
        gate_weights = F.softmax(gate_logits, dim=-1)
        speech_repr = torch.sum(reps * gate_weights.unsqueeze(-1), dim=1)

        aux["speech_expert_names"] = expert_names
        aux["speech_expert_weights"] = gate_weights.detach().cpu()
        return speech_repr, aux

    def encode_text(
        self,
        text_feats: Dict[str, torch.Tensor],
        text_available: Dict[str, torch.Tensor],
    ):
        expert_reprs = []
        expert_masks = []
        expert_names = []
        aux = {}

        for original_name in self.text_expert_names:
            if original_name not in text_feats:
                continue

            key = self.text_name_to_key[original_name]
            encoder = self.text_encoders[key]

            repr_i, aux_i = encoder(text_feats[original_name])
            expert_reprs.append(repr_i)
            expert_masks.append(text_available[original_name])
            expert_names.append(original_name)
            aux[f"text_{safe_module_name(original_name)}"] = aux_i

        if not expert_reprs:
            raise RuntimeError("No text expert features found.")

        reps = torch.stack(expert_reprs, dim=1)
        available = torch.stack(expert_masks, dim=1).to(
            device=reps.device,
            dtype=torch.bool,
        )

        gate_logits = self.text_gate(reps).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~available, -1e9)
        gate_weights = F.softmax(gate_logits, dim=-1)
        text_repr = torch.sum(reps * gate_weights.unsqueeze(-1), dim=1)

        aux["text_expert_names"] = expert_names
        aux["text_expert_weights"] = gate_weights.detach().cpu()
        return text_repr, aux

    def evidence_to_opinion(self, evidence: torch.Tensor):
        """Convert non-negative evidence into a Dirichlet opinion."""
        alpha = evidence + 1.0
        strength = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / strength
        belief = evidence / strength
        uncertainty = self.num_classes / strength

        return {
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "belief": belief,
            "uncertainty": uncertainty,
            "strength": strength,
        }

    def normalize_evidence_direction(
        self,
        evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split evidence into strength and a unit-sum class direction.

        ReLU can theoretically produce an all-zero vector; in that case the
        direction falls back to the uniform distribution.
        """
        evidence_strength = evidence.sum(dim=-1, keepdim=True)
        safe_strength = evidence_strength.clamp_min(self.eps)
        direction = evidence / safe_strength

        uniform = torch.full_like(evidence, 1.0 / self.num_classes)
        direction = torch.where(
            evidence_strength > self.eps,
            direction,
            uniform,
        )
        return direction, evidence_strength

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        tcp_ratio: float = 1.0,
    ):
        """
        Args:
            batch: cached multimodal feature batch.
            tcp_ratio:
                0.0 -> exact raw two-branch evidence accumulation.
                1.0 -> full fixed-S TCP dynamic fusion.
                Values in between provide gradual activation.
        """
        if not 0.0 <= float(tcp_ratio) <= 1.0:
            raise ValueError(f"tcp_ratio must be in [0, 1], got {tcp_ratio}")

        device = next(self.parameters()).device

        speech_feats = {
            key: value.to(device)
            for key, value in batch["speech_feats"].items()
        }
        speech_masks = {
            key: value.to(device)
            for key, value in batch["speech_masks"].items()
        }
        text_feats = {
            key: value.to(device)
            for key, value in batch["text_feats"].items()
        }
        text_available = {
            key: value.to(device)
            for key, value in batch["text_available"].items()
        }

        speech_repr, speech_aux = self.encode_speech(
            speech_feats=speech_feats,
            speech_masks=speech_masks,
        )
        text_repr, text_aux = self.encode_text(
            text_feats=text_feats,
            text_available=text_available,
        )
        # Language-independent head inputs. lang_id may remain in the
        # cached batch for compatibility, but it is intentionally unused.
        speech_input = speech_repr
        text_input = text_repr

        # Raw speech/text evidence. There is no fusion-evidence head e_f.
        raw_speech_evidence = self.evidence_activation(
            self.speech_evidence_head(speech_input)
        )
        raw_text_evidence = self.evidence_activation(
            self.text_evidence_head(text_input)
        )

        raw_speech_opinion = self.evidence_to_opinion(raw_speech_evidence)
        raw_text_opinion = self.evidence_to_opinion(raw_text_evidence)

        speech_direction, speech_evidence_strength = (
            self.normalize_evidence_direction(raw_speech_evidence)
        )
        text_direction, text_evidence_strength = (
            self.normalize_evidence_direction(raw_text_evidence)
        )

        raw_total_evidence_strength = (
            speech_evidence_strength + text_evidence_strength
        )  # S_raw - K

        # Absolute TCP confidence estimates in (0, 1).
        speech_tcp_logit = self.speech_tcp_head(speech_input)
        text_tcp_logit = self.text_tcp_head(text_input)
        speech_tcp_confidence = torch.sigmoid(speech_tcp_logit)
        text_tcp_confidence = torch.sigmoid(text_tcp_logit)

        tcp_confidences = torch.cat(
            [speech_tcp_confidence, text_tcp_confidence],
            dim=-1,
        )
        tcp_weights = tcp_confidences / tcp_confidences.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        # Baseline weights reproduce raw e_s + e_t exactly.
        base_strengths = torch.cat(
            [speech_evidence_strength, text_evidence_strength],
            dim=-1,
        )
        base_weights = base_strengths / raw_total_evidence_strength.clamp_min(
            self.eps
        )

        ratio = torch.as_tensor(
            float(tcp_ratio),
            device=device,
            dtype=raw_speech_evidence.dtype,
        )
        used_weights = (
            (1.0 - ratio) * base_weights
            + ratio * tcp_weights
        )
        used_weights = used_weights / used_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        speech_weight = used_weights[:, 0:1]
        text_weight = used_weights[:, 1:2]

        # Fixed-S evidence allocation.
        speech_fused_evidence = (
            raw_total_evidence_strength
            * speech_weight
            * speech_direction
        )
        text_fused_evidence = (
            raw_total_evidence_strength
            * text_weight
            * text_direction
        )
        total_evidence = speech_fused_evidence + text_fused_evidence
        total_opinion = self.evidence_to_opinion(total_evidence)

        # Numerical diagnostic: should be approximately zero.
        expected_total_strength = (
            raw_total_evidence_strength + self.num_classes
        )
        strength_preservation_error = torch.abs(
            total_opinion["strength"] - expected_total_strength
        )

        out = {
            # Final fused Dirichlet opinion.
            "evidence": total_opinion["evidence"],
            "alpha": total_opinion["alpha"],
            "prob": total_opinion["prob"],
            "belief": total_opinion["belief"],
            "uncertainty": total_opinion["uncertainty"],
            "strength": total_opinion["strength"],

            # Raw unimodal opinions: used for auxiliary and TCP supervision.
            "speech_evidence": raw_speech_opinion["evidence"],
            "speech_alpha": raw_speech_opinion["alpha"],
            "speech_prob": raw_speech_opinion["prob"],
            "speech_belief": raw_speech_opinion["belief"],
            "speech_uncertainty": raw_speech_opinion["uncertainty"],
            "speech_strength": raw_speech_opinion["strength"],

            "text_evidence": raw_text_opinion["evidence"],
            "text_alpha": raw_text_opinion["alpha"],
            "text_prob": raw_text_opinion["prob"],
            "text_belief": raw_text_opinion["belief"],
            "text_uncertainty": raw_text_opinion["uncertainty"],
            "text_strength": raw_text_opinion["strength"],

            # Evidence contributions after fixed-S TCP allocation.
            "speech_fused_evidence": speech_fused_evidence,
            "text_fused_evidence": text_fused_evidence,
            "speech_fused_strength": speech_fused_evidence.sum(
                dim=-1,
                keepdim=True,
            ),
            "text_fused_strength": text_fused_evidence.sum(
                dim=-1,
                keepdim=True,
            ),
            "raw_total_evidence_strength": raw_total_evidence_strength,
            "expected_total_strength": expected_total_strength,
            "strength_preservation_error": strength_preservation_error,

            # TCP outputs.
            "speech_tcp_logit": speech_tcp_logit,
            "text_tcp_logit": text_tcp_logit,
            "speech_tcp_confidence": speech_tcp_confidence,
            "text_tcp_confidence": text_tcp_confidence,
            "tcp_weights": tcp_weights,
            "base_modality_weights": base_weights,
            "modality_weights": used_weights,
            "tcp_ratio": ratio,

            # Speech position-encoding configuration.
            "position_encoding_type": self.position_encoding_type,

            # Representations retained for analysis.
            "speech_repr": speech_repr,
            "text_repr": text_repr,
            "uses_language_embedding": False,
        }

        aux = {}
        aux.update(speech_aux)
        aux.update(text_aux)

        aux["tcp_ratio"] = float(tcp_ratio)
        aux["position_encoding_type"] = self.position_encoding_type
        aux["uses_language_embedding"] = False
        aux["max_speech_chunks"] = self.max_speech_chunks
        aux["speech_position_dropout"] = self.speech_position_dropout
        aux["speech_tcp_confidence"] = speech_tcp_confidence.detach().cpu()
        aux["text_tcp_confidence"] = text_tcp_confidence.detach().cpu()
        aux["tcp_weights"] = tcp_weights.detach().cpu()
        aux["base_modality_weights"] = base_weights.detach().cpu()
        aux["modality_weights"] = used_weights.detach().cpu()

        aux["raw_speech_evidence"] = raw_speech_evidence.detach().cpu()
        aux["raw_text_evidence"] = raw_text_evidence.detach().cpu()
        aux["speech_fused_evidence"] = speech_fused_evidence.detach().cpu()
        aux["text_fused_evidence"] = text_fused_evidence.detach().cpu()
        aux["total_evidence"] = total_evidence.detach().cpu()
        aux["total_prob"] = total_opinion["prob"].detach().cpu()
        aux["total_uncertainty"] = total_opinion["uncertainty"].detach().cpu()

        return out, aux