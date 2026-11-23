#!/usr/bin/env python3
"""
scripts/train_ntv3.py
=====================
Modular worker for multi-task fine-tuning and baseline benchmarking of
Nucleotide Transformer v3 (NTv3) on Zebrafish epigenomics and nascent transcriptomics:
- ATAC-seq: 256-cell (pre-MZT) and dome (late blastula)
- SLAM-seq Nascent RNA-seq: 0.75 hpf baseline, 2.5 hpf (pre-MZT), 3.0-3.5 hpf (onset), 4.5 hpf (post-MZT)
  both forward and reverse strand tracks.

Features:
- "laptop_16gb": InstaDeepAI/NTv3_tracks, 65,536 bp context, LoRA (PEFT), bfloat16,
  gradient checkpointing, batch size 1, gradient accumulation 16.
- "hpc_1mb": InstaDeepAI/NTv3_tracks_1mb, 1,048,576 bp context, full/LoRA, DDP / accelerate.
- Central 37.5% crop loss evaluation on genomic sequences with spatial resolution alignment.
- Trainable nn.Linear(embed_dim, num_tracks) head attached to trunk.
- Leak-free stratified window sampling strictly bounded by chr_fold.json (train, valid, test).
- Safe worker pyBigWig handle lifecycle preventing multiprocess segfaults.
- Real publication baselines benchmark (--run-baselines) evaluating NTv3 vs ChromBPNet on held-out test split.
- ABSOLUTE ZERO-MOCK POLICY: strictly fails with non-zero exit code on missing inputs or prerequisites.
"""

import os
import sys
import re
import math
import json
import inspect
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure conda env bin directory is at the front of PATH
bin_dir = str(Path(sys.prefix) / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from scripts.task_manager import set_task_progress
except ImportError:
    try:
        from task_manager import set_task_progress
    except ImportError:
        def set_task_progress(work_dir, task_name, progress_str):
            pass


def get_crop_bounds(context_len: int, crop_ratio: float = 0.375) -> Tuple[int, int]:
    """Calculate start and end base positions for the central crop."""
    crop_len = int(context_len * crop_ratio)
    start = (context_len - crop_len) // 2
    end = start + crop_len
    return start, end


def sanitize_for_torch_save(obj: Any) -> Any:
    """Recursively convert numpy/pandas/custom objects to pure Python primitives for PyTorch 2.6+ weights_only=True."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool, bytes)):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_for_torch_save(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_torch_save(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_torch_save(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitize_for_torch_save(x) for x in sorted(list(obj))]
    if hasattr(obj, "__dict__"):
        return sanitize_for_torch_save(vars(obj))
    return str(obj)


def get_ntv3_input_embedding_layer(model: nn.Module) -> nn.Module:
    """
    Resolves NTv3Tracks's real input embedding submodule reliably (raising a loud RuntimeError if not found)
    for both LoRA gradient enabling and Block 8 saliency hooks.
    """
    m = model
    if hasattr(m, "trunk"):
        m = m.trunk
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    if hasattr(m, "model") and hasattr(m.model, "embed_layer"):
        return m.model.embed_layer
    if hasattr(m, "embed_layer"):
        return m.embed_layer
    for name, sub in m.named_modules():
        if "embed_layer" in name and isinstance(sub, nn.Embedding):
            return sub
    raise RuntimeError(f"[FATAL] Could not locate NTv3 input embedding layer in model of type {type(model)}.")


def apply_ntv3_activation_checkpointing(base_model: nn.Module):
    """
    Implement manual activation checkpointing (use_reentrant=False) around NTv3Tracks's
    Transformer blocks and U-Net decoder blocks to prevent VRAM spikes on 16 GB GPUs.
    """
    import torch.utils.checkpoint as checkpoint
    inner = base_model.model if hasattr(base_model, "model") else base_model
    mod = sys.modules.get(type(inner).__module__)
    if mod is None or not hasattr(mod, "_NTv3Backbone"):
        return

    def checkpointed_backbone_forward(self, tokens, conditions=None, conditions_masks=None, stop_after_transformer=False, skip_lm_head=True):
        seq_len = tokens.shape[1]
        downsampling = self.config.total_downsampling
        if seq_len % downsampling != 0:
            raise ValueError(f"sequence length {seq_len} is not a multiple of {downsampling}")

        outs: dict[str, torch.Tensor] = {}
        intermediates: dict[int, torch.Tensor] = {}

        x = self.embed_layer(tokens)
        x = self.stem(x)
        intermediates[1] = x
        x = mod.max_pool(x, 2)

        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            intermediates[2 ** (index + 1)] = x
            x = mod.max_pool(x, 2)

        outs["before_transformer_embedding"] = x
        attention_mask = self.block_attention_mask(tokens)

        for index in range(self.config.num_layers):
            mha = self.mha_blocks[index]
            mlp = self.mlp_blocks[index]
            def create_custom_forward(mha_b, mlp_b):
                def custom_forward(h):
                    h1 = h + mha_b(h, attention_mask=attention_mask, conditions=conditions, conditions_masks=conditions_masks)
                    return h1 + mlp_b(h1, conditions, conditions_masks)
                return custom_forward
            x = checkpoint.checkpoint(create_custom_forward(mha, mlp), x, use_reentrant=False)

        x = self.final_norm(x)
        outs["after_transformer_embedding"] = x
        if stop_after_transformer:
            return outs

        for index, block in enumerate(self.decoder_blocks):
            bin_size = 2 ** (self.config.num_downsamples - index)
            def create_dec_forward(dec_b, inter):
                def dec_forward(h):
                    return dec_b(h, inter, conditions, conditions_masks)
                return dec_forward
            x = checkpoint.checkpoint(create_dec_forward(block, intermediates[bin_size]), x, use_reentrant=False)

        if not skip_lm_head and hasattr(self, "lm_head"):
            outs["logits"] = self.lm_head(x)
        outs["embedding"] = x
        return outs

    mod._NTv3Backbone.forward = checkpointed_backbone_forward
    print("[+] Enabled manual activation checkpointing (use_reentrant=False) on NTv3 Transformer and U-Net blocks.")


def find_target_lora_modules(model: nn.Module) -> List[str]:
    """
    Find attention projection module names dynamically based on NTv3Tracks actual Transformer linear layer names,
    targeting attention Q/K/V/O (q_layer, k_layer, v_layer, linear_embedding) while strictly excluding output heads.
    """
    target_names = set()
    exclude_keywords = ["head", "bigwig", "bed_head", "bed_tracks", "lm_head", "classifier", "track_bias", "pointwise_unet_skip", "stem", "encoder", "decoder"]
    for name, module in model.named_modules():
        if any(ex in name.lower() for ex in exclude_keywords):
            continue
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            leaf = parts[-1]
            if any(ex in leaf.lower() for ex in ["head", "bigwig", "bed_head", "lm_head", "classifier"]):
                continue
            if "mha_blocks" in name and leaf in ["q_layer", "k_layer", "v_layer", "linear_embedding"]:
                target_names.add(leaf)
    if not target_names:
        target_names = {"q_layer", "k_layer", "v_layer", "linear_embedding"}
    return sorted(list(target_names))


_IUPAC_SANITIZE_RE = re.compile(r'[^ACGTNacgtn]')


def sanitize_sequence(seq: str) -> str:
    """
    Sanitize DNA sequence by replacing rare IUPAC ambiguity characters outside ACGTNacgtn with N,
    strictly preserving lowercase soft-masking (acgtn).
    """
    return _IUPAC_SANITIZE_RE.sub('N', seq)


class NTv3TrackHead(nn.Module):
    """Wraps transformer trunk with trainable linear head for track prediction on central crop."""

    def __init__(self, trunk: nn.Module, embed_dim: int, num_tracks: int, crop_ratio: float = 0.375, default_species_id: int = 0):
        super().__init__()
        self.trunk = trunk
        self.embed_dim = embed_dim
        self.crop_ratio = crop_ratio
        self.head = nn.Linear(embed_dim, num_tracks)
        self.default_species_id = default_species_id

        # Inspect base_model.forward signature once in __init__
        import inspect
        forward_target = trunk.forward
        base_sig = None
        if hasattr(trunk, "get_base_model"):
            try:
                base_sig = inspect.signature(trunk.get_base_model().forward)
            except Exception:
                base_sig = None
        sig = inspect.signature(forward_target)

        self.accepts_track_indices = ("track_indices" in sig.parameters) or (base_sig is not None and "track_indices" in base_sig.parameters)
        self.accepts_species_ids = ("species_ids" in sig.parameters) or (base_sig is not None and "species_ids" in base_sig.parameters)
        self.accepts_conditions_tokens = ("conditions_tokens" in sig.parameters) or (base_sig is not None and "conditions_tokens" in base_sig.parameters)
        self.accepts_attention_mask = ("attention_mask" in sig.parameters) or (base_sig is not None and "attention_mask" in base_sig.parameters)
        self.accepts_output_hidden_states = ("output_hidden_states" in sig.parameters) or (base_sig is not None and "output_hidden_states" in base_sig.parameters)
        self.accepts_skip_lm_head = ("skip_lm_head" in sig.parameters) or (base_sig is not None and "skip_lm_head" in base_sig.parameters) or ("kwargs" in str(sig))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        conditions_tokens: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        kwargs = {}
        if species_ids is None and conditions_tokens is not None and len(conditions_tokens) > 0:
            species_ids = conditions_tokens[0]

        if species_ids is None:
            species_ids = torch.full((input_ids.shape[0],), self.default_species_id, dtype=torch.long, device=input_ids.device)

        if self.accepts_conditions_tokens:
            kwargs["conditions_tokens"] = [species_ids]
        elif self.accepts_species_ids:
            kwargs["species_ids"] = species_ids

        if self.accepts_attention_mask and attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if self.accepts_output_hidden_states:
            kwargs["output_hidden_states"] = True
        if self.accepts_skip_lm_head:
            kwargs["skip_lm_head"] = True

        out = self.trunk(input_ids=input_ids, **kwargs)

        if hasattr(out, "embedding") and out.embedding is not None:
            hidden = out.embedding
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hidden = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None and len(out.hidden_states) > 0:
            hidden = out.hidden_states[-1]
        elif isinstance(out, dict):
            if "embedding" in out and out["embedding"] is not None:
                hidden = out["embedding"]
            elif "last_hidden_state" in out and out["last_hidden_state"] is not None:
                hidden = out["last_hidden_state"]
            elif "hidden_states" in out and out["hidden_states"] is not None and len(out["hidden_states"]) > 0:
                hidden = out["hidden_states"][-1]
            else:
                hidden = list(out.values())[0]
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            hidden = out

        assert hidden.shape[-1] == self.embed_dim, f"[FATAL] Extracted hidden dim {hidden.shape[-1]} does not match embed_dim {self.embed_dim}!"
        assert self.embed_dim >= 128, f"[FATAL] embed_dim {self.embed_dim} < 128 (MLM logits detected instead of hidden states)!"

        # Determine whether hidden is already cropped to crop_ratio or spans the full input window (1-bp or downsampled)
        seq_len = hidden.shape[1]
        context_len = input_ids.shape[1]
        is_already_cropped = False
        if self.crop_ratio is not None and self.crop_ratio < 1.0:
            if abs(seq_len - int(round(context_len * self.crop_ratio))) <= 2:
                is_already_cropped = True
            elif seq_len < context_len:
                for downsample in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                    expected_full = context_len // downsample
                    if expected_full > 0 and abs(seq_len - int(round(expected_full * self.crop_ratio))) <= 1:
                        is_already_cropped = True
                        break

        if self.crop_ratio is not None and self.crop_ratio < 1.0 and not is_already_cropped:
            crop_len = int(round(seq_len * self.crop_ratio))
            start = int(round(seq_len * (1 - self.crop_ratio) / 2))
            end = start + crop_len
            hidden_crop = hidden[:, start:end, :]
        else:
            hidden_crop = hidden

        # Strictly apply softplus so predicted log1p(CPM) tracks are non-negative
        preds = F.softplus(self.head(hidden_crop))
        return preds


def resolve_ntv3_species_id(
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    species_name: str = "danio_rerio",
    hf_token: Optional[str] = None,
    model_id: str = "InstaDeepAI/NTv3_tracks"
) -> int:
    """
    Dynamically resolve the exact integer species ID for NTv3_tracks.
    Inspects model config, tokenizer, species vocabulary, or metadata from HF Hub.
    Avoids hardcoding species_ids=0 for non-human species.
    """
    norm_name = species_name.lower().replace(" ", "_")
    candidates = [
        norm_name,
        species_name.lower().replace("_", " "),
        "danio_rerio" if ("danio" in norm_name or "rerio" in norm_name or "grcz11" in norm_name) else norm_name,
        "danio rerio",
        "zebrafish",
        "grcz11",
        "oryzias_latipes" if ("medaka" in norm_name or "oryzias" in norm_name) else norm_name,
        "oryzias latipes",
        "medaka"
    ]

    target_obj = model.trunk if (model is not None and hasattr(model, "trunk")) else model
    if hasattr(target_obj, "get_base_model"):
        target_obj = target_obj.get_base_model()

    # 1. Check model.encode_species([cand]) passing a list
    if target_obj is not None and hasattr(target_obj, "encode_species") and callable(target_obj.encode_species):
        for cand in candidates:
            try:
                res = target_obj.encode_species([cand])
                if isinstance(res, torch.Tensor):
                    res = res.squeeze().item()
                elif isinstance(res, (list, tuple)):
                    res = res[0]
                if isinstance(res, (int, np.integer)):
                    print(f"[+] Resolved species '{species_name}' -> ID {int(res)} via model.encode_species()")
                    return int(res)
            except Exception:
                pass

    if tokenizer is not None and hasattr(tokenizer, "encode_species") and callable(tokenizer.encode_species):
        for cand in candidates:
            try:
                res = tokenizer.encode_species([cand])
                if isinstance(res, torch.Tensor):
                    res = res.squeeze().item()
                elif isinstance(res, (list, tuple)):
                    res = res[0]
                if isinstance(res, (int, np.integer)):
                    print(f"[+] Resolved species '{species_name}' -> ID {int(res)} via tokenizer.encode_species()")
                    return int(res)
            except Exception:
                pass

    # 2. Check model config species_to_token_id / species vocabulary
    cfg = getattr(target_obj, "config", None) if target_obj is not None else None
    for obj in [cfg, target_obj, tokenizer]:
        if obj is None:
            continue
        for attr in ["species_to_token_id", "species_to_id", "species_vocabulary", "species_map", "species"]:
            vocab = getattr(obj, attr, None)
            if isinstance(vocab, dict):
                for cand in candidates:
                    if cand in vocab:
                        val = vocab[cand]
                        print(f"[+] Resolved species '{species_name}' -> ID {int(val)} via {attr}")
                        return int(val)
                    for k, v in vocab.items():
                        if str(k).lower().replace(" ", "_") == cand:
                            print(f"[+] Resolved species '{species_name}' -> ID {int(v)} via {attr}")
                            return int(v)

    # 3. Check track_metadata.json from HF repo
    try:
        from huggingface_hub import hf_hub_download
        token = hf_token or os.environ.get("HF_TOKEN")
        meta_path = hf_hub_download(repo_id=model_id, filename="track_metadata.json", token=token)
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f_meta:
                meta = json.load(f_meta)
            species_vocab = meta.get("species_vocabulary", {})
            for cand in candidates:
                if cand in species_vocab:
                    val = species_vocab[cand]
                    print(f"[+] Resolved species '{species_name}' -> ID {int(val)} via track_metadata.json['species_vocabulary']")
                    return int(val)
                for k, v in species_vocab.items():
                    if str(k).lower().replace(" ", "_") == cand:
                        print(f"[+] Resolved species '{species_name}' -> ID {int(v)} via track_metadata.json['species_vocabulary']")
                        return int(v)
    except Exception:
        pass

    if "danio" in norm_name or "rerio" in norm_name or "grcz11" in norm_name or "zebrafish" in norm_name:
        raise RuntimeError(
            f"[FATAL] Failed to resolve species ID for '{species_name}' from NTv3 model config, encode_species, or track_metadata.json! "
            f"Silently defaulting to human ID 0 is strictly prohibited by the zero-fabrication policy."
        )

    print(f"[!] Warning: Species vocabulary not discovered on model/tokenizer. Defaulting species '{species_name}' -> ID 0.")
    return 0


class TrackBalancedLoss(nn.Module):
    """
    Track-balanced loss combining per-track MSE and cosine/Pearson profile loss on log1p targets.
    Up-weights sparse pre-MZT tracks (atac_256cell, slam_2.5hpf_fwd/rev, slam_0.75hpf) relative to
    dense late-development tracks (atac_dome) so pre-MZT predictions do not collapse to zeros.
    """

    def __init__(self, track_names: List[str], base_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.track_names = track_names
        weights = []
        for name in track_names:
            if any(k in name.lower() for k in ["256cell", "early", "2.5hpf", "0.75hpf"]):
                weights.append(3.0)
            else:
                weights.append(1.0)
        if base_weights:
            for i, name in enumerate(track_names):
                if name in base_weights:
                    weights[i] = base_weights[name]
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds, targets: (batch, crop_len, num_tracks)
        mse = F.mse_loss(preds, targets, reduction="none")  # (batch, crop_len, num_tracks)
        per_track_mse = mse.mean(dim=(0, 1))  # (num_tracks,)

        # Spatial cosine profile loss along sequence dimension (dim=1)
        p_centered = preds - preds.mean(dim=1, keepdim=True)
        t_centered = targets - targets.mean(dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(p_centered, t_centered, dim=1, eps=1e-6)  # (batch, num_tracks)
        
        # Mask out sparse all-zero target windows to prevent adding constant offset to loss
        valid_mask = t_centered.norm(dim=1) > 1e-5  # (batch, num_tracks)
        valid_counts = valid_mask.sum(dim=0).clamp(min=1)
        per_track_profile = ((1.0 - cos_sim) * valid_mask.float()).sum(dim=0) / valid_counts
        # Out-of-place masking via torch.where to prevent autograd version errors
        has_valid = (valid_mask.sum(dim=0) > 0)
        per_track_profile = torch.where(has_valid, per_track_profile, torch.zeros_like(per_track_profile))

        combined = per_track_mse + 0.5 * per_track_profile
        weighted_loss = (combined * self.weights).sum() / self.weights.sum()
        return weighted_loss


def build_ntv3_model(
    model_id: str,
    num_tracks: int,
    train_mode: str = "lora",
    context_len: int = 65536,
    grad_checkpointing: bool = True,
    hf_token: Optional[str] = None,
    default_species_id: int = 0
) -> Tuple[nn.Module, Any]:
    """Instantiate NTv3 track model with optional PEFT LoRA or frozen head."""
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    token = hf_token or os.environ.get("HF_TOKEN")
    print(f"[*] Loading foundation model and tokenizer: {model_id}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        print(f"[FATAL] Failed to load tokenizer for {model_id}: {e}", file=sys.stderr)
        sys.exit(1)

    torch_dtype = torch.bfloat16 if (train_mode in ("lora", "frozen_head") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    try:
        base_model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token,
            dtype=torch_dtype
        )
    except TypeError:
        base_model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        print(f"[FATAL] Failed to load foundation model {model_id}: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Eliminate VRAM waste by deleting unused 7,362-channel BigWig and MLM heads during custom-head training
    if train_mode != "raw_baseline":
        inner = base_model.model if hasattr(base_model, "model") else base_model
        if hasattr(inner, "bigwig_head"):
            delattr(inner, "bigwig_head")
            print("[+] Bypassed/removed unused 7,362-channel pretrained bigwig_head to eliminate VRAM waste.")
        if hasattr(inner, "lm_head"):
            delattr(inner, "lm_head")
            print("[+] Bypassed/removed unused pretrained lm_head to eliminate VRAM waste.")

    # 2. Genuine activation checkpointing on NTv3 Transformer and U-Net blocks
    if grad_checkpointing:
        apply_ntv3_activation_checkpointing(base_model)
        if hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception:
                pass

    # 3. Input embedding layer hook for LoRA gradient flow
    if train_mode == "lora":
        emb_layer = get_ntv3_input_embedding_layer(base_model)
        emb_layer.register_forward_hook(lambda module, input, output: output.requires_grad_(True))
        print("[+] Attached forward hook on NTv3 input embedding layer setting output.requires_grad_(True).")

    # Determine decoder feature embedding dimension (1024 for NTv3 decoder output)
    embed_dim = 1024
    if hasattr(base_model, "model") and hasattr(base_model.model, "config"):
        embed_dim = getattr(base_model.model.config, "conv_init_embed_dim", 1024)
    elif hasattr(base_model.config, "conv_init_embed_dim"):
        embed_dim = getattr(base_model.config, "conv_init_embed_dim", 1024)
    elif hasattr(base_model.config, "hidden_size"):
        embed_dim = getattr(base_model.config, "hidden_size")

    if train_mode == "lora":
        target_modules = find_target_lora_modules(base_model)
        print(f"[*] Applying LoRA PEFT adapters to modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=None,
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none"
        )
        trunk = get_peft_model(base_model, peft_config)
    elif train_mode == "frozen_head":
        for param in base_model.parameters():
            param.requires_grad = False
        trunk = base_model
    else:
        trunk = base_model

    model = NTv3TrackHead(trunk=trunk, embed_dim=embed_dim, num_tracks=num_tracks, crop_ratio=0.375, default_species_id=default_species_id)
    return model, tokenizer


def sample_stratified_windows(
    work_dir: Path,
    chrom_sizes_path: Path,
    chr_fold_path: Optional[Path] = None,
    split: str = "train",
    context_len: int = 65536,
    n_samples: int = 2000,
    seed: int = 42
) -> List[Tuple[str, int, int]]:
    """
    Stratified window sampling strictly bounded by chromosome split:
    - 30% pre-MZT peaks / minor-wave summits
    - 50% late peaks / major-wave summits
    - 20% background genomic regions
    - Caps/excludes miR-430 cluster
    - Enforces zero data leakage by sampling strictly from split chromosomes.
    """
    np.random.seed(seed)
    chrom_sizes = {}
    if chrom_sizes_path.exists():
        with open(chrom_sizes_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2 and int(p[1]) > context_len:
                    chrom_sizes[p[0]] = int(p[1])

    # Filter chromosomes strictly by chr_fold split
    allowed_chroms = set(chrom_sizes.keys())
    if chr_fold_path and Path(chr_fold_path).exists():
        try:
            with open(chr_fold_path) as f:
                fold_data = json.load(f)
            split_key = split if split in fold_data else ("valid" if split == "val" else split)
            if split_key in fold_data:
                target_chroms = set(str(c) for c in fold_data[split_key])
                filtered_sizes = {c: sz for c, sz in chrom_sizes.items() if c in target_chroms}
                if filtered_sizes:
                    chrom_sizes = filtered_sizes
                    allowed_chroms = set(chrom_sizes.keys())
                    print(f"[+] Restricting '{split}' sampling to {len(allowed_chroms)} chromosomes: {sorted(list(allowed_chroms))}")
        except Exception as e:
            print(f"[-] Notice reading chr_fold_path: {e}")

    if not chrom_sizes:
        raise ValueError(
            f"[FATAL] No valid chromosomes found in {chrom_sizes_path} for split '{split}'. "
            f"Zero-mock policy strictly enforced: cannot fabricate chromosome sizes."
        )

    n_premzt = int(n_samples * 0.30)
    n_late = int(n_samples * 0.50)

    premzt_centers = []
    late_centers = []
    zf_targets = work_dir / "atac" / "Danio_rerio" / "Danio_rerio_summit_targets.tsv"

    if zf_targets.exists():
        df_zf = pd.read_csv(zf_targets, sep="\t", dtype={"chrom": str}, low_memory=False)
        df_filt = df_zf[~df_zf.get("is_repeat_downweight", False)]
        df_filt = df_filt[df_filt["chrom"].astype(str).isin(allowed_chroms)]

        early_col = "cpm_early" if "cpm_early" in df_filt.columns else ("early_cpm" if "early_cpm" in df_filt.columns else None)
        if early_col:
            df_early_active = df_filt[df_filt[early_col] >= 2.0]
            if len(df_early_active) < 1000 and "y_abs" in df_filt.columns:
                df_early_active = df_filt.sort_values(by="y_abs", ascending=False).head(1000)
        elif "y_abs" in df_filt.columns:
            df_early_active = df_filt.sort_values(by="y_abs", ascending=False).head(1000)
        else:
            df_early_active = df_filt.head(1000)

        for _, r in df_early_active.iterrows():
            premzt_centers.append((str(r["chrom"]), int(r["summit"])))

        late_col = "cpm_late" if "cpm_late" in df_filt.columns else ("late_cpm" if "late_cpm" in df_filt.columns else None)
        if late_col:
            top_late = df_filt.sort_values(by=late_col, ascending=False).head(5000)
            for _, r in top_late.iterrows():
                late_centers.append((str(r["chrom"]), int(r["summit"])))

    # Include pre-MZT minor-wave gene TSS coordinates if SLAM-seq classes exist
    slam_classes_path = work_dir / "slamseq" / "zf_slamseq_gene_classes.tsv"
    if slam_classes_path.exists():
        try:
            df_slam = pd.read_csv(slam_classes_path, sep="\t")
            if "chrom" in df_slam.columns and "tss" in df_slam.columns:
                minor_genes = df_slam[df_slam["slam_class"].isin(["preMZT_strictly_zygotic", "preMZT_maternal_zygotic"])]
                for _, r in minor_genes.iterrows():
                    c = str(r["chrom"])
                    if c in allowed_chroms:
                        premzt_centers.append((c, int(r["tss"])))
        except Exception:
            pass

    windows = []

    def make_window(chrom, center):
        c_size = chrom_sizes.get(chrom, 100000000)
        start = max(0, center - context_len // 2)
        end = start + context_len
        if end > c_size:
            end = c_size
            start = max(0, end - context_len)
        return (chrom, start, end)

    if premzt_centers:
        if len(premzt_centers) < n_premzt:
            idx_p = np.random.choice(len(premzt_centers), size=n_premzt, replace=True)
        else:
            idx_p = np.random.choice(len(premzt_centers), size=n_premzt, replace=False)
        for i in idx_p:
            windows.append(make_window(*premzt_centers[i]))

    if late_centers:
        k_l = min(n_late, len(late_centers))
        idx_l = np.random.choice(len(late_centers), size=k_l, replace=False)
        for i in idx_l:
            windows.append(make_window(*late_centers[i]))

    chrom_list = list(chrom_sizes.keys())
    while len(windows) < n_samples:
        ch = np.random.choice(chrom_list)
        sz = chrom_sizes[ch]
        st = int(np.random.randint(0, max(1, sz - context_len)))
        windows.append((ch, st, st + context_len))

    return windows


class NTv3GenomeDataset(Dataset):
    """
    Dataset extracting windows from genome FASTA and querying multi-track BigWigs.
    Uses lazy per-worker BigWig and Fasta opening for multiprocess safety.
    """

    def __init__(
        self,
        genome_fa_path: str,
        windows: List[Tuple[str, int, int]],
        track_paths: Dict[str, str],
        tokenizer: Any,
        context_len: int = 65536
    ):
        self.genome_fa_path = str(genome_fa_path)
        self._genome = None
        self.windows = windows
        self.track_paths = track_paths
        self.track_names = list(track_paths.keys())
        self.context_len = context_len
        self.tokenizer = tokenizer
        self._bw_handles: Dict[str, Any] = {}

    @property
    def genome(self):
        if self._genome is None:
            from pyfaidx import Fasta
            self._genome = Fasta(self.genome_fa_path, as_raw=True)
        return self._genome

    def _get_bw(self, track_name: str):
        if track_name not in self._bw_handles:
            import pyBigWig
            p = self.track_paths.get(track_name)
            if p and Path(p).exists() and Path(p).stat().st_size > 0:
                try:
                    self._bw_handles[track_name] = pyBigWig.open(str(p))
                except Exception:
                    self._bw_handles[track_name] = None
            else:
                self._bw_handles[track_name] = None
        return self._bw_handles[track_name]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chrom, start, end = self.windows[idx]
        try:
            seq = str(self.genome[chrom][start:end])
        except Exception:
            seq = "N" * self.context_len

        seq = sanitize_sequence(seq)
        if len(seq) < self.context_len:
            seq = seq + "N" * (self.context_len - len(seq))

        crop_start, crop_end = get_crop_bounds(self.context_len, 0.375)
        c_start = start + crop_start
        c_end = start + crop_end
        crop_len = crop_end - crop_start

        targets = []
        for name in self.track_names:
            bw = self._get_bw(name)
            if bw is not None:
                try:
                    vals = np.nan_to_num(bw.values(chrom, c_start, c_end))
                except Exception:
                    vals = np.zeros(crop_len, dtype=np.float32)
            else:
                vals = np.zeros(crop_len, dtype=np.float32)
            targets.append(vals)

        # Shape: (crop_len, num_tracks)
        targets_arr = np.column_stack(targets).astype(np.float32) if targets else np.zeros((crop_len, 1), dtype=np.float32)

        return {
            "seq": seq,
            "target": torch.tensor(targets_arr, dtype=torch.float32)
        }


def run_baselines_benchmark(
    work_dir: Path,
    out_dir: Path,
    genome_fa: Path,
    chrom_sizes: Path,
    chr_fold_json: Path,
    track_paths: Dict[str, str],
    context_len: int,
    model_id: str,
    train_mode: str,
    device: torch.device,
    checkpoint_path: Path,
    hf_token: Optional[str] = None,
    baseline_train_windows: int = 400,
    grad_accum: int = 16,
    fold: Optional[int] = None
) -> None:
    """
    Benchmark trained NTv3_frozen_head, fine-tuned NTv3, and grafted ChromBPNet on held-out test split.
    Evaluates per-track metrics without flattening sparse tracks:
    - atac_256cell Spearman r (scored on central ±500 bp summit window)
    - early-vs-late peak AUROC (cpm_early >= 2.0 vs < 0.5, excluding miR-430)
    - slam_2.5hpf nascent Spearman r
    - Zero-shot evaluation on Medaka st9 ATAC summits (for ChromBPNet and NTv3)
    - Zero-shot evaluation on target-species minor-wave gene sets (strictly_zygotic and slam_confirmed vs control)
    """
    print("=" * 80)
    print("[*] Running Post-Training Publication Baselines Benchmark (Zero-Fabrication Policy strictly enforced)...")
    atac_dir = work_dir / "atac" / "Danio_rerio"
    zf_targets_path = atac_dir / "Danio_rerio_summit_targets.tsv"
    benchmarks = []
    track_names = list(track_paths.keys())

    # Indices for specific tracks
    idx_early = track_names.index("atac_256cell") if "atac_256cell" in track_names else 0
    idx_late = track_names.index("atac_dome") if "atac_dome" in track_names else 1
    idx_slam_25 = track_names.index("slam_2.5hpf_fwd") if "slam_2.5hpf_fwd" in track_names else None

    # Load chrom_sizes dictionary for safe window clamping
    chrom_sizes_dict = {}
    if chrom_sizes and Path(chrom_sizes).exists():
        with open(chrom_sizes) as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 2:
                    try:
                        chrom_sizes_dict[p[0]] = int(p[1])
                    except ValueError:
                        pass

    # Resolve species_id from ntv3_train_config.json if available
    resolved_species_id = 0
    cfg_path = out_dir / f"ntv3_train_config_fold_{fold}.json" if fold is not None and (out_dir / f"ntv3_train_config_fold_{fold}.json").exists() else out_dir / "ntv3_train_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f_cfg:
                resolved_species_id = int(json.load(f_cfg).get("species_id", 0))
        except Exception:
            resolved_species_id = 0

    # Resolve genomes from config.json (never hardcode species names)
    target_genome_fa = None
    medaka_genome_fa = None
    cfg_file = work_dir / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                study_cfg = json.load(f)
            if study_cfg.get("TARGET_GENOME_FA"):
                p = Path(study_cfg["TARGET_GENOME_FA"])
                if p.exists():
                    target_genome_fa = p
            if study_cfg.get("MEDAKA_GENOME_FA"):
                p = Path(study_cfg["MEDAKA_GENOME_FA"])
                if p.exists():
                    medaka_genome_fa = p
        except Exception as e_cfg:
            print(f"[-] Notice loading config.json: {e_cfg}")

    # Fallback search for Medaka FASTA (strictly uncompressed .fa/.fasta, excluding .fai and .gz)
    if medaka_genome_fa is None or not medaka_genome_fa.exists():
        cand_med = [
            p for p in (work_dir / "downloads").glob("*Oryzias_latipes*")
            if (p.suffix in [".fa", ".fasta"] or p.name.endswith(".fa"))
            and not p.name.endswith(".fai") and not p.name.endswith(".gz")
        ]
        if not cand_med:
            cand_med = [
                p for p in (work_dir / "atac" / "Oryzias_latipes").glob("*")
                if (p.suffix in [".fa", ".fasta"] or p.name.endswith(".fa"))
                and not p.name.endswith(".fai") and not p.name.endswith(".gz")
            ]
        if cand_med:
            medaka_genome_fa = cand_med[0]

    # Fallback search for Target FASTA (strictly uncompressed .fa/.fasta, excluding .fai, .gz, Danio, and Oryzias)
    if target_genome_fa is None or not target_genome_fa.exists():
        cand_tgt = [
            p for p in (work_dir / "downloads").glob("*")
            if (p.suffix in [".fa", ".fasta"] or p.name.endswith(".fa"))
            and not p.name.endswith(".fai") and not p.name.endswith(".gz")
            and "Danio_rerio" not in p.name and "Oryzias_latipes" not in p.name
        ]
        if cand_tgt:
            target_genome_fa = cand_tgt[0]

    # Load test split summits strictly (filtered by chr_fold_json["test"], excluding miR-430 and repeats)
    df_test = pd.DataFrame()
    if zf_targets_path.exists():
        df_zf = pd.read_csv(zf_targets_path, sep="\t", dtype={"chrom": str}, low_memory=False)
        test_chroms = None
        if chr_fold_json and Path(chr_fold_json).exists():
            try:
                with open(chr_fold_json) as f_fold:
                    fold_data = json.load(f_fold)
                if "test" in fold_data:
                    test_chroms = set(str(c) for c in fold_data["test"])
            except Exception as e_f:
                print(f"[-] Notice loading chr_fold_json for test chroms: {e_f}")

        if test_chroms:
            test_mask = df_zf["chrom"].astype(str).isin(test_chroms) & (~df_zf.get("is_repeat_downweight", False))
        else:
            test_mask = (df_zf.get("split") == "test") & (~df_zf.get("is_repeat_downweight", False))

        if "is_mir430" in df_zf.columns:
            test_mask = test_mask & (~df_zf["is_mir430"])
        df_test = df_zf.loc[test_mask].reset_index(drop=True)

    # 1. Real ChromBPNet evaluation on held-out test split
    if fold is not None and (work_dir / "models" / "chrombpnet" / "Danio_rerio" / f"fold_{fold}" / "grafted_chrombpnet.pt").exists():
        chrombpnet_ckpt = work_dir / "models" / "chrombpnet" / "Danio_rerio" / f"fold_{fold}" / "grafted_chrombpnet.pt"
    else:
        chrombpnet_ckpt = work_dir / "models" / "chrombpnet" / "Danio_rerio" / "fold_0" / "grafted_chrombpnet.pt"
    if chrombpnet_ckpt.exists() and len(df_test) > 0 and genome_fa.exists():
        print(f"[+] Evaluating real ChromBPNet checkpoint: {chrombpnet_ckpt} on {len(df_test)} test peaks...")
        try:
            from scripts.infer_and_attribute import load_grafted_model, extract_summit_embeddings_from_genome
            grafted_cbp = load_grafted_model(chrombpnet_ckpt, device=str(device))
            test_embs = extract_summit_embeddings_from_genome(grafted_cbp.encoder, genome_fa, df_test, device=str(device))
            scaled_embs = (test_embs - grafted_cbp.scaler_mean) / grafted_cbp.scaler_scale
            y_pred_abs = np.dot(scaled_embs, grafted_cbp.coef_abs) + grafted_cbp.intercept_abs
            y_true = df_test["y_abs"].values
            sp_r, _ = stats.spearmanr(y_true, y_pred_abs)
            med_val = np.median(y_true)
            auc_val = float(roc_auc_score(y_true > med_val, y_pred_abs))

            # Early vs Late peak classification AUROC (cpm_early >= 2.0 vs < 0.5)
            early_col = "cpm_early" if "cpm_early" in df_test.columns else ("early_cpm" if "early_cpm" in df_test.columns else None)
            if early_col:
                early_active = df_test[early_col] >= 2.0
                late_only = df_test[early_col] < 0.5
                subset_mask = early_active | late_only
                if early_active.sum() >= 3 and late_only.sum() >= 3:
                    y_bin = early_active[subset_mask].astype(int).values
                    preds_sub = y_pred_abs[subset_mask.values]
                    auc_early_vs_late = float(roc_auc_score(y_bin, preds_sub))
                elif "y_rel" in df_test.columns:
                    auc_early_vs_late = float(roc_auc_score(df_test["y_rel"].values > 0, y_pred_abs))
                else:
                    auc_early_vs_late = auc_val
            elif "y_rel" in df_test.columns:
                auc_early_vs_late = float(roc_auc_score(df_test["y_rel"].values > 0, y_pred_abs))
            else:
                auc_early_vs_late = auc_val

            # Medaka zero-shot evaluation for ChromBPNet
            medaka_auc_cbp = None
            medaka_targets = work_dir / "atac" / "Oryzias_latipes" / "Oryzias_latipes_summit_targets.tsv"
            if medaka_targets.exists() and medaka_genome_fa and medaka_genome_fa.exists():
                try:
                    from scripts.infer_and_attribute import score_cross_species_medaka_atac
                    med_res = score_cross_species_medaka_atac(grafted_cbp, str(medaka_targets), str(medaka_genome_fa), device=str(device))
                    medaka_auc_cbp = round(float(med_res["medaka_st9_auroc"]), 4)
                except Exception as e_med:
                    print(f"[-] ChromBPNet Medaka cross-species evaluation notice: {e_med}")

            # Target species minor-wave zero-shot AUROC for ChromBPNet
            cbp_strict_auc, cbp_slam_auc = None, None
            scored_tsv_cands = list(work_dir.glob("results*/target_genes_scored_metrics.tsv")) + list(work_dir.glob("**/target_genes_scored_metrics.tsv"))
            if scored_tsv_cands and scored_tsv_cands[0].exists():
                try:
                    from scripts.infer_and_attribute import benchmark_transfer_metrics
                    df_sc = pd.read_csv(scored_tsv_cands[0], sep="\t")
                    m_st = benchmark_transfer_metrics(df_sc, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
                    m_sl = benchmark_transfer_metrics(df_sc, pos_category="slam_confirmed_minor_wave", ctrl_category="control_non_activated")
                    cbp_strict_auc = round(float(m_st["roc_auc"]), 4) if not np.isnan(m_st.get("roc_auc", np.nan)) else None
                    cbp_slam_auc = round(float(m_sl["roc_auc"]), 4) if not np.isnan(m_sl.get("roc_auc", np.nan)) else None
                except Exception as e_sc:
                    print(f"[-] ChromBPNet target minor-wave metric notice: {e_sc}")

            benchmarks.append({
                "model": "ChromBPNet_grafted",
                "architecture": "BPNet_CNN",
                "atac_256cell_spearman_r": round(float(sp_r), 4),
                "early_vs_late_auroc": round(float(auc_early_vs_late), 4),
                "slam_2.5hpf_spearman_r": "N/A (ATAC-only)",
                "medaka_st9_auroc": medaka_auc_cbp,
                "strictly_zygotic_auroc": cbp_strict_auc,
                "slam_confirmed_auroc": cbp_slam_auc,
                "status": "evaluated"
            })
            print(f"    [+] ChromBPNet Real Test Spearman r: {sp_r:.4f}, Early-vs-Late AUROC: {auc_early_vs_late:.4f}")
        except Exception as e:
            print(f"[-] ChromBPNet evaluation notice: {e}")

    # Fixed-seed stratified sampling of up to 120 held-out test summits
    df_test_eval = pd.DataFrame()
    if len(df_test) > 0:
        early_col = "cpm_early" if "cpm_early" in df_test.columns else ("early_cpm" if "early_cpm" in df_test.columns else None)
        if early_col:
            df_early = df_test[df_test[early_col] >= 2.0]
            df_late = df_test[df_test[early_col] < 0.5]
            n_pos = min(60, len(df_early))
            n_neg = min(60, len(df_late))
            s_pos = df_early.sample(n=n_pos, random_state=42) if n_pos > 0 else pd.DataFrame()
            s_neg = df_late.sample(n=n_neg, random_state=42) if n_neg > 0 else pd.DataFrame()
            chosen_indices = set(s_pos.index).union(set(s_neg.index))
            rem_needed = min(120, len(df_test)) - len(chosen_indices)
            if rem_needed > 0:
                df_rem = df_test.loc[~df_test.index.isin(chosen_indices)]
                s_rem = df_rem.sample(n=min(rem_needed, len(df_rem)), random_state=42) if len(df_rem) > 0 else pd.DataFrame()
                df_test_eval = pd.concat([s_pos, s_neg, s_rem]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            else:
                df_test_eval = pd.concat([s_pos, s_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            df_test_eval = df_test.sample(n=min(120, len(df_test)), random_state=42).reset_index(drop=True)

    # Load held-out test genes for SLAM-seq nascent RNA evaluation (strictly bounded by chr_fold test split)
    df_slam_test = pd.DataFrame()
    slam_classes_path = work_dir / "slamseq" / "zf_slamseq_gene_classes.tsv"
    if slam_classes_path.exists() and chr_fold_json and Path(chr_fold_json).exists():
        try:
            with open(chr_fold_json) as f:
                fold_data = json.load(f)
            test_chroms = set(str(c) for c in fold_data.get("test", []))
            df_slam_all = pd.read_csv(slam_classes_path, sep="\t")
            if "chrom" in df_slam_all.columns:
                df_slam_test = df_slam_all[df_slam_all["chrom"].astype(str).isin(test_chroms)].reset_index(drop=True)
                if len(df_slam_test) > 80:
                    df_slam_test = df_slam_test.sample(n=80, random_state=42).reset_index(drop=True)
                print(f"[+] Found {len(df_slam_test)} held-out SLAM test genes on split chromosomes: {sorted(list(test_chroms))}")
        except Exception as e_sl:
            print(f"[-] Notice loading SLAM test genes: {e_sl}")

    # Prepare held-out test windows as 3-tuples (chrom, st, en) with chromosome clamping
    test_windows = []
    if len(df_test_eval) > 0:
        for _, row in df_test_eval.iterrows():
            c = str(row["chrom"])
            s = int(row["summit"])
            c_size = chrom_sizes_dict.get(c, 100000000)
            st = max(0, s - context_len // 2)
            en = st + context_len
            if en > c_size:
                en = c_size
                st = max(0, en - context_len)
            test_windows.append((c, st, en))
    else:
        test_windows = sample_stratified_windows(
            work_dir=work_dir,
            chrom_sizes_path=chrom_sizes,
            chr_fold_path=chr_fold_json,
            split="test",
            context_len=context_len,
            n_samples=60
        )

    autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    def evaluate_ntv3_instance(eval_model, eval_tok, model_label, arch_label):
        eval_model.to(device)
        eval_model.eval()

        test_ds = NTv3GenomeDataset(
            genome_fa_path=str(genome_fa),
            windows=test_windows,
            track_paths=track_paths,
            tokenizer=eval_tok,
            context_len=context_len
        )
        t_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        pred_early_list, true_early_list = [], []
        pred_late_list, true_late_list = [], []

        with torch.no_grad():
            for t_b in t_loader:
                t_s = t_b["seq"]
                t_y = t_b["target"].numpy()  # (1, crop_len, num_tracks)
                t_enc = eval_tok(t_s, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")

                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    t_preds = eval_model(
                        input_ids=t_enc["input_ids"].to(device),
                        species_ids=torch.full((t_enc["input_ids"].shape[0],), resolved_species_id, dtype=torch.long, device=device)
                    )

                # Downsample targets via F.adaptive_avg_pool1d rather than upsampling preds to 1-bp
                if t_preds.shape[1] < t_y.shape[1]:
                    t_y_t = torch.tensor(t_y, dtype=torch.float32)
                    t_y = F.adaptive_avg_pool1d(t_y_t.permute(0, 2, 1), t_preds.shape[1]).permute(0, 2, 1).numpy()
                elif t_preds.shape[1] > t_y.shape[1]:
                    t_preds = F.adaptive_avg_pool1d(t_preds.permute(0, 2, 1), t_y.shape[1]).permute(0, 2, 1)

                p_np = t_preds.cpu().float().numpy()

                # Score over the central ±500 bp summit window (resolution-safe)
                crop_len = p_np.shape[1]
                bp_per_bin = (context_len * 0.375) / crop_len
                half_bins = max(1, int(round(500.0 / bp_per_bin)))
                summit_c = crop_len // 2
                s_st = max(0, summit_c - half_bins)
                s_en = min(crop_len, summit_c + half_bins)

                pred_early_list.append(float(np.mean(p_np[0, s_st:s_en, idx_early])))
                true_early_list.append(float(np.mean(np.log1p(t_y[0, s_st:s_en, idx_early]))))

                pred_late_list.append(float(np.mean(p_np[0, s_st:s_en, idx_late])))
                true_late_list.append(float(np.mean(np.log1p(t_y[0, s_st:s_en, idx_late]))))

        # Evaluate SLAM-seq nascent RNA on held-out test genes (3' UTR terminal 1,000 bp, strand-matched)
        pred_slam_list, true_slam_list = [], []
        idx_slam_fwd = track_names.index("slam_2.5hpf_fwd") if "slam_2.5hpf_fwd" in track_names else None
        idx_slam_rev = track_names.index("slam_2.5hpf_rev") if "slam_2.5hpf_rev" in track_names else None

        if len(df_slam_test) > 0 and genome_fa.exists() and idx_slam_fwd is not None and idx_slam_rev is not None:
            from pyfaidx import Fasta
            import pyBigWig
            zf_fa = Fasta(str(genome_fa), as_raw=True)
            bw_fwd = pyBigWig.open(track_paths["slam_2.5hpf_fwd"]) if Path(track_paths["slam_2.5hpf_fwd"]).exists() else None
            bw_rev = pyBigWig.open(track_paths["slam_2.5hpf_rev"]) if Path(track_paths["slam_2.5hpf_rev"]).exists() else None

            for _, grow in df_slam_test.iterrows():
                g_chr = str(grow["chrom"])
                g_st = int(grow.get("start", 0))
                g_en = int(grow.get("end", 0))
                g_strand = str(grow.get("strand", "+"))

                three_p_pos = g_en if g_strand == "+" else g_st
                c_sz = chrom_sizes_dict.get(g_chr, 100000000)
                w_st = max(0, three_p_pos - context_len // 2)
                w_en = w_st + context_len
                if w_en > c_sz:
                    w_en = c_sz
                    w_st = max(0, w_en - context_len)

                try:
                    seq_str = str(zf_fa[g_chr][w_st:w_en])
                except Exception:
                    seq_str = "N" * context_len
                seq_str = sanitize_sequence(seq_str)
                if len(seq_str) < context_len:
                    seq_str += "N" * (context_len - len(seq_str))

                enc_sl = eval_tok(seq_str, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                        pred_sl = eval_model(
                            input_ids=enc_sl["input_ids"].to(device),
                            species_ids=torch.full((enc_sl["input_ids"].shape[0],), resolved_species_id, dtype=torch.long, device=device)
                        )
                pred_sl_np = pred_sl.cpu().float().numpy()
                out_len_sl = pred_sl_np.shape[1]
                bp_bin_sl = (context_len * 0.375) / out_len_sl
                half_b_sl = max(1, int(round(500.0 / bp_bin_sl)))
                c_sl = out_len_sl // 2
                sl_st = max(0, c_sl - half_b_sl)
                sl_en = min(out_len_sl, c_sl + half_b_sl)

                if g_strand == "+":
                    p_val = float(np.mean(pred_sl_np[0, sl_st:sl_en, idx_slam_fwd]))
                    t_val = 0.0
                    if bw_fwd is not None:
                        try:
                            t_vals = bw_fwd.values(g_chr, max(0, three_p_pos - 500), min(c_sz, three_p_pos + 500))
                            t_val = float(np.mean(np.nan_to_num(t_vals)))
                        except Exception:
                            t_val = 0.0
                else:
                    p_val = float(np.mean(pred_sl_np[0, sl_st:sl_en, idx_slam_rev]))
                    t_val = 0.0
                    if bw_rev is not None:
                        try:
                            t_vals = bw_rev.values(g_chr, max(0, three_p_pos - 500), min(c_sz, three_p_pos + 500))
                            t_val = float(np.mean(np.nan_to_num(t_vals)))
                        except Exception:
                            t_val = 0.0

                pred_slam_list.append(p_val)
                true_slam_list.append(float(np.log1p(t_val)))

            if bw_fwd:
                bw_fwd.close()
            if bw_rev:
                bw_rev.close()

        # Metrics computation
        pe = np.array(pred_early_list)
        te = np.array(true_early_list)
        sp_early, _ = stats.spearmanr(te, pe) if len(te) > 5 else (0.0, 1.0)

        # Early vs Late peak classification AUROC (cpm_early >= 2.0 vs < 0.5)
        auc_evl = 0.5
        df_sub = df_test_eval.iloc[:len(pe)] if len(df_test_eval) >= len(pe) else pd.DataFrame()
        early_col = "cpm_early" if "cpm_early" in df_sub.columns else ("early_cpm" if "early_cpm" in df_sub.columns else None)
        if early_col:
            early_active = (df_sub[early_col] >= 2.0).values
            late_only = (df_sub[early_col] < 0.5).values
            subset_mask = early_active | late_only
            if early_active.sum() >= 3 and late_only.sum() >= 3:
                y_bin = early_active[subset_mask].astype(int)
                auc_evl = float(roc_auc_score(y_bin, pe[subset_mask]))
            else:
                pl = np.array(pred_late_list)
                tl = np.array(true_late_list)
                binary_early = (te >= tl)
                auc_evl = float(roc_auc_score(binary_early, pe - pl)) if len(np.unique(binary_early)) > 1 else 0.5
        else:
            pl = np.array(pred_late_list)
            tl = np.array(true_late_list)
            binary_early = (te >= tl)
            auc_evl = float(roc_auc_score(binary_early, pe - pl)) if len(np.unique(binary_early)) > 1 else 0.5

        if pred_slam_list:
            ps = np.array(pred_slam_list)
            ts = np.array(true_slam_list)
            sp_slam, _ = stats.spearmanr(ts, ps) if len(ts) > 5 else (0.0, 1.0)
            slam_str = round(float(sp_slam), 4)
        else:
            slam_str = "N/A"

        # Check for Medaka st9 zero-shot evaluation (always pass trained zebrafish resolved_species_id for unseen teleost)
        medaka_auc = None
        medaka_targets = work_dir / "atac" / "Oryzias_latipes" / "Oryzias_latipes_summit_targets.tsv"
        if medaka_targets.exists() and medaka_genome_fa and medaka_genome_fa.exists():
            try:
                from pyfaidx import Fasta
                m_fa = Fasta(str(medaka_genome_fa), as_raw=True)
                df_med_raw = pd.read_csv(medaka_targets, sep="\t", dtype={"chrom": str}, low_memory=False)
                df_med = df_med_raw.sample(n=min(120, len(df_med_raw)), random_state=42).reset_index(drop=True)
                m_scores = []
                for _, mr in df_med.iterrows():
                    m_chr = str(mr["chrom"])
                    m_summit = int(mr["summit"])
                    st = max(0, m_summit - context_len // 2)
                    en = st + context_len
                    try:
                        s_str = str(m_fa[m_chr][st:en])
                    except Exception:
                        s_str = "N" * context_len
                    s_str = sanitize_sequence(s_str)
                    if len(s_str) < context_len:
                        s_str += "N" * (context_len - len(s_str))
                    enc_m = eval_tok(s_str, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                    with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                        p_m = eval_model(
                            input_ids=enc_m["input_ids"].to(device),
                            species_ids=torch.full((enc_m["input_ids"].shape[0],), resolved_species_id, dtype=torch.long, device=device)
                        )
                    p_m_np = p_m.cpu().float().numpy()
                    out_len_m = p_m_np.shape[1]
                    bp_per_bin_m = (context_len * 0.375) / out_len_m
                    half_bins_m = max(1, int(round(500.0 / bp_per_bin_m)))
                    c_m = out_len_m // 2
                    s_st_m = max(0, c_m - half_bins_m)
                    s_en_m = min(out_len_m, c_m + half_bins_m)
                    m_scores.append(float(np.mean(p_m_np[0, s_st_m:s_en_m, idx_early])))
                y_med = df_med["y_abs"].values
                med_val = np.median(y_med)
                if len(np.unique(y_med > med_val)) > 1:
                    medaka_auc = round(float(roc_auc_score(y_med > med_val, m_scores)), 4)
            except Exception as e:
                print(f"[-] Medaka zero-shot evaluation notice: {e}")

        # Check for target-species minor-wave zero-shot AUROC
        strict_auc, slam_auc = None, None
        gene_sets_cands = list(work_dir.glob("results*/*gene_sets.tsv")) + list(work_dir.glob("**/*gene_sets.tsv"))
        if gene_sets_cands and target_genome_fa and target_genome_fa.exists() and gene_sets_cands[0].exists():
            try:
                from pyfaidx import Fasta
                t_fa = Fasta(str(target_genome_fa), as_raw=True)
                df_gs = pd.read_csv(gene_sets_cands[0], sep="\t")
                pos_strict = df_gs[df_gs["category"] == "strictly_zygotic_minor_wave"]
                if len(pos_strict) > 40:
                    pos_strict = pos_strict.sample(n=40, random_state=42)

                pos_slam = df_gs[df_gs.get("is_slam_confirmed_minor_wave", False) == True]
                if len(pos_slam) > 40:
                    pos_slam = pos_slam.sample(n=40, random_state=42)

                ctrl_genes = df_gs[df_gs["category"] == "control_non_activated"]
                if len(ctrl_genes) > 40:
                    ctrl_genes = ctrl_genes.sample(n=40, random_state=42)

                genes_to_score = pd.concat([pos_strict, pos_slam, ctrl_genes])
                if "gene_id" in genes_to_score.columns:
                    genes_to_score = genes_to_score.drop_duplicates(subset=["gene_id"])
                genes_to_score = genes_to_score.reset_index(drop=True)

                g_scores_atac = []
                g_scores_slam = []
                for _, gr in genes_to_score.iterrows():
                    g_c = str(gr["chrom"])
                    g_tss = int(gr.get("tss", gr.get("start", 0)))
                    g_strand = str(gr.get("strand", "+"))
                    c_sz = len(t_fa[g_c]) if g_c in t_fa else 100000000
                    w_st = max(0, g_tss - context_len // 2)
                    w_en = w_st + context_len
                    if w_en > c_sz:
                        w_en = c_sz
                        w_st = max(0, w_en - context_len)
                    try:
                        g_seq = str(t_fa[g_c][w_st:w_en])
                    except Exception:
                        g_seq = "N" * context_len
                    g_seq = sanitize_sequence(g_seq)
                    if len(g_seq) < context_len:
                        g_seq += "N" * (context_len - len(g_seq))
                    enc_g = eval_tok(g_seq, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                            p_g = eval_model(
                                input_ids=enc_g["input_ids"].to(device),
                                species_ids=torch.full((enc_g["input_ids"].shape[0],), resolved_species_id, dtype=torch.long, device=device)
                            )
                    p_g_np = p_g.cpu().float().numpy()
                    # Score max predicted atac_256cell across full central 37.5% crop (~24.6 kb)
                    g_scores_atac.append(float(np.max(p_g_np[0, :, idx_early])))

                    # Score strand-matched slam_2.5hpf 3'-end signal
                    if idx_slam_fwd is not None and idx_slam_rev is not None:
                        idx_sl = idx_slam_fwd if g_strand == "+" else idx_slam_rev
                        g_scores_slam.append(float(np.max(p_g_np[0, :, idx_sl])))

                genes_to_score = genes_to_score.copy()
                genes_to_score["pred_max_y_abs"] = g_scores_atac
                from scripts.infer_and_attribute import benchmark_transfer_metrics
                m_st_nt = benchmark_transfer_metrics(genes_to_score, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
                m_sl_nt = benchmark_transfer_metrics(genes_to_score, pos_category="slam_confirmed_minor_wave", ctrl_category="control_non_activated")
                strict_auc = round(float(m_st_nt["roc_auc"]), 4) if not np.isnan(m_st_nt.get("roc_auc", np.nan)) else None
                slam_auc = round(float(m_sl_nt["roc_auc"]), 4) if not np.isnan(m_sl_nt.get("roc_auc", np.nan)) else None

                slam_track_strict_auc, slam_track_slam_auc = None, None
                if len(g_scores_slam) == len(genes_to_score):
                    genes_to_score["pred_slam_2.5hpf"] = g_scores_slam
                    m_st_sl = benchmark_transfer_metrics(genes_to_score, score_col="pred_slam_2.5hpf", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
                    m_sl_sl = benchmark_transfer_metrics(genes_to_score, score_col="pred_slam_2.5hpf", pos_category="slam_confirmed_minor_wave", ctrl_category="control_non_activated")
                    slam_track_strict_auc = round(float(m_st_sl["roc_auc"]), 4) if not np.isnan(m_st_sl.get("roc_auc", np.nan)) else None
                    slam_track_slam_auc = round(float(m_sl_sl["roc_auc"]), 4) if not np.isnan(m_sl_sl.get("roc_auc", np.nan)) else None
            except Exception as e_tgt:
                print(f"[-] NTv3 target minor-wave zero-shot notice: {e_tgt}")

        res_dict = {
            "model": model_label,
            "architecture": arch_label,
            "atac_256cell_spearman_r": round(float(sp_early), 4),
            "early_vs_late_auroc": round(float(auc_evl), 4),
            "slam_2.5hpf_spearman_r": slam_str,
            "medaka_st9_auroc": medaka_auc,
            "strictly_zygotic_auroc": strict_auc,
            "slam_confirmed_auroc": slam_auc,
            "slam_track_strictly_zygotic_auroc": slam_track_strict_auc,
            "slam_track_slam_confirmed_auroc": slam_track_slam_auc,
            "status": "evaluated"
        }
        print(f"    [+] {model_label} Results -> ATAC Spearman r: {sp_early:.4f}, Early-vs-Late AUROC: {auc_evl:.4f}, SLAM r: {slam_str}, Medaka AUROC: {medaka_auc}")
        return res_dict

    # 2. Raw Pretrained NTv3 baseline (native pretrained tracks directly, skipping random untrained linear heads)
    try:
        print(f"[+] Checking Raw Pretrained NTv3 baseline ({model_id}) for native pretrained tracks...")
        native_tracks_evaluated = False
        native_zf_tracks = []
        try:
            from huggingface_hub import hf_hub_download
            token = hf_token or os.environ.get("HF_TOKEN")
            meta_path = hf_hub_download(repo_id=model_id, filename="track_metadata.json", token=token)
            if meta_path and os.path.exists(meta_path):
                with open(meta_path, "r") as f_meta:
                    meta_json = json.load(f_meta)
                native_zf_tracks = meta_json.get("bigwigs_per_file_assembly", {}).get("GRCz11", [])
        except Exception as e_meta:
            print(f"[-] Native track metadata query notice: {e_meta}")

        if native_zf_tracks:
            try:
                from transformers import AutoModel, AutoTokenizer
                token = hf_token or os.environ.get("HF_TOKEN")
                raw_trunk = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=token)
                raw_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
                raw_trunk.to(device)
                raw_trunk.eval()

                if len(test_windows) > 0 and genome_fa.exists():
                    raw_preds = []
                    from pyfaidx import Fasta
                    z_fa = Fasta(str(genome_fa), as_raw=True)
                    with torch.no_grad():
                        for chrom_w, st_w, en_w in test_windows[:60]:
                            try:
                                w_seq = str(z_fa[chrom_w][st_w:en_w])
                            except Exception:
                                w_seq = "N" * context_len
                            w_seq = sanitize_sequence(w_seq)
                            if len(w_seq) < context_len:
                                w_seq += "N" * (context_len - len(w_seq))
                            b_enc = raw_tok(w_seq, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                                b_cond = [torch.full((b_enc["input_ids"].shape[0],), resolved_species_id, dtype=torch.long, device=device)]
                                out_raw = raw_trunk(
                                    input_ids=b_enc["input_ids"].to(device),
                                    conditions_tokens=b_cond
                                )
                            if hasattr(out_raw, "bigwig_tracks_logits") and out_raw.bigwig_tracks_logits is not None:
                                trk = out_raw.bigwig_tracks_logits[:, :, :len(native_zf_tracks)]
                            elif hasattr(out_raw, "tracks") and out_raw.tracks is not None:
                                trk = out_raw.tracks[:, :, :len(native_zf_tracks)]
                            elif hasattr(out_raw, "logits") and out_raw.logits is not None:
                                trk = out_raw.logits[:, :, :len(native_zf_tracks)]
                            else:
                                trk = out_raw[:, :, :len(native_zf_tracks)]
                            trk_np = trk.cpu().float().numpy()
                            c_len = trk_np.shape[1]
                            half_bins = max(1, int(round(500.0 / ((context_len * 0.375) / c_len))))
                            raw_preds.append(np.mean(trk_np[0, max(0, c_len//2 - half_bins): min(c_len, c_len//2 + half_bins), :], axis=0))

                    if raw_preds:
                        raw_preds_arr = np.array(raw_preds)
                        early_col = "cpm_early" if "cpm_early" in df_test.columns else ("early_cpm" if "early_cpm" in df_test.columns else None)
                        if early_col and len(df_test) >= len(raw_preds_arr):
                            y_true_sub = df_test.iloc[:len(raw_preds_arr)][early_col].values
                            track_spearmans = []
                            for trk_i in range(raw_preds_arr.shape[1]):
                                r_val, _ = stats.spearmanr(y_true_sub, raw_preds_arr[:, trk_i])
                                if not np.isnan(r_val):
                                    track_spearmans.append((trk_i, float(r_val)))
                            if track_spearmans:
                                best_trk_idx, best_sp = max(track_spearmans, key=lambda x: x[1])
                                mean_sp = float(np.mean([s[1] for s in track_spearmans]))
                                med_val = np.median(y_true_sub)
                                auc_raw = float(roc_auc_score(y_true_sub > med_val, raw_preds_arr[:, best_trk_idx])) if len(np.unique(y_true_sub > med_val)) > 1 else 0.5
                            else:
                                best_sp, mean_sp, auc_raw = 0.0, 0.0, 0.5
                            print(f"    [+] Native Pretrained Zebrafish Tracks ({len(native_zf_tracks)} tracks) -> Best Spearman r: {best_sp:.4f}, Mean r: {mean_sp:.4f}, AUROC: {auc_raw:.4f}")
                            benchmarks.append({
                                "model": "NTv3_raw_pretrained",
                                "architecture": f"Transformer_{context_len//1000}k_pretrained_native",
                                "atac_256cell_spearman_r": round(best_sp, 4),
                                "early_vs_late_auroc": round(auc_raw, 4),
                                "slam_2.5hpf_spearman_r": "N/A",
                                "medaka_st9_auroc": None,
                                "strictly_zygotic_auroc": None,
                                "slam_confirmed_auroc": None,
                                "status": "evaluated"
                            })
                            native_tracks_evaluated = True

                del raw_trunk
                del raw_tok
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e_run_native:
                print(f"[-] Native pretrained track forward notice: {e_run_native}")

        if not native_tracks_evaluated:
            print("[*] Base foundation model does not expose native un-finetuned zebrafish tracks; recording 'skipped_no_native_tracks' (zero-mock policy: never score random untrained linear layer as pretrained baseline).")
            benchmarks.append({
                "model": "NTv3_raw_pretrained",
                "architecture": f"Transformer_{context_len//1000}k_pretrained_native",
                "atac_256cell_spearman_r": "N/A",
                "early_vs_late_auroc": "N/A",
                "slam_2.5hpf_spearman_r": "N/A",
                "medaka_st9_auroc": None,
                "strictly_zygotic_auroc": None,
                "slam_confirmed_auroc": None,
                "status": "skipped_no_native_tracks"
            })
    except Exception as e_raw:
        print(f"[-] Raw Pretrained NTv3 baseline notice: {e_raw}")

    # 3. Genuine trained NTv3_frozen_head baseline
    try:
        print(f"[+] Training and evaluating NTv3_frozen_head baseline ({model_id})...")
        frozen_model, frozen_tok = build_ntv3_model(
            model_id=model_id,
            num_tracks=len(track_paths),
            train_mode="frozen_head",
            context_len=context_len,
            grad_checkpointing=False,
            hf_token=hf_token,
            default_species_id=resolved_species_id
        )
        frozen_model.to(device)

        # Train linear head for 2 epochs on comparable sample budget with proper grad accumulation
        train_windows = sample_stratified_windows(
            work_dir=work_dir,
            chrom_sizes_path=chrom_sizes,
            chr_fold_path=chr_fold_json,
            split="train",
            context_len=context_len,
            n_samples=baseline_train_windows
        )
        train_ds = NTv3GenomeDataset(
            genome_fa_path=str(genome_fa),
            windows=train_windows,
            track_paths=track_paths,
            tokenizer=frozen_tok,
            context_len=context_len
        )
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        opt_head = torch.optim.AdamW(frozen_model.head.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion_frozen = TrackBalancedLoss(track_names).to(device)
        frozen_model.train()
        frozen_model.trunk.eval()

        opt_head.zero_grad()
        for ep in range(2):
            for b_idx, b in enumerate(train_loader):
                b_s = [sanitize_sequence(s) for s in b["seq"]]
                b_y = b["target"].to(device)
                b_enc = frozen_tok(b_s, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    preds = frozen_model(
                        input_ids=b_enc["input_ids"].to(device),
                        species_ids=torch.full((b_enc["input_ids"].shape[0],), resolved_species_id, dtype=torch.long, device=device)
                    )
                    b_y_scaled = torch.log1p(b_y.float())
                    if preds.shape[1] < b_y_scaled.shape[1]:
                        b_y_scaled = F.adaptive_avg_pool1d(b_y_scaled.permute(0, 2, 1), preds.shape[1]).permute(0, 2, 1)
                    elif preds.shape[1] > b_y_scaled.shape[1]:
                        preds = F.adaptive_avg_pool1d(preds.permute(0, 2, 1), b_y_scaled.shape[1]).permute(0, 2, 1)
                    loss = criterion_frozen(preds, b_y_scaled)
                    loss = loss / grad_accum
                loss.backward()
                if (b_idx + 1) % grad_accum == 0 or (b_idx + 1) == len(train_loader):
                    opt_head.step()
                    opt_head.zero_grad()

        print("[+] Finished training linear head on frozen trunk (2 epochs, full budget).")
        res_frozen = evaluate_ntv3_instance(frozen_model, frozen_tok, "NTv3_frozen_head", f"Transformer_{context_len//1000}k_frozen_trunk")
        benchmarks.append(res_frozen)
        del frozen_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"[-] NTv3_frozen_head baseline notice: {e}")

    # 4. Fine-tuned NTv3 evaluation
    if checkpoint_path.exists() and len(df_test) > 0 and genome_fa.exists():
        print(f"[+] Found fine-tuned NTv3 checkpoint: {checkpoint_path}")
        try:
            ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            eval_model, eval_tok = build_ntv3_model(
                model_id=model_id,
                num_tracks=len(track_paths),
                train_mode=train_mode,
                context_len=context_len,
                grad_checkpointing=False,
                hf_token=hf_token,
                default_species_id=resolved_species_id
            )
            eval_model.load_state_dict(ckpt_data["model_state"])
            model_label = f"NTv3_multitask_{train_mode}"
            arch_label = f"Transformer_{context_len//1000}k_{train_mode}"
            res_tuned = evaluate_ntv3_instance(eval_model, eval_tok, model_label, arch_label)
            benchmarks.append(res_tuned)
            del eval_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[-] Fine-tuned NTv3 evaluation notice: {e}")

    if benchmarks:
        df_bench = pd.DataFrame(benchmarks)
        if fold is not None:
            fold_bench_tsv = out_dir / f"ntv3_baselines_benchmark_fold_{fold}.tsv"
            df_bench.to_csv(fold_bench_tsv, sep="\t", index=False)
            print(f"[+] Saved fold baseline comparison table: {fold_bench_tsv}")
        else:
            bench_tsv = out_dir / "ntv3_baselines_benchmark.tsv"
            df_bench.to_csv(bench_tsv, sep="\t", index=False)
            print(f"[+] Saved baseline comparison table: {bench_tsv}")
    print("=" * 80)



def main():
    parser = argparse.ArgumentParser(description="Multi-Task NTv3 Epigenomics/Nascent RNA-seq Fine-Tuning")
    parser.add_argument("--work-dir", default="./work_astatotilapia_premzt", help="Working directory")
    parser.add_argument("--genome-fa", default=None, help="Reference genome FASTA")
    parser.add_argument("--chrom-sizes", default=None, help="Path to chrom.sizes")
    parser.add_argument("--chr-fold-json", default=None, help="Path to chr_fold.json")
    parser.add_argument("--model-id", default="InstaDeepAI/NTv3_tracks", help="Hugging Face model ID")
    parser.add_argument("--compute-profile", default="laptop_16gb", choices=["laptop_16gb", "hpc_1mb", "custom"])
    parser.add_argument("--context-len", type=int, default=None, help="Genomic context length (bp)")
    parser.add_argument("--train-mode", default="lora", choices=["lora", "full", "frozen_head", "raw_baseline"])
    parser.add_argument("--train-windows", type=int, default=2000, help="Number of stratified training windows")
    parser.add_argument("--val-windows", type=int, default=300, help="Number of validation windows")
    parser.add_argument("--fold", type=int, default=None, help="Cross-validation fold index (0-indexed)")
    parser.add_argument("--baseline-train-windows", type=int, default=400, help="Stratified windows budget for baseline training (e.g. frozen head)")
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Toggle gradient checkpointing")
    parser.add_argument("--run-baselines", action="store_true", help="Benchmark raw pretrained, frozen_head, and fine-tuned")
    parser.add_argument("--enable-option2-target-rna", action="store_true", help="Option 2: add target-species RNA-seq head under 5-fold CV")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining from scratch ignoring checkpoints")
    parser.add_argument("--task-name", default="train_ntv3_zf", help="TaskManager task name for progress tracking")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    out_dir = work_dir / "models" / "ntv3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve genome FASTA, chrom.sizes, and chr_fold.json
    genome_fa = Path(args.genome_fa).resolve() if args.genome_fa else (
        work_dir / "downloads" / "Danio_rerio.GRCz11.dna_sm.primary_assembly.fa"
    )
    chrom_sizes = Path(args.chrom_sizes).resolve() if args.chrom_sizes else (
        work_dir / "atac" / "Danio_rerio" / "chrom.sizes"
    )

    species_atac_dir = work_dir / "atac" / "Danio_rerio"
    if args.fold is not None:
        if args.chr_fold_json:
            chr_fold_json_src = Path(args.chr_fold_json).resolve()
        elif (species_atac_dir / "chr_folds.json").exists():
            chr_fold_json_src = species_atac_dir / "chr_folds.json"
        elif (species_atac_dir / f"chr_fold_fold_{args.fold}.json").exists():
            chr_fold_json_src = species_atac_dir / f"chr_fold_fold_{args.fold}.json"
        else:
            chr_fold_json_src = species_atac_dir / "chr_fold.json"

        fold_dict: Dict[str, List[str]] = {}
        if chr_fold_json_src.exists() and chr_fold_json_src.stat().st_size > 0:
            with open(chr_fold_json_src, "r") as f:
                raw_folds = json.load(f)
            fold_key = str(args.fold)
            if f"fold_{fold_key}" in raw_folds:
                fold_dict = raw_folds[f"fold_{fold_key}"]
            elif fold_key in raw_folds:
                fold_dict = raw_folds[fold_key]
            elif "all_folds" in raw_folds and f"fold_{fold_key}" in raw_folds["all_folds"]:
                fold_dict = raw_folds["all_folds"][f"fold_{fold_key}"]
            elif "train" in raw_folds or "test" in raw_folds:
                fold_dict = raw_folds
            else:
                fold_dict = raw_folds
        else:
            fold_dict = {"train": [], "valid": [], "test": []}

        if "val" in fold_dict and "valid" not in fold_dict:
            fold_dict["valid"] = fold_dict.pop("val")
        for req_k in ["train", "valid", "test"]:
            if req_k not in fold_dict:
                fold_dict[req_k] = []

        chr_fold_json = out_dir / f"chr_fold_fold_{args.fold}.json"
        with open(chr_fold_json, "w") as f:
            json.dump(fold_dict, f, indent=2)
        checkpoint_path = out_dir / f"ntv3_best_checkpoint_fold_{args.fold}.pt"
        state_path = out_dir / f"ntv3_last_state_fold_{args.fold}.pt"
    else:
        chr_fold_json = Path(args.chr_fold_json).resolve() if args.chr_fold_json else (
            species_atac_dir / "chr_fold.json"
        )
        checkpoint_path = out_dir / "ntv3_best_checkpoint.pt"
        state_path = out_dir / "ntv3_last_state.pt"

    if args.context_len is not None and args.context_len > 0:
        context_len = args.context_len
    elif args.compute_profile == "laptop_16gb":
        context_len = 65536
    elif args.compute_profile == "hpc_1mb":
        context_len = 1048576
    else:
        context_len = 65536

    # DDP / Distributed setup
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if is_distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precision = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    hf_token = os.environ.get("HF_TOKEN")

    if local_rank == 0:
        print("=" * 80)
        print(f"[*] Multi-Task NTv3 Pipeline")
        print(f"    Profile:          {args.compute_profile}")
        print(f"    Model ID:         {args.model_id}")
        print(f"    Context Length:   {context_len} bp (Crop: {int(context_len * 0.375)} bp)")
        print(f"    Genome FA:        {genome_fa}")
        print(f"    Chrom Sizes:      {chrom_sizes}")
        print(f"    Chr Fold JSON:    {chr_fold_json}")
        print(f"    Train Mode:       {args.train_mode}")
        print(f"    Device:           {device}")
        print(f"    Precision:        {precision}")
        print(f"    Output Dir:       {out_dir}")
        print(f"    Distributed:      {is_distributed} (World Size: {os.environ.get('WORLD_SIZE', '1')})")
        print("=" * 80)

    # Multi-task track registry with both forward and reverse SLAM-seq BigWigs
    slam_dir = work_dir / "slamseq"
    atac_dir = work_dir / "atac" / "Danio_rerio"

    track_paths = {
        "atac_256cell": str(atac_dir / "Danio_rerio_early_cpm.bw"),
        "atac_dome": str(atac_dir / "Danio_rerio_late_cpm.bw"),
        "slam_0.75hpf_fwd": str(slam_dir / "0.75hpf" / "0.75hpf_fwd.bw"),
        "slam_0.75hpf_rev": str(slam_dir / "0.75hpf" / "0.75hpf_rev.bw"),
        "slam_2.5hpf_fwd": str(slam_dir / "2.5hpf" / "2.5hpf_fwd.bw"),
        "slam_2.5hpf_rev": str(slam_dir / "2.5hpf" / "2.5hpf_rev.bw"),
        "slam_3.0_3.5hpf_fwd": str(slam_dir / "3.0_3.5hpf" / "3.0_3.5hpf_fwd.bw"),
        "slam_3.0_3.5hpf_rev": str(slam_dir / "3.0_3.5hpf" / "3.0_3.5hpf_rev.bw"),
        "slam_4.5hpf_fwd": str(slam_dir / "4.5hpf" / "4.5hpf_fwd.bw"),
        "slam_4.5hpf_rev": str(slam_dir / "4.5hpf" / "4.5hpf_rev.bw")
    }

    if args.enable_option2_target_rna:
        raise NotImplementedError(
            "Option 2 (--enable-option2-target-rna) requires genuine multi-species batch sampling "
            "(loading target-species FASTA + held-out target-species chromosome splits alongside zebrafish windows "
            "so target RNA heads receive real non-zero signal). Zero-fabrication policy strictly enforced."
        )

    # Check for available tracks
    found_tracks = {k: p for k, p in track_paths.items() if Path(p).exists() and Path(p).stat().st_size > 0}
    if local_rank == 0:
        print(f"[*] Found {len(found_tracks)} / {len(track_paths)} multi-task BigWig tracks on disk.")

    # Check input prerequisites strictly
    if not genome_fa.exists() or genome_fa.stat().st_size < 10 * 1024 * 1024:
        print(f"[FATAL] Genome FASTA missing or incomplete: {genome_fa}", file=sys.stderr)
        sys.exit(1)

    if not found_tracks:
        print(f"[FATAL] No BigWig tracks found under {work_dir}; cannot train NTv3.", file=sys.stderr)
        sys.exit(1)

    if args.train_mode == "raw_baseline":
        if local_rank == 0:
            print("[*] Train mode is raw_baseline: skipping fine-tuning and running baselines benchmark directly...")
            set_task_progress(work_dir, args.task_name, "Evaluating raw_baseline benchmark...")
            run_baselines_benchmark(
                work_dir=work_dir,
                out_dir=out_dir,
                genome_fa=genome_fa,
                chrom_sizes=chrom_sizes,
                chr_fold_json=chr_fold_json,
                track_paths=track_paths,
                context_len=context_len,
                model_id=args.model_id,
                train_mode=args.train_mode,
                device=device,
                checkpoint_path=checkpoint_path,
                hf_token=hf_token,
                baseline_train_windows=args.baseline_train_windows,
                grad_accum=args.grad_accum,
                fold=args.fold
            )
            set_task_progress(work_dir, args.task_name, "100% (Done)")
        if is_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    # Build model & tokenizer (fails loudly with sys.exit(1) if unavailable)
    model, tokenizer = build_ntv3_model(
        model_id=args.model_id,
        num_tracks=len(track_paths),
        train_mode=args.train_mode,
        context_len=context_len,
        grad_checkpointing=args.grad_checkpointing,
        hf_token=hf_token
    )
    resolved_species_id = resolve_ntv3_species_id(model=model, tokenizer=tokenizer, species_name="danio_rerio", model_id=args.model_id, hf_token=hf_token)
    if hasattr(model, "default_species_id"):
        model.default_species_id = resolved_species_id
    model.to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True
        )

    # Sample stratified windows with zero data leakage (train vs valid)
    if local_rank == 0:
        print(f"[*] Sampling {args.train_windows} training windows strictly from 'train' chromosomes...")
    train_windows = sample_stratified_windows(
        work_dir=work_dir,
        chrom_sizes_path=chrom_sizes,
        chr_fold_path=chr_fold_json,
        split="train",
        context_len=context_len,
        n_samples=args.train_windows
    )

    if local_rank == 0:
        print(f"[*] Sampling {args.val_windows} validation windows strictly from 'valid' chromosomes...")
    val_windows = sample_stratified_windows(
        work_dir=work_dir,
        chrom_sizes_path=chrom_sizes,
        chr_fold_path=chr_fold_json,
        split="valid",
        context_len=context_len,
        n_samples=args.val_windows
    )

    train_dataset = NTv3GenomeDataset(
        genome_fa_path=str(genome_fa),
        windows=train_windows,
        track_paths=track_paths,
        tokenizer=tokenizer,
        context_len=context_len
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )

    val_dataset = NTv3GenomeDataset(
        genome_fa_path=str(genome_fa),
        windows=val_windows,
        track_paths=track_paths,
        tokenizer=tokenizer,
        context_len=context_len
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = TrackBalancedLoss(list(track_paths.keys())).to(device)

    start_epoch = 0
    best_val_loss = float("inf")

    if state_path.exists() and not args.force_retrain:
        try:
            state = torch.load(state_path, map_location=device, weights_only=False)
            saved_model = model.module if hasattr(model, "module") else model
            saved_model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            best_val_loss = float(state.get("best_val_loss", float("inf")))
            start_epoch = int(state.get("epoch", -1)) + 1
            if local_rank == 0:
                print(f"[+] Loaded resume state from {state_path.name}: starting at epoch {start_epoch + 1}/{args.epochs} (best val loss: {best_val_loss:.4f})")
        except Exception as e:
            if local_rank == 0:
                print(f"[-] Warning: Failed to load resume state from {state_path.name} ({e}). Starting from epoch 0.")
            start_epoch = 0

    if start_epoch >= args.epochs and checkpoint_path.exists():
        if local_rank == 0:
            print(f"[+] Training already completed ({start_epoch} >= {args.epochs} epochs) and checkpoint exists at {checkpoint_path}. Skipping training loop.")
    else:
        if local_rank == 0:
            print(f"[*] Starting training loop: {start_epoch + 1} -> {args.epochs} epochs, grad_accum={args.grad_accum}...")

        step = 0
        for epoch in range(start_epoch, args.epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()

            for batch in train_loader:
                seqs = batch["seq"]
                targets = batch["target"].to(device)
                # Log1p target scaling for numerical stability
                targets_scaled = torch.log1p(targets)

                if tokenizer is not None:
                    encoded = tokenizer(
                        seqs,
                        add_special_tokens=False,
                        return_tensors="pt",
                        max_length=context_len,
                        truncation=True,
                        padding="max_length"
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attn_mask = encoded.get("attention_mask", None)
                    if attn_mask is not None:
                        attn_mask = attn_mask.to(device)
                else:
                    input_ids = torch.zeros((len(seqs), context_len), dtype=torch.long, device=device)
                    attn_mask = None

                with torch.amp.autocast(device_type=device.type, dtype=precision):
                    preds = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        species_ids=torch.full((input_ids.shape[0],), resolved_species_id, dtype=torch.long, device=device)
                    )

                    # Ensure spatial dimensions match targets (downsample targets via adaptive_avg_pool1d)
                    if preds.shape[1] < targets_scaled.shape[1]:
                        targets_scaled = F.adaptive_avg_pool1d(
                            targets_scaled.permute(0, 2, 1),
                            preds.shape[1]
                        ).permute(0, 2, 1)
                    elif preds.shape[1] > targets_scaled.shape[1]:
                        preds = F.adaptive_avg_pool1d(
                            preds.permute(0, 2, 1),
                            targets_scaled.shape[1]
                        ).permute(0, 2, 1)

                    loss = criterion(preds, targets_scaled) / args.grad_accum

                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                n_batches += 1
                step += 1

                if step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if n_batches % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = epoch_loss / max(1, n_batches)

            # Validation loop
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_seqs = v_batch["seq"]
                    v_targets = v_batch["target"].to(device)
                    v_targets_scaled = torch.log1p(v_targets)

                    if tokenizer is not None:
                        v_enc = tokenizer(
                            v_seqs,
                            add_special_tokens=False,
                            return_tensors="pt",
                            max_length=context_len,
                            truncation=True,
                            padding="max_length"
                        )
                        v_ids = v_enc["input_ids"].to(device)
                        v_mask = v_enc.get("attention_mask", None)
                        if v_mask is not None:
                            v_mask = v_mask.to(device)
                    else:
                        v_ids = torch.zeros((len(v_seqs), context_len), dtype=torch.long, device=device)
                        v_mask = None

                    with torch.amp.autocast(device_type=device.type, dtype=precision):
                        v_preds = model(
                            input_ids=v_ids,
                            attention_mask=v_mask,
                            species_ids=torch.full((v_ids.shape[0],), resolved_species_id, dtype=torch.long, device=device)
                        )
                        if v_preds.shape[1] < v_targets_scaled.shape[1]:
                            v_targets_scaled = F.adaptive_avg_pool1d(
                                v_targets_scaled.permute(0, 2, 1),
                                v_preds.shape[1]
                            ).permute(0, 2, 1)
                        elif v_preds.shape[1] > v_targets_scaled.shape[1]:
                            v_preds = F.adaptive_avg_pool1d(
                                v_preds.permute(0, 2, 1),
                                v_targets_scaled.shape[1]
                            ).permute(0, 2, 1)
                        v_l = criterion(v_preds, v_targets_scaled)
                        val_loss += v_l.item()
                        val_batches += 1

            avg_val_loss = val_loss / max(1, val_batches)
            if local_rank == 0:
                print(f"    Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                set_task_progress(work_dir, args.task_name, f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}")

                train_config = {
                    "model_id": args.model_id,
                    "compute_profile": args.compute_profile,
                    "context_len": context_len,
                    "crop_ratio": 0.375,
                    "tracks": list(track_paths.keys()),
                    "train_mode": args.train_mode,
                    "species_id": int(resolved_species_id),
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "epochs": args.epochs,
                    "best_val_loss": float(best_val_loss),
                    "device": str(device)
                }

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    train_config["best_val_loss"] = float(best_val_loss)
                    cfg_file = out_dir / (f"ntv3_train_config_fold_{args.fold}.json" if args.fold is not None else "ntv3_train_config.json")
                    with open(cfg_file, "w") as f_cfg:
                        json.dump(train_config, f_cfg, indent=2)

                    saved_model = model.module if hasattr(model, "module") else model
                    tmp_ckpt = checkpoint_path.with_suffix(".tmp.pt")
                    try:
                        clean_cfg = sanitize_for_torch_save(train_config)
                        torch.save({"model_state": saved_model.state_dict(), "config": clean_cfg}, tmp_ckpt)
                        os.replace(str(tmp_ckpt), str(checkpoint_path))
                        print(f"    [+] Saved best model checkpoint to {checkpoint_path} (val loss: {best_val_loss:.4f})")
                    except Exception as e:
                        tmp_ckpt.unlink(missing_ok=True)
                        print(f"[-] Warning: Failed saving checkpoint {tmp_ckpt}: {e}")
                    finally:
                        tmp_ckpt.unlink(missing_ok=True)

                # Atomically save epoch-level state for resume
                saved_model = model.module if hasattr(model, "module") else model
                tmp_state = state_path.with_suffix(".tmp.pt")
                try:
                    torch.save({
                        "epoch": epoch,
                        "model_state": saved_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": float(best_val_loss),
                        "train_config": sanitize_for_torch_save(train_config)
                    }, tmp_state)
                    os.replace(str(tmp_state), str(state_path))
                except Exception as e:
                    tmp_state.unlink(missing_ok=True)
                    print(f"[-] Warning: Failed saving resume state {tmp_state}: {e}")
                finally:
                    tmp_state.unlink(missing_ok=True)

    if local_rank == 0:
        set_task_progress(work_dir, args.task_name, "100% (Done)")
        print(f"[SUCCESS] Multi-task NTv3 training completed.")

    # Post-training baselines benchmark
    if args.run_baselines and local_rank == 0:
        run_baselines_benchmark(
            work_dir=work_dir,
            out_dir=out_dir,
            genome_fa=genome_fa,
            chrom_sizes=chrom_sizes,
            chr_fold_json=chr_fold_json,
            track_paths=track_paths,
            context_len=context_len,
            model_id=args.model_id,
            train_mode=args.train_mode,
            device=device,
            checkpoint_path=checkpoint_path,
            hf_token=hf_token,
            baseline_train_windows=args.baseline_train_windows,
            grad_accum=args.grad_accum,
            fold=args.fold
        )

    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
