#!/usr/bin/env python
"""Main training script for LCTSCap with curriculum learning.

Supports phases 1-4 with automatic checkpoint resumption.

Usage:
    python scripts/train.py --config configs/train/phase1.yaml
    python scripts/train.py --config configs/train/phase2.yaml --resume
"""

import argparse
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.config import load_config
from lctscap.data.collator import LCTSCapCollator
from lctscap.data.dataset import LCTSCapDataset
from lctscap.models.full_model import LCTSCapModel, ModelConfig
from lctscap.models.losses import clip_infonce, coverage_loss, event_loss, compute_total_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    """Construct the LCTSCapModel from config."""
    model_cfg = ModelConfig(
        d_model=cfg.model.d_model,
        num_channels=cfg.data.channels_for(cfg.data.dataset),
        patch_size=cfg.model.patch_size,
        local_encoder_layers=cfg.model.num_layers_local,
        local_encoder_heads=cfg.model.num_heads,
        planner_layers=cfg.model.num_layers_planner,
        planner_heads=cfg.model.num_heads,
        segment_size=cfg.model.segment_size,
        decoder_layers=cfg.model.decoder_layers,
        decoder_heads=cfg.model.num_heads,
        vocab_size=cfg.model.decoder_vocab_size,
        n_event_types=cfg.model.n_event_types,
        max_events=cfg.model.max_events,
        d_align=cfg.model.d_align,
        text_model_name=cfg.model.text_model_name,
        no_hierarchy=cfg.no_hierarchy,
        no_event=cfg.no_event,
        no_align=cfg.no_align,
    )
    return LCTSCapModel(model_cfg)


def build_optimizer(model, cfg):
    """Build AdamW optimizer with separate learning rates."""
    adapter_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder" in name:
            adapter_params.append(param)
        else:
            new_params.append(param)

    param_groups = []
    if new_params:
        param_groups.append({"params": new_params, "lr": cfg.train.lr_new})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": cfg.train.lr_adapter})
    return AdamW(param_groups, weight_decay=cfg.train.weight_decay)


def build_scheduler(optimizer, total_steps, warmup_ratio):
    """Build cosine scheduler with linear warmup."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    decay_steps = total_steps - warmup_steps

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, decay_steps))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def load_datasets(cfg, context_lens):
    """Load datasets for the specified context lengths."""
    data_root = Path(cfg.data.data_root) / "processed" / cfg.data.dataset / "annotations"
    datasets = {}
    precomputed_dir = None
    if cfg.data.use_precomputed_embeddings:
        if cfg.data.precomputed_embeddings_dir:
            precomputed_dir = cfg.data.precomputed_embeddings_dir
        else:
            precomputed_dir = str(
                Path(cfg.data.data_root) / "processed" / cfg.data.dataset / "embeddings"
            )

    for ctx_len in context_lens:
        # Find annotated manifest for this context length
        pattern = f"*_{ctx_len}_*_annotated.jsonl"
        jsonl_files = list(data_root.glob(pattern))
        if not jsonl_files:
            logger.warning("No annotated data for ctx_len=%d, skipping.", ctx_len)
            continue

        # Merge JSONL files into a single manifest JSON for the dataset
        merged = []
        for jf in jsonl_files:
            with open(jf, "r") as f:
                for line in f:
                    if line.strip():
                        merged.append(json.loads(line))

        # Write temporary manifest
        tmp_dir = Path(tempfile.gettempdir()) / "lctscap_manifests"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_manifest = tmp_dir / f"{cfg.data.dataset}_ctx{ctx_len}.json"
        with open(tmp_manifest, "w") as f:
            json.dump(merged, f)

        ds = LCTSCapDataset(
            str(tmp_manifest),
            context_len=ctx_len,
            precomputed_embeddings_dir=precomputed_dir,
        )
        datasets[ctx_len] = ds
        logger.info("Loaded ctx_len=%d: %d samples", ctx_len, len(ds))

    return datasets


def get_loss_weights(cfg):
    """Get loss weights from config."""
    return {
        "caption": cfg.train.loss_weights.get("caption", 1.0),
        "align": cfg.train.loss_weights.get("align", 0.5),
        "event": cfg.train.loss_weights.get("event", 0.3),
        "coverage": cfg.train.loss_weights.get("coverage", 0.1),
    }


def set_trainable_modules(model: nn.Module, module_modes: Dict[str, str]) -> None:
    """Freeze/unfreeze modules according to config."""
    if not module_modes:
        return

    for module_name, mode in module_modes.items():
        if not hasattr(model, module_name):
            logger.warning("Unknown module in config.modules: %s", module_name)
            continue

        module = getattr(model, module_name)
        if module is None:
            continue

        trainable = str(mode).lower() == "train"
        for param in module.parameters():
            param.requires_grad = trainable

        logger.info(
            "Module %s set to %s",
            module_name,
            "trainable" if trainable else "frozen",
        )


def load_model_weights(model: nn.Module, path: str) -> None:
    """Load model weights only, without optimizer/scheduler state."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered_state[key] = value

    model.load_state_dict(filtered_state, strict=False)
    logger.info(
        "Initialized model weights from %s (loaded=%d, skipped=%d)",
        path,
        len(filtered_state),
        len(skipped),
    )


def build_tokenizer(cfg):
    """Build a tokenizer for decoder-based phases."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_name,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as exc:
        logger.warning(
            "Falling back to simple offline tokenizer for '%s': %s",
            cfg.model.tokenizer_name,
            exc,
        )

        class SimpleTokenizer:
            def __init__(self, vocab_size: int):
                self.vocab_size = vocab_size
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def _encode_text(self, text: str, max_length: int) -> list[int]:
                token_ids = [self.eos_token_id]
                for token in text.lower().split():
                    token_ids.append(2 + (abs(hash(token)) % max(2, self.vocab_size - 2)))
                token_ids.append(self.eos_token_id)
                return token_ids[:max_length]

            def __call__(
                self,
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ):
                encoded = [self._encode_text(text or "", max_length) for text in texts]
                max_len = max(len(seq) for seq in encoded) if padding else max(len(seq) for seq in encoded)
                input_ids = []
                attention_mask = []
                for seq in encoded:
                    pad_len = max_len - len(seq)
                    input_ids.append(seq + [self.pad_token_id] * pad_len)
                    attention_mask.append([1] * len(seq) + [0] * pad_len)
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }

        return SimpleTokenizer(cfg.model.decoder_vocab_size)


def compute_early_stop_score(
    val_loss: float,
    val_components: Dict[str, float],
    cfg,
) -> Tuple[float, str]:
    """Compute the metric used for early stopping.

    For composite early stopping we use inverse validation losses as a stable
    proxy before the full evaluation pipeline is wired into training.
    """
    metric_name = getattr(cfg.train, "early_stop_metric", "loss")
    if metric_name != "composite":
        return -val_loss, "val_loss"

    weights = cfg.train.early_stop_weights or {}
    grounding = 1.0 / (1.0 + val_components.get("event", val_loss))
    semantic = 1.0 / (1.0 + val_components.get("align", val_loss))
    lexical = 1.0 / (1.0 + val_components.get("caption", val_loss))
    score = (
        weights.get("grounding", 0.45) * grounding
        + weights.get("semantic", 0.35) * semantic
        + weights.get("lexical", 0.20) * lexical
    )
    return score, "composite"


def train_one_epoch(model, dataloader, optimizer, scheduler, device, cfg, grad_accum, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components_accum = {}
    n_batches = 0
    optimizer.zero_grad()

    loss_weights = get_loss_weights(cfg)

    total_steps = len(dataloader)
    log_interval = max(1, total_steps // 20)  # log ~20 times per epoch

    for step, batch in enumerate(dataloader):
        ts_input = batch["ts_input"].to(device)

        # Raw caption strings for the aligner
        captions = batch.get("caption_short")

        # Tokenized targets for the decoder (only if tokenizer was used)
        decoder_input_ids = batch.get("decoder_input_ids")
        decoder_attention_mask = batch.get("decoder_attention_mask")
        target_ids = batch.get("target_ids")
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(device)
        if target_ids is not None:
            target_ids = target_ids.to(device)

        events_gt = batch.get("events_gt")
        if events_gt is not None:
            events_gt = {k: v.to(device) for k, v in events_gt.items()}

        # Forward pass
        outputs = model(
            x=ts_input,
            captions=captions if not cfg.no_align else None,
            target_ids=decoder_input_ids,
            target_mask=decoder_attention_mask,
            events_gt=events_gt,
        )

        # Compute loss
        cap_logits = outputs.get("caption_logits")
        if cap_logits is not None and target_ids is not None:
            # Full loss (decoder active)
            cov_loss = None
            if not cfg.no_coverage and events_gt is not None:
                event_types_gt = events_gt.get("type_labels")
                if event_types_gt is not None:
                    cov_loss = coverage_loss(cap_logits, event_types_gt)

            loss, components = compute_total_loss(
                cap_logits=cap_logits,
                cap_targets=target_ids,
                z_ts=outputs.get("z_ts"),
                z_text=outputs.get("z_text"),
                logit_scale=outputs.get("logit_scale"),
                event_preds=outputs.get("event_proposals"),
                event_targets=events_gt,
                coverage_score=cov_loss,
                weights=loss_weights,
            )
        else:
            # Encoder-only phases (phase 1): alignment + event loss only
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            components = {}

            if outputs.get("z_ts") is not None and outputs.get("z_text") is not None:
                align_loss = clip_infonce(outputs["z_ts"], outputs["z_text"], outputs["logit_scale"])
                loss = loss + loss_weights["align"] * align_loss
                components["align"] = align_loss.item()

            if outputs.get("event_proposals") is not None and events_gt is not None:
                ev_loss = event_loss(
                    outputs["event_proposals"]["event_type_logits"],
                    outputs["event_proposals"]["span_logits"],
                    events_gt,
                    ts_input.size(1),
                )
                loss = loss + loss_weights["event"] * ev_loss
                components["event"] = ev_loss.item()

            components["total"] = loss.item()

        # Gradient accumulation
        scaled_loss = loss / grad_accum
        scaled_loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += components.get("total", 0.0)
        for k, v in components.items():
            loss_components_accum[k] = loss_components_accum.get(k, 0.0) + v
        n_batches += 1

        if step % log_interval == 0 or step == total_steps - 1:
            logger.info("  [%d/%d] loss=%.4f %s", step, total_steps,
                        components.get("total", 0.0),
                        " ".join(f"{k}={v:.4f}" for k, v in components.items() if k != "total"))

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components_accum.items()}
    return avg_loss, avg_components


@torch.no_grad()
def validate(model, dataloader, device, cfg):
    """Run validation and return average loss + components."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    loss_weights = get_loss_weights(cfg)
    loss_components_accum: Dict[str, float] = {}

    for batch in dataloader:
        ts_input = batch["ts_input"].to(device)
        decoder_input_ids = batch.get("decoder_input_ids")
        decoder_attention_mask = batch.get("decoder_attention_mask")
        target_ids = batch.get("target_ids")
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(device)
        if target_ids is not None:
            target_ids = target_ids.to(device)

        captions = batch.get("caption_short")
        events_gt = batch.get("events_gt")
        if events_gt is not None:
            events_gt = {k: v.to(device) for k, v in events_gt.items()}

        outputs = model(
            x=ts_input,
            captions=captions if not cfg.no_align else None,
            target_ids=decoder_input_ids,
            target_mask=decoder_attention_mask,
            events_gt=events_gt,
        )

        cap_logits = outputs.get("caption_logits")
        if cap_logits is not None and target_ids is not None:
            _, components = compute_total_loss(
                cap_logits=cap_logits,
                cap_targets=target_ids,
                z_ts=outputs.get("z_ts"),
                z_text=outputs.get("z_text"),
                logit_scale=outputs.get("logit_scale"),
                event_preds=outputs.get("event_proposals"),
                event_targets=events_gt,
                coverage_score=None,
                weights=loss_weights,
            )
            total_loss += components.get("total", 0.0)
        else:
            # Encoder-only phases: compute align + event loss
            batch_loss = 0.0
            components = {}
            if outputs.get("z_ts") is not None and outputs.get("z_text") is not None:
                align_loss = clip_infonce(outputs["z_ts"], outputs["z_text"], outputs["logit_scale"])
                batch_loss += loss_weights["align"] * align_loss.item()
                components["align"] = align_loss.item()

            if outputs.get("event_proposals") is not None and events_gt is not None:
                ev_loss = event_loss(
                    outputs["event_proposals"]["event_type_logits"],
                    outputs["event_proposals"]["span_logits"],
                    events_gt,
                    ts_input.size(1),
                )
                batch_loss += loss_weights["event"] * ev_loss.item()
                components["event"] = ev_loss.item()

            components["total"] = batch_loss
            total_loss += batch_loss

        for k, v in components.items():
            loss_components_accum[k] = loss_components_accum.get(k, 0.0) + v
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components_accum.items()}
    return avg_loss, avg_components


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info("Resumed from checkpoint: %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Train LCTSCap model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Determine context lengths for this phase from config
    context_lens = cfg.data.context_lens
    logger.info("Training context lengths: %s", context_lens)
    logger.info("Loss weights: %s", cfg.train.loss_weights)

    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    set_trainable_modules(model, cfg.model.modules)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # Load datasets
    logger.info("Loading datasets...")
    datasets = load_datasets(cfg, context_lens)
    if not datasets:
        logger.error("No datasets loaded. Run preprocess + generate_annotations first.")
        sys.exit(1)

    # Build dataloaders (one per context length)
    # Phase 1 (no decoder) doesn't need a tokenizer
    has_decoder = cfg.train.loss_weights.get("caption", 0.0) > 0
    if has_decoder:
        tokenizer = build_tokenizer(cfg)
        collator = LCTSCapCollator(tokenizer=tokenizer, convert_events_to_per_token=True)
    else:
        collator = LCTSCapCollator(tokenizer=None, convert_events_to_per_token=True)

    dataloaders = {}
    for ctx_len, ds in datasets.items():
        train_ds = ds.get_split("train")
        val_ds = ds.get_split("val") if cfg.data.dataset != "harth" else ds.get_split("test")
        bs = cfg.train.batch_size_for(ctx_len)
        dataloaders[ctx_len] = {
            "train": DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collator,
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=True),
            "val": DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collator,
                              num_workers=max(1, cfg.num_workers // 2), pin_memory=True),
        }
        logger.info("ctx_len=%d: train=%d, val=%d, bs=%d",
                     ctx_len, len(train_ds), len(val_ds), bs)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    total_train_samples = sum(len(dl["train"].dataset) for dl in dataloaders.values())
    avg_bs = max(1, total_train_samples // max(1, sum(1 for dl in dataloaders.values())))
    steps_per_epoch = max(1, total_train_samples // max(1, avg_bs))
    total_steps = steps_per_epoch * cfg.train.max_epochs
    scheduler = build_scheduler(optimizer, total_steps, cfg.train.warmup_ratio)

    # Resume
    start_epoch = 0
    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    if args.resume:
        if latest.exists():
            start_epoch = load_checkpoint(model, optimizer, scheduler, str(latest)) + 1
    elif cfg.model.init_from:
        init_path = Path(cfg.model.init_from)
        if init_path.exists():
            load_model_weights(model, str(init_path))
        else:
            logger.warning("Configured init_from checkpoint does not exist: %s", init_path)

    # Training loop
    best_val_loss = float("inf")
    best_score = -float("inf")
    patience_counter = 0

    logger.info("=== Starting training from epoch %d ===", start_epoch)
    for epoch in range(start_epoch, cfg.train.max_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_batches = 0

        # Iterate over context lengths (round-robin within epoch)
        for ctx_len in sorted(dataloaders.keys()):
            dl = dataloaders[ctx_len]
            avg_loss, components = train_one_epoch(
                model, dl["train"], optimizer, scheduler, device,
                cfg, cfg.train.grad_accum, epoch,
            )
            epoch_loss += avg_loss
            epoch_batches += 1
            logger.info("  Epoch %d ctx=%d: loss=%.4f %s",
                         epoch, ctx_len, avg_loss,
                         " ".join(f"{k}={v:.4f}" for k, v in components.items()))

        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)

        # Validation
        val_losses = []
        val_components_all: Dict[str, float] = {}
        for ctx_len in sorted(dataloaders.keys()):
            dl = dataloaders[ctx_len]
            val_loss, val_components = validate(model, dl["val"], device, cfg)
            val_losses.append(val_loss)
            for k, v in val_components.items():
                val_components_all[k] = val_components_all.get(k, 0.0) + v
            logger.info("  Epoch %d ctx=%d: val_loss=%.4f %s", epoch, ctx_len, val_loss,
                        " ".join(f"{k}={v:.4f}" for k, v in val_components.items()))

        avg_val = sum(val_losses) / max(len(val_losses), 1)
        avg_val_components = {
            k: v / max(len(dataloaders), 1) for k, v in val_components_all.items()
        }
        score, score_name = compute_early_stop_score(avg_val, avg_val_components, cfg)
        elapsed = time.time() - epoch_start
        logger.info("Epoch %d done in %.1fs. train_loss=%.4f val_loss=%.4f %s=%.4f",
                     epoch, elapsed, avg_epoch_loss, avg_val, score_name, score)

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, avg_val,
                        str(ckpt_dir / "latest.pt"))

        improved = False
        if score_name == "composite":
            if score > best_score:
                best_score = score
                best_val_loss = avg_val
                improved = True
        elif avg_val < best_val_loss:
            best_val_loss = avg_val
            best_score = score
            improved = True

        if improved:
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, avg_val,
                            str(ckpt_dir / "best.pt"))
            logger.info("  New best model saved (%s=%.4f, val_loss=%.4f).", score_name, score, avg_val)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch, cfg.train.patience)
                break

    logger.info("=== Training complete. Best val_loss=%.4f best_score=%.4f ===", best_val_loss, best_score)


if __name__ == "__main__":
    main()
