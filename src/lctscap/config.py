"""Configuration system for LCTSCap using Python dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset: str = "capture24"
    data_root: str = "/path/to/lctscap_data/"
    sample_rate: int = 50
    window_sec: int = 10
    context_lens: List[int] = field(default_factory=lambda: [128, 256, 512])
    strides: List[int] = field(default_factory=lambda: [32, 64])
    use_precomputed_embeddings: bool = False
    precomputed_embeddings_dir: Optional[str] = None
    caption_source: str = "template"
    channels_capture24: int = 3
    channels_harth: int = 6

    @property
    def window_samples(self) -> int:
        """Number of samples per window."""
        return self.sample_rate * self.window_sec

    def channels_for(self, dataset: str) -> int:
        """Return the number of channels for the given dataset."""
        if dataset == "capture24":
            return self.channels_capture24
        elif dataset == "harth":
            return self.channels_harth
        else:
            raise ValueError(f"Unknown dataset: {dataset}")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    d_model: int = 512
    d_align: int = 256
    num_layers_local: int = 4
    num_layers_planner: int = 4
    num_heads: int = 8
    patch_size: int = 25
    segment_size: int = 32
    n_event_types: int = 20
    max_events: int = 16
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    decoder_vocab_size: int = 50257
    decoder_layers: int = 6
    tokenizer_name: str = "gpt2"
    init_from: Optional[str] = None
    modules: Dict[str, str] = field(default_factory=dict)

    @property
    def d_ff(self) -> int:
        """Feed-forward dimension (4x model dimension)."""
        return self.d_model * 4


@dataclass
class TrainConfig:
    """Configuration for training."""

    lr_new: float = 3e-4
    lr_adapter: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    scheduler: str = "cosine"
    batch_sizes: Dict[int, int] = field(
        default_factory=lambda: {128: 32, 256: 16, 512: 8}
    )
    batch_size: Optional[int] = None  # single override for all context_lens
    grad_accum: int = 2
    max_epochs: int = 30
    patience: int = 5
    early_stop_metric: str = "loss"
    early_stop_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "caption_loss": 0.4,
            "event_f1": 0.3,
            "retrieval_r@5": 0.3,
        }
    )
    checkpoint_dir: Optional[str] = None
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "caption": 1.0,
            "align": 0.5,
            "event": 0.3,
            "coverage": 0.1,
        }
    )

    def batch_size_for(self, context_len: int) -> int:
        """Return the batch size for a given context length."""
        if self.batch_size is not None:
            return self.batch_size
        if context_len in self.batch_sizes:
            return self.batch_sizes[context_len]
        # Fall back to the smallest batch size for unknown context lengths
        return min(self.batch_sizes.values())


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics."""

    ks: List[int] = field(default_factory=lambda: [1, 5, 10])
    duration_bins: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "short": [0.0, 60.0],
            "medium": [60.0, 300.0],
            "long": [300.0, float("inf")],
        }
    )


@dataclass
class LCTSCapConfig:
    """Top-level configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Ablation flags
    no_align: bool = False
    no_hierarchy: bool = False
    no_event: bool = False
    no_coverage: bool = False

    experiment_name: str = "default"
    phase: Optional[int] = None
    seed: int = 42
    num_workers: int = 4
    output_dir: str = "./outputs"

    @property
    def checkpoint_dir(self) -> Path:
        if self.train.checkpoint_dir:
            return Path(self.train.checkpoint_dir)
        return Path(self.output_dir) / self.experiment_name / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name / "logs"

    def active_ablations(self) -> List[str]:
        """Return a list of active ablation flag names."""
        flags = []
        if self.no_align:
            flags.append("no_align")
        if self.no_hierarchy:
            flags.append("no_hierarchy")
        if self.no_event:
            flags.append("no_event")
        if self.no_coverage:
            flags.append("no_coverage")
        return flags


def _merge_dict_into_dataclass(dc_class: type, raw: Dict[str, Any]) -> Any:
    """Recursively create a dataclass instance from a dictionary,
    using defaults for any missing fields."""
    import dataclasses

    field_defaults = {f.name: f for f in dataclasses.fields(dc_class)}
    kwargs = {}
    for fname, fobj in field_defaults.items():
        if fname in raw:
            val = raw[fname]
            # If the field type is itself a dataclass, recurse
            if dataclasses.is_dataclass(fobj.type if isinstance(fobj.type, type) else None):
                val = _merge_dict_into_dataclass(fobj.type, val)
            kwargs[fname] = val
        # Otherwise the dataclass default will be used
    return dc_class(**kwargs)


def load_config(path: str) -> LCTSCapConfig:
    """Load configuration from a YAML file.

    Missing fields are filled with their default values from the dataclass
    definitions. If the file does not exist, returns the default config.

    Supports two YAML layouts:
    1. Nested: top-level keys ``data``, ``model``, ``train``, ``eval``
    2. Flat: all keys at root level (e.g. per-dataset configs like capture24.yaml)

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A fully populated LCTSCapConfig instance.
    """
    config_path = Path(path)
    if not config_path.exists():
        return LCTSCapConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Build sub-configs — support both nested and flat layouts
    data_raw = raw.pop("data", {})
    model_raw = raw.pop("model", {})
    train_raw = raw.pop("train", {}) or raw.pop("training", {})
    eval_raw = raw.pop("eval", {}) or raw.pop("evaluation", {})

    if isinstance(model_raw, dict) and "_config" in model_raw:
        model_cfg_path = Path(model_raw.pop("_config"))
        if not model_cfg_path.is_absolute():
            model_cfg_path = config_path.parent / model_cfg_path
        if model_cfg_path.exists():
            with open(model_cfg_path, "r") as f:
                base_model_yaml = yaml.safe_load(f) or {}
            base_model_raw = base_model_yaml.get("model", {})

            local_encoder_raw = base_model_raw.get("local_encoder", {})
            planner_raw = base_model_raw.get("planner", {})
            event_head_raw = base_model_raw.get("event_head", {})
            aligner_raw = base_model_raw.get("aligner", {})
            decoder_raw = base_model_raw.get("decoder", {})

            flattened_base = {}
            if isinstance(local_encoder_raw, dict):
                for src_key, dst_key in {
                    "d_model": "d_model",
                    "patch_size": "patch_size",
                    "num_layers": "num_layers_local",
                    "num_heads": "num_heads",
                }.items():
                    if src_key in local_encoder_raw:
                        flattened_base[dst_key] = local_encoder_raw[src_key]
            if isinstance(planner_raw, dict):
                if "num_layers" in planner_raw:
                    flattened_base["num_layers_planner"] = planner_raw["num_layers"]
                if "segment_size" in planner_raw:
                    flattened_base["segment_size"] = planner_raw["segment_size"]
            if isinstance(event_head_raw, dict):
                if "n_event_types" in event_head_raw:
                    flattened_base["n_event_types"] = event_head_raw["n_event_types"]
                if "max_events" in event_head_raw:
                    flattened_base["max_events"] = event_head_raw["max_events"]
            if isinstance(aligner_raw, dict):
                if "d_align" in aligner_raw:
                    flattened_base["d_align"] = aligner_raw["d_align"]
                if "text_model_name" in aligner_raw:
                    flattened_base["text_model_name"] = aligner_raw["text_model_name"]
            if isinstance(decoder_raw, dict):
                if "vocab_size" in decoder_raw:
                    flattened_base["decoder_vocab_size"] = decoder_raw["vocab_size"]
                if "num_layers" in decoder_raw:
                    flattened_base["decoder_layers"] = decoder_raw["num_layers"]
                if "tokenizer_name" in decoder_raw:
                    flattened_base["tokenizer_name"] = decoder_raw["tokenizer_name"]

            flattened_base.update(model_raw)
            model_raw = flattened_base

    # For flat YAML files (like capture24.yaml), map known keys into data_raw
    _flat_data_keys = {
        "dataset", "data_root", "sample_rate", "window_sec",
        "context_lens", "strides", "channels_capture24", "channels_harth",
        "use_precomputed_embeddings", "precomputed_embeddings_dir", "caption_source",
        "original_sample_rate", "target_sample_rate", "num_channels",
        "train_ratio", "val_ratio", "test_ratio", "split_seed",
        "segment_size", "label_map", "activities",
    }
    for key in list(raw.keys()):
        if key in _flat_data_keys:
            data_raw.setdefault(key, raw.pop(key))

    # Map target_sample_rate -> sample_rate if present
    if "target_sample_rate" in data_raw and "sample_rate" not in data_raw:
        data_raw["sample_rate"] = data_raw["target_sample_rate"]

    # Handle single context_len -> context_lens list
    if "context_len" in data_raw and "context_lens" not in data_raw:
        data_raw["context_lens"] = [data_raw.pop("context_len")]
    elif "context_len" in data_raw:
        data_raw.pop("context_len")

    # Handle single stride -> strides list
    if "stride" in data_raw and "strides" not in data_raw:
        data_raw["strides"] = [data_raw.pop("stride")]
    elif "stride" in data_raw:
        data_raw.pop("stride")

    # Flatten nested training config (optimizer.lr_new, scheduler.warmup_ratio, etc.)
    if isinstance(train_raw, dict):
        opt = train_raw.pop("optimizer", {})
        if isinstance(opt, dict):
            for k in ("lr_new", "lr_adapter", "weight_decay"):
                if k in opt and k not in train_raw:
                    train_raw[k] = opt[k]
        sched = train_raw.pop("scheduler", {})
        if isinstance(sched, dict):
            for k in ("warmup_ratio",):
                if k in sched and k not in train_raw:
                    train_raw[k] = sched[k]
            if "type" in sched and "scheduler" not in train_raw:
                train_raw["scheduler"] = sched["type"]

        # Rename grad_accumulation -> grad_accum
        if "grad_accumulation" in train_raw and "grad_accum" not in train_raw:
            train_raw["grad_accum"] = train_raw.pop("grad_accumulation")

        if "batch_size_per_ctx" in train_raw and "batch_sizes" not in train_raw:
            train_raw["batch_sizes"] = train_raw.pop("batch_size_per_ctx")

        # Map loss_weights keys: cap -> caption
        lw = train_raw.get("loss_weights")
        if isinstance(lw, dict) and "cap" in lw:
            lw["caption"] = lw.pop("cap")

    if isinstance(model_raw, dict):
        decoder_raw = model_raw.get("decoder")
        if isinstance(decoder_raw, dict):
            if "vocab_size" in decoder_raw and "decoder_vocab_size" not in model_raw:
                model_raw["decoder_vocab_size"] = decoder_raw["vocab_size"]
            if "num_layers" in decoder_raw and "decoder_layers" not in model_raw:
                model_raw["decoder_layers"] = decoder_raw["num_layers"]
            if "tokenizer_name" in decoder_raw and "tokenizer_name" not in model_raw:
                model_raw["tokenizer_name"] = decoder_raw["tokenizer_name"]

        aligner_raw = model_raw.get("aligner")
        if isinstance(aligner_raw, dict):
            if "d_align" in aligner_raw and "d_align" not in model_raw:
                model_raw["d_align"] = aligner_raw["d_align"]
            if "text_model_name" in aligner_raw and "text_model_name" not in model_raw:
                model_raw["text_model_name"] = aligner_raw["text_model_name"]

    data_cfg = _merge_dict_into_dataclass(DataConfig, data_raw) if data_raw else DataConfig()
    model_cfg = _merge_dict_into_dataclass(ModelConfig, model_raw) if model_raw else ModelConfig()
    train_cfg = _merge_dict_into_dataclass(TrainConfig, train_raw) if train_raw else TrainConfig()
    eval_cfg = _merge_dict_into_dataclass(EvalConfig, eval_raw) if eval_raw else EvalConfig()

    def _resolve_path(value: Optional[str]) -> Optional[str]:
        if not value or not isinstance(value, str):
            return value
        return value.replace("${data_root}", data_cfg.data_root)

    model_cfg.init_from = _resolve_path(model_cfg.init_from)
    train_cfg.checkpoint_dir = _resolve_path(train_cfg.checkpoint_dir)
    data_cfg.precomputed_embeddings_dir = _resolve_path(data_cfg.precomputed_embeddings_dir)

    # Build top-level config with remaining keys (ablation flags, seed, etc.)
    top_kwargs = {}
    for key in ("no_align", "no_hierarchy", "no_event", "no_coverage",
                "experiment_name", "phase", "seed", "num_workers", "output_dir"):
        if key in raw:
            top_kwargs[key] = raw[key]

    # Derive output_dir from checkpoint_dir if specified in training config
    ckpt_dir = train_cfg.checkpoint_dir or ""
    if isinstance(ckpt_dir, str) and ckpt_dir:
        # Set output_dir to parent of checkpoints dir
        ckpt_path = Path(ckpt_dir)
        if ckpt_path.name == "checkpoints":
            top_kwargs.setdefault("output_dir", str(ckpt_path.parent.parent))
            top_kwargs.setdefault("experiment_name", ckpt_path.parent.name)

    # Use phase number as experiment_name if available
    phase = raw.get("phase")
    if phase is not None and "experiment_name" not in top_kwargs:
        top_kwargs["experiment_name"] = f"phase{phase}"

    return LCTSCapConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        eval=eval_cfg,
        **top_kwargs,
    )
