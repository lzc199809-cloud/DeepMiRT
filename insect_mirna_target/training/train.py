#!/usr/bin/env python3
"""
miRNA-target prediction model -- main training entry script

[Usage]
  # Smoke test (run only 1 batch to verify the pipeline):
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --fast-dev-run

  # Full training:
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml

  # Override configuration parameters:
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --override training.lr=5e-5 --override data.batch_size=64

  # Resume training from a checkpoint (restores epoch, optimizer state, etc.):
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --resume checkpoints/last.ckpt

  # Load pretrained weights but restart training (Phase 2 fine-tuning):
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --ckpt-path checkpoints/epoch-4-0.8923.ckpt \\
    --override unfreezing.enabled=true \\
    --override training.lr=5e-5

[Training Pipeline Orchestration]
  1. Load the YAML configuration file
  2. Set the random seed (for reproducibility)
  3. Create the DataModule (handles data loading)
  4. Create the LightningModule (handles model + optimizer + metrics)
  5. Configure Callbacks (checkpoint, early stopping, staged unfreezing)
  6. Create the Trainer (orchestrates the training loop)
  7. trainer.fit() starts training

[Key Design Decisions]

  1. Why use YAML configuration instead of command-line arguments?
     - There are many hyperparameters (20+); command-line usage would be verbose
     - YAML is readable, maintainable, and convenient for version control and experiment comparison
     - --override provides flexibility for quick experiments

  2. Why TensorBoard instead of wandb?
     - TensorBoard: local storage, no login required, privacy-friendly
     - wandb: cloud sync, requires network and account, but has more powerful comparison tools
     - TensorBoard was chosen for simpler deployment and privacy protection

  3. Why DDP instead of DP (DataParallel)?
     - DP: single-process multi-GPU, the primary GPU becomes a bottleneck, high communication overhead
     - DDP: multi-process multi-GPU, each GPU runs an independent process, efficient communication
     - DDP is the modern standard for distributed training

  4. Why mixed precision (16-mixed)?
     - float16: halves memory usage, 2-3x faster computation
     - float32: numerically stable, precise optimizer states
     - 16-mixed: forward/backward in float16, optimizer in float32 -- optimal balance

  5. Checkpoint naming convention {epoch}-{val_auroc:.4f}
     - Makes it easy to quickly identify the best model (epoch with highest AUROC)
     - When disk space is limited, save_top_k=3 keeps only the top-3 best models

  6. Purpose of --fast-dev-run
     - Lightning runs exactly 1 training batch + 1 validation batch
     - Use case: quickly verify that data loading, model forward pass, and loss computation work correctly
     - Does not actually train, but catches 90% of code bugs
"""

import argparse
import os
import sys
from pathlib import Path

# Design decision: add the project root directory to the Python path
# This ensures the insect_mirna_target package can be correctly imported regardless of where the script is run from
# e.g.: python insect_mirna_target/training/train.py --config ...
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# DDP/NCCL configuration
# These environment variables only affect the current process, not other users
os.environ.setdefault("NCCL_P2P_DISABLE", "1")                    # Disable P2P direct transfer (two L20s across PCIe bus; P2P would busy-wait and deadlock)
os.environ.setdefault("NCCL_TIMEOUT", "1800")                     # NCCL operation timeout: 30 minutes
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800") # Watchdog heartbeat timeout: 30 minutes

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from insect_mirna_target.data_module.datamodule import MiRNATargetDataModule
from insect_mirna_target.training.callbacks import StagedUnfreezeCallback
from insect_mirna_target.training.lightning_module import MiRNATargetLitModule


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: Configuration file does not exist
        yaml.YAMLError: YAML format error
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    return config


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Apply command-line override parameters.

    Format: key.subkey=value
    Examples: training.lr=5e-5, data.batch_size=64

    Args:
        config: Original configuration dictionary
        overrides: List of override parameters in the format "key.subkey=value"

    Returns:
        Modified configuration dictionary

    Raises:
        ValueError: Override parameter format error or key path does not exist
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format (expected key=value): {override}")
        
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        
        # Navigate to the parent dictionary
        d = config
        for k in keys[:-1]:
            if k not in d:
                raise ValueError(f"Configuration key does not exist: {k} (in path {key_path})")
            d = d[k]
        
        # Set the final value with type inference
        final_key = keys[-1]
        if final_key not in d:
            raise ValueError(f"Configuration key does not exist: {final_key} (in path {key_path})")
        
        # Type inference: try bool -> int -> float -> str
        try:
            # Try boolean (must be before float, since "true" contains "e" which could be misinterpreted as float)
            if value_str.lower() in ("true", "false"):
                d[final_key] = value_str.lower() == "true"
            # Try integer
            elif value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
                d[final_key] = int(value_str)
            # Try float
            elif "." in value_str or "e" in value_str.lower():
                d[final_key] = float(value_str)
            # Default to string
            else:
                d[final_key] = value_str
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Cannot parse override parameter {override}: {e}")
    
    return config


def build_callbacks(config: dict) -> list:
    """
    Build all callbacks based on the configuration.

    Callbacks are "hooks" that Lightning calls at specific points in the training loop.
    This function creates the following callbacks based on the config:
    - ModelCheckpoint: save the best models
    - EarlyStopping: early stopping to prevent overfitting
    - LearningRateMonitor: monitor learning rate changes
    - StagedUnfreezeCallback: staged unfreezing of the backbone network

    Args:
        config: Complete configuration dictionary

    Returns:
        List of callbacks
    """
    callbacks = []
    
    # 1. ModelCheckpoint -- save the best models
    # Design decision: rank by val_auroc, keep top-3 best models
    # Naming convention {epoch}-{val_auroc:.4f} for quick identification of the best epoch
    ckpt_cfg = config["checkpointing"]
    callbacks.append(
        ModelCheckpoint(
            monitor=ckpt_cfg["monitor"],
            mode=ckpt_cfg["mode"],
            save_top_k=ckpt_cfg["save_top_k"],
            save_last=ckpt_cfg["save_last"],
            dirpath=ckpt_cfg["dirpath"],
            filename="{epoch}-{val_auroc:.4f}",
        )
    )
    
    # 2. EarlyStopping -- early stopping (if enabled)
    # Design decision: monitor val_loss; stop if no improvement for 5 consecutive epochs
    # Prevents overfitting and saves compute resources
    if config.get("early_stopping", {}).get("enabled", False):
        es_cfg = config["early_stopping"]
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg["monitor"],
                patience=es_cfg["patience"],
                mode=es_cfg["mode"],
            )
        )
    
    # 3. LearningRateMonitor -- monitor learning rate
    # Design decision: log once per step to observe the effect of the LR schedule
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # 4. StagedUnfreezeCallback -- staged unfreezing (if enabled)
    # Design decision: Phase 1 freezes the backbone so new layers converge quickly
    #                  Phase 2 unfreezes the top N layers to fine-tune the backbone
    if config.get("unfreezing", {}).get("enabled", False):
        unf_cfg = config["unfreezing"]
        callbacks.append(
            StagedUnfreezeCallback(
                unfreeze_at_epoch=unf_cfg["unfreeze_at_epoch"],
                num_layers_to_unfreeze=unf_cfg["num_layers"],
                unfreeze_interval=unf_cfg.get("unfreeze_interval", 3),
                warmup_epochs=unf_cfg.get("warmup_epochs", 1),
            )
        )
    
    return callbacks


def main():
    """Main training entry function."""
    parser = argparse.ArgumentParser(
        description="miRNA-target prediction model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (verify pipeline, run only 1 batch)
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --fast-dev-run

  # Full training
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml

  # Override parameters
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --override training.lr=5e-5 --override data.batch_size=64

  # Resume interrupted training from a checkpoint
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --resume checkpoints/last.ckpt

  # Phase 2: load Phase 1 weights and restart fine-tuning
  python insect_mirna_target/training/train.py \\
    --config insect_mirna_target/configs/default.yaml \\
    --ckpt-path checkpoints/epoch-4-0.8923.ckpt \\
    --override unfreezing.enabled=true \\
    --override training.lr=5e-5
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Quick validation mode (runs only 1 training batch + 1 validation batch, for debugging)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint (restores epoch, optimizer state, LR scheduler, etc.). "
        "Used to continue after an interruption. Example: --resume checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Load model weights from a checkpoint but restart training (epoch starts at 0, "
        "optimizer is re-initialized). Used for Phase 2 fine-tuning with Phase 1 weights. "
        "Example: --ckpt-path checkpoints/epoch-4-0.8923.ckpt",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override configuration parameters, format: key.subkey=value (can be used multiple times)",
    )
    
    args = parser.parse_args()
    
    # ── Mutual exclusion check ──
    # --resume and --ckpt-path cannot be used together:
    # --resume: restores the full training state (epoch, optimizer, scheduler, random state)
    # --ckpt-path: loads only model weights; everything else starts from scratch
    if args.resume and args.ckpt_path:
        parser.error("--resume and --ckpt-path cannot be used together.\n"
                      "  --resume: resume interrupted training (full state)\n"
                      "  --ckpt-path: load model weights only (restart training)")
    
    # ── Step 1: Load configuration ──
    print(f"[INFO] Loading configuration file: {args.config}")
    config = load_config(args.config)
    
    # ── Step 2: Apply command-line overrides ──
    if args.override:
        print(f"[INFO] Applying {len(args.override)} override parameter(s)")
        config = apply_overrides(config, args.override)
    
    # ── Step 3: Set random seed ──
    # Design decision: global seed ensures experiment reproducibility
    # Covers Python random, NumPy, PyTorch, and CUDA random number generators
    seed = config.get("seed", 42)
    print(f"[INFO] Setting random seed: {seed}")
    pl.seed_everything(seed)

    # Optimize Tensor Core utilization (L20 supports TF32) and suppress Lightning warning
    torch.set_float32_matmul_precision("medium")

    # ── Step 4: Create DataModule ──
    print("[INFO] Creating DataModule")
    data_cfg = config["data"]
    dm = MiRNATargetDataModule(
        data_dir=data_cfg["data_dir"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
    )
    
    # ── Step 5: Create LightningModule ──
    # --ckpt-path mode: load model weights from a checkpoint but use the current config's hyperparameters
    # This allows modifying learning rate, unfreezing strategy, etc. in Phase 2 while retaining Phase 1 weights
    if args.ckpt_path:
        ckpt_file = Path(args.ckpt_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")
        print(f"[INFO] Loading model weights from checkpoint: {args.ckpt_path}")
        print("[INFO] Optimizer and LR scheduler will be re-initialized (loading model weights only)")

        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        
        # First create the module with current config (ensuring hyperparameters use the new config)
        lit_model = MiRNATargetLitModule(config)
        
        # Extract and load model weights from the checkpoint
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing weight keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected weight keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print("[INFO] Model weights loaded successfully")
    else:
        print("[INFO] Creating LightningModule")
        lit_model = MiRNATargetLitModule(config)
    
    # ── Step 6: Configure Callbacks ──
    print("[INFO] Configuring Callbacks")
    callbacks = build_callbacks(config)
    
    # ── Step 7: Create Trainer ──
    print("[INFO] Creating Trainer")
    trainer_cfg = config["trainer"]
    log_cfg = config.get("logging", {})
    train_cfg = config["training"]
    
    # Configure logging backend
    # Design decision: TensorBoard uses local storage, no login required, privacy-friendly
    logger = None
    if log_cfg.get("logger") == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=log_cfg.get("log_dir", "lightning_logs/"),
            name="mirna_target",
        )
    else:
        logger = True  # Use Lightning's default logger
    
    trainer = pl.Trainer(
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", 2),
        strategy=trainer_cfg.get("strategy", "ddp"),
        max_epochs=train_cfg.get("max_epochs", 30),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        precision=train_cfg.get("precision", "16-mixed"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 50),
        fast_dev_run=args.fast_dev_run,
    )
    
    # ── Step 8: Start training ──
    # --resume mode: pass ckpt_path to trainer.fit(); Lightning automatically restores:
    #   - Model weights
    #   - Optimizer state (momentum, adaptive learning rate, etc.)
    #   - Learning rate scheduler state
    #   - Current epoch and global_step
    #   - Callback states (e.g., EarlyStopping counter)
    #   - Random number generator states (to ensure consistent data order)
    resume_path = None
    if args.resume:
        resume_file = Path(args.resume)
        if not resume_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
        resume_path = args.resume
        print(f"[INFO] Resuming training from checkpoint: {resume_path}")
    
    print("[INFO] Starting training...")
    trainer.fit(lit_model, dm, ckpt_path=resume_path)
    
    print("[INFO] Training complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr)
        sys.exit(1)
