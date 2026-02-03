# Training script for language model
# Features:
# Ability to configure and control the various model and optimizer hyperparameters.
# Memory-efficient loading of training and validation large datasets with flag mmap_mode='r' to np.load.
# Serializing checkpoints to a user-provided path.
# Periodically logging training and validation performance via Weights & Biases (wandb).
import json
import os
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import numpy.typing as npt
import torch
import wandb
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

from myoperator import (
    get_batch,
    gradient_clipping,
    save_checkpoint,
    cross_entropy,
    TransformerLM,
    AdamW,
    LRCosineScheduler,
)

from ablations import (
    TransformerLM_noRMSNorm,
    TransformerLM_postnorm,
    TransformerLM_noPE,
    TransformerLM_siluFFN,
    TransformerLM_weight_tying,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train(config_path: Path) -> None:
    # get training config
    config: dict[str, Any] = _load_json(config_path)
    print("Training config:")
    print(json.dumps(config, indent=4))
    run_name: str = config["run_name"]
    run_name = run_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if config["device"] == "cuda" and torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    elif config["device"] == "mps" and torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    # get wandb config
    wandb_config = config["wandb"]
    wandb_project = wandb_config["project"]
    wandb_tags = wandb_config["tags"]
    wandb_watch_log_freq = wandb_config["watch_log_freq"]
    wandb_watch_log_freq = int(wandb_watch_log_freq)
    # get optimizer config
    if config["optimizer"]["type"] != "adamw":
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']['type']}")
    optim_config = config["optimizer"]
    lr = optim_config["lr"]
    betas = tuple(optim_config["betas"])
    eps = optim_config["eps"]
    weight_decay = optim_config["weight_decay"]
    # get model config
    model_config = config["model"]
    vocab_size = model_config["vocab_size"]
    context_length = model_config["context_length"]
    d_model = model_config["d_model"]
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    d_ff = model_config["dff"]
    rope_theta = model_config["rope_theta"]
    if model_config["dtype"] == "float32":
        model_dtype = torch.float32
    elif model_config["dtype"] == "float16":
        model_dtype = torch.float16
    elif model_config["dtype"] == "bfloat16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32
    ablation = model_config.get("ablation", None)
    # get lr scheduler config
    scheduler_config = config["lr_scheduler"]
    if scheduler_config["type"] != "cosine":
        raise ValueError(f"Unsupported lr scheduler type: {scheduler_config['type']}")
    lr_max = scheduler_config["lr_max"]
    lr_min = scheduler_config["lr_min"]
    T_warmup = scheduler_config["T_warmup"]
    T_c = scheduler_config["T_c"]
    # get training config
    batch_size = config["training"]["batch_size"]
    max_iters = config["training"]["max_iters"]
    grad_clip_norm = config["training"]["grad_clip_norm"]
    run_valid_interval = config["training"]["run_valid_interval"]
    save_checkpoint_interval = config["training"]["save_checkpoint_interval"]
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir = checkpoint_dir / run_name
    train_dataset_path = Path(config["training"]["train_dataset_path"])
    valid_dataset_path = Path(config["training"]["valid_dataset_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # initialize wandb
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=config,
        tags=wandb_tags,
        mode="online",
    )

    try:
        # load datasets
        train_data: npt.NDArray = np.load(train_dataset_path, mmap_mode="r")
        valid_data: npt.NDArray = np.load(valid_dataset_path, mmap_mode="r")

        # initialize model and optimizer
        if ablation == "no_rmsnorm":
            print("Using TransformerLM_noRMSNorm ablation model")
            model = TransformerLM_noRMSNorm(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                device=device,
                dtype=model_dtype,
            )
        elif ablation == "postnorm":
            print("Using TransformerLM_postnorm ablation model")
            model = TransformerLM_postnorm(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                device=device,
                dtype=model_dtype,
            )
        elif ablation == "nope":
            print("Using TransformerLM_noPE ablation model")
            model = TransformerLM_noPE(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                device=device,
                dtype=model_dtype,
            )
        elif ablation == "silu_ffn":
            print("Using TransformerLM_siluFFN ablation model")
            model = TransformerLM_siluFFN(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                rope_theta=rope_theta,
                device=device,
                dtype=model_dtype,
            )
        elif ablation == "weight_tying":
            print("Using TransformerLM_weight_tying ablation model")
            model = TransformerLM_weight_tying(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                device=device,
                dtype=model_dtype,
            )
        else:
            print("Using standard TransformerLM model")
            model = TransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                device=device,
                dtype=model_dtype,
            )
        wandb.watch(
            model,
            log="all",
            log_freq=wandb_watch_log_freq,
        )
        optimizer = AdamW(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        scheduler = LRCosineScheduler(
            optimizer, lr_max=lr_max, lr_min=lr_min, T_warmup=T_warmup, T_c=T_c
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            training_task = progress.add_task("Training...", total=max_iters)
            for iter in range(1, max_iters + 1):
                model.train()
                xb, yb = get_batch(train_data, batch_size, context_length, device)
                logits = model(xb)
                loss = cross_entropy(logits, yb).mean()
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    gradient_clipping(model.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()
                # update progress bar
                progress.update(training_task, advance=1)
                # log training loss
                wandb.log({"train/loss": loss.item()}, step=iter)
                progress.console.log(f"Iter {iter}: train loss = {loss.item():.4f}")
                # run validation
                if iter % run_valid_interval == 0 or iter == max_iters:
                    model.eval()
                    with torch.no_grad():
                        xb_val, yb_val = get_batch(
                            valid_data, batch_size, context_length, device
                        )
                        logits_val = model(xb_val)
                        val_loss = cross_entropy(logits_val, yb_val).mean()
                        wandb.log({"valid/loss": val_loss.item()}, step=iter)
                        progress.console.log(f"Iter {iter}: valid loss = {val_loss.item():.4f}")
                        
                        # compute final validation loss with more samples at the end of training
                        if iter == max_iters:
                            final_losses = []
                            for _ in range(100):
                                xb_val, yb_val = get_batch(
                                    valid_data, batch_size, context_length, device
                                )
                                logits_val = model(xb_val)
                                val_loss = cross_entropy(logits_val, yb_val).mean()
                                final_losses.append(val_loss.item())
                            final_valid_loss = np.mean(final_losses)
                            wandb.log({"valid/final_loss": final_valid_loss}, step=iter)
                            progress.console.log(f"Final validation loss (100 batches): {final_valid_loss:.4f}")

                # save checkpoint
                if iter % (save_checkpoint_interval) == 0 or iter == max_iters:
                    save_checkpoint(
                        model,
                        optimizer,
                        iter,
                        checkpoint_dir / f"checkpoint_iter{iter}.pt",
                        model_config=model_config,
                    )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    train(Path("local_training_config.json"))
