from __future__ import annotations
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .checkpoints import load_model, resolve_checkpoint, save_checkpoint
from .config import model_config, read_config
from .data import BatchStream, CachedDataset, Collator, move_to_device
from .evaluation import evaluate_model, validate_cache_tokenizer
from .losses import count_supervision, security_loss
from .model import SecTrace
from .runtime import (autocast_context, choose_precision, restore_rng, rng_state, runtime_info,
                      seed_all, select_device, synchronize)
from .util import atomic_json


def learning_rate_factor(step, seen_tokens, settings):
    warmup = max(1, int(settings.get("warmup_steps", 200)))
    if step <= warmup:
        return step / warmup
    if settings.get("target_tokens", 0):
        fraction = seen_tokens / settings["target_tokens"]
    else:
        fraction = (step - warmup) / max(1, settings.get("max_steps", 1000) - warmup)
    fraction = min(1.0, max(0.0, fraction))
    floor = float(settings.get("min_lr_fraction", .1))
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * fraction))


def make_optimizer(model, settings, stage):
    lr = float(settings.get("learning_rate", 3e-4 if stage == "pretrain" else 2e-5))
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = p.ndim >= 2 and not any(part in name for part in ("token.", "kind.", "role.", "language.", "relation.", "bias."))
        scale = float(settings.get("head_lr_multiplier", 1.0)) if name.startswith("heads.") else 1.0
        groups.setdefault((decay, scale), []).append(p)
    param_groups = [{"params": params, "weight_decay": float(settings.get("weight_decay", .1)) if decay else 0.0,
                     "lr": lr * scale, "base_lr": lr * scale} for (decay, scale), params in groups.items()]
    return torch.optim.AdamW(param_groups, betas=(.9, .95), eps=1e-8, foreach=False, fused=False)


def train(config_path, train_data, validation_data, output_dir, init=None, resume=None,
          device="auto", precision=None, stop_after=None):
    if init and resume:
        raise ValueError("Use --init for a new stage OR --resume for the same run, not both")
    config = read_config(config_path)
    settings = config["training"]
    stage = settings.get("stage", "pretrain")
    if stage not in ("pretrain", "sft", "align"):
        raise ValueError("stage must be pretrain, sft, or align")
    cfg = model_config(config)
    train_set, validation_set = CachedDataset(train_data), CachedDataset(validation_data)
    for ds, purpose in ((train_set, "train"), (validation_set, "validation")):
        if ds.manifest.get("purpose") != purpose:
            raise ValueError(f"Expected {purpose} data, found {ds.manifest.get('purpose')}. Reserve test/calibration data.")
        if ds.manifest["stage"] != ("pretrain" if stage == "pretrain" else "security"):
            raise ValueError("Cache stage mismatch; use prepare --stage pretrain or security")
        if ds.manifest["max_length"] > cfg.max_length:
            raise ValueError("Cache exceeds model context capacity")
    if train_set.manifest["input_sha256"] == validation_set.manifest["input_sha256"]:
        raise ValueError("Training and validation point to the same source data")
    if train_set.manifest["tokenizer_sha256"] != validation_set.manifest["tokenizer_sha256"]:
        raise ValueError("Train/validation tokenizer mismatch")
    if train_set.manifest["vocab_size"] > cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model embedding capacity")
    run = Path(output_dir)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "latest.json").exists() and not resume:
        raise FileExistsError("Run already has checkpoints; use --resume or a new output directory")
    dev = select_device(device, float(settings.get("mps_memory_fraction", .85)))
    prec, precision_message = choose_precision(dev, precision or settings.get("precision", "auto"))
    seed = int(settings.get("seed", 17))
    seed_all(seed, dev)
    previous, checkpoint_state = {}, None
    if init or resume:
        model, previous, directory = load_model(resume or init, dev, "latest" if resume else "best")
        # Context chunk/checkpoint settings may change for a new stage, not tensor dimensions.
        structural = ("vocab_size", "dim", "layers", "heads", "ffn_dim", "graph_dim", "graph_heads",
                      "relations", "task_dim", "edge_dim", "families", "cwes", "evidence_roles", "global_every")
        if any(getattr(cfg, k) != getattr(model.cfg, k) for k in structural):
            raise ValueError("Requested architecture is incompatible with checkpoint tensors")
        validate_cache_tokenizer(model, previous, train_set)
        if resume:
            if previous["stage"] != stage or previous["train_data_sha256"] != train_set.manifest["data_sha256"]:
                raise ValueError("Resume requires the same stage and training cache; use --init for curriculum/stage changes")
            if previous.get("validation_data_sha256") != validation_set.manifest["data_sha256"]:
                raise ValueError("Resume validation cache changed")
            if previous["model_config"] != cfg.to_dict() or previous["training_config"] != settings:
                raise ValueError("Exact resume requires unchanged configuration. Use --init for a new experiment.")
            checkpoint_state = torch.load(directory / "trainer.pt", map_location="cpu", weights_only=True)
        else:
            # Apply nonstructural runtime options consistently to all modules.
            model.cfg = cfg
            for module in model.modules():
                if hasattr(module, "cfg"):
                    module.cfg = cfg
    else:
        model = SecTrace(cfg).to(dev)
        if stage != "pretrain":
            print("WARNING: starting supervised training from random initialization, not pre-trained weights")
    model.set_stage(stage)
    optimizer = make_optimizer(model, settings, stage)
    microbatch = int(settings.get("microbatch_size", 1))
    accumulation = int(settings.get("gradient_accumulation", 16))
    if microbatch < 1 or accumulation < 1:
        raise ValueError("Batch and accumulation sizes must be positive")
    stream = BatchStream(train_set, microbatch, seed, checkpoint_state.get("stream") if checkpoint_state else None)
    task = "mlm" if stage == "pretrain" else "security"
    collator = Collator(task, train_set.manifest["random_vocab_size"], seed=seed + 1,
                        mask_probability=float(settings.get("mask_probability", .15)),
                        graph_dropout=float(settings.get("graph_dropout", .1)))
    step, seen_tokens, best_score, bad_evals = 0, 0, None, 0
    supervision = previous.get("supervision_presentations", {}) if stage != "pretrain" else {}
    if checkpoint_state:
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        step, seen_tokens = checkpoint_state["step"], checkpoint_state["seen_tokens"]
        best_score, bad_evals = checkpoint_state["best_score"], checkpoint_state["bad_evals"]
        collator.generator.set_state(checkpoint_state["collator_rng"])
        restore_rng(checkpoint_state["rng"], dev)
    info = {**runtime_info(dev), "precision": prec, "precision_probe": precision_message,
            "parameter_counts": model.parameter_counts(), "stage": stage,
            "effective_record_batch": microbatch * accumulation,
            "note": "Verified paired records expand the number of encoded sequences."}
    atomic_json(run / "environment.json", info)
    atomic_json(run / "resolved_config.json", config)
    print(json.dumps(info, indent=2), flush=True)
    log = (run / "metrics.jsonl").open("a", encoding="utf-8")
    model.train()
    max_steps = int(settings.get("max_steps", 1000))
    target_tokens = int(settings.get("target_tokens", 0))
    eval_every = max(1, int(settings.get("evaluate_every", 200)))
    log_every = max(1, int(settings.get("log_every", 10)))
    eval_limit = int(settings.get("validation_max_samples", 2000))
    patience = int(settings.get("early_stopping_patience", 0))
    start_step, last_save = step, step if checkpoint_state else -1
    clock, last_tokens = time.perf_counter(), seen_tokens
    last_validation = None
    try:
        while step < max_steps and (not target_tokens or seen_tokens < target_tokens):
            if stop_after is not None and step - start_step >= stop_after:
                break
            next_step = step + 1
            factor = learning_rate_factor(next_step, seen_tokens, settings)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            cpu_batches = [collator(stream.next()) for _ in range(accumulation)]
            total_mlm = sum(b["mlm_labels"].numel() for b in cpu_batches) if stage == "pretrain" else 0
            step_loss, real_tokens = 0.0, 0
            for cpu_batch in cpu_batches:
                real_tokens += int(cpu_batch["attention_mask"].sum())
                if stage != "pretrain":
                    count_supervision(cpu_batch, supervision)
                batch = move_to_device(cpu_batch, dev)
                with autocast_context(dev, prec):
                    out = model(batch, task)
                    if stage == "pretrain":
                        # Exact masked-token normalization across accumulated microbatches.
                        loss = F.cross_entropy(out["mlm"].float(), batch["mlm_labels"], reduction="sum") / total_mlm
                    else:
                        loss, _ = security_loss(out, batch, settings.get("loss_weights"))
                        loss = loss / accumulation
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite loss: lower LR, inspect data, or use --precision fp32")
                loss.backward()
                step_loss += float(loss.detach().cpu())
                del out, batch, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(settings.get("clip_norm", 1.0)),
                error_if_nonfinite=True, foreach=False)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = next_step
            seen_tokens += real_tokens
            stop = step >= max_steps or (target_tokens and seen_tokens >= target_tokens) or (stop_after is not None and step - start_step >= stop_after)
            event = {"step": step, "seen_input_tokens_including_markers": seen_tokens,
                     "loss": step_loss, "gradient_norm": float(grad_norm.cpu()), "lr": optimizer.param_groups[0]["lr"],
                     "epoch": stream.epoch}
            if step % log_every == 0 or stop or step == 1:
                synchronize(dev)
                now = time.perf_counter()
                event["end_to_end_tokens_per_second"] = (seen_tokens - last_tokens) / max(1e-9, now - clock)
                clock, last_tokens = now, seen_tokens
                print(json.dumps(event), flush=True)
            if step % eval_every == 0 or stop:
                metrics, _ = evaluate_model(model, validation_set, dev, prec, stage,
                                             int(settings.get("evaluation_batch_size", 1)), eval_limit)
                last_validation = metrics
                score = -metrics["validation_loss"] if stage == "pretrain" or metrics.get("auprc") is None else metrics["auprc"]
                improved = best_score is None or score > best_score + float(settings.get("min_delta", 0.0))
                if improved:
                    best_score, bad_evals = score, 0
                else:
                    bad_evals += 1
                event["validation"] = metrics
                print(json.dumps({"step": step, "validation": metrics, "best": improved}), flush=True)
                metadata = {"format": 1, "stage": stage, "model_config": model.cfg.to_dict(),
                            "training_config": settings, "tokenizer_sha256": train_set.manifest["tokenizer_sha256"],
                            "train_data_sha256": train_set.manifest["data_sha256"],
                            "validation_data_sha256": validation_set.manifest["data_sha256"],
                            "max_training_length": max(train_set.manifest["max_length"], previous.get("max_training_length", 0)),
                            "supervision_presentations": supervision, "validation": metrics,
                            "step": step, "seen_tokens": seen_tokens, "diagnostic_tokenizer": train_set.manifest["diagnostic_tokenizer"],
                            "training_status": "experimental; benchmark performance not certified"}
                state = {"step": step, "seen_tokens": seen_tokens, "best_score": best_score, "bad_evals": bad_evals,
                         "stream": stream.state_dict(), "collator_rng": collator.generator.get_state(), "rng": rng_state(dev)}
                save_checkpoint(run, model, optimizer, state, metadata, best=improved,
                                keep=int(settings.get("keep_checkpoints", 3)))
                last_save = step
                if patience and bad_evals >= patience:
                    stop = True
                    event["early_stopped"] = True
            log.write(json.dumps(event) + "\n")
            log.flush()
            if stop:
                break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError("Memory exhausted. Keep allocator limits enabled; reduce sequence length/query_chunk or microbatch size, use checkpointing, and restart from the last completed checkpoint.") from e
        raise
    finally:
        log.close()
        train_set.close()
        validation_set.close()
    return {"step": step, "seen_tokens": seen_tokens, "last_saved_step": last_save,
            "validation": last_validation, "run": str(run)}
