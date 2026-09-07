from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import CachedDataset, Collator, encode_record, move_to_device
from .checkpoints import load_model
from .losses import security_loss
from .metrics import binary_metrics, group_bootstrap_ap, sigmoid, threshold_for_fpr
from .runtime import autocast_context, choose_precision, select_device
from .schema import CWE, ROLES
from .tokenization import CodeTokenizer
from .util import atomic_json, digest, sha256_file


@torch.no_grad()
def evaluate_model(model, dataset, device, precision, stage, batch_size=1, max_samples=0, seed=991):
    was_training = model.training
    model.eval()
    collator = Collator("mlm" if stage == "pretrain" else "security", dataset.manifest["random_vocab_size"], seed=seed)
    order = np.arange(len(dataset))
    if max_samples and max_samples < len(order):
        order = np.random.default_rng(seed).permutation(order)[:max_samples]
    total_loss, denominator = 0.0, 0
    predictions = []
    for begin in range(0, len(order), batch_size):
        rows = [dataset[int(i)] for i in order[begin:begin + batch_size]]
        batch = move_to_device(collator(rows), device)
        with autocast_context(device, precision):
            out = model(batch, "mlm" if stage == "pretrain" else "security")
            if stage == "pretrain":
                loss = F.cross_entropy(out["mlm"].float(), batch["mlm_labels"], reduction="sum")
                total_loss += float(loss.cpu())
                denominator += batch["mlm_labels"].numel()
            else:
                loss, _ = security_loss(out, batch)
                total_loss += float(loss.cpu()) * len(batch["meta"])
                denominator += len(batch["meta"])
                logits = out["risk"].float().cpu().tolist()
                adequate = out["adequacy"].float().cpu().tolist()
                labels = batch["label"].cpu().tolist()
                for info, logit, label, adequacy in zip(batch["meta"], logits, labels, adequate):
                    predictions.append({"id": info["id"], "group_id": info["group_id"],
                                        "label": int(label), "logit": logit, "adequacy_logit": adequacy})
    model.train(was_training)
    metrics = {"validation_loss": total_loss / max(1, denominator),
               "evaluated_cache_records": len(order), "available_cache_records": len(dataset)}
    if stage != "pretrain":
        metrics.update(binary_metrics([p["label"] for p in predictions], sigmoid([p["logit"] for p in predictions])))
    return metrics, predictions


def validate_cache_tokenizer(model, meta, dataset):
    if meta["tokenizer_sha256"] != dataset.manifest["tokenizer_sha256"]:
        raise ValueError("Checkpoint/tokenizer mismatch")
    if dataset.manifest["vocab_size"] > model.cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
    if dataset.manifest["max_length"] > model.cfg.max_length:
        raise ValueError("Cache exceeds model context capacity")


def evaluate_checkpoint(checkpoint, data, output, device="auto", precision="auto", calibration=None, bootstrap=0):
    dev = select_device(device)
    prec, _ = choose_precision(dev, precision)
    model, meta, folder = load_model(checkpoint, dev)
    dataset = CachedDataset(data)
    validate_cache_tokenizer(model, meta, dataset)
    metrics, predictions = evaluate_model(model, dataset, dev, prec, meta["stage"])
    if meta["stage"] != "pretrain":
        logits = np.asarray([p["logit"] for p in predictions])
        labels = [p["label"] for p in predictions]
        threshold = .5
        if calibration:
            cal = json.loads(Path(calibration).read_text())
            if cal["weights_sha256"] != sha256_file(folder / "model.safetensors"):
                raise ValueError("Calibration belongs to different weights; recalibrate after every fine-tune")
            if dataset.manifest["data_sha256"] == cal["calibration_data_sha256"]:
                raise ValueError("Refusing to present calibration-set metrics as an independent test")
            logits = logits / cal["temperature"] + cal["bias"]
            threshold = cal["threshold"]
        probabilities = sigmoid(logits)
        metrics.update(binary_metrics(labels, probabilities, threshold))
        if bootstrap:
            metrics["group_bootstrap_auprc_95pct_interval"] = group_bootstrap_ap(
                labels, probabilities, [p["group_id"] for p in predictions], bootstrap)
        for r, p in zip(predictions, probabilities):
            r["risk_score"] = float(p)
    report = {"metrics": metrics, "checkpoint": str(folder), "stage": meta["stage"],
              "precision": prec, "device": str(dev), "cache_manifest": dataset.manifest,
              "calibrated": bool(calibration),
              "warning": "Metrics cover only admitted complete analysis units, not the entire source corpus."}
    atomic_json(output, report)
    predfile = Path(str(output) + ".predictions.jsonl")
    with predfile.open("w") as f:
        for r in predictions:
            f.write(json.dumps(r) + "\n")
    dataset.close()
    print(json.dumps(report, indent=2))
    return report


def calibrate(checkpoint, data, output, device="auto", precision="auto", target_fpr=.01, allow_small=False):
    dev = select_device(device)
    prec, _ = choose_precision(dev, precision)
    model, meta, folder = load_model(checkpoint, dev)
    if meta["stage"] == "pretrain":
        raise ValueError("A pre-training-only checkpoint has no trained security classifier")
    dataset = CachedDataset(data)
    if dataset.manifest.get("purpose") != "calibration":
        raise ValueError("Calibration requires a dedicated calibration cache, never train/validation/test")
    validate_cache_tokenizer(model, meta, dataset)
    _, predictions = evaluate_model(model, dataset, dev, prec, "sft")
    valid = [p for p in predictions if p["label"] >= 0]
    labels = np.asarray([p["label"] for p in valid])
    if len(set(labels.tolist())) != 2:
        raise ValueError("Calibration requires both positive and negative examples")
    negatives, positives = int((labels == 0).sum()), int((labels == 1).sum())
    if not allow_small and (negatives < max(100, int(1 / max(target_fpr, 1e-8))) or positives < 20):
        raise ValueError("Calibration pool is too small. --allow-small is for smoke tests only, not operating-rate claims.")
    x = torch.tensor([p["logit"] for p in valid], dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    # CPU, two-parameter calibration. This cannot repair model discrimination.
    log_t = torch.nn.Parameter(torch.zeros(()))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([log_t, bias], lr=.03)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        t = log_t.exp().clamp(.05, 20)
        loss = F.binary_cross_entropy_with_logits(x / t + bias, y)
        loss.backward()
        optimizer.step()
    temperature = float(log_t.detach().exp().clamp(.05, 20))
    intercept = float(bias.detach())
    p = sigmoid(x.numpy() / temperature + intercept)
    threshold = threshold_for_fpr(labels, p, target_fpr)
    report = {"temperature": temperature, "bias": intercept, "threshold": threshold,
              "target_fpr": target_fpr, "weights_sha256": sha256_file(folder / "model.safetensors"),
              "tokenizer_sha256": meta["tokenizer_sha256"],
              "calibration_data_sha256": dataset.manifest["data_sha256"],
              "calibration_metrics_not_test_metrics": binary_metrics(labels, p, threshold),
              "small_pool_diagnostic_only": bool(allow_small), "device": str(dev), "precision": prec,
              "warning": "Observed calibration FPR is not a guarantee on unseen repositories. Preserve deployment sampling prevalence."}
    atomic_json(output, report)
    dataset.close()
    print(json.dumps(report, indent=2))
    return report


def supported_slots(metadata, head):
    value = metadata.get("supervision_presentations", {}).get(head, {})
    return [i for i, (p, n) in enumerate(zip(value.get("positive", []), value.get("negative", []))) if p > 0 and n > 0]


def predict(checkpoint, tokenizer_path, code_path, output=None, device="auto", precision="auto", calibration=None,
            max_length=512, with_clang=False):
    dev = select_device(device)
    prec, _ = choose_precision(dev, precision)
    model, metadata, folder = load_model(checkpoint, dev)
    if metadata["tokenizer_sha256"] != sha256_file(tokenizer_path):
        raise ValueError("Checkpoint/tokenizer mismatch")
    if not supported_slots(metadata, "risk"):
        raise ValueError("The risk head has not observed both labels; train it before running vulnerability predictions")
    model.eval()
    tokenizer = CodeTokenizer(tokenizer_path)
    code = Path(code_path).read_text(encoding="utf-8")
    record = {"id": digest(code), "group_id": "inference", "code": code, "label": None,
              "language": "c++" if Path(code_path).suffix in (".cpp", ".cc", ".cxx") else "c"}
    if with_clang:
        from .frontend import clang_graph
        record["graph"], record["frontend_status"] = clang_graph(code, record["language"])
    limit = min(max_length, model.cfg.max_length)
    rows = encode_record(record, tokenizer, limit)
    if not rows:
        report = {"status": "insufficient_context", "risk_score": None,
                  "reason": "Input exceeds the configured analysis-unit length; no truncated-prefix prediction was made.",
                  "max_length": limit}
    else:
        batch = move_to_device(Collator("security", tokenizer.random_vocab_size)(rows), dev)
        with torch.no_grad(), autocast_context(dev, prec):
            pred = model(batch, "security")
        raw = float(pred["risk"][0].float().cpu())
        probability = float(sigmoid([raw])[0])
        threshold = None
        if calibration:
            cal = json.loads(Path(calibration).read_text())
            if cal["weights_sha256"] != sha256_file(folder / "model.safetensors"):
                raise ValueError("Calibration weights do not match this checkpoint")
            probability = float(sigmoid([raw / cal["temperature"] + cal["bias"]])[0])
            threshold = cal["threshold"]
        status = "uncalibrated_risk_score" if threshold is None else "finding_candidate" if probability >= threshold else "no_finding_under_supplied_context"
        report = {"status": status, "risk_score": probability, "calibrated": bool(calibration),
                  "threshold": threshold, "source": str(code_path), "source_hash": digest(code),
                  "context_length": len(rows[0]["input_ids"]), "graph_diagnostics": rows[0]["diagnostics"],
                  "security_proof": False, "independent_validation_required": True}
        if supported_slots(metadata, "adequacy"):
            report["context_adequacy_score_uncalibrated"] = float(pred["adequacy"][0].float().sigmoid().cpu())
        if limit > metadata.get("max_training_length", 512):
            report["context_warning"] = "Requested length exceeds the largest recorded training curriculum length"
        if record.get("frontend_status", {}).get("parse_failed"):
            report["frontend_warning"] = record["frontend_status"]
        slots = supported_slots(metadata, "cwe")
        if slots:
            scores = pred["cwe"][0].float().sigmoid().cpu().tolist()
            report["weakness_candidates_uncalibrated"] = sorted(
                [{"cwe": CWE[i], "score": scores[i]} for i in slots], key=lambda x: x["score"], reverse=True)[:5]
        slots = supported_slots(metadata, "roles")
        if slots:
            scores = pred["roles"][0].float().sigmoid().cpu()
            candidates = []
            for token, (a, b) in enumerate(rows[0]["offsets"]):
                if b > a:
                    for role in slots:
                        candidates.append({"start": a, "end": b, "line": code.count("\n", 0, a) + 1,
                                           "role": ROLES[role], "score": float(scores[token, role])})
            report["evidence_candidates_not_proof"] = sorted(candidates, key=lambda x: x["score"], reverse=True)[:12]
        else:
            report["evidence_status"] = "No evidence role has both positive and negative supervision; evidence predictions are suppressed."
    if output:
        atomic_json(output, report)
    print(json.dumps(report, indent=2))
    return report
