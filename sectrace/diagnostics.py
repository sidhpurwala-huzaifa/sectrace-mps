from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import time
import torch
from .config import ModelConfig
from .data import Collator, encode_record, move_to_device
from .losses import security_loss
from .model import SecTrace
from .runtime import autocast_context, choose_precision, runtime_info, seed_all, select_device, synchronize
from .tokenization import CodeTokenizer, train_tokenizer
from .util import atomic_json


def diagnostic_record():
    code = "int f(int i){ int a[4]={1,2,3,4}; return a[i]; }"
    start = code.index("a[i]")
    return {"id": "diagnostic", "group_id": "diagnostic", "code": code, "label": 1,
            "adequate": 1, "cwes": ["CWE-125"], "known_cwes": ["CWE-416"],
            "evidence": [{"start": start, "end": start + 4, "roles": {"risky_operation": 1, "guard": 0}}],
            "graph": {"nodes": [{"start": 0, "end": len(code)}, {"start": start, "end": start + 4}],
                      "edges": [{"src": 0, "dst": 1, "type": "ast_child", "evidence": 1}]}}


def doctor(device="auto", precision="auto", full=False, steps=2, length=128, output=None):
    if steps < 1 or length < 64 or length % 8:
        raise ValueError("Use steps >= 1 and a length >= 64 divisible by 8")
    dev = select_device(device)
    prec, message = choose_precision(dev, precision)
    seed_all(17, dev)
    cfg = ModelConfig() if full else ModelConfig.tiny()
    cfg.query_chunk = min(64, length)
    report = {**runtime_info(dev), "precision": prec, "precision_probe": message,
              "full_100m_model": full, "benchmark_sequence_length": length}
    with tempfile.TemporaryDirectory() as directory:
        tokenizer_path = str(Path(directory) / "byte.json")
        train_tokenizer([], tokenizer_path, diagnostic=True)
        tokenizer = CodeTokenizer(tokenizer_path)
        record = diagnostic_record()
        record["code"] += "\n/* " + "x " * length + "*/"
        # Keep source spans in the original prefix; trim padding only in this
        # clearly synthetic performance fixture, not in supervised dataset import.
        record["code"] = record["code"][:length - 2]
        record["target_span"] = [0, len(record["code"])]
        encoded = encode_record(record, tokenizer, length)
        security_batch = Collator("security", 272)(encoded)
        mlm_batch = Collator("mlm", 272)(encoded)
        model = SecTrace(cfg).to(dev)
        report["parameters"] = model.parameter_counts()
        if not full and dev.type != "cpu":
            reference = copy.deepcopy(model).cpu().eval()
            model.eval()
            with torch.no_grad():
                expected = reference(security_batch)["risk"]
                with autocast_context(dev, prec):
                    actual = model(move_to_device(security_batch, dev))["risk"].float().cpu()
            report["cpu_comparison_max_abs_risk_logit_error"] = float((expected - actual).abs().max())
            tolerance = .02 if prec == "bf16" else .001
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            del reference
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False, fused=False)
        sec, mlm = move_to_device(security_batch, dev), move_to_device(mlm_batch, dev)
        elapsed, losses = [], []
        for i in range(steps + 1):
            synchronize(dev)
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(dev, prec):
                sec_out = model(sec)
                sec_loss, _ = security_loss(sec_out, sec)
                mlm_out = model(mlm, "mlm")
                mlm_loss = torch.nn.functional.cross_entropy(mlm_out["mlm"].float(), mlm["mlm_labels"])
                loss = sec_loss + mlm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
            optimizer.step()
            synchronize(dev)
            if i:
                elapsed.append(time.perf_counter() - begin)
                losses.append(float(loss.detach().cpu()))
        report["passed"] = True
        report["two_objective_optimizer_step_seconds"] = sum(elapsed) / len(elapsed)
        report["input_tokens_per_second_two_forwards"] = 2 * length / report["two_objective_optimizer_step_seconds"]
        report["finite_losses"] = losses
        report["runtime_after_steps"] = runtime_info(dev)
        report["interpretation"] = "Implementation/performance diagnostic only. No vulnerability benchmark accuracy is measured."
    if output:
        atomic_json(output, report)
    print(json.dumps(report, indent=2))
    return report
