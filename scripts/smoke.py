"""Offline three-stage integration test. Synthetic fixtures, NOT model training evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from sectrace.data import prepare
from sectrace.training import train
from sectrace.evaluation import calibrate, evaluate_checkpoint
from sectrace.tokenization import train_tokenizer
from sectrace.util import atomic_json, write_jsonl


def fixture(i, split, paired=False):
    name = f"f{i}"
    bad = f"int {name}(int i){{int a[4]={{1,2,3,4}};return a[i];}}"
    good = f"int {name}(int i){{int a[4]={{1,2,3,4}};if(i<0||i>=4)return -1;return a[i];}}"
    label = i % 2
    code = bad if label else good
    start = code.index("a[i]")
    r = {"id": f"toy:{i}", "group_id": f"toy-group:{i}", "repo": f"toy-project-{i}",
         "code": code, "label": label, "split": split,
         "adequate": 1, "cwes": ["CWE-125"] if label else [], "known_cwes": ["CWE-125", "CWE-416"],
         "evidence": [{"start": start, "end": start + 4, "roles": {"risky_operation": 1, "guard": 0}}],
         "graph": {"nodes": [{"start": 0, "end": len(code)}, {"start": start, "end": start + 4}],
                   "edges": [{"src": 0, "dst": 1, "type": "ast_child"}]},
         "source": "synthetic integration fixture; not a vulnerability benchmark"}
    if paired:
        r["code"], r["label"] = bad, 1
        r.pop("graph")
        r.pop("evidence")
        r["mate"] = {"id": f"toy:{i}:fixed", "group_id": r["group_id"], "repo": r["repo"],
                     "code": good, "label": 0, "split": split, "adequate": 1,
                     "known_cwes": ["CWE-125", "CWE-416"]}
        r["pair_relation"], r["pair_verified"] = "higher", True
    return r


def run(output, device="auto"):
    out = Path(output)
    if (out / "summary.json").exists():
        raise FileExistsError("Use a new smoke output directory")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = str(out / "tokenizer.json")
    train_tokenizer([], tokenizer, diagnostic=True)
    paths = {}
    for n, (split, count) in enumerate((("train", 32), ("validation", 12), ("calibration", 40), ("test", 12))):
        path = out / f"{split}.jsonl"
        write_jsonl(path, (fixture(i + n * 100, split) for i in range(count)))
        paths[split] = path
        prepare(path, tokenizer, out / f"security_{split}", "security", 128)
        if split in ("train", "validation"):
            prepare(path, tokenizer, out / f"mlm_{split}", "pretrain", 128)
    align_path = out / "alignment.jsonl"
    write_jsonl(align_path, (fixture(i + 1000, "train", paired=True) for i in range(16)))
    prepare(align_path, tokenizer, out / "alignment_cache", "security", 128)
    reports = []
    previous = None
    for stage in ("pretrain", "sft", "align"):
        cfg = {"model": {"preset": "tiny", "dropout": 0.0}, "training": {
            "stage": stage, "seed": 17, "microbatch_size": 2, "gradient_accumulation": 2,
            "max_steps": 4, "learning_rate": .0005 if stage == "pretrain" else .0001,
            "warmup_steps": 1, "evaluate_every": 2, "validation_max_samples": 0,
            "graph_dropout": 0.0, "log_every": 1, "keep_checkpoints": 2}}
        config_path = out / f"{stage}.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        training_cache = out / ("mlm_train" if stage == "pretrain" else "security_train" if stage == "sft" else "alignment_cache")
        validation_cache = out / ("mlm_validation" if stage == "pretrain" else "security_validation")
        result = train(config_path, training_cache, validation_cache, out / stage, init=previous,
                       device=device, precision="fp32" if device == "cpu" else None)
        previous = str(out / stage)
        reports.append(result)
    calibration = out / "calibration.json"
    calibrate(previous, out / "security_calibration", calibration, device=device, allow_small=True)
    evaluate_checkpoint(previous, out / "security_test", out / "evaluation.json", device=device, calibration=calibration)
    summary = {"passed": True, "device_requested": device, "stages": reports,
               "interpretation": "Pipeline execution test only: synthetic code, diagnostic byte tokenizer, tiny model. NOT evidence of security quality."}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/smoke")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = p.parse_args()
    run(args.output, args.device)
