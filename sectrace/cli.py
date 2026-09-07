from __future__ import annotations
import argparse
import os
# Must be set before importing torch. Do not silently move unsupported ops to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sectrace", description="SecTrace: experimental 100M security encoder")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("doctor", help="Exercise the actual model/backend and benchmark optimizer steps")
    q.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    q.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    q.add_argument("--full", action="store_true")
    q.add_argument("--steps", type=int, default=2)
    q.add_argument("--length", type=int, default=128)
    q.add_argument("--output")
    q = sub.add_parser("import", help="Convert a local public dataset release to canonical JSONL")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--format", choices=["primevul", "megavul", "secvuleval", "code", "canonical"], required=True)
    q.add_argument("--split", choices=["train", "validation", "calibration", "test"])
    q = sub.add_parser("import-hf", help="Stream approved public code or supported vulnerability datasets")
    q.add_argument("--dataset", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--revision")
    q.add_argument("--data-dir")
    q.add_argument("--split", default="train")
    q.add_argument("--limit", type=int, default=20000)
    q.add_argument("--format", choices=["code", "primevul", "secvuleval"], default="code")
    q.add_argument("--licenses", nargs="+")
    q.add_argument("--acknowledge-terms", action="store_true")
    q = sub.add_parser("import-sqlite", help="Read-only CVEfixes or custom SELECT export; labels stay unknown")
    q.add_argument("--database", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--query-file")
    q = sub.add_parser("split", help="Group and deduplicate all source datasets together, before tokenization")
    q.add_argument("--inputs", nargs="+", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--group-key", choices=["repo", "incident"], default="repo")
    q.add_argument("--seed", type=int, default=17)
    q.add_argument("--respect-splits", action="store_true")
    q.add_argument("--alias-file")
    q = sub.add_parser("audit", help="Fail on declared group/repository/lexical overlap across partitions")
    q.add_argument("--inputs", nargs="+", required=True)
    q.add_argument("--output")
    q.add_argument("--group-key", choices=["repo", "incident"], default="repo")
    q.add_argument("--alias-file")
    q = sub.add_parser("tokenizer", help="Train the BPE tokenizer on TRAINING code only")
    q.add_argument("--inputs", nargs="+", default=[])
    q.add_argument("--output", required=True)
    q.add_argument("--vocab-size", type=int, default=32768)
    q.add_argument("--diagnostic-byte", action="store_true")
    q = sub.add_parser("graph", help="Add Clang AST/declaration-reference graphs (not full dataflow)")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--compiler", default="clang")
    q = sub.add_parser("prepare", help="Build a seekable token cache without unsafe label truncation")
    q.add_argument("--input", required=True)
    q.add_argument("--tokenizer", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--stage", choices=["pretrain", "security"], required=True)
    q.add_argument("--max-length", type=int, default=512)
    q.add_argument("--max-nodes", type=int, default=256)
    q.add_argument("--max-neighbors", type=int, default=16)
    q.add_argument("--purpose", choices=["train", "validation", "calibration", "test"])
    q = sub.add_parser("train", help="Pretrain, SFT or reliability alignment according to the config")
    q.add_argument("--config", required=True)
    q.add_argument("--train-data", required=True)
    q.add_argument("--validation-data", required=True)
    q.add_argument("--output-dir", required=True)
    group = q.add_mutually_exclusive_group()
    group.add_argument("--init")
    group.add_argument("--resume")
    q.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    q.add_argument("--precision", choices=["auto", "fp32", "bf16"])
    q.add_argument("--stop-after", type=int, help="Stop cleanly after N steps without changing the planned LR schedule")
    q = sub.add_parser("evaluate", help="Evaluate an independent cache; never tune deployment thresholds on test data")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    q.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    q.add_argument("--calibration")
    q.add_argument("--bootstrap", type=int, default=0)
    q = sub.add_parser("calibrate", help="Fit held-out temperature+bias and select an empirical FPR threshold")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    q.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    q.add_argument("--target-fpr", type=float, default=.01)
    q.add_argument("--allow-small", action="store_true")
    q = sub.add_parser("predict", help="Return a structured candidate, not a proof or a repository-wide safety claim")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--tokenizer", required=True)
    q.add_argument("--code", required=True)
    q.add_argument("--output")
    q.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    q.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    q.add_argument("--calibration")
    q.add_argument("--max-length", type=int, default=512)
    q.add_argument("--with-clang", action="store_true")
    args = vars(p.parse_args(argv))
    command = args.pop("command")
    if command == "doctor":
        from .diagnostics import doctor
        return doctor(**args)
    if command == "import":
        from .importers import convert_file
        args["input_path"], args["output_path"] = args.pop("input"), args.pop("output")
        return convert_file(**args)
    if command == "import-hf":
        from .importers import import_hf
        args["license_allowlist"] = args.pop("licenses")
        return import_hf(**args)
    if command == "import-sqlite":
        from .importers import import_sqlite
        return import_sqlite(**args)
    if command == "split":
        from .splitting import split_corpus
        return split_corpus(**args)
    if command == "audit":
        from .splitting import audit_files
        return audit_files(**args)
    if command == "tokenizer":
        from .tokenization import train_tokenizer
        args["diagnostic"] = args.pop("diagnostic_byte")
        return train_tokenizer(**args)
    if command == "graph":
        from .frontend import enrich
        args["input_path"] = args.pop("input")
        return enrich(**args)
    if command == "prepare":
        from .data import prepare
        args["input_path"], args["tokenizer_path"] = args.pop("input"), args.pop("tokenizer")
        return prepare(**args)
    if command == "train":
        from .training import train
        args["config_path"] = args.pop("config")
        return train(**args)
    if command == "evaluate":
        from .evaluation import evaluate_checkpoint
        return evaluate_checkpoint(**args)
    if command == "calibrate":
        from .evaluation import calibrate
        return calibrate(**args)
    if command == "predict":
        from .evaluation import predict
        args["tokenizer_path"], args["code_path"] = args.pop("tokenizer"), args.pop("code")
        return predict(**args)


if __name__ == "__main__":
    main()
