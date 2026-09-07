# SecTrace-MPS

**An experimental graph-aware, 100-million-parameter security encoder, with a runnable PyTorch pre-training → SFT → reliability-alignment pipeline.**

This is source code and execution-tested infrastructure, **not pre-trained weights, a validated vulnerability scanner, or a state-of-the-art result**. Training on public datasets does not, by itself, establish useful security accuracy. The default operating scope is C/C++ source code, not binaries or exploit generation.

## What is implemented

| Component | Implementation |
|---|---|
| Encoder | 16 layers, width 640, 10 attention heads, SwiGLU width 1,664, pre-RMSNorm, RoPE |
| Attention | Exact ±256-token local attention; every fourth layer is global; query-chunked SDPA |
| Graph adapters | Four 128-dimensional relation-aware modules, bounded gather-based neighbor lists |
| Heads | Risk, 12 family slots, 64 CWE slots, eight evidence roles, evidence edges, context adequacy |
| Parameter budget | **99,553,002** without the MLM-only head; **99,996,650** including it |
| Pre-training | Dynamic span-masked language modeling; tied vocabulary head; graph-conditioned when graphs are present |
| SFT | Masked-label multi-task loss, example weights, optional verified vulnerable/fixed pair ranking |
| Alignment | Reviewed corrections, verified invariance/ranking pairs, evidence/adequacy supervision, optional teacher soft labels |
| Mac execution | Native PyTorch MPS; BF16 runtime probe or FP32; activation checkpointing; gradient accumulation |
| Reliability controls | Partition checks, exact lexical deduplication, unknown-label masking, no supervised-prefix truncation |
| State and evaluation | Safetensors, optimizer/RNG/cursor resume, held-out calibration, low-FPR diagnostics, group-bootstrap AUPRC |

The configured maximum context is **8,192 tokens**. The Mac starting curriculum is **512 tokens**. Configuring 8K does not train 8K competence or establish that 8K training will fit your Mac. The 99.55M inference count excludes the MLM-only head; ordinary training checkpoints retain that head for continuing pre-training.

## 1. Install on a native Apple-silicon Python environment

Use a native `arm64` Terminal and Python, preferably Python 3.11 or 3.12. Do not run the Python environment under Rosetta. You need internet access for installation and public datasets.

```bash
cd sectrace-mps
bash scripts/setup_mac.sh
source .venv/bin/activate
```

The script pins PyTorch **2.10.0**, the CPU-tested API baseline, installs the project and data/test dependencies, records exact installed versions, and runs a small MPS diagnostic. It does not download training datasets or start a long training run. Mac compatibility still needs the actual diagnostic result.

Manual installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-tested-torch.txt -e '.[data,dev]'
python -m sectrace doctor --device mps --output mac-doctor.json
```

Run the full-size hardware check after the tiny check:

```bash
python -m sectrace doctor --device mps --full --length 128 --steps 2 \
  --output mac-full-doctor.json
python -m pytest -q
python scripts/smoke.py --device mps --output runs/smoke-mps
```

The smoke test uses a tiny model, a diagnostic byte tokenizer and synthetic code. It exercises all three training stages, calibration and evaluation **without downloading a dataset**. Its metrics are not security benchmark results. The optional BPE unit test exercises the real tokenizer dependency when installed.

If the BF16/full-model check fails, retry with `--precision fp32`. If MPS reports an unsupported operation, retain the error and use CPU diagnostics to isolate it; do not silently enable fallback and call the resulting timing MPS performance.

## 2. Obtain a manageable public-code pilot

Start with BigCode's **The Stack Smol** C/C++ subsets to exercise the real BPE/data path. They are a small pilot, not the previously proposed multi-billion-token research corpus. Review the upstream terms and current release, obtain access where required, then authenticate through the installed Hugging Face CLI or an environment credential. Never store credentials in the dataset or repository.

```bash
mkdir -p data/raw data/canonical
hf auth login

python -m sectrace import-hf \
  --dataset bigcode/the-stack-smol --data-dir data/c \
  --format code --limit 10000 --acknowledge-terms \
  --licenses mit apache-2.0 bsd-2-clause bsd-3-clause isc \
  --output data/canonical/code-c.jsonl

python -m sectrace import-hf \
  --dataset bigcode/the-stack-smol --data-dir 'data/c++' \
  --format code --limit 10000 --acknowledge-terms \
  --licenses mit apache-2.0 bsd-2-clause bsd-3-clause isc \
  --output data/canonical/code-cpp.jsonl
```

The allowlist is an illustrative **filter**, not a legal conclusion or a substitute for file-level provenance review. The importer requires explicit approval, rejects unknown license metadata, resolves an exact dataset revision, and records it. A multi-license file is conservatively accepted only when all listed identifiers are in the approved list.

**The Stack v2 is not a drop-in content download.** Its published records contain Software Heritage identifiers. Bulk content access requires the upstream agreement/access route. After authorized hydration, convert records with actual code to canonical JSONL; the importer refuses identifier-only records. See `docs/DATA.md` for provenance fields and source links.

## 3. Import vulnerability labels

Download the original **PrimeVul** release through its official project instructions. Then convert the local JSONL files; do not use a third-party relabeling without reviewing it.

```bash
python -m sectrace import --format primevul \
  --input data/raw/primevul_train.jsonl \
  --output data/canonical/primevul-train.jsonl --split train

python -m sectrace import --format primevul \
  --input data/raw/primevul_valid.jsonl \
  --output data/canonical/primevul-validation.jsonl --split validation

python -m sectrace import --format primevul \
  --input data/raw/primevul_test.jsonl \
  --output data/canonical/primevul-test.jsonl --split test
```

There are also local importers for **MegaVul** and **SecVulEval**, a streaming SecVulEval importer, and a read-only **CVEfixes SQLite** exporter. The SQLite exporter produces **unreviewed candidates with unknown labels**; it does not call every changed function vulnerable. MoreFixes PostgreSQL, ARVO, D2A and Juliet require conversion/review through the documented canonical schema; they do not have native one-command downloads in this release.

## 4. Establish global partitions before tokenization or pre-training

Two valid evaluation designs are supported, but do not confuse them.

**A new project-held-out research experiment:** include all source pools together and make new global partitions. This is not an official PrimeVul benchmark split.

```bash
python -m sectrace split \
  --inputs data/canonical/code-c.jsonl data/canonical/code-cpp.jsonl \
           data/canonical/primevul-train.jsonl \
           data/canonical/primevul-validation.jsonl \
           data/canonical/primevul-test.jsonl \
  --output-dir data/splits --group-key repo

python -m sectrace audit --inputs data/splits/train.jsonl \
  data/splits/validation.jsonl data/splits/calibration.jsonl data/splits/test.jsonl \
  --group-key repo --output data/splits/audit.json
```

The default split is approximately 80/10/5/5 by connected component, **not guaranteed by record count**. Inspect split counts and class distributions; tiny corpora can produce empty or single-class partitions.

**Preserving an existing benchmark:** use `split --respect-splits --group-key incident` in a new output directory. Conflicting source partitions fail instead of silently moving test data into training. Reserve a calibration portion from training groups before this operation; the tool does not quietly repurpose benchmark validation/test examples. Add repository-family exclusions for a stricter unseen-project test.

Neither mode is automatically chronological. Exact lexical deduplication does not catch near-clones or a held-out function embedded in a longer file. Supply repository alias mappings and perform broader contamination checks before publishing results. Read `docs/EVALUATION.md`.

## 5. Train the tokenizer, then prepare caches

Only training code is used to learn tokenizer merges:

```bash
python -m sectrace tokenizer --inputs data/splits/train.jsonl \
  --vocab-size 32768 --output data/tokenizer.json

for part in train validation; do
  python -m sectrace prepare --stage pretrain \
    --input "data/splits/$part.jsonl" --tokenizer data/tokenizer.json \
    --max-length 512 --output-dir "data/cache/pretrain-$part"
done

for part in train validation calibration test; do
  python -m sectrace prepare --stage security \
    --input "data/splits/$part.jsonl" --tokenizer data/tokenizer.json \
    --max-length 512 --output-dir "data/cache/security-$part"
done
```

Pre-training windows long, unlabeled code without inheriting security labels. Supervised preparation rejects a whole overlength example, **not just its suffix**. Review admission statistics: evaluation covers admitted analysis units, not rejected long functions. A calibration or test partition containing no usable labeled examples must be corrected through the declared sampling plan, not ignored.

To add the optional local graph frontend before caching:

```bash
xcode-select --install   # only when Command Line Tools are missing
python -m sectrace graph --input data/splits/train.jsonl \
  --output data/splits/train-with-ast.jsonl
```

This helper extracts **Clang AST and declaration-reference relationships**, not alias-sensitive dataflow, CFG, taint, or whole-repository call graphs. Use the same frontend policy for train/validation/test. Function-only snippets may fail parsing; failures are recorded. The model also accepts richer externally supplied graphs in the canonical format.

## 6. Run the three stages

### A. Pre-training

```bash
python -m sectrace train --config configs/pretrain_m3.yaml \
  --train-data data/cache/pretrain-train \
  --validation-data data/cache/pretrain-validation \
  --output-dir runs/pretrain --device mps
```

The supplied Mac config is a **pilot**: at most 300 million presented input tokens or 40,000 optimizer updates, whichever comes first. It is not sufficient evidence of a well-pretrained model. It cycles the available training cache; verify unique-token coverage and avoid excessive repetition. `configs/pretrain_research.yaml` supplies a proposed 10B-token budget, not a promise that training it on one Mac is practical or optimal.

### B. Supervised fine-tuning

```bash
python -m sectrace train --config configs/sft_m3.yaml \
  --init runs/pretrain \
  --train-data data/cache/security-train \
  --validation-data data/cache/security-validation \
  --output-dir runs/sft --device mps
```

`--init` loads the prior stage's best validation checkpoint and starts a fresh optimizer/schedule. Labels not actually annotated remain unknown. In particular, binary vulnerability datasets do **not** automatically train accurate evidence or context-adequacy heads.

### C. Reviewed security-reliability alignment

Run the SFT model on a designated **training/development mining pool**, inspect false positives/negatives and evidence/context errors, then create reviewed canonical records. Do not mine the final test set. Put independent reviewed alignment records into:

```
data/reviewed/align-train.jsonl
data/reviewed/align-validation.jsonl
```

These are user-supplied reviewed files, **not fabricated data included with this project**. `examples/reviewed-schema.json` illustrates the fields and verified pair format using a synthetic example. Run the global leakage audit against held-out pools after adding reviewed records.

```bash
python -m sectrace prepare --stage security --purpose train \
  --input data/reviewed/align-train.jsonl --tokenizer data/tokenizer.json \
  --max-length 512 --output-dir data/cache/align-train

python -m sectrace prepare --stage security --purpose validation \
  --input data/reviewed/align-validation.jsonl --tokenizer data/tokenizer.json \
  --max-length 512 --output-dir data/cache/align-validation

python -m sectrace train --config configs/align_m3.yaml \
  --init runs/sft --train-data data/cache/align-train \
  --validation-data data/cache/align-validation \
  --output-dir runs/align --device mps
```

The alignment engine is executable; the expert judgments must be supplied. Simply renaming the SFT dataset an “alignment dataset” does not create reliable alignment.

## 7. Calibrate, test, and inspect a candidate

```bash
mkdir -p reports
python -m sectrace calibrate --checkpoint runs/align \
  --data data/cache/security-calibration --target-fpr 0.01 \
  --output reports/calibration.json --device mps

python -m sectrace evaluate --checkpoint runs/align \
  --data data/cache/security-test --calibration reports/calibration.json \
  --bootstrap 200 --output reports/test.json --device mps

python -m sectrace predict --checkpoint runs/align \
  --tokenizer data/tokenizer.json --code examples/example.c \
  --calibration reports/calibration.json --max-length 512 \
  --output reports/example.json --device mps
```

Calibration records must represent the intended deployment sampling distribution. A 1% empirical calibration FPR is not a 1% deployment guarantee. The built-in minimum sample guard is a sanity check, not a power analysis. Calibration is tied to exact model weights and must be redone after fine-tuning. `--allow-small` is for diagnostics, not quality claims.

Predictions are candidates, not vulnerability proofs. Heads with no observed positive **and** negative supervision for a slot are suppressed. Adequacy is reported as uncalibrated when trained; no arbitrary learned adequacy threshold is silently treated as a reliable abstention rule. Overlength inputs are explicitly declined instead of scanned partially.

## Resume and longer-context curricula

```bash
# Same config, stage, tokenizer and cache; restores latest optimizer/RNG/cursor.
python -m sectrace train --config configs/pretrain_m3.yaml \
  --train-data data/cache/pretrain-train \
  --validation-data data/cache/pretrain-validation \
  --output-dir runs/pretrain --resume runs/pretrain --device mps
```

Use `--stop-after N` to end after N additional updates and save cleanly without changing the planned LR schedule. A signal/crash resumes from the last completed saved checkpoint; there is no promise of saving in-progress gradients.

For 1,024/2,048-token continuation, prepare **new** caches, use `configs/pretrain_context_m3.yaml`, `--init runs/pretrain`, and a new run directory. Optimizer/schedule reset is deliberate. Check the actual full-model diagnostic at the intended context before selecting it.

## Verification and limitations

See `verification/RESULTS.md`, the actual test XML, and `verification/full_cpu.json`.

- Full 100M forward/backward/AdamW diagnostic: passed on CPU at 64 tokens.
- Unit/regression tests: 30 passed; two skipped (MPS hardware and optional BPE dependency unavailable in the build environment).
- Tiny offline integration: pre-training, SFT, alignment, calibration and evaluation passed.
- Real MPS execution, Hugging Face downloads and full BPE training were **not** run in the build environment.
- No real-corpus pre-training, benchmark sweep, model weights or state-of-the-art results are included.

The cache implementation is inspectable JSONL plus memory-mapped offsets, not an optimized distributed token-shard system. Global splitting still keeps fingerprints/components in RAM. Benchmark ingestion throughput and storage on a bounded corpus before any billion-token scale-up; do not assume the current single-machine preprocessing is a production-scale data platform.

## Documentation

`docs/ARCHITECTURE.md` explains computation and objectives. `docs/DATA.md` specifies import semantics and annotation contracts. `docs/MAC.md` covers memory and MPS operation. `docs/EVALUATION.md` defines the evidence needed for a competitive-model claim. `docs/SOURCES.md` lists primary-source references.

Original code is MIT licensed. That license does **not** relicense third-party datasets or source code.
