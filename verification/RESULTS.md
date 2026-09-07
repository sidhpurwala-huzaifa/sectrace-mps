# Verification results

Executed on 6 September 2026, Linux x86-64, Python 3.13.5, PyTorch 2.10.0+cpu.

| Check | Result |
|---|---|
| Unit/regression tests | **30 passed, 2 skipped** |
| Full 99,996,650-parameter training model | Graph-conditioned security + MLM forward/backward/clipping/AdamW passed at 64 tokens on CPU |
| Exact parameter count | 99,996,650 full / 99,553,002 excluding MLM-only parameters |
| Exact query-chunked attention | Outputs and gradients match a dense local/global reference across chunk sizes |
| Unknown-label masking | Unannotated targets contribute zero gradient |
| Graph/checkpoint gradients | Adapter gradients are nonzero; checkpointed/direct gradients match |
| Padding and evaluation dropout | Finite padded rows and deterministic eval attention |
| Dataset semantics | Before/fix distinction, missing CWE targets, privileged metadata exclusion and cross-source GitHub identity covered |
| Data split/cache controls | Conflict quarantine, cross-partition overlap and cache tamper tests passed |
| Exact resume | Interrupted/resumed CPU weights match uninterrupted run bit-for-bit |
| Three-stage integration | Tiny pretrain → SFT → alignment → calibration → test passed offline |

The MPS unit test was skipped because this environment has no Apple GPU. The real BPE dependency test was skipped because `tokenizers` could not be installed without network access. Live Hugging Face downloads and upstream archives were not retrieved into this runtime. The Mac setup installs those dependencies and the included tests/doctor should be rerun there.

No M3 throughput or memory-fit measurement is claimed. CPU diagnostic timing is specific to this environment and its short synthetic batch. No trained model weights or real security benchmark results are supplied. Synthetic smoke metrics verify pipeline execution only, not useful detector quality.

Raw reports: `pytest.xml`, `pytest.txt`, `full_cpu.json`, `offline-smoke.json`, `summary.json`.
