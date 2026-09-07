# Mac M3 execution and memory

## Native backend

The model uses **PyTorch's `mps` device**, backed by Apple's Metal Performance Shaders/MPSGraph infrastructure. This is the Apple-specific acceleration path for this PyTorch project. There is no Intel extension, CUDA emulation or MLX model conversion.

The launcher checks native architecture/backend availability. `doctor` reports the Python/PyTorch versions, physical memory, backend availability, precision probe and, on MPS, allocator/driver/recommended working-set counters. Its full-model mode performs real forward, backward, clipping and AdamW updates using both MLM and graph-conditioned security losses.

## Precision

Master parameters and Adam moments remain FP32. `--precision auto` tries BF16 autocast, attention backward and optimizer behavior. It uses BF16 only if that runtime probe succeeds; otherwise it chooses FP32. The real-model doctor adds a model-specific check and a tiny CPU/MPS output comparison. A probe on one batch cannot certify every shape or kernel; keep an FP32 control run.

`--precision bf16` fails clearly if the capability test does not pass. No untested FP16/GradScaler training path is enabled. Precision behavior depends on installed PyTorch, macOS and hardware, not just the text “M3.”

## Conservative starting points, not measured fit guarantees

| Unified memory | First context-length experiment | Microbatch | Accumulation |
|---|---:|---:|---:|
| 8 GB | 128, then 256 only after measurement | 1 | 16 |
| 16–18 GB | 512 | 1 | 16 |
| 24–36 GB | 512, then 1,024 after measurement | 1 | 16 |
| More memory | Profile 1,024/2,048 deliberately | 1 | 16 |

These are proposals, not measurements from an M3. Background applications and paired examples affect available memory. Keep activation checkpointing enabled. Generate caches at the selected length; changing only the model's maximum does not shorten cached examples.

Approximately 100M FP32 parameters, gradients and two FP32 Adam moment tensors account for about **1.6 GB decimal** before activations, workspaces, temporary checkpoint loads, CPU cache objects and the operating system. This arithmetic is not total peak memory. BF16 autocast does not halve FP32 optimizer-state storage.

Each full training checkpoint can occupy roughly 1.2 GB decimal for FP32 weights plus two moment tensors, with additional metadata/state overhead. The retention policy keeps recent checkpoints and protects the best one, so the retained count can exceed `keep_checkpoints` by one. Keep ample free disk space for atomic writes and dataset caches.

## Deliberate optimizations

- Exact query-chunked local/global attention controls temporary matrix sizes while retaining the selected attention pattern.
- Non-reentrant activation checkpointing trades recomputation for activation memory.
- Selected-position MLM logits avoid allocating a full `[B,T,V]` tensor for unmasked positions.
- Bounded neighbor-list gathers avoid CUDA-only sparse/graph dependencies.
- CPU tokenization/collation and memory-mapped index arrays avoid loading all decoded records on the device.
- Length bucketing reduces padding. Gradient accumulation increases effective batch size without retaining all forward activations.
- FP32 normalization/graph accumulations improve numerical robustness; no claim that every operation runs in BF16.
- No worker-forking, CUDA-only fused Adam, FlashAttention dependency, or unconditional `torch.compile` path.

These are implemented memory/portability choices. Only measurements on the actual Mac establish speedups.

## Environment controls

The CLI defaults to:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTORCH_MPS_FAST_MATH=0
```

It limits allocation to 85% of `recommendedMaxWorkingSetSize` by default, not 85% of all installed RAM. Do not set the high watermark to zero to force a larger run; PyTorch documents that disabling the limit can cause system-wide allocation failure.

`PYTORCH_MPS_PREFER_METAL=1` selects Metal matmul kernels instead of the MPSGraph path where applicable. It is **not assumed faster** and is not forced. Compare the same doctor/training batches with and without it, in fresh processes, and retain the setting only after numerical and performance checks:

```bash
PYTORCH_MPS_PREFER_METAL=1 python -m sectrace doctor \
  --device mps --full --precision fp32 --length 512 --steps 5 \
  --output mac-metal-matmul.json
```

The diagnostic times two forwards (MLM and security) per optimizer update. Its token/sec number is not a measured single-stage pre-training throughput. Actual training logs emit end-to-end non-padding input tokens/sec. Use several stable intervals; validation/checkpoint activity may affect timing windows.

## Failures and reproducibility

For OOM, reduce context length first, reduce `query_chunk`, keep microbatch one, and remember that verified pairs expand sequences. Restart from the last complete checkpoint. Never hide arbitrary runtime errors by pretending that the optimizer update succeeded.

A normal `--resume` requires the same config, data, tokenizer and stage, and restores data cursor, CPU/device RNG, collator RNG and optimizer. CPU interrupted/resumed weights were tested bit-for-bit identical. Floating-point behavior can differ across device/backend/version changes; **bit-for-bit MPS determinism is not promised**.

Use the saved `installed-mac-versions.txt`, run environment JSON, source manifests and config with your results. Test version upgrades in a separate environment and rerun doctor, unit tests and numerical controls.
