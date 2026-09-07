# Architecture and training objectives

## Model, shapes and budget

`model.py` implements an encoder, not a decoder/chatbot. At maximum configured settings:

- Tokens: `[batch, length]`, vocabulary 32,768, hidden width 640.
- Sixteen blocks: pre-RMSNorm → multi-head attention → residual → pre-RMSNorm → SwiGLU → residual.
- Ten heads of width 64. SwiGLU intermediate 1,664, three bias-free weight projections (two packed together).
- Real-valued, interleaved RoPE. No trainable positional table.
- Local radius 256 except layers 4/8/12/16, which are fully global within the supplied unit.
- Four graph adapters of width 128, four heads of width 32. One learned residual gate per adapter.

| Parameters | Count |
|---|---:|
| Token embedding | 20,971,520 |
| Metadata embeddings | 30,720 |
| Attention projections | 26,214,400 |
| SwiGLU projections | 51,118,080 |
| Backbone norms | 21,120 |
| Graph adapters | 929,028 |
| Task heads | 268,134 |
| Inference architecture total | **99,553,002** |
| Additional MLM transform/norm/bias | 443,648 |
| Full training model | **99,996,650** |

The vocabulary decoder is a functional linear operation tied to the token embedding, not a second 21M-parameter matrix. Checkpoints retain MLM-only parameters; the reported inference count means those parameters are not needed for security inference. There is no standalone stripped-export command in this version.

## Memory-aware attention

Queries are processed in chunks. Each local query chunk reads only keys within its possible radius, with an exact distance mask. A query does not attend to all keys in its chunk's bounding interval indiscriminately. Global layers process all keys for each chunk. Global computation remains quadratic in sequence length; chunking does not eliminate that cost or guarantee a fused Metal attention kernel.

Boolean SDPA masks use **True for allowed attention**. Padded queries receive a safe dummy key, then their outputs are zeroed. This avoids undefined all-masked softmax rows. SDPA dropout is explicitly zero during evaluation. Gradient/output equivalence against a dense reference is regression tested.

Activation checkpointing covers transformer blocks and graph adapters. This recomputes activations during backward; it reduces saved activations at a compute cost. Query chunking plus checkpointing is a portability-oriented implementation, not a claim to outperform every MPS attention backend.

## Graph input

Character spans are mapped to token spans. Pooling uses FP32 prefix sums; messages use dense gathers over sparse bounded neighbor lists. No CUDA-only graph library is required. Each receiving node attends to supplied incoming neighbors with relation embeddings and biases. Nodes are projected back to tokens through a bounded, normalized token-to-node gather.

The adapter supports a 16-relation vocabulary. Data-cache defaults cap 256 nodes, 16 incoming neighbors (including self), and four node memberships per token. The original proposed upper bounds are not silently assumed: discarded nodes/edges/overlaps are recorded. Increase cache limits deliberately when your graph distribution/hardware warrants it. Graph caps change input information, not parameter count.

The included compiler helper provides AST containment and declaration-reference edges only. Its `definition_use` tag means syntactic declaration reference in this helper, **not** reaching-definition or alias-sensitive dataflow. Richer graph extraction and repository-context assembly are external responsibilities. Extract train and inference graphs consistently; never provide vulnerability labels as graph features.

Graphs are optional. A batch without graphs still trains the encoder; it does not train graph adapters through those examples. Global graph dropout is available as a robustness regularizer. A partial graph is not a proof of missing dependence.

## Stage A: pre-training

Implemented objective: span-masked language modeling, normally 15% selected tokens, with span lengths 1–5 and 80/10/10 masking/random/unchanged replacements. Only selected positions produce vocabulary logits. The loss is normalized by the total selected-token count across gradient accumulation.

Graph-conditioned MLM trains adapters when graphs are present. **There is no separate held-out graph-edge reconstruction objective in this release.** The evidence-edge head is supervised later when reviewed edge labels exist. No hidden teacher, distillation service, or extra large model is required.

## Stages B/C: structured supervision

The risk head uses binary BCE. Family/CWE/evidence/adequacy heads use independently masked BCE. Unknown targets are `-1`; they contribute zero gradient. Loss is normalized over known targets within each example, then weighted across examples. This avoids assuming all unannotated code is safe.

The shared task projection is 640→256. Risk/CWE/family/adequacy use the average of the final analysis-token and target-token mean representations; token evidence uses local states. The final graph adapter can therefore influence the aggregate score through target pooling.

Verified pair losses:

- `higher`: softplus(margin − (risk_vulnerable − risk_fixed)). Both are separate encoder inputs; no fix is needed during ordinary detection.
- `invariant`: squared difference of sigmoid risk scores under an adjudicated semantics-preserving transformation.

Do not assert a risk ranking when the label scope does not support it. The code requires `pair_verified: true`; that flag is a caller assertion, not automatic verification.

Teacher probabilities are optional. Their default loss weight is zero. They are not produced by this package and must not replace expert/dynamic evidence without validation.

## Optimization

AdamW, FP32 master weights and moments, linear warmup, cosine LR decay, gradient clipping, zero-grad-to-none, length-bucketed examples and gradient accumulation. Head LR multipliers apply to newly trained task heads. Non-decayed parameter groups include embeddings, norms and biases. `foreach=False` and `fused=False` choose portability over unverified backend optimizations.

Token budgets count presented non-padding input tokens including markers, not unique corpus tokens. Token budget and maximum steps are independent stopping ceilings. Supervised loss accumulation averages microbatch objectives; it is not claimed to be identical to a single global batch with differently distributed sparse annotations.

Best checkpoints are chosen on held-out validation AUPRC for classification, or validation loss for MLM. Calibration and test are not used to select training checkpoints. No hyperparameter setting here has been established as optimal.
