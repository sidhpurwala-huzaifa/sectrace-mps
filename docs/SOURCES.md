# Primary sources consulted

Checked while preparing this implementation, 6 September 2026. Links document mechanisms/dataset schemas; they do not validate SecTrace's accuracy or Mac throughput.

## Runtime and tokenizer

- Apple, accelerated PyTorch training on Mac: https://developer.apple.com/metal/pytorch/
- PyTorch 2.10 MPS environment variables: https://docs.pytorch.org/docs/2.10/mps_environment_variables.html
- PyTorch 2.10 automatic mixed precision: https://docs.pytorch.org/docs/2.10/amp.html
- PyTorch scaled-dot-product attention: https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch activation checkpointing: https://docs.pytorch.org/docs/2.10/checkpoint.html
- Hugging Face Tokenizers pre-tokenizers: https://huggingface.co/docs/tokenizers/api/pre-tokenizers
- Clang LibTooling: https://clang.llvm.org/docs/LibTooling.html

## Datasets and evaluation

- The Stack v2, current access/licensing/removal terms and identifier-only schema: https://huggingface.co/datasets/bigcode/the-stack-v2
- The Stack Smol, small code-content pilot: https://huggingface.co/datasets/bigcode/the-stack-smol
- PrimeVul official repository and data instructions: https://github.com/DLVulDet/PrimeVul
- PrimeVul paper: https://arxiv.org/abs/2403.18624
- MegaVul official repository: https://github.com/icyrockton/MegaVul
- MegaVul field specification: https://github.com/icyrockton/MegaVul/blob/main/SPECIFICATION.md
- SecVulEval official dataset card and schema: https://huggingface.co/datasets/arag0rn/SecVulEval
- CVEfixes official repository: https://github.com/secureIT-project/CVEfixes
- MoreFixes official repository and PostgreSQL release: https://github.com/JafarAkhondali/morefixes
- ARVO official repository: https://github.com/n132/ARVO
- D2A official repository: https://github.com/ibm/D2A
- NIST Juliet C/C++ suite: https://samate.nist.gov/SARD/test-suites/112
- DiverseVul paper: https://arxiv.org/abs/2304.00409
- Magma diagnostic benchmark: https://hexhive.epfl.ch/magma/

## Architectural precedents, not performance guarantees

- GraphCodeBERT: https://arxiv.org/abs/2009.08366
- ModernBERT: https://arxiv.org/abs/2412.13663
- GLU/SwiGLU variants: https://arxiv.org/abs/2002.05202
- Rotary position embeddings: https://arxiv.org/abs/2104.09864
- Temperature scaling/calibration: https://arxiv.org/abs/1706.04599
