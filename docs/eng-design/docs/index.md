# Contents

- [AMD Print Lessons Learned](amd-printf-lessons-learned)

    This document describes the technical challenges and solutions
    involved in implementing print statement debugging for AMD GPU
    kernels in Mojo by porting OpenCL hostcall functionality to avoid
    dependencies on AMD's device-libs and additional LLVM copies.

- [GenAI and Paged Attention](genai-paged-attention)

    This document explains PagedAttention, a memory management
    technique for LLM inference that fragments KV cache into reusable
    pages and enables prefix sharing between sequences with common
    prompts, resulting in improved memory efficiency and faster
    time-to-first-token performance.

- [Multi-Head Latent Attention](multi-head-latent-attention)

    This design document presents an optimized Multi-head Latent
    Attention (MLA) implementation that reduces KV cache memory usage
    to just 576 values per token by storing compressed latent
    representations instead of full K and V tensors.

- [PyTorch Layers to MAX Mapping Guide](pytorch-to-max-mapping-guide)

    This guide provides mappings between common PyTorch layers used in
    HuggingFace `transformers` and their equivalent MAX graph operations and
    layer abstractions.

- [Token sampling](token-sampling)

    This design document provides a comprehensive overview of token
    sampling techniques in LLM inference, covering algorithms like
    greedy sampling, top-k, top-p, and min-p sampling that control the
    randomness and diversity of text generation by determining how the
    next token is selected from probability distributions.
