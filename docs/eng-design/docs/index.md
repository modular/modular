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

- [Matrix Multiplication to Flash Attention](matmul-to-flash-attention)

    This document explains how Flash Attention can be understood as an extension
    of fast matrix multiplication techniques for Ampere hardware, using
    asynchronous data transfer instructions and online softmax computation to
    achieve memory-efficient attention processing without materializing large
    intermediate matrices.

- [Multi-Head Flash Attention](multi-head-flash-attention)

    This document describes the implementation of multi-head attention using
    Flash Attention algorithms, progressing from the basic self-attention
    mechanism, through Flash Attention 2's memory-efficient tiling approach, to
    Flash Attention 3's specialized optimizations for Hopper architecture with
    asynchronous operations and warp-group specialization.

- [U/WGMMA Flash Decoding](uwgmma-flash-decoding)

    This document explores "U/WGMMA Flash Decoding," proposing to transpose matrix
    operations in Flash Attention 3 to better utilize GPU hardware by operating
    on 64+ rows at once instead of wasting computation on smaller group sizes,
    while analyzing the trade-offs between improved throughput and increased
    memory/synchronization costs.

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
