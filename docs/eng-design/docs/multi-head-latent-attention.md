---
title: "Multi-Head Latent Attention"
author: Shouzheng Liu
date: February 19, 2025
---

- **Author:** Shouzheng Liu
- **Date:** February 19, 2025

## Background

### Multi-head Attention (MHA)

Attached is the compute graph of an MHA block, where square nodes represent
tensors materialized in memory, and edges denote certain operations. Once `Q`,
`K` and `V` are prepared, they are sent to the `mha` kernel. `K` and `V` can be
cached for reusing during the decoding stage.

![Compute graph of
MHA](img/multi-head-latent-attention/img01-multi-head-attention.png)
/// caption
Compute graph of multi-head attention
///

### Multi-head Latent Attention (MLA)

The key idea of MLA is to use a "latent" vector (`KV_lora`) to store a
compressed representation of the `K` and `V` tensors during inference. Instead
of directly computing `K` and `V` at full precision, the hidden states are
first **down-projected** to a much lower-dimension space and then
**up-projected** to the full dimension of `head_dim * num_heads` . `K` and `V`
share the same compressed representation `KV_lora`.

![original compute graph of multi-head latent
attention](img/multi-head-latent-attention/img02-multi-head-latent-attention.png)
/// caption
Original compute graph of multi-head latent attention
///

#### Rotary Position Embedding (ROPE) in MLA

Applying ROPE in MLA is somewhat non-trivial because not all elements of an
attention head undergo rotary encoding.

- For `Q` : ROPE is only applied to the last 64 (`rope_dim`) elements of each
  attention head, which has a total size of 192 (`no_dim+rope_dim`).
- For `K` : Instead of applying ROPE to the entire latent vector (`KV_lora'`),
  we extract the last 64 elements of each token, apply ROPE, and broadcast the
  results to all attention heads. Then, the `K` tensor is constructed by
  concatenating the roped part with the remaining dimensions that do not
  undergo rotary encoding.

When computing attention scores, the unroped elements of `Q` are multiplied
with the unroped elements of `K`. The roped elements of `Q` are multiplied with
the roped elements of `K`.

#### KV cache

While MLA aims to reduce computation and memory usage, its original
implementation does not actually reduce the KV cache size. This is because the
full `K` and `V` tensors are still materialized after up-projection, rather
than being stored in a compressed format throughout the process.

#### Optimized Attention Computation in MLA

![Optimized compute graph of multi-head latent
attention](img/multi-head-latent-attention/img03-optimized-multi-head-latent-attention.png)
/// caption
Optimized compute graph of multi-head latent attention
///

We calculate attention scores using

 $p=q^{T}k$,

where:

 $q=W_{qup}Q_{lora}$ , $k=W_{kup}K_{lora}$.

This allows us to rewrite as:

 $q=W_{kup}^TW_{qup}Q_{lora}$ and $k=K_{lora}$

while still keeping the results unchanged.

Similarly, instead of storing and passing the full `V` tensors, we can simply
reuse the compressed `KV_lora` tensor and apply the up-projection after the MHA
computation.

In this way, we only need to cache the `KV_lora` and the `K_roped` tensors.
(This reduces KV cache size to only 576 values per token!)

## Detailed Design

### New attention kernel for MLA

In the optimized MLA compute graph, the attention kernel effectively performs
multi-query attention (MQA). Specifically:

- The `Q` input has a shape of `[seq_len, num_heads, 576]` .
- The `K` input has a shape of `[seq_len, 1, 576]` .
- The `V` tensor is derived by reusing `K` , where `V = K[:, :, :512]`

A few points:

1. A `head_dim` of 576 feels kinda big, although if there will be a performance
   impact is unclear.
2. Here, we have `num_kv_heads=1` . By default we parallelize across KV heads,
   but with only one KV head, this would lead to severe hardware
   under-utilization. We can either apply split-k or parallelize across
   queries. In the latter case, different thread blocks will issue repeated
   loads of the same `K` , but since `K` is quite small, this might not be a
   major issue.
3. We don’t need to reserve shared memory or registers for `V` because `V` is
   `K`.

### Update KV cache

The KV cache manager needs to be updated to support models that only have `K`
cache. Currently, the implementation assumes the presence of both `K` and `V`
caches.

### Multi-GPU

We can still distribute the workload across multiple devices by splitting
across the queries heads. However, since there is only one KV head, we cannot
further split it, meaning the KV cache must be duplicated on every device.

We can also explore how MQA (multi-query attention) is optimized for multi-GPU
setups to see if we can borrow some ideas.
