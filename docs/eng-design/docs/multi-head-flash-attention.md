---
title: "Multi-Head Flash Attention" 
author: Chris Elrod
date: May 7, 2025
---

- **Author:** Chris Elrod
- **Date:** May 7, 2025

This document descibes the implementation of Multi-Head Attention (MHA) using
Flash Attention 3.

## Background

The self-attention mechanism is defined as:

```none
A = softmax(Q@K')@V
```

Where `Q`, `K`, and `V` are the set of queries, keys, and values for that
attention head.

Multi-head attention extends this, adding the parameters `q_heads` and
`kv_heads`, where `q_heads % kv_heads == 0`. Let `group = q_heads // kv_heads`.

Then, we have:

```none
A = softmax(Q_{q_head} @ K_{kv_head}') @ V_{kv_head} 
  = softmax(Q_{q_head} @ K_{q_head//group}') @ V_{q_head//group}
```

Thus, we can index arrays using only `q_head`.  We additionally have a
`batch_idx`, meaning the operation we want to perform is:

```none
softmax(Q_{batch_idx, q_head_idx} @ K_{batch_idx,q_head_idx//group}') @ V_{batch_idx,q_head_idx//group}
```

Thus, we must essentially do `batch_size * num_q_head` number of attention
evaluations, although we only need to load `batch_size * num_kv_head` unique
`K` and `V` matrices.

We can view `Q`, `K`, and `V` as 4-D (ragged) arrays:

```none
Q: batch_size x seq_len x num_q_heads x depth
K: batch_size x num_keys x kv_num_heads x depth
V: batch_size x num_keys x kv_num_heads x depth
```

Indexing with `batch_idx` and `q_head_idx` we:

```none
S = Q @ K'
P = softmax(S)
O = P @ V
```

1. Multiply a `seq_len` x `depth` matrix `Q` with a `depth` x `num_keys` matrix
   `K'`.
2. Evaluate row-wise `softmax` of the `seq_len` x `num_keys` output matrix.
3. Multiply the `seq_len` x `num_keys` matrix with the `num_keys` x `depth`
   matrix `V`.

## Flash Attention 2

This naive algorithm is costly in terms of data movement. `depth` tends to be
small, e.g. `128`, while both `seq_len` and `num_keys` can be large, e.g. up to
`8192` and `119132` respectively in llama3.3.70b. Thus, materializing an `8192`
x `119132` matrix would impose a high memory bandwidth cost that the reduction
over `128` elements of `depth` cannot hide.

The innovation of flash attention is to avoid materializing this array, holding
the output in registers, and performing an online `softmax`.

```none
softmax(S[i,j]) = exp(S[i,j]) / sum(exp(S[i,k]) for k in range(num_keys))
                = exp(S[i,j]-S[i,a]) / sum(exp(S[i,k]-S[i,a]) for k in range(num_keys))
```

where `a` is the rowwise maximum value. For a 32-bit values of `x >= 88.72284`,
`exp(x)=Inf`. To prevent overflow and preserve numerical accuracy, the
subtraction guarantees that the largest exponential value is `1.0`.

The online algorithm allows us to split this into batches. First, we avoid
applying the denominator until the end, which we apply to the final output
array.

Thus we focus only on updating the numerator, as well as the outputs computed
from previous tiles. Let `b` be the old maximum index from prior batches.

```none
exp(S[i,j]-S[i,a]) = exp(S[i,j]-S[i,b]+S[i,b]-S[i,a])
                   = exp(S[i,j]-S[i,b])*exp(S[i,b]-S[i,a])
```

To update old values, we simply scale them by the
correction factor `exp(S[i,b]-S[i,a])`.

This requires keeping track of the `rowmax` values through the online algorithm,
as well as the exponential sum which we use as the denominator at the end.

With this, we’re able to tile `K'` by columns. The Flash Attention 2 algorithm
is essentially:

```mojo
row_max = [-Inf for _ in range(seq_len)]
row_sum = [0 for _ in range(seq_len)]
O = matrix(seq_len, depth).fill(0)

for kv_start in range(0, num_keys, BN):
    block_range = range(kv_start, kv_start+BN)
    S = mask_function(Q @ K'[:, block_range]) # apply mask, e.g. CausalMask()
    old_rowmax = rowmax
    row_max = max(old_rowmax, rowmax(S))
    P = exp(S - row_max)
    correction = exp(old_rowmax - row_max)
    row_sum = row_sum * correction + rowsum(P)
    O = correction*O + P @ V[block_range, :]
    
O /= row_sum
```

Note that there is no communication across rows, therefore it is natural to
block across `seq_len`. Doing this, the size of all temporaries is bounded; we
pick values such that all temporaries (`row_max`, `row_sum`, `S`, `P`, and `O`)
can be held in registers.

This avoids materializing the large array, and reading it to and from memory.
Our only writes are the final answer at the end, and our only reads are the
inputs, `Q`, `K`, and `V`.

An important special case is token generation. Using a KV-cache, we can save
prior results, and future work uses `seq_len=1`, incrementally computing new
results.

With this, we index our arrays using `kv_head_idx`, operating on `group` rows
at a time. Otherwise, the algorithm is similar.

Further optimizations include pipelining with buffering, e.g. using
asynchronous copies of global to shared memory. This can help hide latency:
while computing one iteration, we’re copying the data used
`num_pipeline_stages - 1` in advance.

## Flash Attention 3

FA3 is Flash Attention 3, the flash attention algorithm specialized for the
Hopper (sm90) architecture. Hopper adds asynchronous `wgmma` instructions, for
computing matrix multiply `@` operations asynchronously. It also adds
shared-memory barriers and dynamic register allocation/deallocation, which
allow for warp-specialization.

Warp group specialized kernels are also often referred to as “ping pong”
kernels.

One warp group specializes on launching asynchronous copies to shared memory
(and deallocates most of its registers).

Others perform computation. The GPU’s execution can bounce or “ping pong”
between them, while their operations asynchronously run in the background. By
running two compute warp groups, we allow their matrix multiplies to overlap
the `softmax`-related instructions such as exponentials, as these use separate
units on the hardware that are able to fully run in parallel.

To further optimize within a warpgroup (which is also the only option for
context decoding, where we don’t have enough rows to `Q` to divide it into two
warpgroups), we can pipeline the loop above:

```python
for kv_start in range(BN, num_keys, BN):
    # copy from `P` (f32 register tile) to a bf16 tile
    # this frees the f32 register tile for `S = Q @ K'[...]`
    S = Q @ K'[:, range(kv_start, kv_start+BN)]
    O += P @ V[range(kv_start-BN, kv_start), :] # from the previous iteration!
    S.wait()
    S = mask_function(S) # apply mask, e.g. CausalMask()
    old_rowmax = rowmax
    row_max = max(old_rowmax, rowmax(S))
    P = exp(S - row_max)
    correction = exp(old_rowmax - row_max)
    row_sum = row_sum * correction + rowsum(P)
    O.wait() # frees `bf16` register tile
    O = correction*O
    
# NOTE: the `P` in `P @ V` is a truncation to `bfloat16`, so the registers
#       do not alias `S` or `P` elsewhere; 
```

Now, the `P @ V[subset, :]` wgmma instructions are capable of overlapping the
bulk of the vector instructions within a kernel, while all these operations
additionally overlap the memory transfers handled by the other warp group.

This allows us to use a great deal of the hardware’s resources in parallel.
