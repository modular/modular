---
title: "U/WGMMA Flash Decoding"
author: Chris Elrod
date: May 8, 2025
---

- **Author:** Chris Elrod
- **Date:** May 8, 2025

## Flash Attention 3 Refresher

The hot compute-loop for Flash Attention 3 (FA3) is roughly, in pseudo-code.

```mojo
for kv_start in range(BN, num_keys, BN):
    # copy from `P` (f32 register tile) to a bf16 tile
    # this frees the f32 register tile for `S = Q @ K'[...]`
    S = Q @ (K[range(kv_start, kv_start+BN),:])'
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

See [Multi-Head Flash Attention](../multi-head-flash-attention) for details.

When doing context decoding, `Q` and `P` have `group = num_q_heads //
num_kv_heads` rows. In our models so far, common values are `4` or `8`, with
one model having `16`, and no models with `group > 16`.  Each `@`
instruction is thus performed on a matrix with `group` rows.

Each of the `@` operations executes a WGMMA instruction on Hopper (sm90) or a
UMMA instruction on Blackwell (sm100).

## WGMMA and UMMA operate on 64 rows at a time

From a performance perspective, this is problematic, as WGMMA instructions all
execute on `64` rows, while UMMA instructions execute on `64`, `128`, or `256`
rows at a time (for valid UMMA shapes, see [Nvidia's
documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-kind-shapes)),
assuming `bfloat16` or `float16` inputs.

We are thus limited to 1/16, 1/8, and 1/4 of peak wgmma throughput when using
`group=4`, `8`, and `16`, respectively, as most of the computation is wasted in
every case.

WGMMA and UMMA instructions both accept inputs with as few as `8` columns,
accepting anything in `range(8, 256+8, 8)`.

## What if we transpose our operations?

That is, what if instead of performing roughly

```python
S = Q @ (K[range(kv_start, kv_start+BN),:])'
row_max = rowmax(S)
P = exp(S - row_max)
O = exp(old_row_max - row_max) * O
old_row_max = row_max
O += P @ V[range(kv_start, kv_start+BN), :]
# write corrected O when done
```

we perform

```python
S = K[range(kv_start, kv_start+BN),:] @ Q'
col_max = colmax(S)
P = exp(S - col_max)
O = exp(old_col_max - col_max) * O
old_col_max = col_max
O += (V[range(kv_start, kv_start+BN), :])' @ P
# write corrected O' when done
```

In this case, we can use `BN=64` (or a multiple thereof), thereby executing `64
x K` by `K x group` WGMMA or UMMA instructions, thus avoiding wasting any
computation.

However, there are issues.

### Throughput

I don’t know about Blackwell, but we have Hopper instruction throughput
(measured in TFLOPS) from [Luo, et al (Table
X)](https://arxiv.org/pdf/2402.13499v1), where `SS` denotes "the WGMMA
instruction that loads both A and B from shared memory, while `RS` is used for
the instruction that loads A from the register file":

- Dense throughput, `SS`, `N = 8`: 157.6
- Dense throughput, `SS`, `N = 16`: 283.5
- Dense throughput, `SS`, `N = 128`: 659.8 (currently used for `S = Q @ K'`)
  - or 728.5, for zero-initialization
- Dense throughput, `RS`, `N = 128`: 661.7 # currently used for `O += P @ V`

The WGMMA instructions we are currently using achieve over 4x (about 4.2x) the
throughput as the instructions we now propose.

With `group ≤ 8`, we waste 8x less work, but perform the work 4.2x less quickly.
Expected benefit = 8/4.2 = 1.9x throughput increase.

With `group == 16`, we waste 4x less work, but perform the work 2.33x less
quickly. Expected benefit = 4 / 2.3 = 1.7x throughput increase.

Expected MMA throughput increases of 1.7-1.9x are far less dramatic than the
≥4x implied by the amount of wasted work, but they are nothing to scoff at.

However, table VII reports a throughput of 490.7 for the old `m16n8k16` MMA
instruction, substantially higher than what it reports for the `n=8` WGMMA
instructions in table X. I am not sure why this is. We saw a dramatic
performance increase when switching to FA3 for decoding, which executes WGMMA
(waste = 7/8 for group=8) in place of MMA (waste = 1/2 for group=8). The
authors noted the performance difference between `RS` and `SS` was higher for
smaller shapes, which couldn’t hide memory latency as well. I suspect that the
benchmarks of Hopper executing the old sm80 MMA instructions were done using
only register inputs, ignoring the cost of loading.

This may be worth taking a look at. Without reproducing the benchmarks
ourselves, we can’t be too confident in their accuracy, particularly when
comparing across tables, where conditions may have changed (or I could
otherwise misinterpret the comparability of results between tables).

### Memory

WGMMA exists in `SS` and `RS` forms. When calculating:

`D (+)= A @ B`

`D` must be in registers and `B` in shared memory, while `A` can be in shared
memory (S) or registers (R).

For UMA, `D` must be in tensor memory, `B` in shared memory, and `A` in either
shared or tensor memory.

Focusing on FA3, the non-transposed algorithm lets us keep `S = Q @ K'` in
registers, all the way through rescaling, truncation to `bfloat16`, and being
the `A` operand in `O += P @ V`.

When transposing `S` , so that we have `S = K @ Q'`, we have a couple options.

For the following examples, let `W` be the number of consumer warps, and an
integer multiple of `4` (e.g. `4` or `8`), and let `BN = 16*W*R` for integer
`R`.

When computing `S = K @ Q'`, we perform `R` WGMMA operations, one per 64 rows.
Row `r` is located in the registers of warp `(r % 16W) // 16`, e.g. warp `0`
will contain rows `0-15, 64-79`, etc.

### Option `0`: Copy `P` to shared memory, `O` is transposed

Whether on Hopper or Blackwell, operand `B` must be in shared memory.

When performing `O += V' @ P`, our reduction would be across rows of `P` (i.e.
row `r` is in warp `(r % 16W) // 16`).  As the `B` operand must be in shared
memory, this means we write `P` to shared memory and synchronize on every
iteration.

This unfortunately means we must pay synchronization and memory transfer costs,
and must again use the `SS` form of wgmma.

The TFLOPS we could achieve with this matmul would thus be around 150 or 280,
minus the cost of writing to smem and synchronizing, for `group = 8` and `16`,
respectively. That is, we’re hopefully ≥1.5x faster for the second MMA, just as
we hopefully are for the first.

For FA4 and Blackwell, we will take this route as we transfer memory between
registers and tensor memory regardless.

### Option `1`: Perform split-k reduction, do not transpose `O`

To keep `P` in registers and avoid the need for synchronizing on every
iteration, we can avoid transposing the second matmul via doing a warp-level
split-k reduction. That is, we can calculate `O += P' @ V`, with `P` in
registers, and the reduction split across warps.

For example, warp `w` will perform `O += (P[range(16) + 16*w + 16*W*k, :])' @
V[range(16) + 16*w + 16*W*k, :]` for `k` in `range(R)`.

After the main loop, we must combine the accumulators of all warps, performing
corrections, as in Ampere warp-level split-k.

However, we’re left again with the problem of only `group` rows in the
operation.

We can’t naively use sm80’s old MMA instructions here, because overlapping the
second MMA with the `softmax` calculation is an important optimization in FA3.

However, we could still consider the possibility. One can think of a series of
sm80 MMAs being necessarily less efficient than WGMMA for a few reasons:

1. Memory reuse. Multiple warps must load the same data into memory to use MMAs
   on large blocks. WGMMA probably does something more efficient under the
   hood.
2. Asynchronous - allows overlapping computation. This is important to FA3’s
   performance, allowing MMA and `softmax` to overlap.
3. Less work for the scheduler, as instructions are much larger.
4. Being larger, it can probably be implemented more efficiently, moving less
   data around in the silicon.

Problem `0.` does not apply to this case: `P` s are already in registers, and
`V` is not reused across any warps.

The primary problem for performance may be `1`, as `2` and `3` may be mitigated
by the fact that we’re reducing the amount of wasted computation. We could
address `1` at the cost of increasing register pressure, by overlapping `K @
Q'` computations with the `softmax` . Normally, we overlap the `softmax` with
the second MMA, as this can be done with a minimal cost to register use.
Overlapping the first `MMA` with `softmax` is also possible — in terms of
dependency graph, it is trivial, as each of the first `MMA`s are independent —
but the register cost is higher.

This may be possible, as our register use is otherwise going to be smaller than
when encoding, due to processing only `group` rows of `Q` at a time. Warp-level
split-k eats into that, however, given enough time to experiment, it would be
interesting to see how well this approach works.

Another alternative is using WGMMA here, same as in the current decoding
implementation, where we use only `group/64` rows per wgmma.

Between option `0` (copying to shared memory and synchronizing) and option `1`,
I think option `0` should have a higher priority for exploration. This is
because

1. We have to transfer memory regardless on Blackwell.
2. If this approach (transposing) is really worth while, option `0` lets us
   transpose for both MMAs. Thus, if it is a big win to do so, then we should
   try to do for both MMAs / the win will be big enough to compensate for the
   synchronization.

A final point I’d like to make is that in decoding currently, warps `1..3`
currently branch to skip the softmax so the scheduler isn’t busy assigning
useless work to them. The approach proposed here lets all warps to contribute
to the softmax calculation (but they must synchronize to communicate the
`col_max` results).
