# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Structured state space duality (SSD) output-recombination kernel.

This implements stage 4 (the final "output recombination") of the SSD
chunk-scan algorithm used by Mamba2. After stage 3 (``ssd_chunk_scan``) has
threaded the inter-chunk recurrence to produce the state *entering* each chunk,
this stage combines that inter-chunk contribution with the per-chunk diagonal
output to form the final per-token output.

For a single ``(batch, chunk, head)`` slice with chunk length ``L``, state dim
``N`` and head dim ``P``:

- Inputs:
  - ``C: [b, nc, h, L, N]`` (the per-token C projection).
  - ``entering_state: [b, nc, h, P, N]`` (state entering this chunk, the
    stage-3 output).
  - ``A: [b, nc, h, L]`` (per-token log-decay within the chunk).
  - ``Y_diag: [b, nc, h, L, P]`` (the intra-chunk diagonal output, from
    stage 1).
- Compute (per token ``l`` and head-dim index ``p``):
  ```
  A_cumsum[l]       = sum_{k <= l} A[k]           # inclusive prefix sum
  state_decay_out[l]= exp(A_cumsum[l])
  Y_off[l, p]       = state_decay_out[l] * sum_n C[l, n] * entering_state[p, n]
  Y[l, p]           = Y_diag[l, p] + Y_off[l, p]
  ```
- Output:
  - ``Y: [b, nc, h, L, P]``.

The computation is fully element-wise parallel over ``(b, nc, h, l, p)``: one
GPU thread owns a single output scalar ``Y[b, nc, h, l, p]``. The only sequential
dependency is the per-``(b, nc, h, l)`` inclusive prefix sum ``A_cumsum[l]``,
which each thread recomputes up to its own ``l`` (``l`` is at most the chunk
length, so this is cheap and avoids any cross-thread synchronization).
"""

from layout import (
    Idx,
    Layout,
    LayoutTensor,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape
from std.math import ceildiv, exp
from max.gpu.sync import barrier as gpu_barrier
from std.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.memory import unsafe_stack_allocation as mem_stack_allocation
from std.utils.index import IndexList
from linalg.bmm import batched_matmul

# ── Fused single-pass MMA tiling (Mamba2 profile, tuned to mirror stage 2) ──
# Output tile is the whole [L, P] per slice; warps tile it WARPS_M x WARPS_N.
# block_dim = WARPS_M * WARPS_N * WARP_SIZE must be >= chunk_len for the
# shared-memory decay scan. 8x1 warps = 256 threads covers chunk_len=256.
comptime COMBINE_FUSED_WARPS_M = 8
comptime COMBINE_FUSED_WARPS_N = 1
# K (state-dim) step for the fused MMA; multiple of MMA_K (8), divides state_dim.
comptime COMBINE_FUSED_BK = 16

# Stride helpers for indexing the flat row-major buffers.
comptime Strides4D = IndexList[4]
comptime Strides5D = IndexList[5]

# Upper bound on chunk_len for the shared-memory decay scratch (one Float32 per
# token). The static path launches one block per slice with
# ``block_dim == chunk_len``, so this also bounds that block size. Mirrors
# ``STATE_MAX_CHUNK`` in ``ssd_chunk_state.mojo``.
comptime COMBINE_MAX_CHUNK = 1024


def _row_major_strides_4d(d0: Int, d1: Int, d2: Int, d3: Int) -> Strides4D:
    """Row-major strides for a 4D shape ``(d0, d1, d2, d3)``."""
    return Strides4D(d1 * d2 * d3, d2 * d3, d3, 1)


def _row_major_strides_5d(
    d0: Int, d1: Int, d2: Int, d3: Int, d4: Int
) -> Strides5D:
    """Row-major strides for a 5D shape ``(d0, d1, d2, d3, d4)``."""
    return Strides5D(d1 * d2 * d3 * d4, d2 * d3 * d4, d3 * d4, d4, 1)


def ssd_output_recombination_fwd_cpu[
    kernel_dtype: DType,
    c_layout: Layout,
    entering_layout: Layout,
    a_layout: Layout,
    y_diag_layout: Layout,
    y_layout: Layout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
    C: LayoutTensor[kernel_dtype, c_layout, MutAnyOrigin],
    entering_state: LayoutTensor[kernel_dtype, entering_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, a_layout, MutAnyOrigin],
    Y_diag: LayoutTensor[kernel_dtype, y_diag_layout, MutAnyOrigin],
    Y: LayoutTensor[kernel_dtype, y_layout, MutAnyOrigin],
):
    """Compute the SSD output recombination (stage 4) on CPU.

    Combines the intra-chunk diagonal output ``Y_diag`` with the inter-chunk
    contribution (``C`` projected against the entering state, decayed by the
    in-chunk cumulative decay) to form the final per-token output ``Y``.

    Tensor shapes (row-major):
        - ``C``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``entering_state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``Y_diag``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``Y``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        c_layout: Layout of the ``C`` tensor.
        entering_layout: Layout of the ``entering_state`` tensor.
        a_layout: Layout of the ``A`` tensor.
        y_diag_layout: Layout of the ``Y_diag`` tensor.
        y_layout: Layout of the ``Y`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        head_dim: Head dimension (``P``).
        state_dim: State dimension (``N``).
        C: Per-token C projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        entering_state: State entering this chunk,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        A: Per-token in-chunk log-decay,
            shape ``[batch, n_chunks, n_heads, L]``.
        Y_diag: Intra-chunk diagonal output,
            shape ``[batch, n_chunks, n_heads, L, P]``.
        Y: Output, shape ``[batch, n_chunks, n_heads, L, P]``.
    """
    # Row-major strides for the flat backing buffers.
    var c_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, state_dim
    )
    var ent_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, head_dim, state_dim
    )
    var a_strides = _row_major_strides_4d(batch, n_chunks, n_heads, chunk_len)
    var y_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, head_dim
    )

    for bi in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var a_base = (
                    bi * a_strides[0] + c * a_strides[1] + h * a_strides[2]
                )
                var c_base = (
                    bi * c_strides[0] + c * c_strides[1] + h * c_strides[2]
                )
                var ent_base = (
                    bi * ent_strides[0]
                    + c * ent_strides[1]
                    + h * ent_strides[2]
                )
                var y_base = (
                    bi * y_strides[0] + c * y_strides[1] + h * y_strides[2]
                )

                # Inclusive prefix sum of A over the chunk -> decay per token.
                var a_cumsum = Float32(0.0)
                for l in range(chunk_len):
                    var a_off = a_base + l * a_strides[3]
                    a_cumsum += A.ptr[a_off].cast[DType.float32]()
                    var decay = exp(a_cumsum)

                    for p in range(head_dim):
                        # Y_off[l, p] = decay * sum_n C[l, n] * entering[p, n].
                        var acc = Float32(0.0)
                        for n in range(state_dim):
                            var c_off = (
                                c_base + l * c_strides[3] + n * c_strides[4]
                            )
                            var ent_off = (
                                ent_base
                                + p * ent_strides[3]
                                + n * ent_strides[4]
                            )
                            var c_val = C.ptr[c_off].cast[DType.float32]()
                            var ent_val = entering_state.ptr[ent_off].cast[
                                DType.float32
                            ]()
                            acc += c_val * ent_val

                        var y_off = decay * acc

                        # Y[l, p] = Y_diag[l, p] + Y_off[l, p].
                        var yd_off = (
                            y_base + l * y_strides[3] + p * y_strides[4]
                        )
                        var y_diag_val = Y_diag.ptr[yd_off].cast[
                            DType.float32
                        ]()
                        Y.ptr[yd_off] = (y_diag_val + y_off).cast[
                            kernel_dtype
                        ]()


# ===----------------------------------------------------------------------=== #
# GPU Implementation
# ===----------------------------------------------------------------------=== #


def ssd_output_recombination_fwd_gpu[
    kernel_dtype: DType,
    c_LT: TensorLayout,
    entering_LT: TensorLayout,
    a_LT: TensorLayout,
    y_diag_LT: TensorLayout,
    y_LT: TensorLayout,
](
    batch_i32: Int32,
    n_chunks_i32: Int32,
    n_heads_i32: Int32,
    chunk_len_i32: Int32,
    head_dim_i32: Int32,
    state_dim_i32: Int32,
    C: TileTensor[kernel_dtype, c_LT, MutUntrackedOrigin],
    entering_state: TileTensor[kernel_dtype, entering_LT, MutUntrackedOrigin],
    A: TileTensor[kernel_dtype, a_LT, MutUntrackedOrigin],
    Y_diag: TileTensor[kernel_dtype, y_diag_LT, MutUntrackedOrigin],
    Y: TileTensor[kernel_dtype, y_LT, MutUntrackedOrigin],
):
    """Compute the SSD output recombination (stage 4) on GPU.

    The computation is element-wise parallel: each GPU thread owns a single
    output scalar ``Y[bi, c, h, l, p]``. The only sequential dependency is the
    per-``(bi, c, h, l)`` inclusive prefix sum ``A_cumsum[l]``, which each thread
    recomputes up to its own ``l`` (cheap, and avoids cross-thread sync).

    The same math as the CPU kernel, for the scalar ``Y[l, p]``:
        - ``A_cumsum[l] = sum_{k <= l} A[k]``
        - ``decay = exp(A_cumsum[l])``
        - ``Y_off[l, p] = decay * sum_n C[l, n] * entering_state[p, n]``
        - ``Y[l, p] = Y_diag[l, p] + Y_off[l, p]``

    Tensor shapes (row-major):
        - ``C``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``entering_state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``Y_diag``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``Y``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        c_LT: Layout of the ``C`` tensor.
        entering_LT: Layout of the ``entering_state`` tensor.
        a_LT: Layout of the ``A`` tensor.
        y_diag_LT: Layout of the ``Y_diag`` tensor.
        y_LT: Layout of the ``Y`` output tensor.

    Args:
        batch_i32: Number of batch elements.
        n_chunks_i32: Number of chunks per sequence.
        n_heads_i32: Number of heads.
        chunk_len_i32: Tokens per chunk (``L``).
        head_dim_i32: Head dimension (``P``).
        state_dim_i32: State dimension (``N``).
        C: Per-token C projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        entering_state: State entering this chunk,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        A: Per-token in-chunk log-decay,
            shape ``[batch, n_chunks, n_heads, L]``.
        Y_diag: Intra-chunk diagonal output,
            shape ``[batch, n_chunks, n_heads, L, P]``.
        Y: Output, shape ``[batch, n_chunks, n_heads, L, P]``.
    """
    var batch = Int(batch_i32)
    var n_chunks = Int(n_chunks_i32)
    var n_heads = Int(n_heads_i32)
    var chunk_len = Int(chunk_len_i32)
    var head_dim = Int(head_dim_i32)
    var state_dim = Int(state_dim_i32)
    var flat_idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total_threads = batch * n_chunks * n_heads * chunk_len * head_dim
    if flat_idx >= total_threads:
        return

    # Decompose the flat thread index into (batch, chunk, head, l, p).
    var p = flat_idx % head_dim
    var l = (flat_idx // head_dim) % chunk_len
    var h = (flat_idx // (head_dim * chunk_len)) % n_heads
    var c = (flat_idx // (head_dim * chunk_len * n_heads)) % n_chunks
    var bi = flat_idx // (head_dim * chunk_len * n_heads * n_chunks)

    # Row-major strides for the flat backing buffers.
    # C: [batch, n_chunks, n_heads, chunk_len, state_dim].
    var c_n_stride = 1
    var c_l_stride = state_dim
    var c_h_stride = chunk_len * c_l_stride
    var c_c_stride = n_heads * c_h_stride
    var c_b_stride = n_chunks * c_c_stride

    # entering_state: [batch, n_chunks, n_heads, head_dim, state_dim].
    var ent_n_stride = 1
    var ent_p_stride = state_dim
    var ent_h_stride = head_dim * ent_p_stride
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride

    # A: [batch, n_chunks, n_heads, chunk_len].
    var a_l_stride = 1
    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    # Y / Y_diag: [batch, n_chunks, n_heads, chunk_len, head_dim].
    var y_p_stride = 1
    var y_l_stride = head_dim
    var y_h_stride = chunk_len * y_l_stride
    var y_c_stride = n_heads * y_h_stride
    var y_b_stride = n_chunks * y_c_stride

    # Inclusive prefix sum of A over [0, l] for this (bi, c, h).
    var a_base = bi * a_b_stride + c * a_c_stride + h * a_h_stride
    var a_cumsum = Float32(0.0)
    for k in range(l + 1):
        a_cumsum += A.ptr[a_base + k * a_l_stride].cast[DType.float32]()
    var decay = exp(a_cumsum)

    # Y_off[l, p] = decay * sum_n C[l, n] * entering_state[p, n].
    var c_base = (
        bi * c_b_stride + c * c_c_stride + h * c_h_stride + l * c_l_stride
    )
    var ent_base = (
        bi * ent_b_stride
        + c * ent_c_stride
        + h * ent_h_stride
        + p * ent_p_stride
    )
    var acc = Float32(0.0)
    for n in range(state_dim):
        var c_val = C.ptr[c_base + n * c_n_stride].cast[DType.float32]()
        var ent_val = entering_state.ptr[ent_base + n * ent_n_stride].cast[
            DType.float32
        ]()
        acc += c_val * ent_val
    var y_off = decay * acc

    # Y[l, p] = Y_diag[l, p] + Y_off[l, p].
    var y_off_flat = (
        bi * y_b_stride
        + c * y_c_stride
        + h * y_h_stride
        + l * y_l_stride
        + p * y_p_stride
    )
    var y_diag_val = Y_diag.ptr[y_off_flat].cast[DType.float32]()
    Y.ptr[y_off_flat] = (y_diag_val + y_off).cast[kernel_dtype]()


# ===----------------------------------------------------------------------=== #
# Static-shape tensor-core path
# ===----------------------------------------------------------------------=== #


def _ssd_combine_decay_gpu[
    kernel_dtype: DType,
    a_LT: TensorLayout,
    decay_LT: TensorLayout,
](
    chunk_len_i32: Int32,
    A: TileTensor[kernel_dtype, a_LT, MutAnyOrigin],
    decay_out: TileTensor[DType.float32, decay_LT, MutAnyOrigin],
):
    """Per-token decay ``decay[slice, l] = exp(A_cumsum[l])`` for every slice.

    One block per ``(batch, chunk, head)`` slice (``block_dim == chunk_len``).
    Computes the inclusive prefix sum of ``A`` over the chunk via a Hillis-Steele
    scan in shared memory, exponentiates, and writes the per-token decay to a
    ``[num_slices, chunk_len]`` scratch buffer (read back by the coalesced add
    kernel). Cheap: ``num_slices * chunk_len`` elements total.
    """
    var chunk_len = Int(chunk_len_i32)
    var slice_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var decay = mem_stack_allocation[
        COMBINE_MAX_CHUNK, Float32, address_space=AddressSpace.SHARED
    ]()
    var a_base = slice_idx * chunk_len
    if tid < chunk_len:
        decay[tid] = A.ptr[a_base + tid].cast[DType.float32]()
    gpu_barrier()

    var offset = 1
    while offset < chunk_len:
        var add: Float32 = 0.0
        if tid < chunk_len and tid >= offset:
            add = decay[tid - offset]
        gpu_barrier()
        if tid < chunk_len and tid >= offset:
            decay[tid] += add
        gpu_barrier()
        offset *= 2

    if tid < chunk_len:
        decay_out.ptr[a_base + tid] = exp(decay[tid])


def _ssd_combine_add_gpu[
    kernel_dtype: DType,
    mt_LT: TensorLayout,
    decay_LT: TensorLayout,
    y_diag_LT: TensorLayout,
    y_LT: TensorLayout,
](
    chunk_len_i32: Int32,
    head_dim_i32: Int32,
    num_slices_i32: Int32,
    M_T: TileTensor[kernel_dtype, mt_LT, MutAnyOrigin],
    decay: TileTensor[DType.float32, decay_LT, MutAnyOrigin],
    Y_diag: TileTensor[kernel_dtype, y_diag_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, y_LT, MutAnyOrigin],
):
    """Coalesced add: ``Y[l, p] = Y_diag[l, p] + decay[l] * M_T[p, l]``.

    Flat element-wise launch -- one thread per ``(slice, l, p)`` output scalar,
    so the grid has ``num_slices * chunk_len * head_dim`` threads (vs the
    one-block-per-slice epilogue, which starved GB10's SMs). Consecutive threads
    own consecutive ``p`` (the contiguous output axis), so both the ``Y_diag``
    read and the ``Y`` write are coalesced. The transposed ``M_T[p, l]`` read is
    strided by ``chunk_len`` across ``p`` but L2-resident.
    """
    var chunk_len = Int(chunk_len_i32)
    var head_dim = Int(head_dim_i32)
    var num_slices = Int(num_slices_i32)
    var flat_idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total = num_slices * chunk_len * head_dim
    if flat_idx >= total:
        return

    var p = flat_idx % head_dim
    var l = (flat_idx // head_dim) % chunk_len
    var slice_idx = flat_idx // (head_dim * chunk_len)

    var d = decay.ptr[slice_idx * chunk_len + l]
    var mt_val = M_T.ptr[
        slice_idx * head_dim * chunk_len + p * chunk_len + l
    ].cast[DType.float32]()
    var y_off = slice_idx * chunk_len * head_dim + l * head_dim + p
    var yd_val = Y_diag.ptr[y_off].cast[DType.float32]()
    Y.ptr[y_off] = (yd_val + d * mt_val).cast[kernel_dtype]()


def ssd_output_recombination_fwd_gpu_static[
    kernel_dtype: DType,
    chunk_len_ct: Int,
    state_dim_ct: Int,
    head_dim_ct: Int,
    c_LT: TensorLayout,
    entering_LT: TensorLayout,
    a_LT: TensorLayout,
    y_diag_LT: TensorLayout,
    y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    C: TileTensor[kernel_dtype, c_LT, MutAnyOrigin],
    entering_state: TileTensor[kernel_dtype, entering_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, a_LT, MutAnyOrigin],
    Y_diag: TileTensor[kernel_dtype, y_diag_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, y_LT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Static-shape, tensor-core variant of ``ssd_output_recombination_fwd_gpu``.

    Computes the output recombination as a batched matmul on tensor cores
    followed by a cheap decay + add epilogue:

        ``M_T[p, l] = sum_n entering_state[p, n] * C[l, n]``   (batched_matmul)
        ``Y[l, p]   = Y_diag[l, p] + exp(A_cumsum[l]) * M_T[p, l]``  (epilogue)

    The decay multiplies an *output* row (``l``), not the reduction dim, so the
    matmul stays a plain ``entering_state @ Cᵀ`` (``transpose_b=True``, no decay
    materialisation). Knowing ``chunk_len`` / ``state_dim`` / ``head_dim`` at
    compile time lets ``batched_matmul`` see ``has_static_NK``; the output has
    ``chunk_len`` (== matmul ``N``) columns and ``state_dim`` (== ``K``) is a
    multiple of 32 and ``>= 128``, so it takes the A100 multistage tensor-core
    batched path (one launch for all ``batch * n_chunks * n_heads`` slices)
    instead of the scalar reduction in ``ssd_output_recombination_fwd_gpu``.

    The scalar kernel remains the all-shape-correct fallback for shapes that
    miss the gate (e.g. ``chunk_len`` not a multiple of 128).

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        chunk_len_ct: Tokens per chunk (``L``), compile-time.
        state_dim_ct: State dimension (``N``), compile-time.
        head_dim_ct: Head dimension (``P``), compile-time.
        c_LT: Layout of the ``C`` tensor.
        entering_LT: Layout of the ``entering_state`` tensor.
        a_LT: Layout of the ``A`` tensor.
        y_diag_LT: Layout of the ``Y_diag`` tensor.
        y_LT: Layout of the ``Y`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        C: Per-token C projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        entering_state: State entering this chunk,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        A: Per-token in-chunk log-decay,
            shape ``[batch, n_chunks, n_heads, L]``.
        Y_diag: Intra-chunk diagonal output,
            shape ``[batch, n_chunks, n_heads, L, P]``.
        Y: Output, shape ``[batch, n_chunks, n_heads, L, P]``.
        ctx: The device context to enqueue work on.
    """
    var num_slices = batch * n_chunks * n_heads

    # Scratch for the off-diagonal matmul result, transposed: [num_slices, P, L].
    var mt_device = ctx.enqueue_create_buffer[kernel_dtype](
        num_slices * head_dim_ct * chunk_len_ct
    )

    # Stage 1: M_T[p, l] = entering_state[p, n] @ C[l, n]ᵀ (batched tensor-core
    # matmul, one launch). Output N == chunk_len (% 128 == 0), K == state_dim
    # (% 32 == 0, >= 128) hits the A100 multistage path.
    var ent3 = TileTensor(
        entering_state.ptr,
        row_major(num_slices, Idx[head_dim_ct], Idx[state_dim_ct]),
    )
    var C3 = TileTensor(
        C.ptr,
        row_major(num_slices, Idx[chunk_len_ct], Idx[state_dim_ct]),
    )
    var MT3 = TileTensor(
        mt_device,
        row_major(num_slices, Idx[head_dim_ct], Idx[chunk_len_ct]),
    )
    with ctx.push_context():
        batched_matmul[target="gpu", transpose_b=True](
            MT3, ent3, C3, context=ctx
        )

    var MT_tt = TileTensor(
        mt_device,
        row_major(batch, n_chunks, n_heads, head_dim_ct, chunk_len_ct),
    )

    # Stage 2: per-token decay = exp(A_cumsum) into a [num_slices, L] scratch
    # (one block per slice, block_dim == L; the only sequential part).
    var decay_device = ctx.enqueue_create_buffer[DType.float32](
        num_slices * chunk_len_ct
    )
    var decay_tt = TileTensor(
        decay_device, row_major(batch, n_chunks, n_heads, chunk_len_ct)
    )
    var decay_compiled = ctx.compile_function[
        _ssd_combine_decay_gpu[kernel_dtype, a_LT, decay_tt.LayoutType]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            decay_compiled,
            Int32(chunk_len_ct),
            A,
            decay_tt,
            grid_dim=(num_slices,),
            block_dim=(chunk_len_ct,),
        )

    # Stage 3: coalesced add Y[l,p] = Y_diag[l,p] + decay[l] * M_T[p,l]
    # (flat element-wise launch -- one thread per output, many small blocks).
    comptime ADD_BLOCK = 256
    var total_out = num_slices * chunk_len_ct * head_dim_ct
    var add_compiled = ctx.compile_function[
        _ssd_combine_add_gpu[
            kernel_dtype,
            MT_tt.LayoutType,
            decay_tt.LayoutType,
            y_diag_LT,
            y_LT,
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            add_compiled,
            Int32(chunk_len_ct),
            Int32(head_dim_ct),
            Int32(num_slices),
            MT_tt,
            decay_tt,
            Y_diag,
            Y,
            grid_dim=(ceildiv(total_out, ADD_BLOCK),),
            block_dim=(ADD_BLOCK,),
        )


# ===----------------------------------------------------------------------=== #
# Fused single-pass tensor-core path
# ===----------------------------------------------------------------------=== #


def _ssd_combine_fused_mma_gpu[
    kernel_dtype: DType,
    L: Int,
    P: Int,
    N: Int,
    BK: Int,
    num_warps_m: Int,
    num_warps_n: Int,
    c_LT: TensorLayout,
    entering_LT: TensorLayout,
    a_LT: TensorLayout,
    y_LT: TensorLayout,
](
    C: TileTensor[kernel_dtype, c_LT, MutAnyOrigin],
    entering_state: TileTensor[kernel_dtype, entering_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, a_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, y_LT, MutAnyOrigin],
):
    """Fused, single-pass tensor-core off-diagonal recombination (one block/slice).

    Computes ``M[l,p] = decay[l] * sum_n C[l,n] * entering_state[p,n]`` for one
    ``(batch, chunk, head)`` slice as a transpose-of-entering tensor-core GEMM,
    **without** materialising the matmul result in DRAM (the cost the 2-pass
    ``..._fwd_gpu_static`` pays). Per K-tile the block streams ``C``/``entering``
    once, folds ``decay[l]`` into the ``C`` (``As``) tile while loading (so the
    decay is on the matmul ``M`` rows, no post-pass), transposes ``entering``
    into the ``Bs`` tile, and accumulates the warp-tiled MMA. The decayed
    ``M[l,p]`` is written to ``Y``; the small ``+ Y_diag`` is a separate
    same-layout coalesced add (``_ssd_combine_add_diag_gpu``).

    Compile-time tiling: block tile is the whole ``[L, P]`` output; warps tile it
    ``num_warps_m x num_warps_n``; ``BK`` is the K (state-dim) step. Slices are
    contiguous in every buffer, so ``slice_idx`` alone indexes them. Mirrors
    ``_ssd_chunk_state_fused_mma_gpu`` in ``ssd_chunk_state.mojo``.
    """
    comptime accum_type = DType.float32
    comptime mma_shape = get_mma_shape[kernel_dtype, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = L // num_warps_m
    comptime WN = P // num_warps_n
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime num_k_mmas = BK // MMA_K
    comptime fs = get_fragment_size[mma_shape]()
    comptime a_frag_size = fs[0]
    comptime b_frag_size = fs[1]
    comptime c_frag_size = fs[2]
    comptime num_threads = num_warps_m * num_warps_n * WARP_SIZE

    var slice_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    var c_base = slice_idx * L * N
    var ent_base = slice_idx * P * N
    var a_base = slice_idx * L
    var y_base = slice_idx * L * P

    # ── Per-token decay into shared memory (Hillis-Steele, then exp). ─────────
    # Stage 4 uses exp(A_cumsum[l]) directly (inclusive prefix sum, no cum_end).
    var decay = mem_stack_allocation[
        COMBINE_MAX_CHUNK, Float32, address_space=AddressSpace.SHARED
    ]()
    if tid < L:
        decay[tid] = A.ptr[a_base + tid].cast[DType.float32]()
    gpu_barrier()
    var offset = 1
    while offset < L:
        var add: Float32 = 0.0
        if tid < L and tid >= offset:
            add = decay[tid - offset]
        gpu_barrier()
        if tid < L and tid >= offset:
            decay[tid] += add
        gpu_barrier()
        offset *= 2
    if tid < L:
        decay[tid] = exp(decay[tid])
    gpu_barrier()

    # ── Shared tiles: As is decay-scaled C [L, BK]; Bs is entering transposed
    # to [BK, P]. ─────────────────────────────────────────────────────────────
    var As = LayoutTensor[
        kernel_dtype,
        Layout.row_major(L, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var Bs = LayoutTensor[
        kernel_dtype,
        Layout.row_major(BK, P),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var c_reg = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )
    var a_reg = LayoutTensor[
        kernel_dtype,
        Layout.row_major(num_m_mmas, a_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var b_reg = LayoutTensor[
        kernel_dtype,
        Layout.row_major(num_n_mmas, b_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    var mma_op = TensorCore[accum_type, kernel_dtype, mma_shape, False]()

    # ── K-loop over the state dimension N ────────────────────────────────────
    var k0 = 0
    while k0 < N:
        # As[l, kl] = decay[l] * C[l, k0+kl] (coalesced read over kl per row l).
        var idx = tid
        while idx < L * BK:
            var l = idx // BK
            var kl = idx % BK
            As[l, kl] = (
                decay[l]
                * C.ptr[c_base + l * N + (k0 + kl)].cast[DType.float32]()
            ).cast[kernel_dtype]()
            idx += num_threads
        # Bs[kl, p] = entering[p, k0+kl] (coalesced read over kl per row p).
        idx = tid
        while idx < P * BK:
            var p = idx // BK
            var kl = idx % BK
            Bs[kl, p] = entering_state.ptr[ent_base + p * N + (k0 + kl)]
            idx += num_threads
        gpu_barrier()

        var a_warp_tile = As.tile[WM, BK](warp_y, 0)
        var b_warp_tile = Bs.tile[BK, WN](0, warp_x)

        comptime for k_mma in range(num_k_mmas):
            mma_op.load_a(a_warp_tile, a_reg.vectorize[1, a_frag_size](), k_mma)
            mma_op.load_b(
                b_warp_tile, b_reg.vectorize[1, b_frag_size](), k_mma, warp_x
            )
            mma_op.mma(
                a_reg.vectorize[1, a_frag_size](),
                b_reg.vectorize[1, b_frag_size](),
                c_reg.vectorize[1, c_frag_size](),
            )
        gpu_barrier()
        k0 += BK

    # ── Store the warp's [WM, WN] output tile (decayed M) to Y. ──────────────
    var y_slice = LayoutTensor[
        kernel_dtype, Layout.row_major(L, P), MutAnyOrigin
    ](Y.ptr + y_base)
    var y_warp_tile = y_slice.tile[WM, WN](warp_y, warp_x)
    copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
        y_warp_tile.vectorize[1, 2](),
        c_reg.vectorize[1, 2]().transpose(),
    )


def _ssd_combine_add_diag_gpu[
    kernel_dtype: DType,
    y_diag_LT: TensorLayout,
    y_LT: TensorLayout,
](
    total_i32: Int32,
    Y_diag: TileTensor[kernel_dtype, y_diag_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, y_LT, MutAnyOrigin],
):
    """Coalesced ``Y[i] += Y_diag[i]`` over every output scalar.

    Both tensors are the same row-major ``[b,nc,h,L,P]`` layout, so this is a
    fully-coalesced element-wise add (no transpose) — far cheaper than the
    transposed-read add of the 2-pass static path.
    """
    var total = Int(total_i32)
    var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if i >= total:
        return
    Y.ptr[i] = (
        Y.ptr[i].cast[DType.float32]() + Y_diag.ptr[i].cast[DType.float32]()
    ).cast[kernel_dtype]()


def ssd_output_recombination_fwd_gpu_fused[
    kernel_dtype: DType,
    chunk_len_ct: Int,
    state_dim_ct: Int,
    head_dim_ct: Int,
    c_LT: TensorLayout,
    entering_LT: TensorLayout,
    a_LT: TensorLayout,
    y_diag_LT: TensorLayout,
    y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    C: TileTensor[kernel_dtype, c_LT, MutAnyOrigin],
    entering_state: TileTensor[kernel_dtype, entering_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, a_LT, MutAnyOrigin],
    Y_diag: TileTensor[kernel_dtype, y_diag_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, y_LT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Fused single-pass tensor-core variant of the stage-4 recombination.

    One block per slice runs ``_ssd_combine_fused_mma_gpu`` (decay folded into
    the matmul, no DRAM round-trip for the off-diagonal result), then a
    coalesced ``Y += Y_diag`` add. Faster than the 2-pass
    ``..._fwd_gpu_static`` (which round-trips ``M_T`` through DRAM and does a
    transposed-read add).

    Requires the fused-MMA tiling constraints: ``chunk_len % (16*num_warps_m)
    == 0``, ``head_dim % (8*num_warps_n) == 0``, ``state_dim % BK == 0``, and
    the launch block (``num_warps_m*num_warps_n*32`` threads) ``>= chunk_len``
    for the decay scan. The 2-pass static path is the fallback for other shapes.
    """
    comptime num_threads = (
        COMBINE_FUSED_WARPS_M * COMBINE_FUSED_WARPS_N * WARP_SIZE
    )
    var num_slices = batch * n_chunks * n_heads

    # Stage 1: fused MMA writes the decayed off-diagonal M[l,p] into Y.
    var fused_compiled = ctx.compile_function[
        _ssd_combine_fused_mma_gpu[
            kernel_dtype,
            chunk_len_ct,
            head_dim_ct,
            state_dim_ct,
            COMBINE_FUSED_BK,
            COMBINE_FUSED_WARPS_M,
            COMBINE_FUSED_WARPS_N,
            c_LT,
            entering_LT,
            a_LT,
            y_LT,
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            fused_compiled,
            C,
            entering_state,
            A,
            Y,
            grid_dim=(num_slices,),
            block_dim=(num_threads,),
        )

    # Stage 2: coalesced Y += Y_diag (same layout, no transpose).
    comptime ADD_BLOCK = 256
    var total_out = num_slices * chunk_len_ct * head_dim_ct
    var add_compiled = ctx.compile_function[
        _ssd_combine_add_diag_gpu[kernel_dtype, y_diag_LT, y_LT]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            add_compiled,
            Int32(total_out),
            Y_diag,
            Y,
            grid_dim=(ceildiv(total_out, ADD_BLOCK),),
            block_dim=(ADD_BLOCK,),
        )
