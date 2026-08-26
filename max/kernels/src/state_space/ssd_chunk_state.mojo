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
"""Structured state space duality (SSD) chunk-state kernel.

This implements stage 2 of the SSD chunk-scan algorithm used by Mamba2: the
"chunk end-state". For each ``(batch, chunk, head)`` it reduces the chunk down
to a single ``[head_dim, state_dim]`` end-state by summing the decay-weighted
outer products of the per-token key-like and value-like projections.

For a chunk of ``L`` tokens with state dim ``N`` and head dim ``P``:

- Inputs: ``B: [L, N]``, ``X: [L, P]``, ``A: [L]`` (a scalar decay per token).
- Cumulative decay: ``A_cumsum[l] = sum_{k=0..l} A[k]``.
- Per-token decay toward the chunk end:
  ``decay[l] = exp(A_cumsum[L-1] - A_cumsum[l])``. Note ``decay[L-1] = 1``.
- End-state: ``state[p, n] = sum_{l=0..L-1} B[l, n] * decay[l] * X[l, p]``.

This matches the reference einsum
``state = einsum("ln,l,lp->pn", B, decay, X)`` (stage 2 of
``ssd_minimal_discrete``), producing one end-state per ``(batch, chunk, head)``.
"""

from layout import (
    Idx,
    Layout,
    LayoutTensor,
    TensorLayout,
    TileTensor,
    row_major,
)
from std.math import exp
from max.gpu.sync import barrier as gpu_barrier
from std.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.memory import unsafe_stack_allocation as mem_stack_allocation
from std.utils.index import IndexList
from linalg.bmm import batched_matmul
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape

# Stride helpers for indexing the flat row-major buffers.
comptime Strides4D = IndexList[4]
comptime Strides5D = IndexList[5]

# Upper bound on chunk_len for the shared-memory decay scratch (one Float32
# per token). The host launches one block per slice with block_dim == chunk_len
# (see ``ssd_chunk_state_fwd_gpu``), so this also bounds the launch block size.
# Mirrors ``SCAN_MAX_CHUNK`` in ``ssd_chunk.mojo``.
comptime STATE_MAX_CHUNK = 1024


def _row_major_strides_4d(d0: Int, d1: Int, d2: Int, d3: Int) -> Strides4D:
    """Row-major strides for a 4D shape ``(d0, d1, d2, d3)``."""
    return Strides4D(d1 * d2 * d3, d2 * d3, d3, 1)


def _row_major_strides_5d(
    d0: Int, d1: Int, d2: Int, d3: Int, d4: Int
) -> Strides5D:
    """Row-major strides for a 5D shape ``(d0, d1, d2, d3, d4)``."""
    return Strides5D(d1 * d2 * d3 * d4, d2 * d3 * d4, d3 * d4, d4, 1)


def ssd_chunk_state_fwd_cpu[
    kernel_dtype: DType,
    B_layout: Layout,
    X_layout: Layout,
    A_layout: Layout,
    state_layout: Layout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    X: LayoutTensor[kernel_dtype, X_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    state: LayoutTensor[kernel_dtype, state_layout, MutAnyOrigin],
):
    """Compute the SSD chunk end-state on CPU.

    Operates per ``(batch, chunk, head)`` over the supplied row-major tensors,
    reducing each chunk down to a single ``[head_dim, state_dim]`` end-state.

    Tensor shapes (row-major):
        - ``B``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``X``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        B_layout: Layout of the ``B`` tensor.
        X_layout: Layout of the ``X`` tensor.
        A_layout: Layout of the ``A`` tensor.
        state_layout: Layout of the ``state`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        state_dim: State dimension (``N``).
        head_dim: Head dimension (``P``).
        B: Key-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        X: Value-like input, shape ``[batch, n_chunks, n_heads, L, P]``.
        A: Per-token scalar decay, shape ``[batch, n_chunks, n_heads, L]``.
        state: Output end-state, shape ``[batch, n_chunks, n_heads, P, N]``.
    """
    # Row-major strides for the flat backing buffers.
    var b_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, state_dim
    )
    var x_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, head_dim
    )
    var a_strides = _row_major_strides_4d(batch, n_chunks, n_heads, chunk_len)
    var state_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, head_dim, state_dim
    )

    for bi in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                # Base offsets for this (batch, chunk, head) slice.
                var b_base = (
                    bi * b_strides[0] + c * b_strides[1] + h * b_strides[2]
                )
                var x_base = (
                    bi * x_strides[0] + c * x_strides[1] + h * x_strides[2]
                )
                var a_base = (
                    bi * a_strides[0] + c * a_strides[1] + h * a_strides[2]
                )
                var state_base = (
                    bi * state_strides[0]
                    + c * state_strides[1]
                    + h * state_strides[2]
                )

                # Inclusive prefix sum of the per-token decay: A_cumsum[l].
                var a_cumsum = List[Float32](length=chunk_len, fill=0.0)
                var running = Float32(0.0)
                for l in range(chunk_len):
                    var a_off = a_base + l * a_strides[3]
                    running += A.ptr[a_off].cast[DType.float32]()
                    a_cumsum[l] = running

                # decay[l] = exp(A_cumsum[L-1] - A_cumsum[l]); decay[L-1] = 1.
                var cum_end = a_cumsum[chunk_len - 1]
                var decay = List[Float32](length=chunk_len, fill=0.0)
                for l in range(chunk_len):
                    decay[l] = exp(cum_end - a_cumsum[l])

                # state[p, n] = sum_l B[l, n] * decay[l] * X[l, p]
                for p in range(head_dim):
                    for n in range(state_dim):
                        var acc = Float32(0.0)
                        for l in range(chunk_len):
                            var bv = B.ptr[
                                b_base + l * b_strides[3] + n * b_strides[4]
                            ].cast[DType.float32]()
                            var xv = X.ptr[
                                x_base + l * x_strides[3] + p * x_strides[4]
                            ].cast[DType.float32]()
                            acc += bv * decay[l] * xv
                        var state_off = (
                            state_base
                            + p * state_strides[3]
                            + n * state_strides[4]
                        )
                        state.ptr[state_off] = acc.cast[kernel_dtype]()


# ===----------------------------------------------------------------------=== #
# GPU Implementation
# ===----------------------------------------------------------------------=== #


def ssd_chunk_state_fwd_gpu[
    kernel_dtype: DType,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    state_LT: TensorLayout,
](
    batch_i32: Int32,
    n_chunks_i32: Int32,
    n_heads_i32: Int32,
    chunk_len_i32: Int32,
    state_dim_i32: Int32,
    head_dim_i32: Int32,
    B: TileTensor[kernel_dtype, B_LT, MutUntrackedOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutUntrackedOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutUntrackedOrigin],
    state: TileTensor[kernel_dtype, state_LT, MutUntrackedOrigin],
):
    """Compute the SSD chunk end-state on GPU.

    Launched over a 2D grid ``(total_slices, n_p_tiles)`` with
    ``block_dim == chunk_len``. ``block_idx.x`` selects the ``(batch, chunk,
    head)`` slice; ``block_idx.y`` selects a tile of ``head_dim`` rows (the tile
    height is ``ceildiv(head_dim, grid_dim.y)``, so the kernel adapts to
    whatever ``grid_dim.y`` the host picks — ``grid_dim.y == 1`` means one block
    computes the whole slice). Splitting each slice across several blocks raises
    the block count and warps-in-flight on GB10's many small SMs.

    Each block first computes the per-token decay
    ``decay[l] = exp(A_cumsum[L-1] - A_cumsum[l])`` once, cooperatively, into
    shared memory (a Hillis-Steele inclusive prefix sum over ``A`` followed by
    the exp; recomputed per p-tile, which is cheap), then the block's threads
    split its tile of the ``head_dim * state_dim`` output elements and each
    accumulates its end-state entry over ``l``.

    This replaces the original one-thread-per-slice port, which launched a
    single block (occupancy/latency bound: ~0.1% compute, ~4% memory of GB10
    peak) and recomputed ``decay[l]`` for every one of the ``head_dim *
    state_dim`` outputs. The decay is now computed once per slice and shared.
    It is still not tensor-core optimized; the reduction reads ``B``/``X`` from
    global memory (the kernel is far from bandwidth bound).

    The same math as the CPU kernel:
        - ``A_cumsum[l] = sum_{k<=l} A[k]``
        - ``decay[l] = exp(A_cumsum[L-1] - A_cumsum[l])``
        - ``state[p, n] = sum_l B[l, n] * decay[l] * X[l, p]``

    Tensor shapes (row-major):
        - ``B``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``X``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        B_LT: Layout of the ``B`` tensor.
        X_LT: Layout of the ``X`` tensor.
        A_LT: Layout of the ``A`` tensor.
        state_LT: Layout of the ``state`` output tensor.

    Args:
        batch_i32: Number of batch elements.
        n_chunks_i32: Number of chunks per sequence.
        n_heads_i32: Number of heads.
        chunk_len_i32: Tokens per chunk (``L``).
        state_dim_i32: State dimension (``N``).
        head_dim_i32: Head dimension (``P``).
        B: Key-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        X: Value-like input, shape ``[batch, n_chunks, n_heads, L, P]``.
        A: Per-token scalar decay, shape ``[batch, n_chunks, n_heads, L]``.
        state: Output end-state, shape ``[batch, n_chunks, n_heads, P, N]``.
    """
    var batch = Int(batch_i32)
    var n_chunks = Int(n_chunks_i32)
    var n_heads = Int(n_heads_i32)
    var chunk_len = Int(chunk_len_i32)
    var state_dim = Int(state_dim_i32)
    var head_dim = Int(head_dim_i32)
    var slice_idx = Int(block_idx.x)
    var total_slices = batch * n_chunks * n_heads
    if slice_idx >= total_slices:
        return

    var tid = Int(thread_idx.x)
    var nthreads = Int(block_dim.x)

    # Decompose the flat slice index into (batch, chunk, head).
    var h = slice_idx % n_heads
    var c = (slice_idx // n_heads) % n_chunks
    var bi = slice_idx // (n_heads * n_chunks)

    # Row-major strides for the flat backing buffers.
    var b_n_stride = 1
    var b_l_stride = state_dim
    var b_h_stride = chunk_len * b_l_stride
    var b_c_stride = n_heads * b_h_stride
    var b_b_stride = n_chunks * b_c_stride

    var x_p_stride = 1
    var x_l_stride = head_dim
    var x_h_stride = chunk_len * x_l_stride
    var x_c_stride = n_heads * x_h_stride
    var x_b_stride = n_chunks * x_c_stride

    var a_l_stride = 1
    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var state_n_stride = 1
    var state_p_stride = state_dim
    var state_h_stride = head_dim * state_p_stride
    var state_c_stride = n_heads * state_h_stride
    var state_b_stride = n_chunks * state_c_stride

    var b_base = bi * b_b_stride + c * b_c_stride + h * b_h_stride
    var x_base = bi * x_b_stride + c * x_c_stride + h * x_h_stride
    var a_base = bi * a_b_stride + c * a_c_stride + h * a_h_stride
    var state_base = (
        bi * state_b_stride + c * state_c_stride + h * state_h_stride
    )

    # ── Per-token decay into shared memory (computed once per slice) ──────────
    # decay[l] = exp(A_cumsum[L-1] - A_cumsum[l]); A_cumsum is the inclusive
    # prefix sum of A. Hillis-Steele scan in shared memory: block_dim ==
    # chunk_len threads, one element each, log2(chunk_len) doubling steps.
    var decay = mem_stack_allocation[
        STATE_MAX_CHUNK, Float32, address_space=AddressSpace.SHARED
    ]()

    if tid < chunk_len:
        decay[tid] = A.ptr[a_base + tid * a_l_stride].cast[DType.float32]()
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

    # decay[] now holds A_cumsum[l]. Convert to the decay weight in place.
    # Every thread reads cum_end before any thread overwrites its slot.
    var cum_end = decay[chunk_len - 1]
    gpu_barrier()
    if tid < chunk_len:
        decay[tid] = exp(cum_end - decay[tid])
    gpu_barrier()

    # ── End-state reduction: state[p, n] = sum_l B[l, n] * decay[l] * X[l, p] ─
    # This block owns p-rows [p_start, p_end); its threads split that tile's
    # (p_end - p_start) * state_dim output elements.
    var n_p_tiles = Int(grid_dim.y)
    var p_per_tile = (head_dim + n_p_tiles - 1) // n_p_tiles
    var p_start = Int(block_idx.y) * p_per_tile
    var p_end = min(p_start + p_per_tile, head_dim)
    if p_start >= head_dim:
        return

    var n_out = (p_end - p_start) * state_dim
    var idx = tid
    while idx < n_out:
        var p = p_start + idx // state_dim
        var n = idx % state_dim
        var acc = Float32(0.0)
        for l in range(chunk_len):
            var bv = B.ptr[b_base + l * b_l_stride + n * b_n_stride].cast[
                DType.float32
            ]()
            var xv = X.ptr[x_base + l * x_l_stride + p * x_p_stride].cast[
                DType.float32
            ]()
            acc += bv * decay[l] * xv
        var state_off = state_base + p * state_p_stride + n * state_n_stride
        state.ptr[state_off] = acc.cast[kernel_dtype]()
        idx += nthreads


# ===----------------------------------------------------------------------=== #
# Tensor-core path: decay-weighted transpose + batched matmul
# ===----------------------------------------------------------------------=== #


def _ssd_chunk_state_decay_xt_gpu[
    kernel_dtype: DType,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    xdt_LT: TensorLayout,
](
    batch_i32: Int32,
    n_chunks_i32: Int32,
    n_heads_i32: Int32,
    chunk_len_i32: Int32,
    head_dim_i32: Int32,
    X: TileTensor[kernel_dtype, X_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutAnyOrigin],
    Xd_T: TileTensor[kernel_dtype, xdt_LT, MutAnyOrigin],
):
    """Materialise the decay-weighted, transposed value tensor.

    One block per ``(batch, chunk, head)`` slice (``block_dim == chunk_len``).
    Computes the per-token decay ``decay[l] = exp(A_cumsum[L-1] - A_cumsum[l])``
    once into shared memory (same scan as ``ssd_chunk_state_fwd_gpu``), then
    writes the **transposed** product

        ``Xd_T[slice, p, l] = decay[l] * X[slice, l, p]``

    into a contiguous ``[num_slices, head_dim, chunk_len]`` buffer. Transposing
    here puts the reduction dim ``l`` last in the left matmul operand, so
    ``state = Xd_T @ B`` (``[P,L] @ [L,N] -> [P,N]``) is a plain
    ``batched_matmul`` with no transpose — and its output ``N`` columns hit the
    A100 tensor-core gate. Thread ``tid`` owns token ``l = tid`` and loops over
    ``p`` so consecutive threads write consecutive ``l`` (coalesced).
    """
    var batch = Int(batch_i32)
    var n_chunks = Int(n_chunks_i32)
    var n_heads = Int(n_heads_i32)
    var chunk_len = Int(chunk_len_i32)
    var head_dim = Int(head_dim_i32)
    var slice_idx = Int(block_idx.x)
    var total_slices = batch * n_chunks * n_heads
    if slice_idx >= total_slices:
        return

    var tid = Int(thread_idx.x)

    var h = slice_idx % n_heads
    var c = (slice_idx // n_heads) % n_chunks
    var bi = slice_idx // (n_heads * n_chunks)

    var x_l_stride = head_dim
    var x_h_stride = chunk_len * x_l_stride
    var x_c_stride = n_heads * x_h_stride
    var x_b_stride = n_chunks * x_c_stride

    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var x_base = bi * x_b_stride + c * x_c_stride + h * x_h_stride
    var a_base = bi * a_b_stride + c * a_c_stride + h * a_h_stride

    # Per-token decay into shared memory (Hillis-Steele inclusive prefix sum of
    # A, then exp) — identical to ssd_chunk_state_fwd_gpu.
    var decay = mem_stack_allocation[
        STATE_MAX_CHUNK, Float32, address_space=AddressSpace.SHARED
    ]()

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

    var cum_end = decay[chunk_len - 1]
    gpu_barrier()
    if tid < chunk_len:
        decay[tid] = exp(cum_end - decay[tid])
    gpu_barrier()

    # Xd_T[slice, p, l] = decay[l] * X[slice, l, p]. Output is row-major
    # [num_slices, head_dim, chunk_len]; this slice starts at slice*P*L.
    if tid < chunk_len:
        var d = decay[tid]
        var xdt_base = slice_idx * head_dim * chunk_len
        for p in range(head_dim):
            var xv = X.ptr[x_base + tid * x_l_stride + p].cast[DType.float32]()
            Xd_T.ptr[xdt_base + p * chunk_len + tid] = (d * xv).cast[
                kernel_dtype
            ]()


def ssd_chunk_state_fwd_gpu_static[
    kernel_dtype: DType,
    chunk_len_ct: Int,
    state_dim_ct: Int,
    head_dim_ct: Int,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    state_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    B: TileTensor[kernel_dtype, B_LT, MutAnyOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutAnyOrigin],
    state: TileTensor[kernel_dtype, state_LT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Static-shape, tensor-core variant of ``ssd_chunk_state_fwd_gpu``.

    Computes the chunk end-state as a batched matmul on tensor cores:

        ``Xd_T[p, l] = decay[l] * X[l, p]``   (decay kernel, one launch)
        ``state[p, n] = sum_l Xd_T[p, l] * B[l, n] = (Xd_T @ B)[p, n]``

    Knowing ``chunk_len`` / ``state_dim`` / ``head_dim`` at compile time lets
    ``batched_matmul`` see ``has_static_NK`` and, since the output has
    ``state_dim`` (== matmul ``N``) columns and ``chunk_len`` (== ``K``) is a
    multiple of 32 and ``>= 128``, pick its A100 multistage tensor-core batched
    path (one launch for all ``batch * n_chunks * n_heads`` slices) instead of
    the scalar reduction in ``ssd_chunk_state_fwd_gpu``.

    The scalar kernel remains the all-shape-correct fallback for shapes that
    miss the gate (e.g. ``state_dim`` not a multiple of 128).

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        chunk_len_ct: Tokens per chunk (``L``), compile-time.
        state_dim_ct: State dimension (``N``), compile-time.
        head_dim_ct: Head dimension (``P``), compile-time.
        B_LT: Layout of the ``B`` tensor.
        X_LT: Layout of the ``X`` tensor.
        A_LT: Layout of the ``A`` tensor.
        state_LT: Layout of the ``state`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        B: Key-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        X: Value-like input, shape ``[batch, n_chunks, n_heads, L, P]``.
        A: Per-token scalar decay, shape ``[batch, n_chunks, n_heads, L]``.
        state: Output end-state, shape ``[batch, n_chunks, n_heads, P, N]``.
        ctx: The device context to enqueue work on.
    """
    var num_slices = batch * n_chunks * n_heads

    # Scratch for the decay-weighted, transposed values: [num_slices, P, L].
    var xdt_device = ctx.enqueue_create_buffer[kernel_dtype](
        num_slices * head_dim_ct * chunk_len_ct
    )
    var Xd_T = TileTensor(
        xdt_device,
        row_major(batch, n_chunks, n_heads, head_dim_ct, chunk_len_ct),
    )

    # Stage 1: Xd_T[p, l] = decay[l] * X[l, p] (one block per slice).
    var decay_compiled = ctx.compile_function[
        _ssd_chunk_state_decay_xt_gpu[
            kernel_dtype,
            X_LT,
            A_LT,
            Xd_T.LayoutType,
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            decay_compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len_ct),
            Int32(head_dim_ct),
            X,
            A,
            Xd_T,
            grid_dim=(num_slices,),
            block_dim=(chunk_len_ct,),
        )

    # Stage 2: state = Xd_T @ B (batched tensor-core matmul, one launch).
    # state[P, N] = Xd_T[P, L] @ B[L, N]; static dims let batched_matmul take
    # the A100 multistage path (c_n=N % 128 == 0, a_k=L % 32 == 0, L >= 128).
    var Xd_T3 = TileTensor(
        xdt_device,
        row_major(num_slices, Idx[head_dim_ct], Idx[chunk_len_ct]),
    )
    var B3 = TileTensor(
        B.ptr,
        row_major(num_slices, Idx[chunk_len_ct], Idx[state_dim_ct]),
    )
    var state3 = TileTensor(
        state.ptr,
        row_major(num_slices, Idx[head_dim_ct], Idx[state_dim_ct]),
    )

    with ctx.push_context():
        batched_matmul[target="gpu", transpose_b=False](
            state3, Xd_T3, B3, context=ctx
        )


# ===----------------------------------------------------------------------=== #
# Fused single-pass tensor-core path
# ===----------------------------------------------------------------------=== #

# Warp grid over the per-slice [head_dim, state_dim] output tile. 2x4 warps =
# 8 warps = 256 threads, which also covers chunk_len=256 for the decay scan.
comptime STATE_FUSED_WARPS_M = 2
comptime STATE_FUSED_WARPS_N = 4
# K (chunk-length) step for the fused MMA. Tuned on GB10 at the Mamba2 profile:
# BK=16 is the sweet spot (nc=4: 0.033 ms) — smaller shared tiles overlap the
# global load with the MMA better at this L2-resident size. 8/32/64 all gave
# 0.043/0.043/0.050 ms. Must be a multiple of MMA_K (8) and divide chunk_len.
comptime STATE_FUSED_BK = 16
# Split each slice's state_dim output across this many blocks (grid.y). Tried as
# a "more blocks" occupancy lever, but it REGRESSED (n_split=2: 0.043 -> 0.069 ms
# at the Mamba2 profile) because each N tile re-reads all of X — the extra
# traffic outweighs the occupancy gain. Kept as infra; default 1 (no split).
comptime STATE_FUSED_N_SPLIT = 1


def _ssd_chunk_state_fused_mma_gpu[
    kernel_dtype: DType,
    P: Int,
    N: Int,
    L: Int,
    BK: Int,
    num_warps_m: Int,
    num_warps_n: Int,
    n_split: Int,
    X_LT: TensorLayout,
    B_LT: TensorLayout,
    A_LT: TensorLayout,
    state_LT: TensorLayout,
](
    X: TileTensor[kernel_dtype, X_LT, MutAnyOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutAnyOrigin],
    state: TileTensor[kernel_dtype, state_LT, MutAnyOrigin],
):
    """Fused, single-pass tensor-core chunk end-state (one block per slice).

    Computes ``state[p,n] = sum_l decay[l] * X[l,p] * B[l,n]`` for one
    ``(batch, chunk, head)`` slice as a transpose-A tensor-core GEMM, **without**
    materialising the decay-weighted transpose in DRAM (the cost the 2-pass
    ``ssd_chunk_state_fwd_gpu_static`` pays). Per K-tile the block streams
    ``X``/``B`` from global once, applies ``decay[l]`` while transposing ``X``
    into a shared tile (``As[p, l]``), and accumulates the warp-tiled MMA. This
    mirrors what Triton's ``_chunk_state_fwd`` does in a single kernel.

    Compile-time tiling: block tile is the whole ``[P, N]`` output; warps tile
    it ``num_warps_m x num_warps_n``; ``BK`` is the K (chunk-length) step. Slices
    are contiguous in every buffer, so ``slice_idx`` alone indexes them.
    """
    comptime accum_type = DType.float32
    comptime mma_shape = get_mma_shape[kernel_dtype, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    # This block computes a [P, Nblk] tile; block_idx.y picks the N tile.
    comptime Nblk = N // n_split
    comptime WM = P // num_warps_m
    comptime WN = Nblk // num_warps_n
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime num_k_mmas = BK // MMA_K
    comptime fs = get_fragment_size[mma_shape]()
    comptime a_frag_size = fs[0]
    comptime b_frag_size = fs[1]
    comptime c_frag_size = fs[2]
    comptime num_threads = num_warps_m * num_warps_n * WARP_SIZE

    var slice_idx = Int(block_idx.x)
    var n_tile = Int(block_idx.y)
    var n_offset = n_tile * Nblk
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    var x_base = slice_idx * L * P
    var b_base = slice_idx * L * N
    var a_base = slice_idx * L
    var state_base = slice_idx * P * N

    # ── Per-token decay into shared memory (Hillis-Steele, then exp) ──────────
    var decay = mem_stack_allocation[
        STATE_MAX_CHUNK, Float32, address_space=AddressSpace.SHARED
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
    var cum_end = decay[L - 1]
    gpu_barrier()
    if tid < L:
        decay[tid] = exp(cum_end - decay[tid])
    gpu_barrier()

    # ── Shared tiles: As is decay-scaled X transposed to [P, BK]; Bs is [BK, N].
    var As = LayoutTensor[
        kernel_dtype,
        Layout.row_major(P, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var Bs = LayoutTensor[
        kernel_dtype,
        Layout.row_major(BK, Nblk),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # ── Register tiles ───────────────────────────────────────────────────────
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

    # ── K-loop over chunk length ─────────────────────────────────────────────
    var k0 = 0
    while k0 < L:
        # Stage As[p, kl] = decay[k0+kl] * X[k0+kl, p] (coalesced read over p).
        var idx = tid
        while idx < P * BK:
            var kl = idx // P
            var p = idx % P
            As[p, kl] = (
                decay[k0 + kl]
                * X.ptr[x_base + (k0 + kl) * P + p].cast[DType.float32]()
            ).cast[kernel_dtype]()
            idx += num_threads
        # Stage Bs[kl, n] = B[k0+kl, n_offset+n] (this block's N tile).
        idx = tid
        while idx < BK * Nblk:
            var kl = idx // Nblk
            var nn = idx % Nblk
            Bs[kl, nn] = B.ptr[b_base + (k0 + kl) * N + n_offset + nn]
            idx += num_threads
        gpu_barrier()

        var a_warp_tile = As.tile[WM, BK](warp_y, 0)
        var b_warp_tile = Bs.tile[BK, WN](0, warp_x)

        # NOTE: As/Bs are written to shared memory unswizzled, so load_a/load_b
        # must read unswizzled too (no swizzle arg). Adding a swizzle would
        # require applying the same permutation on the shared-memory store.
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

    # ── Store the warp's [WM, WN] output tile to global state ────────────────
    # The warp's global N-tile index folds in this block's N offset.
    var c_slice = LayoutTensor[
        kernel_dtype, Layout.row_major(P, N), MutAnyOrigin
    ](state.ptr + state_base)
    var c_warp_tile = c_slice.tile[WM, WN](
        warp_y, n_tile * num_warps_n + warp_x
    )
    copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
        c_warp_tile.vectorize[1, 2](),
        c_reg.vectorize[1, 2]().transpose(),
    )


def ssd_chunk_state_fwd_gpu_fused[
    kernel_dtype: DType,
    chunk_len_ct: Int,
    state_dim_ct: Int,
    head_dim_ct: Int,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    state_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    B: TileTensor[kernel_dtype, B_LT, MutAnyOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutAnyOrigin],
    state: TileTensor[kernel_dtype, state_LT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Fused single-pass tensor-core chunk end-state (Triton-parity target).

    One block per ``(batch, chunk, head)`` slice runs
    ``_ssd_chunk_state_fused_mma_gpu``, which streams ``X``/``B`` once, applies
    decay while transposing ``X`` in shared memory, and accumulates the
    end-state on tensor cores — no DRAM round-trip for the decayed/transposed
    values (unlike ``ssd_chunk_state_fwd_gpu_static``).

    Requires ``head_dim % (16 * num_warps_m) == 0``,
    ``state_dim % (8 * num_warps_n) == 0``, ``chunk_len % BK == 0``, and the
    launch block (``num_warps_m * num_warps_n * 32`` threads) ``>= chunk_len``
    for the shared-memory decay scan. The static batched-matmul path
    (``ssd_chunk_state_fwd_gpu_static``) is the fallback for other shapes.
    """
    comptime num_threads = (
        STATE_FUSED_WARPS_M * STATE_FUSED_WARPS_N * WARP_SIZE
    )
    var num_slices = batch * n_chunks * n_heads

    var compiled = ctx.compile_function[
        _ssd_chunk_state_fused_mma_gpu[
            kernel_dtype,
            head_dim_ct,
            state_dim_ct,
            chunk_len_ct,
            STATE_FUSED_BK,
            STATE_FUSED_WARPS_M,
            STATE_FUSED_WARPS_N,
            STATE_FUSED_N_SPLIT,
            X_LT,
            B_LT,
            A_LT,
            state_LT,
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            compiled,
            X,
            B,
            A,
            state,
            grid_dim=(num_slices, STATE_FUSED_N_SPLIT),
            block_dim=(num_threads,),
        )
