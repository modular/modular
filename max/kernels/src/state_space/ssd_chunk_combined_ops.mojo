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
"""Fused SSD chunk-scan operation registration for Mamba2.

This module registers ``ssd_chunk_scan_combined``, which stitches the four
verified SSD stage kernels — intra-chunk diagonal, chunk end-state, inter-chunk
recurrence, and output recombination — into a single MAX graph op matching the
torch ``mamba_chunk_scan_combined`` contract:

    op(x, dt, A, B, C, chunk_size)
        == ssd_minimal_discrete(x * dt[..., None], A_h * dt, B, C, chunk_size)

with all stage 1-4 work happening behind the registration boundary.

Inputs (row-major):
    - ``x``: ``[batch, seqlen, n_heads, head_dim]`` — value-like input ``X``.
    - ``dt``: ``[batch, seqlen, n_heads]`` — per-token time delta.
    - ``A``: ``[n_heads]`` — scalar log-decay per head.
    - ``B``: ``[batch, seqlen, n_heads, state_dim]``.
    - ``C``: ``[batch, seqlen, n_heads, state_dim]``.

Outputs:
    - ``Y: [batch, seqlen, n_heads, head_dim]`` (same dtype as inputs).
    - ``final_state: [batch, n_heads, head_dim, state_dim]`` — the SSM state
      after the last chunk, used to seed the decode (``selective_scan_update``)
      cache so step-mode continuation picks up where prefill left off.

The op currently assumes ``ngroups == n_heads`` (no group broadcasting yet) and
that ``seqlen`` divides evenly by ``chunk_size``; both restrictions match the
upstream Mamba2 reference at the dimensions we care about for prefill.
"""

import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from std.math import ceildiv
from std.utils.index import Index, IndexList

from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TensorLayout,
    TileTensor,
    row_major,
)

from state_space.ssd_chunk import (
    ssd_intra_chunk_fwd_cpu,
    ssd_intra_chunk_fwd_gpu,
)
from state_space.ssd_chunk_state import (
    ssd_chunk_state_fwd_cpu,
    ssd_chunk_state_fwd_gpu,
    ssd_chunk_state_fwd_gpu_fused,
)
from state_space.ssd_chunk_scan import (
    ssd_chunk_scan_fwd_cpu,
    ssd_chunk_scan_fwd_gpu,
)
from state_space.ssd_chunk_combine import (
    ssd_output_recombination_fwd_cpu,
    ssd_output_recombination_fwd_gpu,
    ssd_output_recombination_fwd_gpu_fused,
)

# Block size for the elementwise precompute/postprocess and the scan/combine
# device-kernel launches (matches the per-stage GPU tests).
comptime _SSD_GPU_BLOCK = 256


# ===----------------------------------------------------------------------=== #
# Pre-stage helpers (CPU): discretization + reshape
# ===----------------------------------------------------------------------=== #


def _ssd_combined_precompute_cpu[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    n_chunks: Int,
    chunk_size: Int,
    x_ptr: UnsafePointer[Scalar[dtype], ...],
    dt_ptr: UnsafePointer[Scalar[dtype], ...],
    A_ptr: UnsafePointer[Scalar[dtype], ...],
    B_ptr: UnsafePointer[Scalar[dtype], ...],
    C_ptr: UnsafePointer[Scalar[dtype], ...],
    X_disc_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    A_disc_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    B_chunk_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    C_chunk_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    chunk_decays_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
):
    """Reshape inputs into chunked form and discretize on CPU.

    Inputs:
        x:  ``[batch, seqlen, n_heads, head_dim]``
        dt: ``[batch, seqlen, n_heads]``
        A:  ``[n_heads]`` (scalar decay per head)
        B:  ``[batch, seqlen, n_heads, state_dim]``
        C:  ``[batch, seqlen, n_heads, state_dim]``

    Produces:
        X_disc:       ``[batch, n_chunks, n_heads, chunk_size, head_dim]``
                      with ``X_disc[b, c, h, l, p] = x[b, c*L+l, h, p] *
                      dt[b, c*L+l, h]``.
        A_disc:       ``[batch, n_chunks, n_heads, chunk_size]`` with
                      ``A_disc[b, c, h, l] = A[h] * dt[b, c*L+l, h]``.
        B_chunk:      reshape of B into chunked form (no value change).
        C_chunk:      reshape of C into chunked form (no value change).
        chunk_decays: ``[batch, n_chunks, n_heads]``,
                      ``= sum over l of A_disc[b, c, h, l]``.
    """
    # Note: all input/output buffers are row-major contiguous.
    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var a_h = A_ptr[h].cast[DType.float32]()
                var chunk_sum = Float32(0.0)
                for l in range(chunk_size):
                    var t = c * chunk_size + l
                    var dt_off = (b * seqlen + t) * n_heads + h
                    var dt_val = dt_ptr[dt_off].cast[DType.float32]()
                    var a_disc_val = a_h * dt_val
                    chunk_sum += a_disc_val

                    var a_disc_off = (
                        (b * n_chunks + c) * n_heads + h
                    ) * chunk_size + l
                    A_disc_ptr[a_disc_off] = a_disc_val.cast[dtype]()

                    # X_disc[b, c, h, l, p] = x[b, t, h, p] * dt[b, t, h]
                    var x_in_base = ((b * seqlen + t) * n_heads + h) * head_dim
                    var x_out_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * head_dim
                    for p in range(head_dim):
                        var x_val = x_ptr[x_in_base + p].cast[DType.float32]()
                        X_disc_ptr[x_out_base + p] = (x_val * dt_val).cast[
                            dtype
                        ]()

                    # B_chunk / C_chunk: pure reshape (same value).
                    var bc_in_base = (
                        (b * seqlen + t) * n_heads + h
                    ) * state_dim
                    var bc_out_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * state_dim
                    for n in range(state_dim):
                        B_chunk_ptr[bc_out_base + n] = B_ptr[bc_in_base + n]
                        C_chunk_ptr[bc_out_base + n] = C_ptr[bc_in_base + n]

                var cd_off = (b * n_chunks + c) * n_heads + h
                chunk_decays_ptr[cd_off] = chunk_sum.cast[dtype]()


def _ssd_combined_postprocess_cpu[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    n_chunks: Int,
    chunk_size: Int,
    Y_chunked_ptr: UnsafePointer[Scalar[dtype], ...],
    Y_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
):
    """Flatten ``Y_chunked: [b, nc, h, L, P]`` back into ``Y: [b, S, h, P]``."""
    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                for l in range(chunk_size):
                    var t = c * chunk_size + l
                    var y_in_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * head_dim
                    var y_out_base = (b * seqlen + t) * n_heads * head_dim + (
                        h * head_dim
                    )
                    for p in range(head_dim):
                        Y_ptr[y_out_base + p] = Y_chunked_ptr[y_in_base + p]


# ===----------------------------------------------------------------------=== #
# CPU dispatch — runs all four stages
# ===----------------------------------------------------------------------=== #


def _ssd_chunk_scan_combined_cpu[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    chunk_size: Int,
    x_ptr: UnsafePointer[Scalar[dtype], ...],
    dt_ptr: UnsafePointer[Scalar[dtype], ...],
    A_ptr: UnsafePointer[Scalar[dtype], ...],
    B_ptr: UnsafePointer[Scalar[dtype], ...],
    C_ptr: UnsafePointer[Scalar[dtype], ...],
    Y_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    final_state_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
) raises:
    """Run the full SSD chunk-scan-combined on CPU.

    ``final_state_ptr`` is filled with the post-last-chunk SSM state, shape
    ``[batch, n_heads, head_dim, state_dim]`` row-major. Callers use it to seed
    the step-mode SSM cache; pass a dummy buffer of the right size if the
    state is not needed.
    """
    if seqlen % chunk_size != 0:
        raise Error(
            "ssd_chunk_scan_combined: seqlen must be divisible by chunk_size"
        )
    var n_chunks = seqlen // chunk_size

    # Heap allocations for intermediates. Sized once and zeroed where needed.
    var x_disc_size = batch * n_chunks * n_heads * chunk_size * head_dim
    var a_disc_size = batch * n_chunks * n_heads * chunk_size
    var bc_chunk_size = batch * n_chunks * n_heads * chunk_size * state_dim
    var y_diag_size = x_disc_size  # same shape as X_disc
    var chunk_states_size = batch * n_chunks * n_heads * head_dim * state_dim
    var entering_size = chunk_states_size
    var final_size = batch * n_heads * head_dim * state_dim
    var cd_size = batch * n_chunks * n_heads
    var y_chunked_size = x_disc_size

    var X_disc = List[Scalar[dtype]](length=x_disc_size, fill=Scalar[dtype](0))
    var A_disc = List[Scalar[dtype]](length=a_disc_size, fill=Scalar[dtype](0))
    var B_chunk = List[Scalar[dtype]](
        length=bc_chunk_size, fill=Scalar[dtype](0)
    )
    var C_chunk = List[Scalar[dtype]](
        length=bc_chunk_size, fill=Scalar[dtype](0)
    )
    var Y_diag = List[Scalar[dtype]](length=y_diag_size, fill=Scalar[dtype](0))
    var chunk_states = List[Scalar[dtype]](
        length=chunk_states_size, fill=Scalar[dtype](0)
    )
    var entering = List[Scalar[dtype]](
        length=entering_size, fill=Scalar[dtype](0)
    )
    var final = List[Scalar[dtype]](length=final_size, fill=Scalar[dtype](0))
    var chunk_decays = List[Scalar[dtype]](
        length=cd_size, fill=Scalar[dtype](0)
    )
    var Y_chunked = List[Scalar[dtype]](
        length=y_chunked_size, fill=Scalar[dtype](0)
    )

    # Stage 0: discretize + chunked reshape.
    _ssd_combined_precompute_cpu[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        state_dim,
        n_chunks,
        chunk_size,
        x_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        X_disc.unsafe_ptr(),
        A_disc.unsafe_ptr(),
        B_chunk.unsafe_ptr(),
        C_chunk.unsafe_ptr(),
        chunk_decays.unsafe_ptr(),
    )

    # Build LayoutTensors over the intermediate buffers (row-major).
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var X_disc_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        X_disc.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, head_dim)
        ),
    )
    var A_disc_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        A_disc.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size)
        ),
    )
    var B_chunk_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        B_chunk.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, state_dim)
        ),
    )
    var C_chunk_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        C_chunk.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, state_dim)
        ),
    )
    var Y_diag_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_diag.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, head_dim)
        ),
    )
    var chunk_states_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        chunk_states.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    var entering_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        entering.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    var final_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        final.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_heads, head_dim, state_dim)
        ),
    )
    var chunk_decays_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        chunk_decays.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_3d].row_major(Index(batch, n_chunks, n_heads)),
    )
    var Y_chunked_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_chunked.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, head_dim)
        ),
    )

    # Stage 1: intra-chunk diagonal -> Y_diag.
    # The kernel accumulates into Y, so Y_diag is pre-zeroed above.
    ssd_intra_chunk_fwd_cpu[
        dtype,
        C_chunk_lt.layout,
        B_chunk_lt.layout,
        X_disc_lt.layout,
        A_disc_lt.layout,
        Y_diag_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_size,
        state_dim,
        head_dim,
        C_chunk_lt,
        B_chunk_lt,
        X_disc_lt,
        A_disc_lt,
        Y_diag_lt,
    )

    # Stage 2: per-chunk end-states.
    ssd_chunk_state_fwd_cpu[
        dtype,
        B_chunk_lt.layout,
        X_disc_lt.layout,
        A_disc_lt.layout,
        chunk_states_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_size,
        state_dim,
        head_dim,
        B_chunk_lt,
        X_disc_lt,
        A_disc_lt,
        chunk_states_lt,
    )

    # Stage 3: inter-chunk recurrence.
    ssd_chunk_scan_fwd_cpu[
        dtype,
        chunk_states_lt.layout,
        chunk_decays_lt.layout,
        entering_lt.layout,
        final_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        head_dim,
        state_dim,
        chunk_states_lt,
        chunk_decays_lt,
        entering_lt,
        final_lt,
    )

    # Stage 4: output recombination (writes into Y_chunked, copying Y_diag in).
    ssd_output_recombination_fwd_cpu[
        dtype,
        C_chunk_lt.layout,
        entering_lt.layout,
        A_disc_lt.layout,
        Y_diag_lt.layout,
        Y_chunked_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_size,
        head_dim,
        state_dim,
        C_chunk_lt,
        entering_lt,
        A_disc_lt,
        Y_diag_lt,
        Y_chunked_lt,
    )

    # Flatten the chunked output back to (batch, seqlen, n_heads, head_dim).
    _ssd_combined_postprocess_cpu[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        n_chunks,
        chunk_size,
        Y_chunked.unsafe_ptr(),
        Y_ptr,
    )

    # Copy the post-last-chunk SSM state into the caller's final_state buffer.
    # Shape is [batch, n_heads, head_dim, state_dim], identical to `final_lt`.
    var final_total = batch * n_heads * head_dim * state_dim
    var final_src = final.unsafe_ptr()
    for i in range(final_total):
        final_state_ptr[i] = final_src[i]


# ===----------------------------------------------------------------------=== #
# GPU dispatch — fully on-device stitched pipeline
# ===----------------------------------------------------------------------=== #


def _ssd_combined_precompute_gpu[
    dtype: DType,
    x_LT: TensorLayout,
    dt_LT: TensorLayout,
    A_LT: TensorLayout,
    bc_LT: TensorLayout,
    xd_LT: TensorLayout,
    ad_LT: TensorLayout,
    bcc_LT: TensorLayout,
](
    batch_i32: Int32,
    seqlen_i32: Int32,
    n_heads_i32: Int32,
    head_dim_i32: Int32,
    state_dim_i32: Int32,
    n_chunks_i32: Int32,
    chunk_size_i32: Int32,
    x: TileTensor[dtype, x_LT, MutAnyOrigin],
    dt: TileTensor[dtype, dt_LT, MutAnyOrigin],
    A: TileTensor[dtype, A_LT, MutAnyOrigin],
    B: TileTensor[dtype, bc_LT, MutAnyOrigin],
    C: TileTensor[dtype, bc_LT, MutAnyOrigin],
    X_disc: TileTensor[dtype, xd_LT, MutAnyOrigin],
    A_disc: TileTensor[dtype, ad_LT, MutAnyOrigin],
    B_chunk: TileTensor[dtype, bcc_LT, MutAnyOrigin],
    C_chunk: TileTensor[dtype, bcc_LT, MutAnyOrigin],
):
    """Stage 0 (GPU): discretize + chunked reshape. One thread per (b,c,h,l).

    Mirrors ``_ssd_combined_precompute_cpu``: ``X_disc = x*dt`` (chunked),
    ``A_disc = A[h]*dt`` (chunked), ``B_chunk``/``C_chunk`` pure reshape. The
    flat ``(b,c,h,l)`` index is row-major over ``A_disc`` so ``A_disc.ptr[idx]``
    is the destination directly.
    """
    var batch = Int(batch_i32)
    var seqlen = Int(seqlen_i32)
    var n_heads = Int(n_heads_i32)
    var head_dim = Int(head_dim_i32)
    var state_dim = Int(state_dim_i32)
    var n_chunks = Int(n_chunks_i32)
    var chunk_size = Int(chunk_size_i32)
    var idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total = batch * n_chunks * n_heads * chunk_size
    if idx >= total:
        return

    var l = idx % chunk_size
    var h = (idx // chunk_size) % n_heads
    var c = (idx // (chunk_size * n_heads)) % n_chunks
    var bi = idx // (chunk_size * n_heads * n_chunks)
    var t = c * chunk_size + l

    var a_h = A.ptr[h].cast[DType.float32]()
    var dt_off = (bi * seqlen + t) * n_heads + h
    var dt_val = dt.ptr[dt_off].cast[DType.float32]()
    A_disc.ptr[idx] = (a_h * dt_val).cast[dtype]()

    var x_in_base = ((bi * seqlen + t) * n_heads + h) * head_dim
    var x_out_base = idx * head_dim
    for p in range(head_dim):
        X_disc.ptr[x_out_base + p] = (
            x.ptr[x_in_base + p].cast[DType.float32]() * dt_val
        ).cast[dtype]()

    var bc_in_base = ((bi * seqlen + t) * n_heads + h) * state_dim
    var bc_out_base = idx * state_dim
    for n in range(state_dim):
        B_chunk.ptr[bc_out_base + n] = B.ptr[bc_in_base + n]
        C_chunk.ptr[bc_out_base + n] = C.ptr[bc_in_base + n]


def _ssd_combined_chunk_decays_gpu[
    dtype: DType,
    ad_LT: TensorLayout,
    cd_LT: TensorLayout,
](
    batch_i32: Int32,
    n_chunks_i32: Int32,
    n_heads_i32: Int32,
    chunk_size_i32: Int32,
    A_disc: TileTensor[dtype, ad_LT, MutAnyOrigin],
    chunk_decays: TileTensor[dtype, cd_LT, MutAnyOrigin],
):
    """Stage 0b (GPU): ``chunk_decays[b,c,h] = sum_l A_disc[b,c,h,l]``."""
    var batch = Int(batch_i32)
    var n_chunks = Int(n_chunks_i32)
    var n_heads = Int(n_heads_i32)
    var chunk_size = Int(chunk_size_i32)
    var idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total = batch * n_chunks * n_heads
    if idx >= total:
        return
    var base = idx * chunk_size
    var s = Float32(0.0)
    for l in range(chunk_size):
        s += A_disc.ptr[base + l].cast[DType.float32]()
    chunk_decays.ptr[idx] = s.cast[dtype]()


def _ssd_combined_postprocess_gpu[
    dtype: DType,
    yc_LT: TensorLayout,
    y_LT: TensorLayout,
](
    batch_i32: Int32,
    seqlen_i32: Int32,
    n_heads_i32: Int32,
    head_dim_i32: Int32,
    n_chunks_i32: Int32,
    chunk_size_i32: Int32,
    Y_chunked: TileTensor[dtype, yc_LT, MutAnyOrigin],
    Y: TileTensor[dtype, y_LT, MutAnyOrigin],
):
    """Stage 5 (GPU): flatten ``Y_chunked[b,c,h,l,p]`` -> ``Y[b,t,h,p]``."""
    var batch = Int(batch_i32)
    var seqlen = Int(seqlen_i32)
    var n_heads = Int(n_heads_i32)
    var head_dim = Int(head_dim_i32)
    var n_chunks = Int(n_chunks_i32)
    var chunk_size = Int(chunk_size_i32)
    var idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total = batch * n_chunks * n_heads * chunk_size * head_dim
    if idx >= total:
        return
    var p = idx % head_dim
    var l = (idx // head_dim) % chunk_size
    var h = (idx // (head_dim * chunk_size)) % n_heads
    var c = (idx // (head_dim * chunk_size * n_heads)) % n_chunks
    var bi = idx // (head_dim * chunk_size * n_heads * n_chunks)
    var t = c * chunk_size + l
    var y_out = ((bi * seqlen + t) * n_heads + h) * head_dim + p
    Y.ptr[y_out] = Y_chunked.ptr[idx]


def _ssd_chunk_scan_combined_gpu[
    dtype: DType,
    chunk_size: Int,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dt_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    C_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    Y_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    final_state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Fully on-device SSD chunk-scan-combined.

    Runs all six stages on the GPU over device-resident buffers (no host
    round-trip): precompute -> intra-chunk diagonal -> chunk end-state (the
    optimized ``ssd_chunk_state`` kernel; the fused tensor-core path for the
    gate dims, else the parallel scalar path) -> inter-chunk scan -> output
    recombination -> postprocess. ``*_ptr`` are device pointers.
    """
    if seqlen % chunk_size != 0:
        raise Error(
            "ssd_chunk_scan_combined: seqlen must be divisible by chunk_size"
        )
    var n_chunks = seqlen // chunk_size
    var num_slices = batch * n_chunks * n_heads

    # Device intermediates (same shapes as the CPU dispatch).
    var xd_n = batch * n_chunks * n_heads * chunk_size * head_dim
    var ad_n = batch * n_chunks * n_heads * chunk_size
    var bc_n = batch * n_chunks * n_heads * chunk_size * state_dim
    var st_n = batch * n_chunks * n_heads * head_dim * state_dim
    var cd_n = batch * n_chunks * n_heads
    var fin_n = batch * n_heads * head_dim * state_dim

    var X_disc_d = ctx.enqueue_create_buffer[dtype](xd_n)
    var A_disc_d = ctx.enqueue_create_buffer[dtype](ad_n)
    var B_chunk_d = ctx.enqueue_create_buffer[dtype](bc_n)
    var C_chunk_d = ctx.enqueue_create_buffer[dtype](bc_n)
    var Y_diag_d = ctx.enqueue_create_buffer[dtype](xd_n)
    var chunk_states_d = ctx.enqueue_create_buffer[dtype](st_n)
    var entering_d = ctx.enqueue_create_buffer[dtype](st_n)
    var final_d = ctx.enqueue_create_buffer[dtype](fin_n)
    var chunk_decays_d = ctx.enqueue_create_buffer[dtype](cd_n)
    var Y_chunked_d = ctx.enqueue_create_buffer[dtype](xd_n)

    # TileTensor views over inputs (device ptrs) and intermediates.
    var x_tt = TileTensor(x_ptr, row_major(batch, seqlen, n_heads, head_dim))
    var dt_tt = TileTensor(dt_ptr, row_major(batch, seqlen, n_heads))
    var A_tt = TileTensor(A_ptr, row_major(n_heads))
    var B_tt = TileTensor(B_ptr, row_major(batch, seqlen, n_heads, state_dim))
    var C_tt = TileTensor(C_ptr, row_major(batch, seqlen, n_heads, state_dim))

    var X_disc = TileTensor(
        X_disc_d, row_major(batch, n_chunks, n_heads, chunk_size, head_dim)
    )
    var A_disc = TileTensor(
        A_disc_d, row_major(batch, n_chunks, n_heads, chunk_size)
    )
    var B_chunk = TileTensor(
        B_chunk_d, row_major(batch, n_chunks, n_heads, chunk_size, state_dim)
    )
    var C_chunk = TileTensor(
        C_chunk_d, row_major(batch, n_chunks, n_heads, chunk_size, state_dim)
    )
    var Y_diag = TileTensor(
        Y_diag_d, row_major(batch, n_chunks, n_heads, chunk_size, head_dim)
    )
    var chunk_states = TileTensor(
        chunk_states_d, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var entering = TileTensor(
        entering_d, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var final = TileTensor(
        final_d, row_major(batch, n_heads, head_dim, state_dim)
    )
    var chunk_decays = TileTensor(
        chunk_decays_d, row_major(batch, n_chunks, n_heads)
    )
    var Y_chunked = TileTensor(
        Y_chunked_d, row_major(batch, n_chunks, n_heads, chunk_size, head_dim)
    )
    var Y_out = TileTensor(Y_ptr, row_major(batch, seqlen, n_heads, head_dim))

    # Stage 0: precompute (discretize + chunked reshape) + chunk decays.
    var pre_total = batch * n_chunks * n_heads * chunk_size
    var pre_compiled = ctx.compile_function[
        _ssd_combined_precompute_gpu[
            dtype,
            x_tt.LayoutType,
            dt_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            X_disc.LayoutType,
            A_disc.LayoutType,
            B_chunk.LayoutType,
        ]
    ]()
    var cd_compiled = ctx.compile_function[
        _ssd_combined_chunk_decays_gpu[
            dtype, A_disc.LayoutType, chunk_decays.LayoutType
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            pre_compiled,
            Int32(batch),
            Int32(seqlen),
            Int32(n_heads),
            Int32(head_dim),
            Int32(state_dim),
            Int32(n_chunks),
            Int32(chunk_size),
            x_tt,
            dt_tt,
            A_tt,
            B_tt,
            C_tt,
            X_disc,
            A_disc,
            B_chunk,
            C_chunk,
            grid_dim=(ceildiv(pre_total, _SSD_GPU_BLOCK),),
            block_dim=(_SSD_GPU_BLOCK,),
        )
        ctx.enqueue_function(
            cd_compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_size),
            A_disc,
            chunk_decays,
            grid_dim=(ceildiv(cd_n, _SSD_GPU_BLOCK),),
            block_dim=(_SSD_GPU_BLOCK,),
        )

    # Stage 1: intra-chunk diagonal -> Y_diag (host launcher).
    ssd_intra_chunk_fwd_gpu[
        dtype,
        C_chunk.LayoutType,
        B_chunk.LayoutType,
        X_disc.LayoutType,
        A_disc.LayoutType,
        Y_diag.LayoutType,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_size,
        state_dim,
        head_dim,
        C_chunk,
        B_chunk,
        X_disc,
        A_disc,
        Y_diag,
        ctx,
    )

    # Stage 2: per-chunk end-states (optimized chunk_state).
    # Fused tensor-core path for the gate dims; parallel scalar path otherwise.
    if state_dim == 128 and head_dim == 64:
        ssd_chunk_state_fwd_gpu_fused[
            dtype,
            chunk_size,
            128,
            64,
            B_chunk.LayoutType,
            X_disc.LayoutType,
            A_disc.LayoutType,
            chunk_states.LayoutType,
        ](batch, n_chunks, n_heads, B_chunk, X_disc, A_disc, chunk_states, ctx)
    else:
        var cs_compiled = ctx.compile_function[
            ssd_chunk_state_fwd_gpu[
                dtype,
                B_chunk.LayoutType,
                X_disc.LayoutType,
                A_disc.LayoutType,
                chunk_states.LayoutType,
            ]
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                cs_compiled,
                Int32(batch),
                Int32(n_chunks),
                Int32(n_heads),
                Int32(chunk_size),
                Int32(state_dim),
                Int32(head_dim),
                B_chunk,
                X_disc,
                A_disc,
                chunk_states,
                grid_dim=(num_slices, 1),
                block_dim=(chunk_size,),
            )

    # Stage 3: inter-chunk recurrence (device kernel).
    var scan_total = batch * n_heads * head_dim * state_dim
    var scan_compiled = ctx.compile_function[
        ssd_chunk_scan_fwd_gpu[
            dtype,
            chunk_states.LayoutType,
            chunk_decays.LayoutType,
            entering.LayoutType,
            final.LayoutType,
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            scan_compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(head_dim),
            Int32(state_dim),
            chunk_states,
            chunk_decays,
            entering,
            final,
            grid_dim=(ceildiv(scan_total, _SSD_GPU_BLOCK),),
            block_dim=(_SSD_GPU_BLOCK,),
        )

    # Stage 4: output recombination (device kernel). Fused single-pass
    # tensor-core path for the gate dims (head_dim=64, state_dim=128); the
    # fused path needs chunk_size a multiple of 128 (comptime tiling), so guard
    # its instantiation at compile time. Scalar fallback otherwise.
    var comb_total = batch * n_chunks * n_heads * chunk_size * head_dim
    var used_fused = False

    comptime if chunk_size % 128 == 0:
        if state_dim == 128 and head_dim == 64:
            used_fused = True
            ssd_output_recombination_fwd_gpu_fused[
                dtype,
                chunk_size,
                128,
                64,
                C_chunk.LayoutType,
                entering.LayoutType,
                A_disc.LayoutType,
                Y_diag.LayoutType,
                Y_chunked.LayoutType,
            ](
                batch,
                n_chunks,
                n_heads,
                C_chunk,
                entering,
                A_disc,
                Y_diag,
                Y_chunked,
                ctx,
            )

    if not used_fused:
        var comb_compiled = ctx.compile_function[
            ssd_output_recombination_fwd_gpu[
                dtype,
                C_chunk.LayoutType,
                entering.LayoutType,
                A_disc.LayoutType,
                Y_diag.LayoutType,
                Y_chunked.LayoutType,
            ]
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                comb_compiled,
                Int32(batch),
                Int32(n_chunks),
                Int32(n_heads),
                Int32(chunk_size),
                Int32(head_dim),
                Int32(state_dim),
                C_chunk,
                entering,
                A_disc,
                Y_diag,
                Y_chunked,
                grid_dim=(ceildiv(comb_total, _SSD_GPU_BLOCK),),
                block_dim=(_SSD_GPU_BLOCK,),
            )

    # Stage 5: postprocess (un-chunk) + copy final state to the output buffer.
    var post_compiled = ctx.compile_function[
        _ssd_combined_postprocess_gpu[
            dtype, Y_chunked.LayoutType, Y_out.LayoutType
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            post_compiled,
            Int32(batch),
            Int32(seqlen),
            Int32(n_heads),
            Int32(head_dim),
            Int32(n_chunks),
            Int32(chunk_size),
            Y_chunked,
            Y_out,
            grid_dim=(ceildiv(comb_total, _SSD_GPU_BLOCK),),
            block_dim=(_SSD_GPU_BLOCK,),
        )
        ctx.enqueue_copy(final_state_ptr, final_d.unsafe_ptr(), fin_n)
    ctx.synchronize()


# ===----------------------------------------------------------------------=== #
# @extensibility.register entry point
# ===----------------------------------------------------------------------=== #


@extensibility.register("ssd_chunk_scan_combined")
struct SSDChunkScanCombined[chunk_size: Int = 256]:
    """Fused Mamba2 SSD chunk-scan combined op.

    Stitches the four verified SSD stage kernels into a single op matching the
    torch ``mamba_chunk_scan_combined`` fused contract.

    Parameters:
        chunk_size: Tokens per chunk. Must divide ``seqlen`` evenly. The
            default (256) matches the upstream Mamba2 prefill setting.

    Tensor Shapes:
        - output Y:           ``(batch, seqlen, n_heads, head_dim)``
        - output final_state: ``(batch, n_heads, head_dim, state_dim)``
        - x:                  ``(batch, seqlen, n_heads, head_dim)``
        - dt:                 ``(batch, seqlen, n_heads)``
        - A:                  ``(n_heads,)``
        - B:                  ``(batch, seqlen, n_heads, state_dim)``
        - C:                  ``(batch, seqlen, n_heads, state_dim)``
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        Y: OutputTensor[dtype=dtype, rank=4, ...],
        final_state: OutputTensor[dtype=dtype, rank=4, ...],
        x: InputTensor[dtype=dtype, rank=4, ...],
        dt: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=1, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if Y.shape() != x.shape():
            raise Error("ssd_chunk_scan_combined: Y shape must match x shape")

        var batch = x.dim_size(0)
        var seqlen = x.dim_size(1)
        var n_heads = x.dim_size(2)
        var head_dim = x.dim_size(3)
        var state_dim = B.dim_size(3)

        if (
            final_state.dim_size(0) != batch
            or final_state.dim_size(1) != n_heads
            or final_state.dim_size(2) != head_dim
            or final_state.dim_size(3) != state_dim
        ):
            raise Error(
                "ssd_chunk_scan_combined: final_state shape must be"
                " [batch, n_heads, head_dim, state_dim]"
            )

        comptime if is_cpu[target]():
            _ssd_chunk_scan_combined_cpu[dtype](
                batch,
                seqlen,
                n_heads,
                head_dim,
                state_dim,
                Self.chunk_size,
                x.unsafe_ptr(),
                dt.unsafe_ptr(),
                A.unsafe_ptr(),
                B.unsafe_ptr(),
                C.unsafe_ptr(),
                Y.unsafe_ptr(),
                final_state.unsafe_ptr(),
            )
        elif is_gpu[target]():
            # Fully on-device stitched dispatch: all six stages run on the GPU
            # over device-resident intermediates (no host round-trip). Stage 2
            # uses the optimized ssd_chunk_state kernel (fused tensor-core path
            # for the gate dims, parallel scalar otherwise).
            _ssd_chunk_scan_combined_gpu[dtype, Self.chunk_size](
                batch,
                seqlen,
                n_heads,
                head_dim,
                state_dim,
                x.unsafe_ptr(),
                dt.unsafe_ptr(),
                A.unsafe_ptr(),
                B.unsafe_ptr(),
                C.unsafe_ptr(),
                Y.unsafe_ptr(),
                final_state.unsafe_ptr(),
                ctx,
            )
        else:
            raise Error("ssd_chunk_scan_combined: unsupported target")


@extensibility.register_shape_function("ssd_chunk_scan_combined")
def ssd_chunk_scan_combined_shape[
    dtype: DType,
](
    x: InputTensor[dtype=dtype, rank=4, ...],
    dt: InputTensor[dtype=dtype, rank=3, ...],
    A: InputTensor[dtype=dtype, rank=1, ...],
    B: InputTensor[dtype=dtype, rank=4, ...],
    C: InputTensor[dtype=dtype, rank=4, ...],
) -> IndexList[4]:
    """Returns the primary (`Y`) output shape for `ssd_chunk_scan_combined`.

    The op has two outputs (`Y` and `final_state`); the graph-side wrapper
    (`functional_ops.ssd_chunk_scan_combined`) passes both `out_types`
    explicitly. Only the primary shape is returned here: a `Tuple` return
    trips the kernel-package loader's parameter-name mapping for `Tuple`
    ("struct decl not present in the loaded package and not in the stdlib
    slot-name table") on the prebuilt toolchain.

    Parameters:
        dtype: Element type of the input tensors.

    Args:
        x: Input tensor with shape `(batch, seqlen, n_heads, head_dim)`.
        dt: Time-step tensor with shape `(batch, seqlen, n_heads)`.
        A: Per-head decay with shape `(n_heads,)`.
        B: Input projection with shape `(batch, seqlen, n_heads, state_dim)`.
        C: Output projection with shape `(batch, seqlen, n_heads, state_dim)`.

    Returns:
        The `Y` output shape, equal to `x.shape()`.
    """
    return x.shape()
