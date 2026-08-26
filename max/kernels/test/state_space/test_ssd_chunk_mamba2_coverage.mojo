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
"""End-to-end Mamba2-dim coverage test for the SSD chunk-scan chain.

Exercises the four SSD stage kernels (intra-chunk diagonal, chunk end-state,
inter-chunk recurrence, output recombination) chained together as
``SSDChunkScanCombined.execute`` does, at Mamba2-representative dimensions
(``head_dim`` in 64-128 and ``state_dim`` in 64-128) plus a small Mamba1
non-regression case. The reference is an in-test unrolled SSD scan that
implements the discretized state-space recurrence directly via nested loops,
so it shares no infrastructure with the stage kernels under test.

RFC 0002 item 8: verify SSD prefill works at Mamba2 dims without regressing
Mamba1. The kernels take dims as runtime args, so this is a test-only
deliverable -- no kernel edits.
"""

from std.math import exp
from std.random import seed

from layout import Layout, LayoutTensor, RuntimeLayout
from layout._fillers import random
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index

from state_space.ssd_chunk import ssd_intra_chunk_fwd_cpu
from state_space.ssd_chunk_state import ssd_chunk_state_fwd_cpu
from state_space.ssd_chunk_scan import ssd_chunk_scan_fwd_cpu
from state_space.ssd_chunk_combine import ssd_output_recombination_fwd_cpu


# ===----------------------------------------------------------------------=== #
# In-test naive reference
# ===----------------------------------------------------------------------=== #


def _naive_ssd_reference[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    x_ptr: UnsafePointer[Scalar[dtype], ...],
    dt_ptr: UnsafePointer[Scalar[dtype], ...],
    A_ptr: UnsafePointer[Scalar[dtype], ...],
    B_ptr: UnsafePointer[Scalar[dtype], ...],
    C_ptr: UnsafePointer[Scalar[dtype], ...],
    Y_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
):
    """Compute Y via the unrolled discretized SSD recurrence.

    For each ``(b, h)``, threads a single token-level recurrence (no chunking)
    over a per-head state ``s`` of shape ``(head_dim, state_dim)``:

        a_t   = exp(A[h] * dt[b, t, h])
        bx_t  = dt[b, t, h] * x[b, t, h, :]    # length head_dim
        s     = a_t * s + outer(bx_t, B[b, t, h, :])
        y_t,p = sum_n C[b, t, h, n] * s[p, n]

    This is intentionally written with plain nested loops -- no LayoutTensor
    strides, no chunk math -- so it serves as an independent oracle for the
    four-stage chained pipeline.

    All buffers are row-major contiguous in the canonical layouts:
        x:  [batch, seqlen, n_heads, head_dim]
        dt: [batch, seqlen, n_heads]
        A:  [n_heads]
        B:  [batch, seqlen, n_heads, state_dim]
        C:  [batch, seqlen, n_heads, state_dim]
        Y:  [batch, seqlen, n_heads, head_dim]
    """
    for b in range(batch):
        for h in range(n_heads):
            var a_h = A_ptr[h].cast[DType.float32]()
            # Per-(b, h) running state, shape (head_dim, state_dim).
            var state = List[Float32](
                length=head_dim * state_dim, fill=Float32(0.0)
            )

            for t in range(seqlen):
                var dt_off = (b * seqlen + t) * n_heads + h
                var dt_val = dt_ptr[dt_off].cast[DType.float32]()
                var a_t = exp(a_h * dt_val)

                var bc_base = ((b * seqlen + t) * n_heads + h) * state_dim
                var x_base = ((b * seqlen + t) * n_heads + h) * head_dim

                # state[p, n] = a_t * state[p, n]
                #             + (dt_val * x[t, p]) * B[t, n]
                for p in range(head_dim):
                    var xv = x_ptr[x_base + p].cast[DType.float32]() * dt_val
                    for n in range(state_dim):
                        var bv = B_ptr[bc_base + n].cast[DType.float32]()
                        var prev = state[p * state_dim + n]
                        state[p * state_dim + n] = a_t * prev + xv * bv

                # y[t, p] = sum_n C[t, n] * state[p, n]
                var y_base = ((b * seqlen + t) * n_heads + h) * head_dim
                for p in range(head_dim):
                    var acc = Float32(0.0)
                    for n in range(state_dim):
                        var cv = C_ptr[bc_base + n].cast[DType.float32]()
                        acc += cv * state[p * state_dim + n]
                    Y_ptr[y_base + p] = acc.cast[dtype]()


# ===----------------------------------------------------------------------=== #
# Pre-stage helpers (CPU): discretization + reshape
# ===----------------------------------------------------------------------=== #


def _precompute_chunked[
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
    """Discretize and chunk-reshape inputs for the four-stage chain.

    Same layout contract as ``_ssd_combined_precompute_cpu`` in the op wrapper:
        X_disc[b, c, h, l, p] = x[b, c*L+l, h, p] * dt[b, c*L+l, h]
        A_disc[b, c, h, l]    = A[h] * dt[b, c*L+l, h]
        B_chunk / C_chunk     = pure reshapes of B / C
        chunk_decays[b, c, h] = sum_l A_disc[b, c, h, l]
    """
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

                    var x_in_base = ((b * seqlen + t) * n_heads + h) * head_dim
                    var x_out_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * head_dim
                    for p in range(head_dim):
                        var x_val = x_ptr[x_in_base + p].cast[DType.float32]()
                        X_disc_ptr[x_out_base + p] = (x_val * dt_val).cast[
                            dtype
                        ]()

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


def _postprocess_chunked[
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
    """Flatten chunked output back to ``[batch, seqlen, n_heads, head_dim]``."""
    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                for l in range(chunk_size):
                    var t = c * chunk_size + l
                    var y_in_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * head_dim
                    var y_out_base = (
                        b * seqlen + t
                    ) * n_heads * head_dim + h * head_dim
                    for p in range(head_dim):
                        Y_ptr[y_out_base + p] = Y_chunked_ptr[y_in_base + p]


# ===----------------------------------------------------------------------=== #
# Drive the four-stage SSD chain directly (CPU).
# ===----------------------------------------------------------------------=== #


def _run_chained_chain[
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
) raises:
    """Run discretize -> stage1 -> stage2 -> stage3 -> stage4 -> flatten."""
    var n_chunks = seqlen // chunk_size

    var x_disc_size = batch * n_chunks * n_heads * chunk_size * head_dim
    var a_disc_size = batch * n_chunks * n_heads * chunk_size
    var bc_chunk_size = batch * n_chunks * n_heads * chunk_size * state_dim
    var chunk_states_size = batch * n_chunks * n_heads * head_dim * state_dim
    var final_size = batch * n_heads * head_dim * state_dim
    var cd_size = batch * n_chunks * n_heads

    var X_disc = List[Scalar[dtype]](length=x_disc_size, fill=Scalar[dtype](0))
    var A_disc = List[Scalar[dtype]](length=a_disc_size, fill=Scalar[dtype](0))
    var B_chunk = List[Scalar[dtype]](
        length=bc_chunk_size, fill=Scalar[dtype](0)
    )
    var C_chunk = List[Scalar[dtype]](
        length=bc_chunk_size, fill=Scalar[dtype](0)
    )
    var Y_diag = List[Scalar[dtype]](length=x_disc_size, fill=Scalar[dtype](0))
    var chunk_states = List[Scalar[dtype]](
        length=chunk_states_size, fill=Scalar[dtype](0)
    )
    var entering = List[Scalar[dtype]](
        length=chunk_states_size, fill=Scalar[dtype](0)
    )
    var final = List[Scalar[dtype]](length=final_size, fill=Scalar[dtype](0))
    var chunk_decays = List[Scalar[dtype]](
        length=cd_size, fill=Scalar[dtype](0)
    )
    var Y_chunked = List[Scalar[dtype]](
        length=x_disc_size, fill=Scalar[dtype](0)
    )

    _precompute_chunked[dtype](
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

    # Stage 1: intra-chunk diagonal.
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

    # Stage 4: output recombination.
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

    _postprocess_chunked[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        n_chunks,
        chunk_size,
        Y_chunked.unsafe_ptr(),
        Y_ptr,
    )


# ===----------------------------------------------------------------------=== #
# Shared test driver
# ===----------------------------------------------------------------------=== #


def _run_case[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    chunk_size: Int,
    seed_val: Int,
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,
) raises:
    """Allocate inputs, run the chain + reference, and compare element-wise."""
    seed(seed_val)

    comptime layout_1d = Layout.row_major[1]()
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()

    # Input allocations -- the chain consumes these flat row-major buffers.
    var x_heap = List[Scalar[dtype]](
        length=batch * seqlen * n_heads * head_dim, fill=Scalar[dtype](0)
    )
    var x_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_heap.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_4d].row_major(
            Index(batch, seqlen, n_heads, head_dim)
        ),
    )

    var dt_heap = List[Scalar[dtype]](
        length=batch * seqlen * n_heads, fill=Scalar[dtype](0)
    )
    var dt_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        dt_heap.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, n_heads)),
    )

    var A_heap = List[Scalar[dtype]](length=n_heads, fill=Scalar[dtype](0))
    var A_lt = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        A_heap.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_1d].row_major(Index(n_heads)),
    )

    var B_heap = List[Scalar[dtype]](
        length=batch * seqlen * n_heads * state_dim, fill=Scalar[dtype](0)
    )
    var B_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_heap.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_4d].row_major(
            Index(batch, seqlen, n_heads, state_dim)
        ),
    )

    var C_heap = List[Scalar[dtype]](
        length=batch * seqlen * n_heads * state_dim, fill=Scalar[dtype](0)
    )
    var C_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_heap.unsafe_ptr().as_unsafe_any_origin(),
        RuntimeLayout[layout_4d].row_major(
            Index(batch, seqlen, n_heads, state_dim)
        ),
    )

    # Fill inputs.
    random(x_lt, min=Scalar[dtype](-1), max=Scalar[dtype](1))
    # Keep dt small and positive so the per-token decay is well-conditioned.
    random(dt_lt, min=Scalar[dtype](0.05), max=Scalar[dtype](0.5))
    random(B_lt, min=Scalar[dtype](-1), max=Scalar[dtype](1))
    random(C_lt, min=Scalar[dtype](-1), max=Scalar[dtype](1))
    # A in [-0.6, -0.1] (negative log-decay per head).
    random(A_lt, min=Scalar[dtype](-0.6), max=Scalar[dtype](-0.1))

    # Outputs.
    var y_size = batch * seqlen * n_heads * head_dim
    var y_chain_heap = List[Scalar[dtype]](length=y_size, fill=Scalar[dtype](0))
    var y_ref_heap = List[Scalar[dtype]](length=y_size, fill=Scalar[dtype](0))

    # Run the four-stage SSD chain.
    _run_chained_chain[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        state_dim,
        chunk_size,
        x_heap.unsafe_ptr(),
        dt_heap.unsafe_ptr(),
        A_heap.unsafe_ptr(),
        B_heap.unsafe_ptr(),
        C_heap.unsafe_ptr(),
        y_chain_heap.unsafe_ptr(),
    )

    # Run the independent naive reference (unrolled recurrence, no chunking).
    _naive_ssd_reference[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        state_dim,
        x_heap.unsafe_ptr(),
        dt_heap.unsafe_ptr(),
        A_heap.unsafe_ptr(),
        B_heap.unsafe_ptr(),
        C_heap.unsafe_ptr(),
        y_ref_heap.unsafe_ptr(),
    )

    for i in range(y_size):
        assert_almost_equal(
            y_chain_heap[i],
            y_ref_heap[i],
            atol=atol,
            rtol=rtol,
        )


# ===----------------------------------------------------------------------=== #
# Test cases
# ===----------------------------------------------------------------------=== #


def test_mamba1_tiny_non_regression() raises:
    """Tiny Mamba1-style case: shows the chain still works at small dims."""
    _run_case[DType.float32](
        batch=1,
        seqlen=8,
        n_heads=1,
        head_dim=2,
        state_dim=3,
        chunk_size=4,
        seed_val=11,
    )


def test_mamba2_hd64_dstate64_seqlen8() raises:
    """Mamba2 lower-bound dims: head_dim=64, state_dim=64, two chunks of 4."""
    _run_case[DType.float32](
        batch=1,
        seqlen=8,
        n_heads=1,
        head_dim=64,
        state_dim=64,
        chunk_size=4,
        seed_val=23,
    )


def test_mamba2_hd64_dstate128_seqlen8() raises:
    """Mamba2 wider state: head_dim=64, state_dim=128."""
    _run_case[DType.float32](
        batch=1,
        seqlen=8,
        n_heads=1,
        head_dim=64,
        state_dim=128,
        chunk_size=4,
        seed_val=37,
    )


def test_mamba2_hd128_dstate128_seqlen8() raises:
    """Mamba2 upper-bound dims: head_dim=128, state_dim=128."""
    _run_case[DType.float32](
        batch=1,
        seqlen=8,
        n_heads=1,
        head_dim=128,
        state_dim=128,
        chunk_size=4,
        seed_val=53,
    )


def test_mamba2_hd64_dstate64_seqlen16_multi_chunk() raises:
    """Multi-chunk case: seqlen=16, chunk_size=8 -> two chunks."""
    _run_case[DType.float32](
        batch=1,
        seqlen=16,
        n_heads=1,
        head_dim=64,
        state_dim=64,
        chunk_size=8,
        seed_val=71,
    )


def test_mamba2_hd64_dstate64_two_heads() raises:
    """Two-head case to exercise the per-head A indexing."""
    _run_case[DType.float32](
        batch=1,
        seqlen=8,
        n_heads=2,
        head_dim=64,
        state_dim=64,
        chunk_size=4,
        seed_val=97,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
