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
"""Structured state space duality (SSD) inter-chunk scan kernel.

This implements stage 3 of the SSD chunk-scan algorithm used by Mamba2: the
"inter-chunk recurrence". After stage 2 (``ssd_chunk_state``) has reduced each
chunk down to a per-chunk end-state, this stage threads a single sequential
recurrence across the chunks, per ``(batch, head)``, to produce the state
*entering* each chunk and the final state after the last chunk.

For a sequence of ``C`` chunks with state dim ``N`` and head dim ``P``:

- Inputs:
  - ``chunk_states: [C, P, N]`` (per-chunk end-states, the stage-2 output).
  - ``chunk_decays: [C]`` (per-chunk total decay = sum of ``A`` over the chunk).
- Recurrence (running ``state`` of shape ``[P, N]``, initialized to 0):
  ```
  entering[c] = state                                # state ENTERING chunk c
  state       = chunk_states[c] + exp(chunk_decays[c]) * state
  ```
- Outputs:
  - ``entering: [C, P, N]`` (the running state entering each chunk; the entry
    for chunk 0 is all zeros).
  - ``final: [P, N]`` (the running state after the last chunk).

This matches the reference recurrence (stage 3 of ``ssd_minimal_discrete``)
that carries inter-chunk information forward so the per-chunk outputs can be
corrected by the contribution of preceding chunks.

The chunk loop is inherently SEQUENTIAL: chunk ``c`` depends on the state left
by chunk ``c-1``, so chunks cannot be parallelized. On GPU we parallelize over
``(batch, head, p, n)`` instead — one thread owns a single ``(p, n)`` scalar of
the running state and walks the chunk axis sequentially.
"""

from layout import Layout, LayoutTensor, TensorLayout, TileTensor
from std.math import exp
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from std.utils.index import IndexList

# Stride helpers for indexing the flat row-major buffers.
comptime Strides3D = IndexList[3]
comptime Strides4D = IndexList[4]
comptime Strides5D = IndexList[5]


def _row_major_strides_3d(d0: Int, d1: Int, d2: Int) -> Strides3D:
    """Row-major strides for a 3D shape ``(d0, d1, d2)``."""
    return Strides3D(d1 * d2, d2, 1)


def _row_major_strides_4d(d0: Int, d1: Int, d2: Int, d3: Int) -> Strides4D:
    """Row-major strides for a 4D shape ``(d0, d1, d2, d3)``."""
    return Strides4D(d1 * d2 * d3, d2 * d3, d3, 1)


def _row_major_strides_5d(
    d0: Int, d1: Int, d2: Int, d3: Int, d4: Int
) -> Strides5D:
    """Row-major strides for a 5D shape ``(d0, d1, d2, d3, d4)``."""
    return Strides5D(d1 * d2 * d3 * d4, d2 * d3 * d4, d3 * d4, d4, 1)


def ssd_chunk_scan_fwd_cpu[
    kernel_dtype: DType,
    chunk_states_layout: Layout,
    chunk_decays_layout: Layout,
    entering_layout: Layout,
    final_layout: Layout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    chunk_states: LayoutTensor[kernel_dtype, chunk_states_layout, MutAnyOrigin],
    chunk_decays: LayoutTensor[kernel_dtype, chunk_decays_layout, MutAnyOrigin],
    entering: LayoutTensor[kernel_dtype, entering_layout, MutAnyOrigin],
    final: LayoutTensor[kernel_dtype, final_layout, MutAnyOrigin],
):
    """Compute the SSD inter-chunk scan on CPU.

    Threads a sequential recurrence across chunks, per ``(batch, head)``, over
    the supplied row-major tensors.

    Tensor shapes (row-major):
        - ``chunk_states``: ``[batch, n_chunks, n_heads, head_dim, state_dim]``
        - ``chunk_decays``: ``[batch, n_chunks, n_heads]``
        - ``entering``: ``[batch, n_chunks, n_heads, head_dim, state_dim]`` (out)
        - ``final``: ``[batch, n_heads, head_dim, state_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        chunk_states_layout: Layout of the ``chunk_states`` tensor.
        chunk_decays_layout: Layout of the ``chunk_decays`` tensor.
        entering_layout: Layout of the ``entering`` output tensor.
        final_layout: Layout of the ``final`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence (``C``).
        n_heads: Number of heads.
        head_dim: Head dimension (``P``).
        state_dim: State dimension (``N``).
        chunk_states: Per-chunk end-states,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        chunk_decays: Per-chunk total decay,
            shape ``[batch, n_chunks, n_heads]``.
        entering: Output state entering each chunk,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        final: Output final state after the last chunk,
            shape ``[batch, n_heads, P, N]``.
    """
    # Row-major strides for the flat backing buffers.
    var cs_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, head_dim, state_dim
    )
    var cd_strides = _row_major_strides_3d(batch, n_chunks, n_heads)
    var ent_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, head_dim, state_dim
    )
    var final_strides = _row_major_strides_4d(
        batch, n_heads, head_dim, state_dim
    )

    for bi in range(batch):
        for h in range(n_heads):
            # Running state for this (batch, head), shape (P, N), init 0.
            var state = List[Float32](length=head_dim * state_dim, fill=0.0)

            for c in range(n_chunks):
                # entering[c] = state (the state ENTERING chunk c).
                var ent_base = (
                    bi * ent_strides[0]
                    + c * ent_strides[1]
                    + h * ent_strides[2]
                )
                for p in range(head_dim):
                    for n in range(state_dim):
                        var ent_off = (
                            ent_base + p * ent_strides[3] + n * ent_strides[4]
                        )
                        entering.ptr[ent_off] = state[p * state_dim + n].cast[
                            kernel_dtype
                        ]()

                # decay = exp(chunk_decays[c]).
                var cd_off = (
                    bi * cd_strides[0] + c * cd_strides[1] + h * cd_strides[2]
                )
                var decay = exp(chunk_decays.ptr[cd_off].cast[DType.float32]())

                # state = chunk_states[c] + decay * state.
                var cs_base = (
                    bi * cs_strides[0] + c * cs_strides[1] + h * cs_strides[2]
                )
                for p in range(head_dim):
                    for n in range(state_dim):
                        var cs_off = (
                            cs_base + p * cs_strides[3] + n * cs_strides[4]
                        )
                        var cs_val = chunk_states.ptr[cs_off].cast[
                            DType.float32
                        ]()
                        var prev = state[p * state_dim + n]
                        state[p * state_dim + n] = cs_val + decay * prev

            # final = state after the last chunk.
            var final_base = bi * final_strides[0] + h * final_strides[1]
            for p in range(head_dim):
                for n in range(state_dim):
                    var final_off = (
                        final_base + p * final_strides[2] + n * final_strides[3]
                    )
                    final.ptr[final_off] = state[p * state_dim + n].cast[
                        kernel_dtype
                    ]()


# ===----------------------------------------------------------------------=== #
# GPU Implementation
# ===----------------------------------------------------------------------=== #


def ssd_chunk_scan_fwd_gpu[
    kernel_dtype: DType,
    chunk_states_LT: TensorLayout,
    chunk_decays_LT: TensorLayout,
    entering_LT: TensorLayout,
    final_LT: TensorLayout,
](
    batch_i32: Int32,
    n_chunks_i32: Int32,
    n_heads_i32: Int32,
    head_dim_i32: Int32,
    state_dim_i32: Int32,
    chunk_states: TileTensor[kernel_dtype, chunk_states_LT, MutUntrackedOrigin],
    chunk_decays: TileTensor[kernel_dtype, chunk_decays_LT, MutUntrackedOrigin],
    entering: TileTensor[kernel_dtype, entering_LT, MutUntrackedOrigin],
    final: TileTensor[kernel_dtype, final_LT, MutUntrackedOrigin],
):
    """Compute the SSD inter-chunk scan on GPU.

    The chunk recurrence is inherently sequential, so we do NOT parallelize over
    chunks. Instead each GPU thread owns a single ``(batch, head, p, n)`` scalar
    of the running state and walks the chunk axis sequentially. This keeps the
    recurrence dependency local to one thread (no cross-thread sync) while still
    exposing ``batch * n_heads * head_dim * state_dim``-way parallelism.

    The same math as the CPU kernel, for the scalar ``state[p, n]``:
        - ``entering[c, p, n] = state`` (before updating)
        - ``state = chunk_states[c, p, n] + exp(chunk_decays[c]) * state``
        - ``final[p, n] = state`` (after the last chunk)

    Tensor shapes (row-major):
        - ``chunk_states``: ``[batch, n_chunks, n_heads, head_dim, state_dim]``
        - ``chunk_decays``: ``[batch, n_chunks, n_heads]``
        - ``entering``: ``[batch, n_chunks, n_heads, head_dim, state_dim]`` (out)
        - ``final``: ``[batch, n_heads, head_dim, state_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        chunk_states_LT: Layout of the ``chunk_states`` tensor.
        chunk_decays_LT: Layout of the ``chunk_decays`` tensor.
        entering_LT: Layout of the ``entering`` output tensor.
        final_LT: Layout of the ``final`` output tensor.

    Args:
        batch_i32: Number of batch elements.
        n_chunks_i32: Number of chunks per sequence (``C``).
        n_heads_i32: Number of heads.
        head_dim_i32: Head dimension (``P``).
        state_dim_i32: State dimension (``N``).
        chunk_states: Per-chunk end-states,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        chunk_decays: Per-chunk total decay,
            shape ``[batch, n_chunks, n_heads]``.
        entering: Output state entering each chunk,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        final: Output final state after the last chunk,
            shape ``[batch, n_heads, P, N]``.
    """
    var batch = Int(batch_i32)
    var n_chunks = Int(n_chunks_i32)
    var n_heads = Int(n_heads_i32)
    var head_dim = Int(head_dim_i32)
    var state_dim = Int(state_dim_i32)
    var flat_idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total_threads = batch * n_heads * head_dim * state_dim
    if flat_idx >= total_threads:
        return

    # Decompose the flat thread index into (batch, head, p, n).
    var n = flat_idx % state_dim
    var p = (flat_idx // state_dim) % head_dim
    var h = (flat_idx // (state_dim * head_dim)) % n_heads
    var bi = flat_idx // (state_dim * head_dim * n_heads)

    # Row-major strides for the flat backing buffers.
    var cs_n_stride = 1
    var cs_p_stride = state_dim
    var cs_h_stride = head_dim * cs_p_stride
    var cs_c_stride = n_heads * cs_h_stride
    var cs_b_stride = n_chunks * cs_c_stride

    var cd_h_stride = 1
    var cd_c_stride = n_heads
    var cd_b_stride = n_chunks * cd_c_stride

    # entering shares chunk_states' shape/strides.
    var ent_n_stride = 1
    var ent_p_stride = state_dim
    var ent_h_stride = head_dim * ent_p_stride
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride

    var final_n_stride = 1
    var final_p_stride = state_dim
    var final_h_stride = head_dim * final_p_stride
    var final_b_stride = n_heads * final_h_stride

    # Sequential recurrence over chunks for this single (p, n) scalar.
    var state = Float32(0.0)
    for c in range(n_chunks):
        # entering[c] = state (before update).
        var ent_off = (
            bi * ent_b_stride
            + c * ent_c_stride
            + h * ent_h_stride
            + p * ent_p_stride
            + n * ent_n_stride
        )
        entering.ptr[ent_off] = state.cast[kernel_dtype]()

        # decay = exp(chunk_decays[c]).
        var cd_off = bi * cd_b_stride + c * cd_c_stride + h * cd_h_stride
        var decay = exp(chunk_decays.ptr[cd_off].cast[DType.float32]())

        # state = chunk_states[c, p, n] + decay * state.
        var cs_off = (
            bi * cs_b_stride
            + c * cs_c_stride
            + h * cs_h_stride
            + p * cs_p_stride
            + n * cs_n_stride
        )
        var cs_val = chunk_states.ptr[cs_off].cast[DType.float32]()
        state = cs_val + decay * state

    # final[p, n] = state after the last chunk.
    var final_off = (
        bi * final_b_stride
        + h * final_h_stride
        + p * final_p_stride
        + n * final_n_stride
    )
    final.ptr[final_off] = state.cast[kernel_dtype]()
