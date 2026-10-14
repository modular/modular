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
"""Graph-op bindings for the KDA (Kimi Delta Attention) kernels.

The kernel math and `kda_chunk`'s host-side launch sequence live in
`//Kernels/lib/kda` (`kda.recurrent`, `kda.chunk_fwd`, `kda.chunk_launch`);
only the `@extensibility.register` wrappers live here, mirroring the `msa.mojo`
binding for `//Kernels/lib/msa`. Registration has to be declared inside the
built-in kernel library itself -- importing the struct from the standalone
lib does not put the op in the graph compiler's registry, so a served graph
resolving `kda_decode` / `kda_chunk` needs this declaration and not just the
dependency.

Two ops, one per prefill regime:

  * `kda_decode` -- the token-sequential recurrence (`kda.recurrent`). Correct
    for any ragged batch (decode's 1-token sequences and a multi-token
    prefill alike), but O(T) sequential steps per sequence.
  * `kda_chunk` -- the chunk-parallel prefill pipeline (`kda.chunk_fwd`,
    driven by `kda.chunk_launch`): L1 `kda_chunk_prepare_gpu` (parallel per
    chunk) -> L2 `kda_chunk_scan_gpu` (sequential, but only O(T/CHUNK_SIZE)
    steps) -> L3 `kda_chunk_output_gpu` (parallel). This is what turns
    prefill from O(T) into O(T/CHUNK_SIZE) sequential depth; the Python KDA
    layer picks it for prefill and keeps `kda_decode` for single-token decode
    (see `kda_attention.py`).
"""

import extensibility

from extensibility import (
    InputTensor,
    OutputTensor,
    _MutableInputTensor as MutableInputTensor,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from std.math import sqrt
from kda.amd_chunk_fwd import K_DIM as AMD_HEAD_DIM
from kda.amd_chunk_launch import kda_amd_chunk_launch
from kda.recurrent import kda_decode_gpu
from kda.chunk_launch import kda_chunk_launch
from kda.kda_helpers import KDA_GATE_LOWER_BOUND


@extensibility.register("kda_decode")
struct KdaDecode:
    """KDA decode recurrence forward pass.

    Implements the Kimi Delta Attention decode recurrence with per-key alpha
    decay and in-kernel gate+beta activation fusion.

    Default parameters (backward-compatible with S1 graph op):
      gate_mode    = "original"  (stable-softplus gate)
      beta_mode    = "logits"    (sigmoid applied to beta_logits)
      state_layout = "K_FIRST"   (pool shape [N, HV, K, V])

    Tensor Shapes:
        - output         : [1, total_T, num_value_heads, value_head_dim]     (OUT)
        - q              : [1, total_T, num_key_heads, key_head_dim]         (bf16)
        - k              : [1, total_T, num_key_heads, key_head_dim]         (bf16)
        - v              : [1, total_T, num_value_heads, value_head_dim]      (bf16)
        - raw_gate       : [1, total_T, num_value_heads, key_head_dim]     (fp32)
        - beta_logits    : [1, total_T, num_value_heads]                   (fp32)
        - a_log          : [num_value_heads]                              (fp32)
        - dt_bias        : [num_value_heads, key_head_dim]                  (fp32)
        - cu_seqlens     : [batch_size + 1]                                 (int32)
        - state_pool     : [max_slots, num_value_heads, key_head_dim, value_head_dim] (MUT)
        - state_indices  : [batch_size]                                     (int32)
    """

    @staticmethod
    def execute[
        qkv_dtype: DType,
        gate_dtype: DType,
        state_dtype: DType,
        output_dtype: DType,
        target: StaticString,
        gate_mode: StaticString = "original",
        beta_mode: StaticString = "logits",
        state_layout: StaticString = "K_FIRST",
    ](
        output: OutputTensor[dtype=output_dtype, rank=4, ...],
        q: InputTensor[dtype=qkv_dtype, rank=4, ...],
        k: InputTensor[dtype=qkv_dtype, rank=4, ...],
        v: InputTensor[dtype=qkv_dtype, rank=4, ...],
        raw_gate: InputTensor[dtype=gate_dtype, rank=4, ...],
        beta_logits: InputTensor[dtype=gate_dtype, rank=3, ...],
        a_log: InputTensor[dtype=gate_dtype, rank=1, ...],
        dt_bias: InputTensor[dtype=gate_dtype, rank=2, ...],
        cu_seqlens: InputTensor[dtype=DType.int32, rank=1, ...],
        # `state_pool` is a slot-indexed in/out pool read+written in place at
        # `state_indices[b]`. It must be a `MutableInputTensor` (not an
        # `OutputTensor`) so the graph binds the caller's persistent pool
        # rather than treating it as a freshly-produced output -- mirroring
        # the `gated_delta_conv1d_fwd` precedent in `kernels.mojo`.
        state_pool: MutableInputTensor[dtype=state_dtype, rank=4, ...],
        state_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert gate_mode == "original" or gate_mode == "safe", (
            "KdaDecode: gate_mode must be 'original' or 'safe', got: "
            + gate_mode
        )
        comptime assert beta_mode == "logits" or beta_mode == "probability", (
            "KdaDecode: beta_mode must be 'logits' or 'probability', got: "
            + beta_mode
        )
        comptime assert (
            state_layout == "K_FIRST" or state_layout == "V_FIRST"
        ), (
            "KdaDecode: state_layout must be 'K_FIRST' or 'V_FIRST', got: "
            + state_layout
        )
        comptime assert is_gpu[target](), "kda_decode is only supported on GPU."

        var num_key_heads = q.dim_size(2)
        var key_head_dim = q.dim_size(3)
        var num_value_heads = v.dim_size(2)
        var value_head_dim = v.dim_size(3)
        # The batch is defined by `state_indices`, which is sized to the live
        # batch; `state_pool` is the long-lived pool and its leading dim is
        # `max_slots`, which is >= batch_size and unrelated to it.
        var batch_size = state_indices.dim_size(0)

        var total_T = q.dim_size(1)

        debug_assert(
            num_value_heads % num_key_heads == 0,
            "kda_decode: num_value_heads must be divisible by num_key_heads",
        )
        debug_assert(
            k.dim_size(1) == total_T,
            "kda_decode: q and k must have same sequence length",
        )
        # Every k offset is built from q's `num_key_heads` / `key_head_dim`
        # (GQA maps a value head to a key head) but scaled by k's own strides,
        # so mismatched k head extents read outside k.
        debug_assert(
            k.dim_size(2) == num_key_heads,
            "kda_decode: k head dim must match q's num_key_heads",
        )
        debug_assert(
            k.dim_size(3) == key_head_dim,
            "kda_decode: k key dim must match q's key_head_dim",
        )
        debug_assert(
            v.dim_size(1) == total_T,
            "kda_decode: q and v must have same sequence length",
        )
        debug_assert(
            raw_gate.dim_size(1) == total_T,
            "kda_decode: raw_gate must have same sequence length as q",
        )
        debug_assert(
            raw_gate.dim_size(2) == num_value_heads,
            "kda_decode: raw_gate head dim must match num_value_heads",
        )
        debug_assert(
            raw_gate.dim_size(3) == key_head_dim,
            "kda_decode: raw_gate key dim must match key_head_dim",
        )
        debug_assert(
            beta_logits.dim_size(1) == total_T,
            "kda_decode: beta_logits must have same sequence length as q",
        )
        debug_assert(
            beta_logits.dim_size(2) == num_value_heads,
            "kda_decode: beta_logits head dim must match num_value_heads",
        )
        debug_assert(
            a_log.dim_size(0) == num_value_heads,
            "kda_decode: a_log length must match num_value_heads",
        )
        debug_assert(
            dt_bias.dim_size(0) == num_value_heads,
            "kda_decode: dt_bias head dim must match num_value_heads",
        )
        debug_assert(
            dt_bias.dim_size(1) == key_head_dim,
            "kda_decode: dt_bias key dim must match key_head_dim",
        )
        debug_assert(
            cu_seqlens.dim_size(0) == batch_size + 1,
            "kda_decode: cu_seqlens must have batch_size + 1 entries",
        )
        debug_assert(
            state_pool.dim_size(1) == num_value_heads,
            "kda_decode: state_pool head dim must match num_value_heads",
        )
        # The kernel indexes pool dims 2 and 3 as (kd, tid) under K_FIRST and
        # (tid, kd) under V_FIRST, with kd < key_head_dim and tid <
        # value_head_dim, so which extent bounds which index follows the layout.
        comptime if state_layout == "K_FIRST":
            debug_assert(
                state_pool.dim_size(2) == key_head_dim,
                "kda_decode: K_FIRST state_pool dim 2 must equal key_head_dim",
            )
            debug_assert(
                state_pool.dim_size(3) == value_head_dim,
                (
                    "kda_decode: K_FIRST state_pool dim 3 must equal"
                    " value_head_dim"
                ),
            )
        else:
            debug_assert(
                state_pool.dim_size(2) == value_head_dim,
                (
                    "kda_decode: V_FIRST state_pool dim 2 must equal"
                    " value_head_dim"
                ),
            )
            debug_assert(
                state_pool.dim_size(3) == key_head_dim,
                "kda_decode: V_FIRST state_pool dim 3 must equal key_head_dim",
            )

        var output_tt = output.to_tile_tensor[DType.int64]()
        var q_tt = q.to_tile_tensor[DType.int64]()
        var k_tt = k.to_tile_tensor[DType.int64]()
        var v_tt = v.to_tile_tensor[DType.int64]()
        var raw_gate_tt = raw_gate.to_tile_tensor[DType.int64]()
        var beta_logits_tt = beta_logits.to_tile_tensor[DType.int64]()
        var a_log_tt = a_log.to_tile_tensor[DType.int64]()
        var dt_bias_tt = dt_bias.to_tile_tensor[DType.int64]()
        var cu_seqlens_tt = cu_seqlens.to_tile_tensor[DType.int64]()
        var state_pool_tt = state_pool.to_tile_tensor[DType.int64]()
        var state_indices_tt = state_indices.to_tile_tensor[DType.int64]()

        var q_strides = q.strides()
        var k_strides = k.strides()
        var v_strides = v.strides()
        var raw_gate_strides = raw_gate.strides()
        var beta_logits_strides = beta_logits.strides()
        var dt_bias_strides = dt_bias.strides()
        var output_strides = output.strides()

        var gpu_ctx = ctx
        # One CTA per (batch_item, value_head); block has value_head_dim threads.
        var num_blocks = batch_size * num_value_heads

        # Head-dim instantiations of the SAME kernel body: every use of the pair
        # below is a compile-time parameter, so a new geometry costs one entry
        # here and no new code. (128, 128) is Kimi-Linear; (32, 32) is Kimi-K3.
        comptime SUPPORTED_HEAD_DIMS = [(32, 32), (128, 128)]
        var dispatched = False

        comptime for head_dims in SUPPORTED_HEAD_DIMS:
            comptime kKD = head_dims[0]
            comptime kVD = head_dims[1]
            if key_head_dim == kKD and value_head_dim == kVD:
                dispatched = True
                gpu_ctx.enqueue_function[
                    kda_decode_gpu[
                        qkv_dtype,
                        gate_dtype,
                        state_dtype,
                        output_dtype,
                        kKD,
                        kVD,
                        output_tt.LayoutType,
                        q_tt.LayoutType,
                        k_tt.LayoutType,
                        v_tt.LayoutType,
                        raw_gate_tt.LayoutType,
                        beta_logits_tt.LayoutType,
                        a_log_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        cu_seqlens_tt.LayoutType,
                        state_pool_tt.LayoutType,
                        state_indices_tt.LayoutType,
                        gate_mode,
                        beta_mode,
                        state_layout,
                    ]
                ](
                    Int32(batch_size),
                    Int32(num_value_heads),
                    Int32(num_key_heads),
                    output_tt,
                    q_tt,
                    k_tt,
                    v_tt,
                    raw_gate_tt,
                    beta_logits_tt,
                    a_log_tt,
                    dt_bias_tt,
                    cu_seqlens_tt,
                    state_pool_tt,
                    state_indices_tt,
                    # q strides
                    UInt32(q_strides[1]),
                    UInt32(q_strides[2]),
                    UInt32(q_strides[3]),
                    # k strides
                    UInt32(k_strides[1]),
                    UInt32(k_strides[2]),
                    UInt32(k_strides[3]),
                    # v strides
                    UInt32(v_strides[1]),
                    UInt32(v_strides[2]),
                    UInt32(v_strides[3]),
                    # raw_gate strides
                    UInt32(raw_gate_strides[1]),
                    UInt32(raw_gate_strides[2]),
                    UInt32(raw_gate_strides[3]),
                    # beta_logits strides
                    UInt32(beta_logits_strides[1]),
                    UInt32(beta_logits_strides[2]),
                    # dt_bias strides
                    UInt32(dt_bias_strides[0]),
                    UInt32(dt_bias_strides[1]),
                    # output strides
                    UInt32(output_strides[1]),
                    UInt32(output_strides[2]),
                    UInt32(output_strides[3]),
                    # state_indices_seq_stride: graph op is spec_mode "none", so
                    # state_indices is 1-D and this stride is unused.
                    UInt32(1),
                    grid_dim=(num_blocks,),
                    block_dim=(kVD,),
                )

        if not dispatched:
            raise Error(
                "kda_decode: unsupported (key_head_dim, value_head_dim) = ("
                + String(key_head_dim)
                + ", "
                + String(value_head_dim)
                + "). Compiled: (32, 32), (128, 128)."
            )


@extensibility.register("kda_chunk")
struct KdaChunk:
    """KDA chunk-parallel prefill forward pass.

    Same algorithm and same numerics-within-tolerance contract as
    `kda_decode` (see `Kernels/lib/kda/chunk_fwd.mojo`'s M2c docstring and
    `test_kda_chunk_parallel.mojo`), but restructures the recurrence so a
    prefill's sequential depth is O(total_T / 16) instead of O(total_T):
    L1 (`kda_chunk_prepare_gpu`) and L3 (`kda_chunk_output_gpu`) run one CTA
    per (chunk, value-head) in parallel; only L2 (`kda_chunk_scan_gpu`)
    carries a sequential dependency, and it carries it per-CHUNK rather than
    per-token. Intended for prefill (multi-token sequences); `kda_decode`
    remains the single-token decode path.

    Takes the SAME tensors as `kda_decode` -- CHUNK_SIZE (16) and the chunk
    map are internal to this op, not part of its call contract, so a caller
    can swap between the two ops without restructuring its inputs.

    Default parameters match `kda_decode`'s (backward-compatible with the S1
    graph op):
      gate_mode    = "original"  (stable-softplus gate)
      beta_mode    = "logits"    (sigmoid applied to beta_logits)
      state_layout = "K_FIRST"   (pool shape [N, HV, K, V])

    Tensor shapes: identical to `kda_decode` (see that op's docstring).

    On AMD the op runs a separate gfx950-only pipeline
    (`kda.amd_chunk_launch`): a fused prologue and a chunk walk over 64-token
    chunks. Its scope is narrower: key_head_dim == value_head_dim == 128,
    num_key_heads == num_value_heads, `gate_mode="safe"`, bf16 q/k/v/output,
    fp32 gates and state, and no compute-bound variant. Both beta modes and
    both state layouts are supported; anything else is rejected at compile
    time or raises.
    """

    @staticmethod
    def execute[
        qkv_dtype: DType,
        gate_dtype: DType,
        state_dtype: DType,
        output_dtype: DType,
        target: StaticString,
        gate_mode: StaticString = "original",
        beta_mode: StaticString = "logits",
        state_layout: StaticString = "K_FIRST",
        use_computebound: Bool = False,
    ](
        output: OutputTensor[dtype=output_dtype, rank=4, ...],
        q: InputTensor[dtype=qkv_dtype, rank=4, ...],
        k: InputTensor[dtype=qkv_dtype, rank=4, ...],
        v: InputTensor[dtype=qkv_dtype, rank=4, ...],
        raw_gate: InputTensor[dtype=gate_dtype, rank=4, ...],
        beta_logits: InputTensor[dtype=gate_dtype, rank=3, ...],
        a_log: InputTensor[dtype=gate_dtype, rank=1, ...],
        dt_bias: InputTensor[dtype=gate_dtype, rank=2, ...],
        cu_seqlens: InputTensor[dtype=DType.int32, rank=1, ...],
        # See `KdaDecode.execute`'s `state_pool` note: mutated in place, so
        # it must stay a `MutableInputTensor`.
        state_pool: MutableInputTensor[dtype=state_dtype, rank=4, ...],
        state_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert gate_mode == "original" or gate_mode == "safe", (
            "KdaChunk: gate_mode must be 'original' or 'safe', got: "
            + gate_mode
        )
        comptime assert beta_mode == "logits" or beta_mode == "probability", (
            "KdaChunk: beta_mode must be 'logits' or 'probability', got: "
            + beta_mode
        )
        comptime assert (
            state_layout == "K_FIRST" or state_layout == "V_FIRST"
        ), (
            "KdaChunk: state_layout must be 'K_FIRST' or 'V_FIRST', got: "
            + state_layout
        )
        comptime assert is_gpu[target](), "kda_chunk is only supported on GPU."

        var num_key_heads = q.dim_size(2)
        var key_head_dim = q.dim_size(3)
        var num_value_heads = v.dim_size(2)
        var value_head_dim = v.dim_size(3)
        var batch_size = state_indices.dim_size(0)
        var total_T = q.dim_size(1)

        debug_assert(
            num_value_heads % num_key_heads == 0,
            "kda_chunk: num_value_heads must be divisible by num_key_heads",
        )
        debug_assert(
            k.dim_size(1) == total_T,
            "kda_chunk: q and k must have same sequence length",
        )
        # These extents bound more here than they do in `kda_decode`, where
        # they only scale in-kernel offsets: `kda_chunk_launch` reinterprets
        # k / raw_gate / beta_logits / dt_bias as flat 2-D TMA views sized
        # from q's head counts and the comptime head dim, so a mismatch makes
        # the descriptor itself span memory the tensor does not own.
        debug_assert(
            k.dim_size(2) == num_key_heads,
            "kda_chunk: k head dim must match q's num_key_heads",
        )
        debug_assert(
            k.dim_size(3) == key_head_dim,
            "kda_chunk: k key dim must match q's key_head_dim",
        )
        debug_assert(
            v.dim_size(1) == total_T,
            "kda_chunk: q and v must have same sequence length",
        )
        debug_assert(
            raw_gate.dim_size(1) == total_T,
            "kda_chunk: raw_gate must have same sequence length as q",
        )
        debug_assert(
            raw_gate.dim_size(2) == num_value_heads,
            "kda_chunk: raw_gate head dim must match num_value_heads",
        )
        debug_assert(
            raw_gate.dim_size(3) == key_head_dim,
            "kda_chunk: raw_gate key dim must match key_head_dim",
        )
        debug_assert(
            beta_logits.dim_size(1) == total_T,
            "kda_chunk: beta_logits must have same sequence length as q",
        )
        debug_assert(
            beta_logits.dim_size(2) == num_value_heads,
            "kda_chunk: beta_logits head dim must match num_value_heads",
        )
        debug_assert(
            a_log.dim_size(0) == num_value_heads,
            "kda_chunk: a_log length must match num_value_heads",
        )
        debug_assert(
            dt_bias.dim_size(0) == num_value_heads,
            "kda_chunk: dt_bias head dim must match num_value_heads",
        )
        debug_assert(
            dt_bias.dim_size(1) == key_head_dim,
            "kda_chunk: dt_bias key dim must match key_head_dim",
        )
        debug_assert(
            cu_seqlens.dim_size(0) == batch_size + 1,
            "kda_chunk: cu_seqlens must have batch_size + 1 entries",
        )
        debug_assert(
            state_pool.dim_size(1) == num_value_heads,
            "kda_chunk: state_pool head dim must match num_value_heads",
        )
        # `kda_chunk_scan_gpu` indexes pool dims 2/3 under the same layout
        # convention as the decode kernel; see `KdaDecode.execute`.
        comptime if state_layout == "K_FIRST":
            debug_assert(
                state_pool.dim_size(2) == key_head_dim,
                "kda_chunk: K_FIRST state_pool dim 2 must equal key_head_dim",
            )
            debug_assert(
                state_pool.dim_size(3) == value_head_dim,
                "kda_chunk: K_FIRST state_pool dim 3 must equal value_head_dim",
            )
        else:
            debug_assert(
                state_pool.dim_size(2) == value_head_dim,
                "kda_chunk: V_FIRST state_pool dim 2 must equal value_head_dim",
            )
            debug_assert(
                state_pool.dim_size(3) == key_head_dim,
                "kda_chunk: V_FIRST state_pool dim 3 must equal key_head_dim",
            )

        comptime if ctx.target.is_amd_gpu():
            # The AMD chunk path (`kda.amd_chunk_launch`): its
            # tile mapping is built for K == V == 128 with one head set, and
            # the bounded ("safe") gate is what keeps its 16-row exp2 bridging
            # finite. It issues CDNA4-only MFMA shapes and assumes wave64.
            comptime assert ctx.target.is_amd_gpu[
                "gfx950"
            ](), "kda_chunk on AMD requires gfx950 (CDNA4); use kda_decode"
            comptime assert gate_mode == "safe", (
                "kda_chunk on AMD supports gate_mode 'safe' only, got: "
                + gate_mode
            )
            comptime assert (
                not use_computebound
            ), "kda_chunk on AMD has no compute-bound variant"
            comptime assert (
                qkv_dtype == .bfloat16
                and output_dtype == .bfloat16
                and gate_dtype == .float32
                and state_dtype == .float32
            ), "kda_chunk on AMD takes bf16 q/k/v/output and fp32 gates/state"
            comptime head_dim = AMD_HEAD_DIM
            if (
                key_head_dim != head_dim
                or value_head_dim != head_dim
                or num_key_heads != num_value_heads
            ):
                raise Error(
                    "kda_chunk on AMD requires key_head_dim == value_head_dim"
                    " == 128 and num_key_heads == num_value_heads, got"
                    " (key_head_dim, value_head_dim, num_key_heads,"
                    " num_value_heads) = ("
                    + String(key_head_dim)
                    + ", "
                    + String(value_head_dim)
                    + ", "
                    + String(num_key_heads)
                    + ", "
                    + String(num_value_heads)
                    + ")"
                )
            # The kernels index every tensor with q's extents, so the shape
            # agreement the generic path only debug-asserts is enforced here.
            # Pool rows are named by `state_indices`, whose values the host
            # cannot see; a pool with fewer rows than sequences cannot give
            # each its own row.
            if (
                q.dim_size(0) != 1
                or k.dim_size(0) != 1
                or k.dim_size(1) != total_T
                or k.dim_size(2) != num_key_heads
                or k.dim_size(3) != head_dim
                or v.dim_size(0) != 1
                or v.dim_size(1) != total_T
                or raw_gate.dim_size(0) != 1
                or raw_gate.dim_size(1) != total_T
                or raw_gate.dim_size(2) != num_value_heads
                or raw_gate.dim_size(3) != head_dim
                or beta_logits.dim_size(0) != 1
                or beta_logits.dim_size(1) != total_T
                or beta_logits.dim_size(2) != num_value_heads
                or a_log.dim_size(0) != num_value_heads
                or dt_bias.dim_size(0) != num_value_heads
                or dt_bias.dim_size(1) != head_dim
                or cu_seqlens.dim_size(0) != batch_size + 1
                or state_pool.dim_size(0) < batch_size
                or state_pool.dim_size(1) != num_value_heads
                or state_pool.dim_size(2) != head_dim
                or state_pool.dim_size(3) != head_dim
                or output.dim_size(0) != 1
                or output.dim_size(1) != total_T
                or output.dim_size(2) != num_value_heads
                or output.dim_size(3) != head_dim
            ):
                raise Error(
                    (
                        "kda_chunk on AMD: k / v / raw_gate / output extents"
                        " must match q's [1, T, H, 128], with beta_logits [1,"
                        " T, H], a_log [H], dt_bias [H, 128], cu_seqlens [N +"
                        " 1] and state_pool [>= N, H, 128, 128] for N ="
                        " len(state_indices); got q "
                    ),
                    q.shape(),
                    ", k ",
                    k.shape(),
                    ", v ",
                    v.shape(),
                    ", raw_gate ",
                    raw_gate.shape(),
                    ", beta_logits ",
                    beta_logits.shape(),
                    ", a_log ",
                    a_log.shape(),
                    ", dt_bias ",
                    dt_bias.shape(),
                    ", cu_seqlens ",
                    cu_seqlens.shape(),
                    ", state_pool ",
                    state_pool.shape(),
                    ", state_indices ",
                    state_indices.shape(),
                    ", output ",
                    output.shape(),
                )
            var q_s = q.strides()
            var k_s = k.strides()
            var v_s = v.strides()
            var g_s = raw_gate.strides()
            var b_s = beta_logits.strides()
            var o_s = output.strides()
            var p_s = state_pool.strides()
            # The kernels issue 16 B loads along the head dim and 8 B stores of
            # the output, so rows must be contiguous and suitably aligned.
            if (
                q_s[3] != 1
                or k_s[3] != 1
                or v_s[3] != 1
                or g_s[3] != 1
                or o_s[3] != 1
                or q_s[1] % 8 != 0
                or q_s[2] % 8 != 0
                or k_s[1] % 8 != 0
                or k_s[2] % 8 != 0
                or v_s[1] % 8 != 0
                or v_s[2] % 8 != 0
                or g_s[1] % 8 != 0
                or g_s[2] % 8 != 0
                or o_s[1] % 4 != 0
                or o_s[2] % 4 != 0
            ):
                raise Error(
                    "kda_chunk on AMD needs unit inner strides and head/token"
                    " strides that are multiples of 8 elements (4 for output)"
                )
            if (
                a_log.strides()[0] != 1
                or cu_seqlens.strides()[0] != 1
                or state_indices.strides()[0] != 1
                or dt_bias.strides()[0] != key_head_dim
                or dt_bias.strides()[1] != 1
                or p_s[3] != 1
                or p_s[2] != head_dim
                or p_s[1] != head_dim * head_dim
                or p_s[0] % 4 != 0
            ):
                raise Error(
                    "kda_chunk on AMD needs a dense a_log / dt_bias /"
                    " cu_seqlens / state_indices and a state_pool whose rows"
                    " are dense [H][128][128] and 16 B aligned"
                )
            comptime scale = Float32(1.0) / sqrt(Float32(head_dim))
            kda_amd_chunk_launch[
                state_layout == "K_FIRST", beta_mode == "logits"
            ](
                ctx,
                num_value_heads,
                total_T,
                batch_size,
                scale,
                KDA_GATE_LOWER_BOUND,
                q.unsafe_ptr[.bfloat16]().as_imm().as_unsafe_any_origin(),
                k.unsafe_ptr[.bfloat16]().as_imm().as_unsafe_any_origin(),
                v.unsafe_ptr[.bfloat16]().as_imm().as_unsafe_any_origin(),
                raw_gate.unsafe_ptr[.float32]().as_imm().as_unsafe_any_origin(),
                beta_logits.unsafe_ptr[.float32]()
                .as_imm()
                .as_unsafe_any_origin(),
                a_log.unsafe_ptr[.float32]().as_imm().as_unsafe_any_origin(),
                dt_bias.unsafe_ptr[.float32]().as_imm().as_unsafe_any_origin(),
                cu_seqlens.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                state_pool.unsafe_ptr[.float32]().as_unsafe_any_origin(),
                state_indices.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                output.unsafe_ptr[.bfloat16]().as_unsafe_any_origin(),
                q_s[1],
                q_s[2],
                k_s[1],
                k_s[2],
                v_s[1],
                v_s[2],
                g_s[1],
                g_s[2],
                b_s[1],
                b_s[2],
                o_s[1],
                o_s[2],
                p_s[0],
            )
        else:
            # Same (key_head_dim, value_head_dim) instantiations as
            # `kda_decode`; see that op for the "one entry per geometry"
            # rationale.
            comptime SUPPORTED_HEAD_DIMS = [(32, 32), (128, 128)]
            var dispatched = False

            comptime for head_dims in SUPPORTED_HEAD_DIMS:
                comptime kKD = head_dims[0]
                comptime kVD = head_dims[1]
                if key_head_dim == kKD and value_head_dim == kVD:
                    dispatched = True
                    kda_chunk_launch[
                        kKD,
                        kVD,
                        gate_mode,
                        beta_mode,
                        state_layout,
                        use_computebound,
                    ](
                        num_key_heads,
                        num_value_heads,
                        batch_size,
                        total_T,
                        output.to_tile_tensor[DType.int64](),
                        q.to_tile_tensor[DType.int64](),
                        k.to_tile_tensor[DType.int64](),
                        v.to_tile_tensor[DType.int64](),
                        raw_gate.to_tile_tensor[DType.int64](),
                        beta_logits.to_tile_tensor[DType.int64](),
                        a_log.to_tile_tensor[DType.int64](),
                        dt_bias.to_tile_tensor[DType.int64](),
                        cu_seqlens.to_device_buffer(ctx),
                        state_pool.to_tile_tensor[DType.int64](),
                        state_indices.to_tile_tensor[DType.int64](),
                        q.to_device_buffer(ctx),
                        k.to_device_buffer(ctx),
                        raw_gate.to_device_buffer(ctx),
                        beta_logits.to_device_buffer(ctx),
                        dt_bias.to_device_buffer(ctx),
                        q.strides(),
                        k.strides(),
                        v.strides(),
                        raw_gate.strides(),
                        beta_logits.strides(),
                        dt_bias.strides(),
                        state_pool.strides(),
                        output.strides(),
                        ctx,
                    )

            if not dispatched:
                raise Error(
                    "kda_chunk: unsupported (key_head_dim, value_head_dim) = ("
                    + String(key_head_dim)
                    + ", "
                    + String(value_head_dim)
                    + "). Compiled: (32, 32), (128, 128)."
                )
