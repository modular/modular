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
"""Graph-op binding for the KDA (Kimi Delta Attention) decode recurrence.

The kernel math lives in `//Kernels/lib/kda` (`kda.recurrent`); only the
`@extensibility.register` wrapper lives here, mirroring the `msa.mojo`
binding for `//Kernels/lib/msa`. Registration has to be declared inside the
built-in kernel library itself -- importing the struct from the standalone
lib does not put the op in the graph compiler's registry, so a served graph
resolving `kda_decode` needs this declaration and not just the dependency.
"""

import extensibility

from extensibility import (
    InputTensor,
    OutputTensor,
    _MutableInputTensor as MutableInputTensor,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu

from kda.recurrent import kda_decode_gpu


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
        var state_pool_strides = state_pool.strides()
        var output_strides = output.strides()

        var gpu_ctx = ctx
        # One CTA per (batch_item, value_head); block has value_head_dim threads.
        var num_blocks = batch_size * num_value_heads

        # The kernel addresses the pool as dim1=outer / dim2=inner regardless
        # of which physical axis is outer; the caller picks which by building
        # the pool with the layout matching `state_layout`.
        var state_dim1_stride = UInt32(state_pool_strides[2])
        var state_dim2_stride = UInt32(state_pool_strides[3])

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
                    # state_pool strides
                    UInt32(state_pool_strides[0]),
                    UInt32(state_pool_strides[1]),
                    state_dim1_stride,
                    state_dim2_stride,
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
