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

"""Gated DeltaNet (linear attention) layer for Qwen3.5.

This implements the linear attention mechanism used in 48 out of 64 layers
of Qwen3.5. It uses a gated delta rule recurrence with causal convolution
for sequence modeling without the quadratic cost of full attention.

State is held in two pools, one per layer per direction, that the kernels
mutate in place at slot ``slot_idx[batch_item]``:

- ``conv_pool``: sliding window of recent inputs for the causal conv1d.
  Shape ``[max_slots, conv_dim, kernel_size - 1]``.
- ``recurrent_pool``: the accumulated key-value memory.
  Shape ``[max_slots, num_v_heads, key_head_dim, value_head_dim]``.

Both prefill (seq_len > 1) and decode (seq_len == 1) are handled by the
same two slot-indexed fused GPU kernels:

- Pass 1 (``gated_delta_conv1d_fwd``): one GPU thread per (batch_item,
  conv_channel). Each thread reads/writes its slot's window in place;
  no gather/scatter, no working buffers.

- Pass 2 (``gated_delta_recurrence_fwd``): one GPU thread per (batch_item,
  value_head, value_dim_element). Each thread owns a KD-element state
  column in registers and iterates over its sequence, applying the
  five-step gated delta rule, then writes the final state back into its
  slot. For decode (seqlen=1) the loop runs once.

This matches vLLM's ``selective_state_update`` design: the kernel does
pointer arithmetic ``state_ptr += slot * stride`` into a long-lived pool,
so there is no per-step pool allocation and no Python-level
gather/scatter loop.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .functional_ops import (
    gated_delta_conv1d_fwd,
    gated_delta_recurrence_fwd,
)


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention layer.

    This replaces standard attention in linear attention layers. It uses:
    1. Input projections (QKV + gate Z + beta B + decay A)
    2. Causal conv1d for local context
    3. Gated delta rule recurrence for long-range memory
    4. Gated RMSNorm on output

    Args:
        hidden_size: Input/output hidden dimension.
        num_key_heads: Number of key heads.
        num_value_heads: Number of value heads.
        key_head_dim: Dimension per key head.
        value_head_dim: Dimension per value head.
        conv_kernel_size: Kernel size for the causal conv1d.
        dtype: Weight data type.
        device: Device for computation.
        rms_norm_eps: Epsilon for the gated RMSNorm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        dtype: DType,
        device: DeviceRef,
        rms_norm_eps: float = 1e-6,
        ssm_dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_kernel_size = conv_kernel_size
        self.dtype = dtype
        self.device = device
        self.ssm_dtype = ssm_dtype

        self.key_dim = key_head_dim * num_key_heads
        self.value_dim = value_head_dim * num_value_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Input projections
        self.in_proj_qkv = Linear(
            in_dim=hidden_size,
            out_dim=self.conv_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.in_proj_z = Linear(
            in_dim=hidden_size,
            out_dim=self.value_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.in_proj_b = Linear(
            in_dim=hidden_size,
            out_dim=num_value_heads,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.in_proj_a = Linear(
            in_dim=hidden_size,
            out_dim=num_value_heads,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        # Causal conv1d weight (depthwise): [conv_dim, 1, kernel_size]
        # Stored as [conv_dim, kernel_size] in the checkpoint
        self.conv1d = Weight(
            "conv1d.weight",
            dtype,
            [self.conv_dim, 1, conv_kernel_size],
            device=DeviceRef.CPU(),
        )

        # Decay parameters
        self.dt_bias = Weight(
            "dt_bias", DType.float32, [num_value_heads], device=DeviceRef.CPU()
        )
        self.A_log = Weight(
            "A_log", DType.float32, [num_value_heads], device=DeviceRef.CPU()
        )

        # Gated RMSNorm: uses DIRECT weight (weight_offset=0.0), not (1+weight).
        # HF Qwen3NextRMSNormGated initializes weight to ones and applies
        # `weight * normalized`, unlike the regular Qwen3.5 RMSNorm which
        # uses (1 + weight) with zero-initialized weights.
        self.norm = RMSNorm(
            value_head_dim,
            dtype=DType.float32,
            eps=rms_norm_eps,
            weight_offset=0.0,
            multiply_before_cast=False,
        )

        # Output projection
        self.out_proj = Linear(
            in_dim=self.value_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        recurrent_pool: BufferValue,
        slot_idx: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass through the Gated DeltaNet layer.

        The conv and recurrent state pools live in graph-input buffers that
        the slot-indexed SSM kernels mutate in place at slot
        ``slot_idx[batch_item]``; there are no graph outputs for the new
        state. This matches vLLM's ``selective_state_update`` design and
        avoids per-decode pool allocation.

        Args:
            x: Input hidden states ``[total_seq_len, hidden_size]``.
            conv_pool: Per-layer conv pool (mutable),
                ``[max_slots, conv_dim, kernel_size - 1]``.
            recurrent_pool: Per-layer recurrent pool (mutable),
                ``[max_slots, num_v_heads, key_head_dim, value_head_dim]``.
            slot_idx: ``[batch_size]`` uint32 slot indices into the pools.
            input_row_offsets: Row offsets ``[batch_size + 1]`` (uint32).

        Returns:
            Output hidden states ``[total_seq_len, hidden_size]``.
        """
        device = x.device
        nv = self.num_value_heads
        vd = self.value_head_dim
        K = self.conv_kernel_size

        # ---- Projections (all tokens, fully parallel) ----
        qkv = self.in_proj_qkv(x)  # [N, conv_dim]
        z = self.in_proj_z(x)  # [N, value_dim]
        b_proj = self.in_proj_b(x)  # [N, nv]
        a_proj = self.in_proj_a(x)  # [N, nv]
        qkv_f32 = ops.cast(qkv, DType.float32)  # [N, conv_dim]

        # ---- Decay / beta params ----
        dt_bias = self.dt_bias.to(device)
        A_log = self.A_log.to(device)
        A = ops.exp(ops.cast(A_log, self.ssm_dtype))
        a_float = ops.cast(a_proj, self.ssm_dtype)  # [N, nv]
        # Stabilised softplus: for x>20 return x directly (avoids float32 overflow)
        x_sp = a_float + ops.cast(dt_bias, self.ssm_dtype)
        softplus_val = ops.where(
            x_sp > ops.constant(20.0, self.ssm_dtype, device=device),
            x_sp,
            ops.log(
                ops.constant(1.0, self.ssm_dtype, device=device) + ops.exp(x_sp)
            ),
        )
        # Cast to float32 for downstream recurrence arithmetic (q/k/v ops always float32).
        decay = ops.exp(ops.cast(-A * softplus_val, DType.float32))  # [N, nv]
        beta = ops.cast(ops.sigmoid(b_proj), DType.float32)  # [N, nv] float32

        # ---- Conv weight (loaded once, shared) ----
        conv_weight_f32 = ops.cast(self.conv1d.to(device), DType.float32)
        conv_weight_flat = ops.reshape(
            conv_weight_f32, [self.conv_dim, K]
        )  # [conv_dim, K]

        # ---- Two-pass fused kernel path (handles both prefill and decode) ----
        # Pass 1: causal conv1d — one GPU thread per (batch_item, conv_channel)
        # Pass 2: gated delta recurrence — one GPU thread per
        #         (batch_item, value_head, vd_element); state column lives in
        #         registers. For decode (seqlen=1) both loops execute once.
        # The pools are mutable graph inputs at the model's native dtype
        # (typically bf16); the kernels cast on read/write so the per-token
        # working tensors stay at fp32.
        offsets_uint32 = ops.cast(input_row_offsets, DType.uint32)
        slot_idx_uint32 = ops.cast(slot_idx, DType.uint32)

        conv_output_ragged = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv_f32,
            conv_weight=conv_weight_flat,
            conv_state=conv_pool,
            slot_idx=slot_idx_uint32,
            input_row_offsets=offsets_uint32,
        )
        conv_output_ragged = ops.silu(conv_output_ragged)

        recurrence_output_flat = gated_delta_recurrence_fwd(
            qkv_conv_output=conv_output_ragged,
            decay_per_token=decay,
            beta_per_token=beta,
            recurrent_state=recurrent_pool,
            slot_idx=slot_idx_uint32,
            input_row_offsets=offsets_uint32,
        )

        output_flat = ops.rebind(
            recurrence_output_flat,
            [x.shape[0], self.value_dim],
            "recurrence_output_flat total_seq_len rebind",
        )

        # ---- Post-process: gated RMSNorm + output projection ----
        output_3d = ops.cast(
            ops.reshape(output_flat, [-1, nv, vd]),
            x.dtype,
        )
        output_normed = self.norm(output_3d)  # [N, nv, vd]

        z_reshaped = ops.reshape(z, [-1, nv, vd])
        z_gate = ops.silu(ops.cast(z_reshaped, DType.float32))
        output_gated = ops.cast(output_normed, DType.float32) * z_gate
        output_gated = ops.cast(output_gated, x.dtype)

        result = self.out_proj(ops.reshape(output_gated, [-1, self.value_dim]))
        return result
