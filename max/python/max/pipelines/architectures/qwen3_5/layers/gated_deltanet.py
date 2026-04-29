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

The layer maintains two types of state:
- conv_state: sliding window of recent inputs for the causal conv1d
  Shape: [batch_size, conv_dim, kernel_size - 1]
- recurrent_state: the accumulated key-value memory
  Shape: [batch_size, num_v_heads, key_head_dim, value_head_dim]

Both prefill (seq_len > 1) and generation (seq_len = 1) are supported.
Batch size >= 1 is supported via two execution paths selected at runtime:

- Decode path (all sequences have exactly 1 token, total_seq_len == batch_size):
  Both conv1d and the delta-rule recurrence are vectorised over the batch
  dimension using standard tensor ops. No sequential loop is required.

- Prefill path (at least one sequence has more than 1 token):
  A two-pass kernel approach is used.  Pass 1 (gated_delta_conv1d_fwd)
  runs the causal conv1d in a single GPU launch.  Pass 2
  (gated_delta_recurrence_fwd) runs the gated delta rule recurrence in a
  second GPU launch.  Each pass dispatches one GPU thread per independent
  unit of work so the full batch is processed concurrently.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, Weight, ops
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
        conv_state: TensorValue,
        recurrent_state: TensorValue,
        input_row_offsets: TensorValue,
        is_decode: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Forward pass through the Gated DeltaNet layer.

        Dispatches between a fast vectorised decode path (all sequences have
        exactly one token) and a two-pass kernel prefill path (one or more
        sequences with potentially different lengths) using ops.cond at runtime.
        Args:
            x: Input hidden states [total_seq_len, hidden_size].
            conv_state: Conv state [batch_size, conv_dim, kernel_size - 1].
            recurrent_state: Recurrent state
                [batch_size, num_v_heads, key_head_dim, value_head_dim].
            input_row_offsets: Row offsets [batch_size + 1] (uint32).
            is_decode: Pre-computed bool scalar on CPU. True when every
                sequence has exactly one token (total_seq_len == batch_size).

        Returns:
            Tuple of (output, updated_conv_state, updated_recurrent_state).
        """
        device = x.device
        nv = self.num_value_heads
        kd = self.key_head_dim
        vd = self.value_head_dim
        K = self.conv_kernel_size

        # ---- Projections (all tokens, fully parallel) ----
        qkv = self.in_proj_qkv(x)  # [N, conv_dim]
        z = self.in_proj_z(x)  # [N, value_dim]
        b_proj = self.in_proj_b(x)  # [N, nv]
        a_proj = self.in_proj_a(x)  # [N, nv]
        qkv_f32 = ops.cast(qkv, DType.float32)  # [N, conv_dim]

        # ---- Decay / beta params (shared between decode and prefill) ----
        dt_bias = self.dt_bias.to(device)
        A_log = self.A_log.to(device)
        # this cast is hard-coded, however it can come from `mamba_ssm_dtype` in HF config
        A = ops.exp(ops.cast(A_log, DType.float32))
        a_float = ops.cast(a_proj, DType.float32)  # [N, nv]
        # Stabilised softplus: for x>20 return x directly (avoids float32 overflow)
        x_sp = a_float + dt_bias
        softplus_val = ops.where(
            x_sp > ops.constant(20.0, DType.float32, device=device),
            x_sp,
            ops.log(
                ops.constant(1.0, DType.float32, device=device) + ops.exp(x_sp)
            ),
        )
        decay = ops.exp(ops.cast(-A * softplus_val, DType.float32))  # [N, nv]
        beta = ops.cast(ops.sigmoid(b_proj), DType.float32)  # [N, nv] float32

        # ---- Conv weight (loaded once, shared) ----
        conv_weight_f32 = ops.cast(self.conv1d.to(device), DType.float32)
        conv_weight_flat = ops.reshape(
            conv_weight_f32, [self.conv_dim, K]
        )  # [conv_dim, K]

        # Cast input states to float32 for computation.
        # States are stored in the model's native dtype (typically bfloat16);
        # all recurrence arithmetic runs in float32 for numerical accuracy.
        # The output states are cast back to the original dtype before returning.
        conv_state_f32 = ops.cast(conv_state, DType.float32)
        recurrent_state_f32 = ops.cast(recurrent_state, DType.float32)

        # ---- Runtime dispatch: decode vs prefill ----
        # is_decode is pre-computed by the caller and shared across all
        # linear attention layers to avoid 48x redundant graph ops.

        out_types = [
            TensorType(
                DType.float32, [x.shape[0], self.value_dim], device
            ),  # output_flat
            TensorType(
                DType.float32,
                [conv_state.shape[0], self.conv_dim, K - 1],
                device,
            ),  # new_conv_state
            TensorType(
                DType.float32,
                [recurrent_state.shape[0], nv, kd, vd],
                device,
            ),  # new_recurrent_state
        ]

        # ------------------------------------------------------------------
        # DECODE BRANCH: all seqlen_b == 1 → vectorised batch ops, no loop
        # ------------------------------------------------------------------
        def _decode_branch() -> list[TensorValue]:
            # B = batch_size (== total_seq_len in this branch)
            B = conv_state_f32.shape[0]

            # Conv1d (vectorised over batch)
            # Rebind [N, conv_dim] → [B, conv_dim] (N==B in decode)
            qkv_b = ops.rebind(
                qkv_f32,
                [B, self.conv_dim],
                "decode: total_seq_len must equal batch_size",
            )  # [B, conv_dim]
            qkv_3d = ops.unsqueeze(qkv_b, -1)  # [B, conv_dim, 1]
            padded = ops.concat(
                [conv_state_f32, qkv_3d], axis=2
            )  # [B, conv_dim, K]

            conv_w_3d = ops.unsqueeze(conv_weight_flat, 0)  # [1, conv_dim, K]
            conv_out_b = ops.silu(
                ops.squeeze(ops.sum(padded * conv_w_3d, axis=-1), -1)
            )  # [B, conv_dim]

            # New conv state: drop oldest, keep last K-1 steps
            new_conv_state_d = ops.slice_tensor(
                padded, [slice(None), slice(None), slice(1, None)]
            )  # [B, conv_dim, K-1]

            # Rebind back to [total_seq_len, conv_dim] for type consistency
            conv_out = ops.rebind(
                conv_out_b,
                [x.shape[0], self.conv_dim],
                "decode: conv_out total_seq_len rebind",
            )  # [N, conv_dim]

            # Q/K/V split (inferred shape = B, where total_seq_len == batch_size in decode)
            query_raw = ops.reshape(
                ops.slice_tensor(
                    conv_out, [slice(None), slice(0, self.key_dim)]
                ),
                [-1, self.num_key_heads, kd],
            )  # [batch_size, num_key_heads, key_head_dim]
            key_raw = ops.reshape(
                ops.slice_tensor(
                    conv_out,
                    [slice(None), slice(self.key_dim, self.key_dim * 2)],
                ),
                [-1, self.num_key_heads, kd],
            )  # [batch_size, num_key_heads, key_head_dim]
            value_raw = ops.reshape(
                ops.slice_tensor(
                    conv_out,
                    [
                        slice(None),
                        slice(
                            self.key_dim * 2,
                            self.key_dim * 2 + self.value_dim,
                        ),
                    ],
                ),
                [-1, nv, vd],
            )  # [batch_size, num_value_heads, value_head_dim]

            # Rebind to batch dimension (decode: total_seq_len == batch_size)
            query_raw = ops.rebind(
                query_raw, [B, self.num_key_heads, kd]
            )  # [B, num_key_heads, key_head_dim]
            key_raw = ops.rebind(
                key_raw, [B, self.num_key_heads, kd]
            )  # [B, num_key_heads, key_head_dim]
            value_raw = ops.rebind(
                value_raw, [B, nv, vd]
            )  # [B, num_value_heads, value_head_dim]

            # GQA head expansion: repeat_interleave each key head heads_ratio
            # times so head ordering matches HF: [h0,h0,...,h1,h1,...].
            # ops.tile has no GPU kernel (GEX-2056); use broadcast_to instead (zero-copy, GPU-native).
            heads_ratio = nv // self.num_key_heads
            if heads_ratio > 1:
                # Expand key heads to value heads via GQA (grouped query attention)
                # unsqueeze: [B, num_key_heads, kd] -> [B, num_key_heads, 1, kd]
                # broadcast: [B, num_key_heads, 1, kd] -> [B, num_key_heads, heads_ratio, kd]
                # reshape:   [B, num_key_heads, heads_ratio, kd] -> [B, num_value_heads, kd]
                nk = self.num_key_heads
                query_raw = ops.reshape(
                    ops.broadcast_to(
                        ops.unsqueeze(query_raw, 2), (B, nk, heads_ratio, kd)
                    ),
                    [B, nv, kd],
                )  # [B, num_value_heads, key_head_dim]
                key_raw = ops.reshape(
                    ops.broadcast_to(
                        ops.unsqueeze(key_raw, 2), (B, nk, heads_ratio, kd)
                    ),
                    [B, nv, kd],
                )  # [B, num_value_heads, key_head_dim]

            # L2 normalise Q and K, scale Q
            q_f32 = ops.cast(
                query_raw, DType.float32
            )  # [B, num_value_heads, key_head_dim]
            k_f32 = ops.cast(
                key_raw, DType.float32
            )  # [B, num_value_heads, key_head_dim]
            v_f32 = ops.cast(
                value_raw, DType.float32
            )  # [B, num_value_heads, value_head_dim]

            q_sq = ops.sum(q_f32 * q_f32, axis=-1)
            q_f32 = q_f32 * ops.rsqrt(
                q_sq + ops.constant(1e-6, DType.float32, device=device)
            )
            k_sq = ops.sum(k_f32 * k_f32, axis=-1)
            k_f32 = k_f32 * ops.rsqrt(
                k_sq + ops.constant(1e-6, DType.float32, device=device)
            )
            scale = 1.0 / (kd**0.5)
            q_f32 = q_f32 * ops.constant(scale, DType.float32, device=device)

            # Rebind beta/decay to [B, nv]
            beta_b = ops.rebind(beta, [B, nv])
            decay_b = ops.rebind(decay, [B, nv])

            # Use q/k/v directly (already [B, nv, kd/vd] after rebind)
            q_b = q_f32
            k_b = k_f32
            v_b = v_f32

            # Batched single-step delta rule (fully vectorised, no loop)
            decay_4d = ops.unsqueeze(
                ops.unsqueeze(decay_b, -1), -1
            )  # [B, nv, 1, 1]
            decayed = recurrent_state_f32 * decay_4d  # [B, nv, kd, vd]
            k_4d = ops.unsqueeze(k_b, -1)  # [B, nv, kd, 1]
            kv_mem = ops.squeeze(
                ops.sum(decayed * k_4d, axis=2), 2
            )  # [B, nv, vd]
            beta_3d = ops.unsqueeze(beta_b, -1)  # [B, nv, 1]
            delta = beta_3d * (v_b - kv_mem)  # [B, nv, vd]
            delta_4d = ops.unsqueeze(delta, 2)  # [B, nv, 1, vd]
            new_state = decayed + k_4d * delta_4d  # [B, nv, kd, vd]
            q_4d = ops.unsqueeze(q_b, -1)  # [B, nv, kd, 1]
            out_3d = ops.squeeze(
                ops.sum(new_state * q_4d, axis=2), 2
            )  # [B, nv, vd]

            output_flat_b = ops.reshape(
                out_3d, [B, self.value_dim]
            )  # [B, value_dim]
            output_flat = ops.rebind(
                output_flat_b,
                [x.shape[0], self.value_dim],
                "decode: output_flat total_seq_len rebind",
            )  # [N, value_dim]

            return [output_flat, new_conv_state_d, new_state]

        # ------------------------------------------------------------------
        # PREFILL BRANCH: batch_size >= 1, two-pass kernel approach
        #
        # Pass 1 — gated_delta_conv1d_fwd:
        #   One GPU thread per (batch_item, conv_channel).  Each thread
        #   processes its full sequence sequentially, reading from
        #   conv_state for the initial look-back window.  Writes the
        #   raw (pre-activation) conv output and the updated conv state.
        #
        # Pass 2 — gated_delta_recurrence_fwd:
        #   One GPU thread per (batch_item, value_head, value_dim_element).
        #   Each thread owns a KD-element state column in registers and
        #   iterates over its sequence, applying the five-step gated delta
        #   rule.  Q/K L2 normalisation and GQA head expansion are fused.
        #
        # ------------------------------------------------------------------
        def _prefill_branch() -> list[TensorValue]:
            # Cast offsets to uint32 once (Mojo ops expect uint32)
            offsets_uint32 = ops.cast(input_row_offsets, DType.uint32)

            # ── Pass 1: causal conv1d ─────────────────────────────────
            # Inputs:
            #   qkv_f32         [total_N, conv_dim]    — projected QKV
            #   conv_weight_flat [conv_dim, K]          — depthwise weights
            #   conv_state       [B, conv_dim, K-1]     — initial state
            #   offsets_uint32   [B+1]                  — ragged offsets
            # Outputs:
            #   conv_output_ragged [total_N, conv_dim]  — raw (pre-activation)
            #   new_conv_state_prefill [B, conv_dim, K-1]
            conv_output_ragged, new_conv_state_prefill = gated_delta_conv1d_fwd(
                qkv_input_ragged=qkv_f32,
                conv_weight=conv_weight_flat,
                conv_state_in=conv_state_f32,
                input_row_offsets=offsets_uint32,
            )

            # SiLU activation (moved from kernel to model level)
            conv_output_ragged = ops.silu(conv_output_ragged)

            # ── Pass 2: gated delta rule recurrence ───────────────────
            # Inputs:
            #   conv_output_ragged [total_N, conv_dim] — from Pass 1 (SiLU-activated)
            #   decay              [total_N, nv]        — exp(-softplus)
            #   beta               [total_N, nv]        — sigmoid
            #   recurrent_state    [B, nv, kd, vd]      — initial state
            #   offsets_uint32     [B+1]                 — ragged offsets
            # Outputs:
            #   recurrence_output_flat [total_N, value_dim]
            #   new_recurrent_state_prefill [B, nv, kd, vd]
            recurrence_output_flat, new_recurrent_state_prefill = (
                gated_delta_recurrence_fwd(
                    qkv_conv_output=conv_output_ragged,
                    decay_per_token=decay,
                    beta_per_token=beta,
                    recurrent_state_in=recurrent_state_f32,
                    input_row_offsets=offsets_uint32,
                )
            )

            output_prefill = ops.rebind(
                recurrence_output_flat,
                [x.shape[0], self.value_dim],
                "prefill kernel: recurrence_output_flat total_seq_len rebind",
            )

            return [
                output_prefill,
                new_conv_state_prefill,
                new_recurrent_state_prefill,
            ]

        output_flat, new_conv_state, new_recurrent_state = ops.cond(
            is_decode,
            out_types,
            _decode_branch,
            _prefill_branch,
        )

        # Cast updated states back to the original storage dtype (model dtype,
        # typically bfloat16).  Computation runs in float32 above; storing in
        # model dtype halves state memory vs float32.
        new_conv_state = ops.cast(new_conv_state, conv_state.dtype)
        new_recurrent_state = ops.cast(
            new_recurrent_state, recurrent_state.dtype
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
        return result, new_conv_state, new_recurrent_state
