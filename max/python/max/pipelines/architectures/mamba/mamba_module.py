# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""Non-legacy Mamba module classes.

Provides MambaPrefill and MambaStep modules that share weight structure
but have separate forward() methods. Both are compiled independently
with the same state_dict.
"""

from __future__ import annotations

import math
from typing import cast as typing_cast

from max import functional as F
from max.dtype import DType
from max.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

from .functional_ops import (
    causal_conv1d,
    causal_conv1d_update,
    rms_norm_fused_residual,
    selective_scan_fwd,
    selective_scan_update,
)
from .model_config import MambaConfig

# Match legacy layer_norm_fn which hardcodes multiply_before_cast=True
# for all layer norms (but NOT the final norm). See fused_norm.py:136,212.
_LAYER_NORM_MULTIPLY_BEFORE_CAST = True


class MambaSSMModule(Module[[Tensor], Tensor]):
    """Non-legacy SSM mixer with separate prefill() and step() methods.

    Weight attribute names match the legacy MambaSSM so the same
    weight_adapters state_dict works for both.
    """

    in_proj: Linear
    x_proj: Linear
    dt_proj: Linear
    out_proj: Linear
    conv1d_weight: Tensor
    A_log: Tensor
    D: Tensor

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Use prefill() or step() directly")

    def __init__(self, config: MambaConfig, layer_idx: int) -> None:
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        d_state = config.d_state
        dt_rank_raw = config.dt_rank
        if dt_rank_raw is None or isinstance(dt_rank_raw, str):
            dt_rank = math.ceil(hidden / 16)
        else:
            dt_rank = dt_rank_raw
        conv_kernel = config.conv_kernel

        x_proj_dim = config.x_proj_dim
        if x_proj_dim is not None:
            calc = x_proj_dim - 2 * d_state
            if calc > 0:
                dt_rank = calc
        else:
            x_proj_dim = dt_rank + 2 * d_state

        self._intermediate = intermediate
        self._d_state = d_state
        self._dt_rank = dt_rank
        self._conv_width = conv_kernel
        self._hidden = hidden
        self._delta_softplus = True

        self.in_proj = Linear(hidden, 2 * intermediate, bias=config.use_bias)
        self.x_proj = Linear(intermediate, x_proj_dim, bias=False)
        self.dt_proj = Linear(dt_rank, intermediate, bias=True)
        self.out_proj = Linear(intermediate, hidden, bias=config.use_bias)

        self.conv1d_weight = Tensor.zeros([intermediate, conv_kernel])
        if config.use_conv_bias:
            self.conv1d_bias: Tensor | None = Tensor.zeros([intermediate])
        else:
            self.conv1d_bias = None
        self.A_log = Tensor.zeros([intermediate, d_state])
        self.D = Tensor.zeros([intermediate])

    def _get_dt_bias(self) -> Tensor | None:
        b = self.dt_proj.bias
        return b if isinstance(b, Tensor) else None

    def prefill(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Prefill: full sequence through SSM, return output + final states.

        Args:
            x: Normalized hidden states (total_seq_len, hidden_size).

        Returns:
            (output, conv_state, ssm_state):
                output (total_seq_len, hidden_size)
                conv_state (1, intermediate, conv_width)
                ssm_state (1, intermediate, d_state)
        """
        seqlen = x.shape[0]
        d = self._intermediate

        # Project: (seqlen, hidden) -> (seqlen, 2*d)
        xz = self.in_proj(x)
        # Reshape to (1, seqlen, 2*d) -> permute to (1, 2*d, seqlen)
        xz = F.reshape(xz, [1, seqlen, 2 * d]).permute([0, 2, 1])

        x_val_raw, z_val_raw = F.split(xz, [d, d], axis=1)
        x_val = typing_cast(Tensor, x_val_raw)
        z_val = typing_cast(Tensor, z_val_raw)

        # Save conv state: last conv_width values of pre-conv input
        conv_state = typing_cast(Tensor, x_val[:, :, -self._conv_width :])

        # Causal conv1d: (1, d, seqlen) -> (1, d, seqlen)
        x_conv = causal_conv1d(
            x_val, self.conv1d_weight, bias=self.conv1d_bias, activation="silu"
        )

        # Flatten: (1, d, seqlen) -> (seqlen, d)
        x_flat = x_conv.permute([0, 2, 1])
        x_flat = F.reshape(x_flat, [seqlen, d])

        # Project to dt, B, C
        x_dbl = self.x_proj(x_flat)
        dt_rank = int(self._dt_rank)
        d_state = int(self._d_state)
        dt_val_raw, B_val_raw, C_val_raw = F.split(
            x_dbl, [dt_rank, d_state, d_state], axis=-1
        )
        dt_val = typing_cast(Tensor, dt_val_raw)
        B_val = typing_cast(Tensor, B_val_raw)
        C_val = typing_cast(Tensor, C_val_raw)

        # dt projection (weight only -- bias passed separately to kernel)
        dt_proj = typing_cast(Tensor, dt_val @ self.dt_proj.weight.T)
        dt_3d = F.reshape(dt_proj, [1, seqlen, d]).permute([0, 2, 1])

        # B, C: (seqlen, d_state) -> (1, 1, d_state, seqlen)
        B_4d = F.reshape(
            F.reshape(B_val, [1, seqlen, self._d_state]).permute([0, 2, 1]),
            [1, 1, self._d_state, seqlen],
        )
        C_4d = F.reshape(
            F.reshape(C_val, [1, seqlen, self._d_state]).permute([0, 2, 1]),
            [1, 1, self._d_state, seqlen],
        )

        A = -F.exp(self.A_log)
        result = selective_scan_fwd(
            u=x_conv,
            delta=dt_3d,
            A=A,
            B=B_4d,
            C=C_4d,
            D=self.D,
            z=z_val,
            delta_bias=self._get_dt_bias(),
            delta_softplus=self._delta_softplus,
            return_last_state=True,
        )
        assert isinstance(result, tuple)
        output, ssm_state = result

        # (1, d, seqlen) -> (seqlen, d) -> out_proj -> (seqlen, hidden)
        out_flat = F.reshape(output.permute([0, 2, 1]), [seqlen, d])
        return self.out_proj(out_flat), conv_state, ssm_state

    def step(
        self, x: Tensor, conv_state: Tensor, ssm_state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Step: single-token update using cached states.

        Args:
            x: Normalized hidden states (batch, hidden_size).
            conv_state: (batch, intermediate, conv_width).
            ssm_state: (batch, intermediate, d_state).

        Returns:
            (output, updated_conv_state, updated_ssm_state).
        """
        batch = x.shape[0]
        d = self._intermediate

        xz = self.in_proj(x)
        x_val_raw, z_val_raw = F.split(xz, [d, d], axis=-1)
        x_val = typing_cast(Tensor, x_val_raw)
        z_val = typing_cast(Tensor, z_val_raw)

        # Conv update: (batch, d) -> (batch, d, 1)
        x_3d = F.reshape(x_val, [batch, d, 1])
        x_conv, updated_conv = causal_conv1d_update(
            x_3d,
            conv_state,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            activation="silu",
        )
        x_flat = F.reshape(x_conv, [batch, d])

        # Project to dt, B, C
        x_dbl = self.x_proj(x_flat)
        dt_rank = int(self._dt_rank)
        d_state = int(self._d_state)
        dt_val_raw, B_val_raw, C_val_raw = F.split(
            x_dbl, [dt_rank, d_state, d_state], axis=-1
        )
        dt_val = typing_cast(Tensor, dt_val_raw)
        B_val = typing_cast(Tensor, B_val_raw)
        C_val = typing_cast(Tensor, C_val_raw)
        dt_proj = typing_cast(Tensor, dt_val @ self.dt_proj.weight.T)

        B_grouped = F.reshape(B_val, [batch, 1, self._d_state])
        C_grouped = F.reshape(C_val, [batch, 1, self._d_state])

        A = -F.exp(self.A_log)
        updated_ssm, y = selective_scan_update(
            state=ssm_state,
            x=x_flat,
            dt=dt_proj,
            A=A,
            B=B_grouped,
            C=C_grouped,
            D=self.D,
            z=z_val,
            dt_bias=self._get_dt_bias(),
            dt_softplus=self._delta_softplus,
        )

        return self.out_proj(y), updated_conv, updated_ssm


class MambaLayer(Module[[Tensor], Tensor]):
    """Norm + SSM mixer for a single Mamba block."""

    norm: RMSNorm
    mixer: MambaSSMModule

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Use via MambaPrefill or MambaStep")

    def __init__(self, config: MambaConfig, layer_idx: int) -> None:
        eps = config.rms_norm_eps or 1e-5
        self.norm = RMSNorm(config.hidden_size, eps=eps)
        self.mixer = MambaSSMModule(config, layer_idx)


class MambaBase(Module[[Tensor, Tensor], tuple[Tensor, ...]]):
    """Shared weight structure for prefill and step modules."""

    embedding: Embedding
    layers: ModuleList[MambaLayer]
    norm: RMSNorm

    def forward(
        self, tokens: Tensor, aux: Tensor
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError("Use MambaPrefill or MambaStep")

    def __init__(self, config: MambaConfig) -> None:
        self.embedding = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList(
            [MambaLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        eps = config.rms_norm_eps or 1e-5
        self.norm = RMSNorm(config.hidden_size, eps=eps)
        self._num_layers = config.num_hidden_layers
        self._residual_in_fp32 = config.residual_in_fp32


class MambaPrefill(MambaBase):
    """Prefill: processes full prompt, extracts per-layer states."""

    def forward(
        self, tokens: Tensor, input_row_offsets: Tensor
    ) -> tuple[Tensor, ...]:
        h = self.embedding(tokens)

        all_states: list[Tensor] = []

        for i in range(self._num_layers):
            layer = self.layers[i]
            if i == 0:
                residual = h
                h_normed = rms_norm(
                    h, layer.norm.weight, layer.norm.eps,
                    multiply_before_cast=_LAYER_NORM_MULTIPLY_BEFORE_CAST,
                )
            else:
                # Match legacy: cast to fp32 for fused add+norm,
                # then cast back to model dtype.
                if self._residual_in_fp32:
                    h_fp32 = h.cast(DType.float32)
                    res_fp32 = residual.cast(DType.float32)
                    h_normed, residual = rms_norm_fused_residual(
                        h_fp32, res_fp32,
                        layer.norm.weight, layer.norm.eps,
                        multiply_before_cast=_LAYER_NORM_MULTIPLY_BEFORE_CAST,
                    )
                    h_normed = h_normed.cast(h.dtype)
                    residual = residual.cast(h.dtype)
                else:
                    h_normed, residual = rms_norm_fused_residual(
                        h, residual, layer.norm.weight, layer.norm.eps,
                        multiply_before_cast=_LAYER_NORM_MULTIPLY_BEFORE_CAST,
                    )
            h, conv_s, ssm_s = layer.mixer.prefill(h_normed)
            all_states.append(conv_s)
            all_states.append(ssm_s)

        # Final: add residual + norm (final norm uses default
        # multiply_before_cast=False, matching legacy RMSNorm.__call__)
        h = h + residual
        h = rms_norm(h, self.norm.weight, self.norm.eps)

        # Gather last-token hidden states per batch element
        last_indices = input_row_offsets[1:] - 1
        last_h = F.gather(h, last_indices, axis=0)

        # Output projection via tied embedding weight
        logits = (last_h @ self.embedding.weight.T).cast(DType.float32)

        return (logits, *all_states)


class MambaStep(MambaBase):
    """Step: processes single new token using cached states."""

    def forward(
        self, tokens: Tensor, *layer_states: Tensor
    ) -> tuple[Tensor, ...]:
        num_layers = self._num_layers
        conv_states = [layer_states[2 * i] for i in range(num_layers)]
        ssm_states = [layer_states[2 * i + 1] for i in range(num_layers)]

        h = self.embedding(tokens)

        updated: list[Tensor] = []

        for i in range(num_layers):
            layer = self.layers[i]
            if i == 0:
                residual = h
                h_normed = rms_norm(
                    h, layer.norm.weight, layer.norm.eps,
                    multiply_before_cast=_LAYER_NORM_MULTIPLY_BEFORE_CAST,
                )
            else:
                if self._residual_in_fp32:
                    h_fp32 = h.cast(DType.float32)
                    res_fp32 = residual.cast(DType.float32)
                    h_normed, residual = rms_norm_fused_residual(
                        h_fp32, res_fp32,
                        layer.norm.weight, layer.norm.eps,
                        multiply_before_cast=_LAYER_NORM_MULTIPLY_BEFORE_CAST,
                    )
                    h_normed = h_normed.cast(h.dtype)
                    residual = residual.cast(h.dtype)
                else:
                    h_normed, residual = rms_norm_fused_residual(
                        h, residual, layer.norm.weight, layer.norm.eps,
                        multiply_before_cast=_LAYER_NORM_MULTIPLY_BEFORE_CAST,
                    )
            h, conv_s, ssm_s = layer.mixer.step(
                h_normed, conv_states[i], ssm_states[i]
            )
            updated.append(conv_s)
            updated.append(ssm_s)

        h = h + residual
        h = rms_norm(h, self.norm.weight, self.norm.eps)

        logits = (h @ self.embedding.weight.T).cast(DType.float32)

        return (logits, *updated)
