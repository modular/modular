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
"""Engram: adds a gated n-gram memory lookup to the residual stream.

Rows gathered from a large hashed n-gram embedding table are projected into
one key per hyper-connection copy and one shared value. A per-copy gate,
computed from the similarity between the hidden state and its key, decides
how much of the value to add to each copy.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.kernels import block_scales_interleave, dynamic_block_scaled_matmul
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from .engram_lookup import EngramOperand

# One E8M0 scale per 32 elements of the gathered rows and of `wkv`.
_MXFP8_BLOCK_SIZE = 32


def _wkv_quant_config() -> QuantConfig:
    """The MXFP8 ``QuantConfig`` for ``engram.wkv``.

    ``block_size=(1, 32)`` declares ``weight_scale`` as
    ``[out_dim, in_dim / 32]`` (one scale per row per 32 columns), not the
    checkpoint's 32x32 block shape.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        # This Linear is always quantized; there are no per-layer sets.
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.MXFP8,
    )


class Engram(Module):
    """Projects, gates, and adds one Engram layer's contribution.

    Owns the ``wkv`` projection (an MXFP8 linear) and the gate's
    ``q_weight`` / ``k_weight``. The embedding table itself is not owned
    here: each call receives the already-gathered rows as an
    :class:`EngramOperand`, still in MXFP8 so they are never dequantized.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        n_hash_cols: int,
        head_dim: int,
        rms_norm_eps: float,
        device: DeviceRef,
        dtype: DType = DType.bfloat16,
    ) -> None:
        """Declares the ``wkv`` and gate weights.

        Args:
            hidden_size: Width of one hyper-connection copy.
            hc_mult: Number of hyper-connection copies.
            n_hash_cols: Table rows gathered per position
                (``(max_ngram_size - 1) * n_heads``).
            head_dim: Width of one table row.
            rms_norm_eps: Floor for the gate's two RMS denominators.
            device: Device the weights live on.
            dtype: Compute dtype of ``wkv``'s output and the residual stream.

        Raises:
            ValueError: If ``head_dim`` is not a multiple of the MXFP8 block
                size.
        """
        super().__init__()
        if head_dim % _MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"head_dim {head_dim} must be a multiple of the MXFP8 block "
                f"size {_MXFP8_BLOCK_SIZE}"
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.n_hash_cols = n_hash_cols
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.device = device
        self.dtype = dtype
        # No `name=`: the attribute already prefixes "wkv.".
        self.wkv = Linear(
            in_dim=n_hash_cols * head_dim,
            out_dim=hidden_size * (hc_mult + 1),
            dtype=DType.float8_e4m3fn,
            device=device,
            quant_config=_wkv_quant_config(),
        )
        self.q_weight = Weight(
            name="q_weight",
            dtype=DType.bfloat16,
            shape=(hc_mult, hidden_size),
            device=device,
        )
        self.k_weight = Weight(
            name="k_weight",
            dtype=DType.bfloat16,
            shape=(hc_mult, hidden_size),
            device=device,
        )

    def shard_columns(self, num_devices: int) -> None:
        """Makes ``wkv`` column-parallel over ``num_devices`` ranks.

        Each rank keeps a contiguous share of the output columns (and their
        scale rows); :meth:`project` then returns only that share, so the
        full output must be gathered on the last axis before :meth:`gate`.
        """
        out_dim = self.hidden_size * (self.hc_mult + 1)
        if out_dim % (num_devices * _MXFP8_BLOCK_SIZE):
            raise ValueError(
                f"wkv's {out_dim} output columns do not split into "
                f"{num_devices} shards of whole {_MXFP8_BLOCK_SIZE}-blocks"
            )
        self.wkv.sharding_strategy = ShardingStrategy.rowwise(num_devices)

    def project(self, operand: EngramOperand) -> TensorValue:
        """Applies ``wkv`` to the gathered rows directly from their MXFP8 codes.

        Returns:
            ``[T, N]`` in this module's dtype, where ``N`` is this rank's
            share of ``hidden_size * (hc_mult + 1)``.
        """
        weight = TensorValue(self.wkv.weight)
        assert self.wkv.weight_scale is not None
        weight_scale = block_scales_interleave(
            TensorValue(self.wkv.weight_scale).to(operand.codes.device),
            sf_vector_size=_MXFP8_BLOCK_SIZE,
        )
        return dynamic_block_scaled_matmul(
            operand.codes,
            weight.to(operand.codes.device),
            operand.scales,
            weight_scale,
            sf_vector_size=_MXFP8_BLOCK_SIZE,
            out_type=self.dtype,
        )

    def _gate(
        self,
        hidden_states: TensorValue,
        kv: TensorValue,
        token_mask: TensorValue | None,
    ) -> TensorValue:
        """Gates ``kv`` into ``hidden_states``, per hyper-connection copy.

        For copy ``c`` with ``h = hidden_states[t, c]``, ``key`` its key and
        ``rms(v) = rsqrt(mean(v * v) + rms_norm_eps)``::

            x = sum(h * q[c] * k[c] * key) * rms(h) * rms(key) / sqrt(hidden)
            gate = sigmoid(sign(x) * sqrt(max(|x|, 1e-6)))
            out = h + gate * value

        One fused kernel computes it in float32 from the bf16 inputs; it
        supports ``hidden_size`` 5120 and ``hc_mult`` 4.

        Args:
            hidden_states: ``[T, hc_mult, hidden_size]``.
            kv: ``[T, hidden_size * (hc_mult + 1)]``, ``wkv``'s output: one
                ``hidden_size``-wide key per copy, then one shared
                ``hidden_size``-wide value.
            token_mask: ``[T]`` bool, ``False`` forces the gate to 0 at that
                position. ``None`` if every position is live.

        Returns:
            ``[T, hc_mult, hidden_size]`` in ``hidden_states``'s dtype.
        """
        device = hidden_states.device
        values = [
            hidden_states,
            kv,
            TensorValue(self.q_weight).to(device),
            TensorValue(self.k_weight).to(device),
        ]
        name = "mo.engram.gate"
        if token_mask is not None:
            values.append(
                token_mask.rebind((hidden_states.shape[0],)).to(device)
            )
            name = "mo.engram.gate_masked"
        return ops.custom(
            name,
            device=device,
            values=values,
            out_types=[hidden_states.type],
            parameters={"epsilon": self.rms_norm_eps},
        )[0].tensor

    def gate(
        self,
        hidden_states: TensorValue,
        kv: TensorValue,
        token_mask: TensorValue | None = None,
    ) -> TensorValue:
        """Validates shapes, then gates ``kv`` into ``hidden_states``."""
        if (
            hidden_states.shape[-2] != self.hc_mult
            or hidden_states.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                f"hidden_states must be [T, {self.hc_mult}, "
                f"{self.hidden_size}], got {hidden_states.shape}"
            )
        kv = kv.rebind(
            (hidden_states.shape[0], self.hidden_size * (self.hc_mult + 1))
        )
        return self._gate(hidden_states, kv, token_mask)

    def __call__(
        self,
        hidden_states: TensorValue,
        operand: EngramOperand,
        token_mask: TensorValue | None = None,
    ) -> TensorValue:
        """Projects through an unsharded ``wkv`` and gates into the residual.

        Args:
            hidden_states: ``[T, hc_mult, hidden_size]``, every
                hyper-connection copy of the residual stream.
            operand: This layer's gathered table rows.
            token_mask: ``[T]`` bool, ``False`` forces the gate to 0 at that
                position. ``None`` if every position is live.

        Returns:
            ``[T, hc_mult, hidden_size]`` in ``hidden_states``'s dtype.
        """
        return self.gate(hidden_states, self.project(operand), token_mask)
