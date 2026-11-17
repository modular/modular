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
"""The decoder's mixture of experts, over the checkpoint's expert stacks."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_linear import QuantizedMLP
from max.nn.common_layers.quant_moe import TensorParallelMoE
from max.nn.common_layers.quant_tensor import FP8BlockTensor, QuantAwareTensor
from max.nn.quant_config import QuantConfig
from max.nn.sequential import ModuleList
from max.tensor import Tensor, default_device, default_dtype, defaults
from max.tensor.sharding import DeviceMapping, Sharded

__all__ = ["Qwen3VLMoE"]


class Qwen3VLRouter(MoEGate):
    """Softmax top-k routing, renormalized over the chosen experts."""

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Returns each token's top-k experts and their weights."""
        logits = self.gate_score(hidden_state).cast(DType.float32)
        weights, idx = F.top_k(
            F.softmax(logits, axis=-1), k=self.num_experts_per_token, axis=-1
        )
        weights = weights / (F.sum(weights, axis=-1) + 1e-20)
        return idx, weights.cast(hidden_state.dtype)


class Qwen3VLStackedExperts(ModuleList[QuantizedMLP]):
    """Every routed expert's weights, stacked as the checkpoint stores them.

    ``gate_up_proj`` is ``[E, hidden, 2 * moe_dim]``, each expert's gate
    columns then its up columns, and ``down_proj`` is ``[E, moe_dim,
    hidden]``. An FP8 checkpoint adds each stack's block scales.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        moe_dim: int,
        quant_config: QuantConfig | None,
    ) -> None:
        super().__init__()
        dtype = DType.float8_e4m3fn if quant_config is not None else None
        self.gate_up_proj = Tensor.zeros(
            [num_experts, hidden_dim, 2 * moe_dim], dtype=dtype
        )
        self.down_proj = Tensor.zeros(
            [num_experts, moe_dim, hidden_dim], dtype=dtype
        )
        if quant_config is not None:
            block_size = quant_config.weight_scale.block_size
            assert block_size is not None
            rows, cols = block_size
            scale_dtype = quant_config.weight_scale.dtype
            self.gate_up_proj_scale_inv = Tensor.zeros(
                [num_experts, hidden_dim // rows, 2 * moe_dim // cols],
                dtype=scale_dtype,
            )
            self.down_proj_scale_inv = Tensor.zeros(
                [num_experts, moe_dim // rows, hidden_dim // cols],
                dtype=scale_dtype,
            )


def _map_leaves(
    weight: QuantAwareTensor, fn: Callable[[Tensor], Tensor]
) -> QuantAwareTensor:
    """Applies ``fn`` to ``weight`` and, when it is FP8, to its scales."""
    if isinstance(weight, FP8BlockTensor):
        return FP8BlockTensor(
            weight=fn(weight.weight),
            weight_scale_inv=fn(weight.weight_scale_inv),
            block_size=weight.block_size,
        )
    assert isinstance(weight, Tensor)
    return fn(weight)


def _matmul_operand(weight: QuantAwareTensor) -> QuantAwareTensor:
    """The ``[E, in, out]`` stack as the ``[E, out, in]`` matmul operand."""
    return _map_leaves(weight, lambda t: t.transpose(1, 2))


class Qwen3VLMoE(TensorParallelMoE):
    """Routed experts with no shared expert, split by ``moe_dim`` under TP."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        quant_config: QuantConfig | None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=Qwen3VLRouter,
            quant_config=quant_config,
        )

    def _init_experts(self) -> ModuleList[QuantizedMLP]:
        dtype, mesh = defaults()
        # As TensorParallelMoE does, keep multi-device experts on the host
        # until each device takes its slice.
        placement = (
            contextlib.nullcontext()
            if mesh.num_devices == 1
            else default_device(CPU())
        )
        with placement, default_dtype(dtype):
            return Qwen3VLStackedExperts(
                self.num_experts,
                self.hidden_dim,
                self.moe_dim,
                self.quant_config,
            )

    def _stack(self, name: str) -> QuantAwareTensor:
        """The expert stack ``name``, with its block scales when FP8."""
        experts = self.experts
        weight = getattr(experts, name)
        if self.quant_config is None:
            return weight
        block_size = self.quant_config.weight_scale.block_size
        assert block_size is not None
        return FP8BlockTensor(
            weight=weight,
            weight_scale_inv=getattr(experts, f"{name}_scale_inv"),
            block_size=block_size,
        )

    def _shard(
        self, stack: QuantAwareTensor, axis: int
    ) -> list[QuantAwareTensor]:
        """Splits ``stack`` evenly along ``axis``, one part per device."""
        if isinstance(stack, FP8BlockTensor):
            return list(stack.shard(axis, self.mesh).local_shards)
        assert isinstance(stack, Tensor)
        mapping = DeviceMapping(self.mesh, (Sharded(axis=axis),))
        return list(stack.to(mapping).local_shards)

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device ``[gate, up]`` weight bundle, sharded along ``moe_dim``."""
        stack = self._stack("gate_up_proj")
        if self.mesh.num_devices == 1:
            return [_matmul_operand(stack)]
        # Split gate and up columns separately so each device's halves stay
        # paired.
        halves = _map_leaves(
            stack, lambda t: t.reshape([t.shape[0], t.shape[1], 2, -1])
        )
        return [
            _matmul_operand(
                _map_leaves(
                    shard, lambda t: t.reshape([t.shape[0], t.shape[1], -1])
                )
            )
            for shard in self._shard(halves, axis=3)
        ]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device down-projection bundle, sharded along ``moe_dim``."""
        stack = self._stack("down_proj")
        if self.mesh.num_devices == 1:
            return [_matmul_operand(stack)]
        return [_matmul_operand(shard) for shard in self._shard(stack, axis=1)]
