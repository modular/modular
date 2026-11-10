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

"""Quantize-aware MoE layers that bind a checkpoint's whole expert stacks."""

from __future__ import annotations

from typing import Any

from max import functional as F
from max.nn import Module
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor, defaults
from max.tensor.sharding import DeviceMapping, Sharded

from . import quant_ops
from .quant_moe import ExpertParallelMoE, QuantizedMoE
from .quant_tensor import NVFP4Tensor, QuantAwareTensor


def _stacked_weight(
    num_experts: int,
    out_dim: int,
    in_dim: int,
    quant_config: QuantConfig | None,
    device: DeviceMapping | None,
    scales_per_expert: int,
) -> QuantAwareTensor:
    if quant_config is None:
        return Tensor.zeros((num_experts, out_dim, in_dim), device=device)
    if not quant_config.is_nvfp4 or quant_config.quantizes_on_load:
        raise ValueError(
            "Stacked experts hold bfloat16 or NVFP4 checkpoint weights, not"
            f" {quant_config.format}"
        )
    block_size = quant_config.weight_scale.block_size
    assert block_size is not None
    return NVFP4Tensor.stacked_zeros(
        num_experts,
        (out_dim, in_dim),
        scales_per_expert=scales_per_expert,
        block_size=block_size,
        device=device,
    )


def _local_experts(w: QuantAwareTensor, device: int | None) -> QuantAwareTensor:
    if device is None:
        return w
    if isinstance(w, NVFP4Tensor):
        return w.local_experts(device)
    assert isinstance(w, Tensor)
    return w.local_shards[device]


class StackedExperts(Module[..., Any]):
    """Every routed expert's weights, stacked as a fused checkpoint stores them.

    ``gate_up_proj`` is ``[E, 2 * moe_dim, hidden_dim]``, each expert's gate
    rows then its up rows, and ``down_proj`` is ``[E, hidden_dim, moe_dim]``.
    An NVFP4 ``gate_up_proj`` keeps both halves' per-tensor scales as
    ``[E, 2]`` ``(gate, up)``.

    Args:
        num_experts: The number of routed experts.
        hidden_dim: The experts' input and output width.
        moe_dim: Each expert's intermediate width.
        quant_config: The experts' quantization, or ``None`` for bfloat16.
        device: The stacks' placement, or ``None`` for the default device.
    """

    gate_up_proj: QuantAwareTensor
    down_proj: QuantAwareTensor

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        moe_dim: int,
        quant_config: QuantConfig | None,
        device: DeviceMapping | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = _stacked_weight(
            num_experts, 2 * moe_dim, hidden_dim, quant_config, device, 2
        )
        self.down_proj = _stacked_weight(
            num_experts, hidden_dim, moe_dim, quant_config, device, 1
        )

    def gate_up(self, device: int | None = None) -> QuantAwareTensor:
        """The fused gate/up stack, all of it or ``device``'s experts."""
        w = _local_experts(self.gate_up_proj, device)
        if isinstance(w, NVFP4Tensor):
            return quant_ops.gate_scaled_nvfp4(w)
        return w

    def down(self, device: int | None = None) -> QuantAwareTensor:
        """The down stack, all of it or ``device``'s experts."""
        down = _local_experts(self.down_proj, device)
        if isinstance(down, NVFP4Tensor):
            gate_up = _local_experts(self.gate_up_proj, device)
            assert isinstance(gate_up, NVFP4Tensor)
            return quant_ops.fold_up_scale_nvfp4(down, gate_up)
        return down


class StackedQuantizedMoE(QuantizedMoE):
    """:class:`QuantizedMoE` whose experts are :class:`StackedExperts`.

    Binding whole stacks gives a layer a few weights instead of a few per
    expert.
    """

    # The base's per-expert ModuleList is the one layout this class replaces.
    experts: StackedExperts  # type: ignore[assignment]

    def _init_experts(self) -> StackedExperts:  # type: ignore[override]
        return StackedExperts(
            self.num_experts, self.hidden_dim, self.moe_dim, self.quant_config
        )

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """The fused gate/up stack, one entry."""
        return [self.experts.gate_up()]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """The down stack, one entry."""
        return [self.experts.down()]


class StackedExpertParallelMoE(ExpertParallelMoE):
    """:class:`ExpertParallelMoE` whose experts are :class:`StackedExperts`.

    The stacks are split whole over the mesh along the expert axis.
    """

    experts: StackedExperts  # type: ignore[assignment]

    def __init__(
        self, *args, ep_batch_manager: EPBatchManager, **kwargs
    ) -> None:
        if ep_batch_manager.config.fused_shared_expert:
            raise ValueError(
                "Stacked experts do not take a fused shared expert yet"
            )
        super().__init__(*args, ep_batch_manager=ep_batch_manager, **kwargs)

    def _init_experts(self) -> StackedExperts:  # type: ignore[override]
        _, mesh = defaults()
        self._check_mesh(mesh, self.num_experts)
        return StackedExperts(
            self.num_experts,
            self.hidden_dim,
            self.moe_dim,
            self.quant_config,
            DeviceMapping(mesh, (Sharded(axis=0),)),
        )

    def _stack_gate_up(self) -> list[QuantAwareTensor]:
        return [self.experts.gate_up(i) for i in range(self.mesh.num_devices)]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device down stacks of the local experts."""
        return [self.experts.down(i) for i in range(self.mesh.num_devices)]

    def _nvfp4_global_input_scale(self) -> Tensor:
        gate_up = self.experts.gate_up()
        assert isinstance(gate_up, NVFP4Tensor)
        return F.max(gate_up.input_scale, axis=0)
