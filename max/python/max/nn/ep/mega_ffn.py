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
"""MegaFFN: the MoE FFN fused with the expert-parallel exchange.

:func:`mega_ffn_fused` runs dispatch, both FFN matmuls and the weighted
combine as one launch per device. :func:`mega_ffn_combine_send` runs both
FFN matmuls on the tokens that :func:`~max.nn.ep.dispatch_finish` grouped
and sends each finished row to its source device, so that
:func:`~max.nn.ep.combine_finish` follows it directly.
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorValue,
    ops,
)
from max.nn._graph_ops import ep as kernels
from max.nn.activation import GatedActivation
from max.nn.ep_config import EPConfig
from max.nn.quant_config import QuantFormat
from max.support.math import ceildiv
from max.tensor import Tensor

from . import _formats
from .buffers import EPBuffers, check_supported
from .exchange import Grouped, Route, _expert_load, _row, _tally

_FusedMoEParameter = bool | int | float | str | DType

# MegaFFN scheduler constants, restated because a graph cannot read them from
# the kernel. `ATOMIC_PAD` is the pool-slot stride in words
# (`MegaFFNScheduler`'s `atomic_pad`) and the token block is the smallest MMA_N
# the tile-regime cascade can select, which is what makes the slot-count bound
# an upper bound across every geometry.
_MEGA_FFN_ATOMIC_PAD = 32
_MEGA_FFN_MIN_TOKEN_BLOCK = 8


def _fused_moe_answer(
    workspace: npt.NDArray[np.uint64], refusal: npt.NDArray[np.uint8]
) -> tuple[int, int, str]:
    """Decodes the fused MoE init or plan op's answer.

    Returns ``(workspace bytes, token block, refusal)``.
    """
    return (
        int(workspace[0]),
        int(workspace[1]),
        bytes(refusal).rstrip(b"\0").decode(),
    )


@functools.cache
def _run_fused_moe_plan(
    parameters: tuple[tuple[str, _FusedMoEParameter], ...],
) -> tuple[int, int, str]:
    with Graph("fused_moe_plan", input_types=[]) as graph:
        graph.output(*kernels._call_mega_ffn_ep_fused_plan(dict(parameters)))
    workspace, refusal = InferenceSession(devices=[CPU()]).load(graph).execute()
    assert isinstance(workspace, Buffer) and isinstance(refusal, Buffer)
    return _fused_moe_answer(workspace.to_numpy(), refusal.to_numpy())


def _plan_fused_moe_workspace(config: EPConfig) -> tuple[int, int, str]:
    """Returns what EP init sets up for the fused MoE under ``config``.

    Allocates nothing: the answer is the workspace bytes per device and the
    token block, or zeros and why the backend refuses. The backend answers
    with the decision ``mega_ffn.ep_fused_init`` uses, computed on the host.
    Cached per parameter set: each new set compiles a small host graph.

    Args:
        config: EP configuration, quantization config included.

    Returns:
        ``(workspace bytes, token block, refusal)``.
    """
    return _run_fused_moe_plan(
        tuple(sorted(kernels._fused_moe_parameters(config).items()))
    )


def _bound_fused_moe_workspace(
    config: EPConfig, dispatch_dtype: DType
) -> tuple[int, str]:
    """Returns the most EP init can allocate for the fused MoE.

    Bounds ``config``'s EP geometry when only the dispatch element type is
    known, as when a memory plan is made before the checkpoint is parsed: the
    workspace bytes per device, or 0 and why the backend serves no format
    with that element type. Unlike :func:`_plan_fused_moe_workspace` this is
    an upper bound, not an exact answer.

    Args:
        config: EP configuration; its dispatch format is ignored.
        dispatch_dtype: The dispatch element type.

    Returns:
        ``(workspace bytes, refusal)``.
    """
    parameters = kernels._fused_moe_bound_parameters(config, dispatch_dtype)
    size, _, refusal = _run_fused_moe_plan(tuple(sorted(parameters.items())))
    return size, refusal


def _device_fused(
    counters: BufferValue,
    tokens: TensorValue,
    ids: TensorValue,
    weights: TensorValue,
    input_scales: TensorValue,
    gate_up_weight: TensorValue,
    gate_up_b_scales: TensorValue,
    down_weight: TensorValue,
    down_b_scales: TensorValue,
    gate_up_expert_scales: TensorValue,
    down_expert_scales: TensorValue,
    c_input_scales: TensorValue,
    count_ptrs: TensorValue,
    config: EPConfig,
    activation: GatedActivation | None,
    tally: BufferValue | None,
    layer_idx: TensorValue | None,
) -> TensorValue:
    """Runs one device's fused MoE, adding to its tally when given."""
    return kernels._call_mega_ffn_ep_fused(
        counters,
        tokens,
        ids,
        weights,
        input_scales,
        gate_up_weight,
        gate_up_b_scales,
        down_weight,
        down_b_scales,
        gate_up_expert_scales,
        down_expert_scales,
        c_input_scales,
        count_ptrs,
        config,
        activation=activation,
        expert_load=_expert_load(tally, layer_idx),
    )


def _arrival_counters(words: int, device: DeviceRef) -> BufferValue:
    """Creates one MoE layer's MegaFFN pool-slot counters.

    ``init_value`` makes them persistent: zeroed once, at load. The kernel
    resets its state by the launch generation, so they must not be zeroed
    per launch. Each call site gets its own buffer, because two MoE layers
    can have the same generation parity.
    """
    return ops.buffer_create(
        BufferType(DType.uint32, shape=[words], device=device), init_value=0
    )


# The one dispatch format the combine send serves: NVFP4 in the NVIDIA
# block-scaled layout, with one global scale for all tokens.
_COMBINE_SEND_FORMAT = (QuantFormat.NVFP4, True, False)


def _check_combine_send(config: EPConfig) -> None:
    """Raises if the combine send cannot serve ``config``.

    It refuses what legacy refuses, with the same errors. It also refuses
    per-token NVFP4 global scales, which legacy MoE layers accept: the kernel
    has no operand for the row scales, so it would quantize with the wrong
    scale.
    """
    kernels._validate_ffn_combine_send_config(config)
    if config.nvfp4_dyn_global_scales:
        raise ValueError(
            "the fused MegaFFN combine send has no operand for per-token"
            " NVFP4 global scales: dispatch with nvfp4_dyn_global_scales off"
        )
    if _formats._key(config) != _COMBINE_SEND_FORMAT:
        raise ValueError("the fused MegaFFN combine send is NVFP4-only")


_call_device_fused = F.functional(_device_fused)
_call_arrival_counters = F.functional(_arrival_counters)
_call_device_combine_send = F.functional(kernels._call_mega_ffn_ep_combine_send)


def mega_ffn_fused(
    config: EPConfig,
    buffers: EPBuffers,
    *,
    tokens: Tensor,
    ids: Tensor,
    weights: Tensor,
    input_scales: Tensor,
    gate_up_weight: Tensor,
    gate_up_b_scales: Tensor,
    down_weight: Tensor,
    down_b_scales: Tensor,
    gate_up_expert_scales: Tensor,
    down_expert_scales: Tensor,
    c_input_scales: Tensor,
    activation: GatedActivation | None = None,
    layer_idx: Tensor | None = None,
) -> Tensor:
    """Runs dispatch, the routed experts' FFN and the combine in one launch.

    Each device runs one kernel, which replaces :func:`~max.nn.ep.dispatch`,
    both grouped matmuls and :func:`~max.nn.ep.combine`. Every device must
    run it for every MoE layer, also with no tokens, because its peers wait
    for its counts. The fused workspace that EP init sets up is shared by
    every MoE layer on a device.

    Args:
        config: The expert-parallel configuration. EP init must have set
            ``config.fused_moe_ready``.
        buffers: The exchange's buffers.
        tokens: Each device's BF16 tokens, of shape ``(tokens, hidden)`` per
            device.
        ids: Each device's ``int32`` top-k expert ids, of shape
            ``(tokens, top_k)`` per device.
        weights: Each device's ``float32`` router weights, of shape
            ``(tokens, top_k)`` per device.
        input_scales: Each device's inverted NVFP4 dispatch input scale. The
            kernel reads element 0. It is 1 for MXFP8.
        gate_up_weight: The local experts' gate/up weights, in the
            sigma-permuted NVFP4 or MXFP8 layout.
        gate_up_b_scales: The gate/up weight scales.
        down_weight: The local experts' NVFP4 or MXFP8 down weights.
        down_b_scales: The down weight scales.
        gate_up_expert_scales: Each local expert's gate/up output scale.
        down_expert_scales: Each local expert's down output scale.
        c_input_scales: Each local expert's inverted scale for the
            requantization between the two matmuls.
        activation: The gated activation that the kernel fuses. Defaults to
            SiLU.
        layer_idx: Each device's ``int32`` index of this MoE layer, as for
            :func:`~max.nn.ep.dispatch`. Needed with
            ``config.eplb_count_expert_load``, and ignored otherwise.

    Returns:
        Each token's weighted sum of its routed experts' BF16 outputs,
        mapped as ``tokens``. The caller adds the shared expert.

    Raises:
        ValueError: If :func:`~max.nn.ep.check_supported` rejects
            ``config``, if EP init did not set up the fused MoE, or if
            ``config`` counts expert load without ``layer_idx`` or without
            :attr:`~max.nn.ep.EPBuffers.expert_load`.
    """
    mesh = tokens.mesh
    check_supported(config, mesh)
    tally = _tally(config, buffers, layer_idx)

    def on_device(
        tokens: Tensor,
        ids: Tensor,
        weights: Tensor,
        input_scales: Tensor,
        gate_up_weight: Tensor,
        gate_up_b_scales: Tensor,
        down_weight: Tensor,
        down_b_scales: Tensor,
        gate_up_expert_scales: Tensor,
        down_expert_scales: Tensor,
        c_input_scales: Tensor,
        counters: Tensor,
        tally: Tensor | None,
        layer_idx: Tensor | None,
    ) -> Tensor:
        return _call_device_fused(
            counters,
            tokens,
            ids,
            weights,
            input_scales,
            gate_up_weight,
            gate_up_b_scales,
            down_weight,
            down_b_scales,
            gate_up_expert_scales,
            down_expert_scales,
            c_input_scales,
            buffers.dispatch.count,
            config,
            activation,
            tally,
            _row(layer_idx),
        )

    return F.call_on_mesh(on_device, mesh, out_specs=tokens.mapping)(
        tokens,
        ids,
        weights,
        input_scales,
        gate_up_weight,
        gate_up_b_scales,
        down_weight,
        down_b_scales,
        gate_up_expert_scales,
        down_expert_scales,
        c_input_scales,
        buffers.counters,
        tally,
        layer_idx,
    )


def mega_ffn_combine_send(
    config: EPConfig,
    buffers: EPBuffers,
    grouped: Grouped,
    route: Route,
    *,
    n_local_experts: int,
    max_recv_tokens: int,
    gate_up_weight: Tensor,
    gate_up_b_scales: Tensor,
    down_weight: Tensor,
    down_b_scales: Tensor,
    gate_up_expert_scales: Tensor,
    down_expert_scales: Tensor,
    c_input_scales: Tensor,
    estimated_total_m: Tensor,
    activation: GatedActivation | None = None,
) -> None:
    """Runs the routed experts' FFN and sends its rows back, one launch each.

    Each device runs one kernel, which replaces both grouped matmuls and
    :func:`~max.nn.ep.combine_start`. The down projection writes each
    finished row to its source device's receive buffer, so the FFN's output
    is never stored on its own device. :func:`~max.nn.ep.combine_finish`
    then waits for the rows.

    Args:
        config: The expert-parallel configuration. It must not set
            ``use_allreduce`` or ``fused_shared_expert``.
        buffers: The exchange's buffers.
        grouped: The NVFP4 tokens that :func:`~max.nn.ep.dispatch_finish`
            grouped.
        route: The route that :func:`~max.nn.ep.dispatch_finish` returned.
        n_local_experts: The number of experts on each device, which is the
            first dimension of the weights.
        max_recv_tokens: The most rows that a device can receive.
        gate_up_weight: The local experts' gate/up weights, in the
            sigma-permuted layout.
        gate_up_b_scales: The gate/up weight scales.
        down_weight: The local experts' down weights.
        down_b_scales: The down weight scales.
        gate_up_expert_scales: Each local expert's gate/up output scale.
        down_expert_scales: Each local expert's down output scale.
        c_input_scales: Each local expert's inverted scale for the
            requantization between the two matmuls.
        estimated_total_m: The ``uint32`` host scalar estimate of the routed
            rows per device, which selects the kernel's tiles.
        activation: The gated activation that the kernel fuses. Defaults to
            SiLU.

    Raises:
        ValueError: If :func:`~max.nn.ep.check_supported` rejects
            ``config``, if ``config`` sets ``use_allreduce`` or
            ``fused_shared_expert``, or if ``config`` dispatches in another
            format than NVFP4 with one global scale. Legacy accepts
            per-token NVFP4 global scales here, but the kernel cannot read
            them, so this refusal is deliberately stricter than legacy.
    """
    mesh = grouped.tokens.mesh
    check_supported(config, mesh)
    _check_combine_send(config)
    # The bound of the graph compiler's fusion; the expert term adds the
    # partial blocks and the send's own words.
    arrival_words = (
        ceildiv(max_recv_tokens, _MEGA_FFN_MIN_TOKEN_BLOCK)
        + n_local_experts
        + 1
    ) * _MEGA_FFN_ATOMIC_PAD

    def on_device(
        tokens: Tensor,
        scales: Tensor,
        expert_start: Tensor,
        expert_ids: Tensor,
        scales_offset: Tensor,
        bounds: Tensor,
        src_info: Tensor,
        gate_up_weight: Tensor,
        gate_up_b_scales: Tensor,
        down_weight: Tensor,
        down_b_scales: Tensor,
        gate_up_expert_scales: Tensor,
        down_expert_scales: Tensor,
        c_input_scales: Tensor,
        counters: Tensor,
        estimated_total_m: Tensor,
    ) -> None:
        _call_device_combine_send(
            _call_arrival_counters(
                arrival_words, DeviceRef.from_device(tokens.device)
            ),
            counters,
            tokens,
            gate_up_weight,
            scales,
            gate_up_b_scales,
            down_weight,
            down_b_scales,
            expert_start,
            expert_ids,
            scales_offset,
            gate_up_expert_scales,
            down_expert_scales,
            c_input_scales,
            src_info,
            buffers.combine.recv,
            buffers.combine.count,
            estimated_total_m,
            bounds[1],
            config,
            activation=activation,
        )

    assert grouped.scales is not None and grouped.scales_offset is not None
    F.call_on_mesh(on_device, mesh)(
        grouped.tokens,
        grouped.scales,
        grouped.expert_start,
        grouped.expert_ids,
        grouped.scales_offset,
        grouped.bounds,
        route.src_info,
        gate_up_weight,
        gate_up_b_scales,
        down_weight,
        down_b_scales,
        gate_up_expert_scales,
        down_expert_scales,
        c_input_scales,
        buffers.counters,
        estimated_total_m,
    )
