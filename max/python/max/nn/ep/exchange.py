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
"""The expert-parallel token exchange: dispatch to the experts, combine back.

:func:`dispatch` and :func:`combine` each run as one launch over the whole
mesh. The ``_start`` and ``_finish`` pairs split each into a launch and a
wait per device, so that other work, such as a shared expert, can run in
between. :func:`dispatch_local` and :func:`combine_local` run one fused
kernel per device; with ``EPConfig.use_allreduce`` they route each device's
tokens to its own experts only, and the caller sums the devices' results.
Dispatch sends BF16 tokens, or quantizes them in transit to the
format of ``EPConfig.dispatch_quant_config``. With
``EPConfig.eplb_count_expert_load``, each dispatch also adds the tokens it
routes to each local expert to :attr:`EPBuffers.expert_load`. Each function
checks its configuration with :func:`check_supported`.
"""

from __future__ import annotations

from collections.abc import Sequence

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferValue, Dim, TensorValue
from max.nn._graph_ops import ep as kernels
from max.nn.ep_config import EPConfig
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    Unknown,
)

from ._formats import DispatchLayout, dispatch_layout
from .buffers import EPBuffers, check_supported


@tree.dataclass
class Grouped:
    """The tokens a device received, grouped by local expert.

    This is the layout that the grouped matmuls read. The optional fields
    that a dispatch fills depend on its format:

    - BF16: no scales.
    - FP8, MXFP4 and MXFP6: :attr:`scales`.
    - MXFP8: :attr:`scales`, and also :attr:`scales_offset` in the NVIDIA
      block-scaled layout.
    - NVFP4: :attr:`scales` and :attr:`scales_offset`. With per-token global
      scales, also :attr:`row_scales`.
    """

    tokens: Tensor
    """The received tokens, the rows of each expert together."""

    expert_start: Tensor
    """The first row of each expert's tokens, then the end of the last."""

    expert_ids: Tensor
    """The local expert of each group of rows."""

    bounds: Tensor
    """A ``(2,)`` ``uint32`` host tensor per device, read inside
    :func:`~max.functional.call_on_mesh`: see :func:`grouped_matmul_bounds`.
    The NVFP4 grouped matmul builds its own bounds, but the MegaFFN combine
    send reads the expert count from these.

    :func:`dispatch` builds one host tensor that every device shares, and
    :func:`dispatch_finish` builds one per device, as the legacy exchange
    does. :func:`dispatch_local`, and a :func:`dispatch` that counts expert
    load, also build one per device. Outside ``call_on_mesh`` the two differ
    in placement.
    """

    scales: Tensor | None = None
    """The scales of quantized tokens, or ``None`` for BF16 tokens."""

    scales_offset: Tensor | None = None
    """Each expert's first row in the padded scales of the NVIDIA
    block-scaled layout, or ``None`` when the scales are not padded."""

    row_scales: Tensor | None = None
    """Each received row's inverse global scale, for NVFP4 tokens quantized
    with per-token global scales, else ``None``."""


@tree.dataclass
class Route:
    """Where each received token came from, which combine sends it back to."""

    src_info: Tensor
    """The source token and top-k slot of each received row."""


def _expert_load(
    tally: BufferValue | None, layer_idx: TensorValue | None
) -> kernels.ExpertLoad | None:
    """Returns the tally row a dispatch adds to, or ``None`` without a tally."""
    if tally is None:
        return None
    assert layer_idx is not None
    return kernels.ExpertLoad(tally, layer_idx)


def _row(layer_idx: Tensor | None) -> Tensor | None:
    """Returns one device's layer index as the ``[1]`` tensor kernels read."""
    return layer_idx.reshape([1]) if layer_idx is not None else None


def _device_dispatch(
    input_tokens: TensorValue,
    topk_ids: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    tally: BufferValue | None,
    layer_idx: TensorValue | None,
    input_scales: TensorValue | None = None,
) -> tuple[TensorValue, ...]:
    """Runs one device's fused dispatch, adding to its tally when given."""
    return kernels.call_ep_dispatch(
        input_tokens,
        topk_ids,
        atomic_counter,
        send_buf_ptrs,
        recv_buf_ptrs,
        recv_count_ptrs,
        config,
        input_scales=input_scales,
        expert_load=_expert_load(tally, layer_idx),
    )


def _dispatch_wait(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_tokens: Dim,
    tally: BufferValue | None,
    layer_idx: TensorValue | None,
) -> tuple[TensorValue, ...]:
    """Waits for one device's dispatch, adding to its tally when given."""
    return kernels.call_ep_dispatch_wait(
        atomic_counter,
        recv_buf_ptrs,
        recv_count_ptrs,
        config,
        num_tokens=num_tokens,
        expert_load=_expert_load(tally, layer_idx),
    )


_call_dispatch = F.functional(kernels.call_distributed_ep_dispatch)
_call_device_dispatch = F.functional(_device_dispatch)
_call_combine = F.functional(kernels.call_distributed_ep_combine)
_call_device_combine = F.functional(kernels.call_ep_combine)
_call_dispatch_async = F.functional(kernels.call_ep_dispatch_async)
_call_dispatch_wait = F.functional(_dispatch_wait)
_call_combine_async = F.functional(kernels.call_ep_combine_async)
_call_combine_wait = F.functional(kernels.call_ep_combine_wait)


def _tally(
    config: EPConfig, buffers: EPBuffers, layer_idx: Tensor | None
) -> Tensor | None:
    """Returns the expert-load tally a dispatch adds to.

    Returns ``None`` when expert-load counting is off.
    """
    if not config.eplb_count_expert_load:
        return None
    if layer_idx is None:
        raise ValueError(
            "eplb_count_expert_load needs the MoE layer index of each dispatch"
        )
    if buffers.expert_load is None:
        raise ValueError(
            "eplb_count_expert_load needs the expert-load tally in"
            " EPBuffers.expert_load"
        )
    return buffers.expert_load


def grouped_matmul_bounds(config: EPConfig) -> Tensor:
    """Returns the grouped matmul's host bounds for the received tokens.

    Args:
        config: The expert-parallel configuration.

    Returns:
        A ``uint32`` host tensor of shape ``(2,)``: the most rows a device can
        receive for one expert, then the number of experts on a device,
        counting a fused shared expert.
    """
    ranks = config.n_gpus_per_node * config.n_nodes
    experts = config.n_experts // ranks + int(config.fused_shared_expert)
    return F.constant(
        [config.max_tokens_per_rank * ranks, experts],
        DType.uint32,
        device=CPU(),
    )


def _own(mesh: DeviceMesh) -> DeviceMapping:
    """Returns the mapping of a result that describes only its own device."""
    return DeviceMapping(mesh, (Unknown(),) * mesh.ndim)


def _received(mesh: DeviceMesh, layout: DispatchLayout) -> list[DeviceMapping]:
    """Returns the mappings of a dispatch's results, the route last.

    The received tokens come first and are split by device. The other
    results describe only their own device's tokens.
    """
    by_device = DeviceMapping(mesh, (Sharded(0),) * mesh.ndim)
    return [by_device, *[_own(mesh)] * len(layout.fields)]


def _grouped(
    layout: DispatchLayout, results: Sequence[Tensor], bounds: Tensor
) -> Grouped:
    """Names a dispatch's results, which precede the route, by ``layout``."""
    fields = dict(zip(layout.fields, results, strict=True))
    return Grouped(**fields, bounds=bounds)


def dispatch(
    config: EPConfig,
    buffers: EPBuffers,
    tokens: Tensor,
    ids: Tensor,
    *,
    input_scales: Tensor | None = None,
    layer_idx: Tensor | None = None,
) -> tuple[Grouped, Route]:
    """Sends each token to the devices of its experts, in one launch.

    A quantized dispatch quantizes the tokens in transit to the format of
    ``config.dispatch_quant_config``. The multi-device kernel cannot count
    expert load, so with ``config.eplb_count_expert_load`` each device runs
    its own fused dispatch kernel instead, as the legacy exchange does.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        tokens: Each device's BF16 tokens, of shape ``(tokens, hidden)`` per
            device.
        ids: Each device's top-k expert ids, of shape ``(tokens, top_k)`` per
            device, in any integer dtype.
        input_scales: Each device's global input scale, which the dispatch
            inverts. NVFP4 needs it, unless it scales each token on its own.
            MXFP8 in the NVIDIA block-scaled layout uses it when given, and 1
            otherwise. The other formats ignore it.
        layer_idx: Each device's ``int32`` index of this MoE layer, a scalar
            or of shape ``(1,)``: the row of :attr:`EPBuffers.expert_load`
            that the dispatch adds to. Needed with
            ``config.eplb_count_expert_load``, and ignored otherwise.

    Returns:
        The tokens each device received, and the route that :func:`combine`
        sends them back by.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, or if
            ``config`` counts expert load without ``layer_idx`` or without
            :attr:`EPBuffers.expert_load`.
    """
    mesh = tokens.mesh
    check_supported(config, mesh)
    tally = _tally(config, buffers, layer_idx)
    if tally is not None:
        return _dispatch_per_device(
            config, buffers, tokens, ids, input_scales, tally, layer_idx
        )
    layout = dispatch_layout(config)
    group = buffers.dispatch

    def on_every_device(
        tokens: list[Tensor],
        ids: list[Tensor],
        counters: list[Tensor],
        input_scales: list[Tensor] | None,
    ) -> list[tuple[Tensor, ...]]:
        results = _call_dispatch(
            tokens,
            [F.cast(device_ids, DType.int32) for device_ids in ids],
            counters,
            group.send,
            group.recv,
            group.count,
            config,
            input_scales=input_scales,
        )
        return [tuple(result) for result in results]

    *results, src_info = F.call_on_mesh(
        on_every_device,
        mesh,
        mesh.axis_names,
        out_specs=_received(mesh, layout),
    )(tokens, ids, buffers.counters, input_scales)
    grouped = _grouped(layout, results, grouped_matmul_bounds(config))
    return grouped, Route(src_info=src_info)


def dispatch_local(
    config: EPConfig,
    buffers: EPBuffers,
    tokens: Tensor,
    ids: Tensor,
    *,
    input_scales: Tensor | None = None,
    layer_idx: Tensor | None = None,
) -> tuple[Grouped, Route]:
    """Sends each token to its experts, with one fused kernel per device.

    With ``config.use_allreduce``, each device routes its own tokens to its
    own experts only, and sends nothing to the other devices. Each device
    gets its own copy of the grouped matmul bounds, as in
    :func:`dispatch_finish`.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        tokens: Each device's tokens, as for :func:`dispatch`.
        ids: Each device's top-k expert ids, as for :func:`dispatch`.
        input_scales: As for :func:`dispatch`.
        layer_idx: As for :func:`dispatch`.

    Returns:
        The tokens each device received, and the route that
        :func:`combine_local` sends them back by.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, or if
            ``config`` counts expert load without ``layer_idx`` or without
            :attr:`EPBuffers.expert_load`.
    """
    check_supported(config, tokens.mesh)
    tally = _tally(config, buffers, layer_idx)
    return _dispatch_per_device(
        config, buffers, tokens, ids, input_scales, tally, layer_idx
    )


def _dispatch_per_device(
    config: EPConfig,
    buffers: EPBuffers,
    tokens: Tensor,
    ids: Tensor,
    input_scales: Tensor | None,
    tally: Tensor | None,
    layer_idx: Tensor | None,
) -> tuple[Grouped, Route]:
    """Runs one fused dispatch kernel per device, in mesh order.

    Each device builds its grouped matmul bounds right after its dispatch,
    where the legacy ``EPBatchManager.ep_dispatch`` builds them.
    """
    mesh = tokens.mesh
    layout = dispatch_layout(config)
    group = buffers.dispatch

    def on_device(
        tokens: Tensor,
        ids: Tensor,
        counters: Tensor,
        input_scales: Tensor | None,
        tally: Tensor | None,
        layer_idx: Tensor | None,
    ) -> tuple[Tensor, ...]:
        results = _call_device_dispatch(
            tokens,
            F.cast(ids, DType.int32),
            counters,
            group.send,
            group.recv,
            group.count,
            config,
            tally,
            _row(layer_idx),
            input_scales=input_scales,
        )
        return (*results, grouped_matmul_bounds(config))

    *results, src_info, bounds = F.call_on_mesh(
        on_device, mesh, out_specs=[*_received(mesh, layout), _own(mesh)]
    )(tokens, ids, buffers.counters, input_scales, tally, layer_idx)
    return _grouped(layout, results, bounds), Route(src_info=src_info)


def combine(
    config: EPConfig,
    buffers: EPBuffers,
    outputs: Tensor,
    route: Route,
    tokens: Tensor,
    weights: Tensor,
) -> Tensor:
    """Returns the experts' outputs to their tokens, weighted, in one launch.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        outputs: The experts' outputs, in the rows of the received tokens.
        route: The route that :func:`dispatch` returned.
        tokens: The tokens that were dispatched, which give each device's
            token count and the result's mapping.
        weights: Each device's router weights, of shape ``(tokens, top_k)``
            per device.

    Returns:
        Each token's weighted sum of its experts' outputs, mapped as
        ``tokens``.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, or if
            ``config`` asks for an MXFP8 combine, which only
            :func:`combine_start` and :func:`combine_finish` send.
    """
    mesh = tokens.mesh
    check_supported(config, mesh)
    _check_exact_combine(config)
    tables = buffers.combine

    def on_every_device(
        outputs: list[Tensor],
        src_info: list[Tensor],
        counters: list[Tensor],
        tokens: list[Tensor],
        weights: list[Tensor],
    ) -> list[Tensor]:
        return _call_combine(
            outputs,
            src_info,
            counters,
            tables.send,
            tables.recv,
            tables.count,
            config,
            [token.shape[0] for token in tokens],
            weights,
        )

    return F.call_on_mesh(
        on_every_device, mesh, mesh.axis_names, out_specs=tokens.mapping
    )(outputs, route.src_info, buffers.counters, tokens, weights)


def combine_local(
    config: EPConfig,
    buffers: EPBuffers,
    outputs: Tensor,
    route: Route,
    weights: Tensor,
    ids: Tensor,
) -> Tensor:
    """Returns the experts' outputs to their tokens, one kernel per device.

    With ``config.fused_shared_expert``, the kernel also adds the shared
    expert's rows of ``outputs`` to each token's result.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        outputs: The experts' outputs, in the rows of the received tokens.
        route: The route that :func:`dispatch_local` returned.
        weights: Each device's router weights, of shape ``(tokens, top_k)``
            per device, which also give each device's token count.
        ids: Each device's top-k expert ids, which the combine reads with
            ``config.use_allreduce``.

    Returns:
        Each token's weighted sum of its experts' outputs. With
        ``config.use_allreduce``, each device holds the sum over its own
        experts only, so the result is ``Partial``; the tokens must then be
        the same on every device. Else it is mapped as ``weights``.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, or if
            ``config.use_allreduce`` and ``weights`` shows that the devices
            hold different tokens.
    """
    mesh = weights.mesh
    check_supported(config, mesh)
    tables = buffers.combine

    def on_device(
        outputs: Tensor,
        src_info: Tensor,
        counters: Tensor,
        weights: Tensor,
        ids: Tensor,
    ) -> Tensor:
        return _call_device_combine(
            outputs,
            src_info,
            counters,
            tables.send,
            tables.recv,
            tables.count,
            config,
            weights.shape[0],
            weights,
            topk_ids=F.cast(ids, DType.int32),
        )

    return F.call_on_mesh(
        on_device, mesh, out_specs=_combined(config, weights)
    )(outputs, route.src_info, buffers.counters, weights, ids)


def _combined(config: EPConfig, weights: Tensor) -> DeviceMapping:
    """Returns the mapping of :func:`combine_local`'s result.

    Under allreduce each device holds its own experts' share of the same
    tokens, so the devices' results sum to the whole.

    A router that runs per device can return ``Unknown`` weights for
    replicated tokens, so only a placement that shows the devices hold
    different tokens is refused.

    Raises:
        ValueError: If ``config.use_allreduce`` and ``weights`` is placed
            other than replicated or unknown.
    """
    if not config.use_allreduce:
        return weights.mapping
    if not all(isinstance(p, Replicated | Unknown) for p in weights.placements):
        raise ValueError(
            "combine_local sums the devices' results only when every device"
            " holds the same tokens: got router weights placed"
            f" {weights.placements}"
        )
    return DeviceMapping(weights.mesh, (Partial(),) * weights.mesh.ndim)


def dispatch_start(
    config: EPConfig,
    buffers: EPBuffers,
    tokens: Tensor,
    ids: Tensor,
    *,
    input_scales: Tensor | None = None,
) -> None:
    """Launches each device's dispatch; :func:`dispatch_finish` waits for it.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        tokens: Each device's tokens, as for :func:`dispatch`.
        ids: Each device's top-k expert ids, as for :func:`dispatch`.
        input_scales: As for :func:`dispatch`.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, or if
            ``config`` quantizes NVFP4 with per-token global scales, which
            only :func:`dispatch` sends.
    """
    check_supported(config, tokens.mesh)
    group = buffers.dispatch

    def on_device(
        tokens: Tensor,
        ids: Tensor,
        counters: Tensor,
        input_scales: Tensor | None,
    ) -> None:
        _call_dispatch_async(
            tokens,
            F.cast(ids, DType.int32),
            counters,
            group.send,
            group.recv,
            group.count,
            config,
            input_scales=input_scales,
        )

    F.call_on_mesh(on_device, tokens.mesh)(
        tokens, ids, buffers.counters, input_scales
    )


def dispatch_finish(
    config: EPConfig,
    buffers: EPBuffers,
    tokens: Tensor,
    *,
    layer_idx: Tensor | None = None,
) -> tuple[Grouped, Route]:
    """Waits for each device's dispatch that :func:`dispatch_start` launched.

    Each device gets its own copy of the grouped matmul bounds, built right
    after its wait, where the legacy ``EPBatchManager.ep_dispatch_wait``
    builds it.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        tokens: The tokens that were dispatched, which give each device's
            token count.
        layer_idx: As for :func:`dispatch`. The wait adds to the tally.

    Returns:
        As :func:`dispatch`.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, if
            ``config`` quantizes NVFP4 with per-token global scales, which
            only :func:`dispatch` sends, or if ``config`` counts expert load
            without ``layer_idx`` or without :attr:`EPBuffers.expert_load`.
    """
    mesh = tokens.mesh
    check_supported(config, mesh)
    tally = _tally(config, buffers, layer_idx)
    layout = dispatch_layout(config)
    group = buffers.dispatch

    def on_device(
        tokens: Tensor,
        counters: Tensor,
        tally: Tensor | None,
        layer_idx: Tensor | None,
    ) -> tuple[Tensor, ...]:
        results = _call_dispatch_wait(
            counters,
            group.recv,
            group.count,
            config,
            tokens.shape[0],
            tally,
            _row(layer_idx),
        )
        return (*results, grouped_matmul_bounds(config))

    *results, src_info, bounds = F.call_on_mesh(
        on_device, mesh, out_specs=[*_received(mesh, layout), _own(mesh)]
    )(tokens, buffers.counters, tally, layer_idx)
    return _grouped(layout, results, bounds), Route(src_info=src_info)


def combine_start(
    config: EPConfig,
    buffers: EPBuffers,
    outputs: Tensor,
    route: Route,
    *,
    total_tokens: Tensor | None = None,
) -> None:
    """Launches each device's combine; :func:`combine_finish` waits for it.

    With ``config.combine_mx_min_tokens`` set, a step of at least that many
    tokens sends the outputs as MXFP8, and a smaller step sends them exact.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        outputs: The experts' outputs, as for :func:`combine`.
        route: The route that :func:`dispatch` returned.
        total_tokens: The step's token count over all devices, a ``uint32``
            host scalar. Needed with ``config.combine_mx_min_tokens``.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, or if
            ``config`` sets ``combine_mx_min_tokens`` without
            ``total_tokens``.
    """
    check_supported(config, outputs.mesh)
    tables = buffers.combine

    def on_device(
        outputs: Tensor,
        src_info: Tensor,
        counters: Tensor,
        total_tokens: Tensor | None,
    ) -> None:
        _call_combine_async(
            outputs,
            src_info,
            counters,
            tables.send,
            tables.recv,
            tables.count,
            config,
            total_tokens=total_tokens,
        )

    F.call_on_mesh(on_device, outputs.mesh)(
        outputs, route.src_info, buffers.counters, total_tokens
    )


def combine_finish(
    config: EPConfig,
    buffers: EPBuffers,
    tokens: Tensor,
    weights: Tensor,
    *,
    total_tokens: Tensor | None = None,
) -> Tensor:
    """Waits for each device's combine that :func:`combine_start` launched.

    Args:
        config: The expert-parallel configuration.
        buffers: The exchange's buffers.
        tokens: The tokens that were dispatched, as for :func:`combine`.
        weights: Each device's router weights, as for :func:`combine`.
        total_tokens: The scalar that :func:`combine_start` got.

    Returns:
        As :func:`combine`.

    Raises:
        ValueError: If :func:`check_supported` rejects ``config``, or if
            ``config`` sets ``combine_mx_min_tokens`` without
            ``total_tokens``.
    """
    check_supported(config, tokens.mesh)
    tables = buffers.combine

    def on_device(
        tokens: Tensor,
        weights: Tensor,
        counters: Tensor,
        total_tokens: Tensor | None,
    ) -> Tensor:
        return _call_combine_wait(
            counters,
            tables.recv,
            tables.count,
            config,
            tokens.shape[0],
            weights,
            total_tokens=total_tokens,
        )

    return F.call_on_mesh(on_device, tokens.mesh, out_specs=tokens.mapping)(
        tokens, weights, buffers.counters, total_tokens
    )


def _check_exact_combine(config: EPConfig) -> None:
    if config.combine_mx_min_tokens > 0:
        raise ValueError(
            "the one-launch combine is always exact: send an MXFP8 combine"
            " (combine_mx_min_tokens) with combine_start and combine_finish"
        )
