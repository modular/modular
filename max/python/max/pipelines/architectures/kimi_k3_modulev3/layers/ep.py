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

BF16 only, mirroring ``minimax_m3_modulev3/ep.py``'s kernel wiring without
its quantized-dispatch fields. ``dispatch``/``combine`` run in one call
across the whole mesh; the ``_start``/``_finish`` pairs split that into a
launch and a wait, so other work can run in between.
"""

from __future__ import annotations

from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
    Unknown,
)
from max.experimental.tensor import Tensor
from max.graph import BufferValue
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep import ep_kernels as kernels


@tree.dataclass(kw_only=True)
class Grouped:
    """Tokens grouped by expert, as the grouped matmuls read them: each
    expert's first row and the expert of each group."""

    tokens: Tensor
    start: Tensor
    ids: Tensor


@tree.dataclass
class Tables:
    """One direction's host tables of every device's send, receive and
    count buffers."""

    send: Tensor
    recv: Tensor
    count: Tensor


@tree.dataclass
class EPBuffers:
    """The exchange's graph inputs: each device's atomic counters, which
    both directions share, and each direction's tables."""

    counters: Tensor
    dispatch: Tables
    combine: Tables


call_dispatch = F.functional(kernels.call_distributed_ep_dispatch)
call_combine = F.functional(kernels.call_distributed_ep_combine)
call_dispatch_async = F.functional(kernels.call_ep_dispatch_async)
call_dispatch_wait = F.functional(kernels.call_ep_dispatch_wait)
call_combine_async = F.functional(kernels.call_ep_combine_async)
call_combine_wait = F.functional(kernels.call_ep_combine_wait)


def grouped_matmul_bounds(config: EPConfig) -> Tensor:
    """The grouped matmul's host bounds for the received tokens: the most
    rows a device can receive and its number of experts."""
    ranks = config.n_gpus_per_node * config.n_nodes
    return F.constant(
        [config.max_tokens_per_rank * ranks, config.n_experts // ranks],
        DType.uint32,
        device=CPU(),
    )


def _received(mesh: DeviceMesh) -> list[DeviceMapping]:
    """Each dispatch result's mapping, in order: the tokens, split by
    device; the expert offsets and ids and ``src_info``, each device's
    own."""
    by_device = DeviceMapping(mesh, (Sharded(0),) * mesh.ndim)
    own = DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
    return [by_device, own, own, own]


def dispatch(
    config: EPConfig, buffers: EPBuffers, tokens: Tensor, ids: Tensor
) -> tuple[Grouped, Tensor]:
    """Sends each token to its experts' devices in one launch over every
    device.

    Returns each device's received tokens and the ``src_info``
    :func:`combine` routes them back with.
    """

    def on_every_device(
        tokens: list[Tensor], ids: list[Tensor], buffers: EPBuffers
    ) -> list[tuple[Tensor, ...]]:
        tables = buffers.dispatch
        results = call_dispatch(
            tokens,
            ids,
            buffers.counters,
            tables.send,
            tables.recv,
            tables.count,
            config,
        )
        return [tuple(result) for result in results]

    mesh = tokens.mesh
    tokens_out, start, ids_out, src_info = F.call_on_mesh(
        on_every_device, mesh, mesh.axis_names, out_specs=_received(mesh)
    )(tokens, ids, buffers)
    return Grouped(tokens=tokens_out, start=start, ids=ids_out), src_info


def combine(
    config: EPConfig,
    buffers: EPBuffers,
    outputs: Tensor,
    src_info: Tensor,
    tokens: Tensor,
    weights: Tensor,
) -> Tensor:
    """Returns the experts' outputs to their tokens, weighted, in one launch
    over every device; every token comes back to the device it left."""

    def on_every_device(
        outputs: list[Tensor],
        src_info: list[Tensor],
        buffers: EPBuffers,
        tokens: list[Tensor],
        weights: list[Tensor],
    ) -> list[Tensor]:
        tables = buffers.combine
        return call_combine(
            outputs,
            src_info,
            buffers.counters,
            tables.send,
            tables.recv,
            tables.count,
            config,
            [token.shape[0] for token in tokens],
            weights,
        )

    mesh = tokens.mesh
    return F.call_on_mesh(
        on_every_device, mesh, mesh.axis_names, out_specs=tokens.mapping
    )(outputs, src_info, buffers, tokens, weights)


def dispatch_start(
    config: EPConfig, buffers: EPBuffers, tokens: Tensor, ids: Tensor
) -> None:
    """Launches each device's dispatch; :func:`dispatch_finish` waits."""

    def on_device(tokens: Tensor, ids: Tensor, buffers: EPBuffers) -> None:
        tables = buffers.dispatch
        call_dispatch_async(
            tokens,
            ids,
            buffers.counters,
            tables.send,
            tables.recv,
            tables.count,
            config,
        )

    F.call_on_mesh(on_device, tokens.mesh)(tokens, ids, buffers)


def dispatch_finish(
    config: EPConfig, buffers: EPBuffers, tokens: Tensor
) -> tuple[Grouped, Tensor]:
    """Waits for each device's dispatched tokens; as :func:`dispatch`."""

    def on_device(tokens: Tensor, buffers: EPBuffers) -> tuple[Tensor, ...]:
        tables = buffers.dispatch
        result = call_dispatch_wait(
            buffers.counters,
            tables.recv,
            tables.count,
            config,
            num_tokens=tokens.shape[0],
        )
        return tuple(result)

    tokens_out, start, ids_out, src_info = F.call_on_mesh(
        on_device, tokens.mesh, out_specs=_received(tokens.mesh)
    )(tokens, buffers)
    return Grouped(tokens=tokens_out, start=start, ids=ids_out), src_info


def combine_start(
    config: EPConfig, buffers: EPBuffers, outputs: Tensor, src_info: Tensor
) -> None:
    """Launches each device's combine; :func:`combine_finish` waits."""

    def on_device(
        outputs: Tensor, src_info: Tensor, buffers: EPBuffers
    ) -> None:
        tables = buffers.combine
        call_combine_async(
            outputs,
            src_info,
            buffers.counters,
            tables.send,
            tables.recv,
            tables.count,
            config,
        )

    F.call_on_mesh(on_device, outputs.mesh)(outputs, src_info, buffers)


def combine_finish(
    config: EPConfig, buffers: EPBuffers, tokens: Tensor, weights: Tensor
) -> Tensor:
    """Waits for each device's combined outputs; as :func:`combine`."""

    def on_device(
        tokens: Tensor, weights: Tensor, buffers: EPBuffers
    ) -> Tensor:
        tables = buffers.combine
        return call_combine_wait(
            buffers.counters,
            tables.recv,
            tables.count,
            config,
            tokens.shape[0],
            weights,
        )

    return F.call_on_mesh(on_device, tokens.mesh, out_specs=tokens.mapping)(
        tokens, weights, buffers
    )


def _host_table(buffer: Buffer) -> Tensor:
    """Wraps one host-side, all-devices pointer table as a plain Tensor.

    One shared value every device's dispatch/combine call reads, not a
    per-device one.
    """
    with ensure_context():
        return F.constant(buffer.to_numpy(), buffer.dtype, device=CPU())


def allocate_ep_buffers(
    config: EPConfig, mesh: DeviceMesh, session: InferenceSession
) -> EPBuffers:
    """Registers the exchange's SHMEM buffers and wraps them for :mod:`ep`.

    Runs ``EPCommInitializer.ep_init`` once, then splits its two buffer
    groups into :attr:`EPBuffers.dispatch` and :attr:`EPBuffers.combine`, so
    combine never reuses dispatch's still-live buffers. The atomic counters
    are shared between both phases, so only one copy is wired in.
    """
    init = EPCommInitializer(config)
    init.ep_init(session)

    n = mesh.num_devices
    with ensure_context():
        counters = Tensor.from_shard_values(
            [
                BufferValue(
                    Tensor(buf.to_numpy(), dtype=buf.dtype, device=device)
                )
                for buf, device in zip(
                    init.atomic_counters[:n], mesh.devices, strict=True
                )
            ],
            DeviceMapping(mesh, (Sharded(axis=0),) * mesh.ndim),
        )
    return EPBuffers(
        counters=counters,
        dispatch=Tables(
            send=_host_table(init.send_buf_ptrs[0]),
            recv=_host_table(init.recv_buf_ptrs[0]),
            count=_host_table(init.recv_count_ptrs[0]),
        ),
        combine=Tables(
            send=_host_table(init.send_buf_ptrs[1]),
            recv=_host_table(init.recv_buf_ptrs[1]),
            count=_host_table(init.recv_count_ptrs[1]),
        ),
    )
