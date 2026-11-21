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
"""The host side of the expert-parallel exchange, which the framework runs.

A decoder graph takes the exchange's buffers as the inputs
:func:`payload_types` declares: one single-device input per buffer and
device, named and ordered as the serving engine binds them.
:func:`payload_buffers` turns those inputs into the
:class:`~max.nn.ep.EPBuffers` a decoder reads, and :func:`flat_buffers`
reads them from a graph that takes them flat. :func:`init_graph` returns
the ``ep_init`` graph that sets SHMEM up, and :func:`start` runs it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import (
    Accelerator,
    Buffer,
    accelerator_api,
    enable_all_peer_access,
)
from max.dtype import DType
from max.engine import Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Type,
    ops,
)
from max.graph.graph import Module as GraphModule
from max.nn._graph_group import EPPayload
from max.nn._graph_ops import ep as kernels
from max.nn.ep_config import (
    _EP_LOCAL_SYNC_RESERVED_WORDS,
    _EP_P2P_SMALL_WORDS,
    NUM_GROUPS,
    EPConfig,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .buffers import (
    EPBuffers,
    EPLBBuffers,
    EPTables,
    _by_device,
    _by_expert,
    _everywhere,
)
from .mega_ffn import _fused_moe_answer

_logger = logging.getLogger("max.pipelines")

__all__ = [
    "EPExchange",
    "EPInitGraph",
    "flat_buffers",
    "init_graph",
    "payload_buffers",
    "payload_types",
    "start",
]

# Mirror ``MAX_GPUS_PER_NODE`` and ``MAX_SMS_PER_DEVICE`` in
# ``max/kernels/src/shmem/ep_comm.mojo``, which Python cannot read.
_MAX_GPUS_PER_NODE = 8
_MAX_SMS_PER_DEVICE = 304

# The most reliable connections per PE that NVSHMEM's IBGDA transport maps
# by warp: one per warp of a block.
_MAX_RCS_PER_PE = {"cuda": 32, "hip": 16}


def _sync_counter_words(n_experts: int) -> int:
    """Int32 words of the counters in ``EPLocalSyncCounters``.

    Mirrors ``EPLocalSyncCounters.total_size()`` in ``ep_comm.mojo``, which
    is the definition of record. It is sized for the worst case over the
    expert-parallel degree, ``n_local_experts == n_experts``.
    """
    dispatch_async = 2 * n_experts + _MAX_GPUS_PER_NODE
    dispatch_wait = 8 * n_experts + 8
    return dispatch_async + dispatch_wait + _MAX_SMS_PER_DEVICE


def _counter_words(config: EPConfig) -> int:
    """Returns the ``int32`` words of one atomic counter buffer.

    The allreduce backend routes within each device, so its counters cover
    the local experts only. A config that gives the experts' FFN width
    (``moe_dim``) ends each buffer in the fused MoE counters' fixed reserve,
    and one that takes the P2P small path ends it in that path's words.
    """
    experts = config.n_experts // (
        config.n_gpus_per_node if config.use_allreduce else 1
    )
    return (
        _sync_counter_words(experts)
        + _EP_LOCAL_SYNC_RESERVED_WORDS * (config.moe_dim > 0)
        + _EP_P2P_SMALL_WORDS * config.uses_p2p_small_path
    )


def _counter_types(config: EPConfig) -> tuple[BufferType, ...]:
    """Both buffer groups' atomic counters, group-major."""
    n_gpus = config.n_gpus_per_node
    return tuple(
        BufferType(
            DType.int32,
            [_counter_words(config)],
            device=DeviceRef.GPU(i % n_gpus),
        )
        for i in range(NUM_GROUPS * n_gpus)
    )


def payload_types(config: EPConfig) -> EPPayload:
    """Returns a decoder's expert-parallel graph inputs, by name.

    ``atomic_counters`` holds both buffer groups' counters, group-major, and
    the three pointer fields hold one host table per group. With EPLB,
    ``eplb_log2phy`` and ``eplb_logcnt`` follow, one buffer per device each,
    and with expert-load counting, ``eplb_expert_load``.

    The serving engine binds the buffers by these names and positions, and
    accepts no input sharded over several devices, so each buffer of each
    device is its own input.

    Args:
        config: The expert-parallel configuration.

    Returns:
        The input types.
    """
    n_gpus = config.n_gpus_per_node
    gpus = [DeviceRef.GPU(i) for i in range(n_gpus)]

    def tables() -> tuple[TensorType, ...]:
        return tuple(
            TensorType(DType.uint64, [n_gpus], device=DeviceRef.CPU())
            for _ in range(NUM_GROUPS)
        )

    def per_device(dtype: DType, shape: list[int]) -> tuple[BufferType, ...]:
        return tuple(BufferType(dtype, shape, device=gpu) for gpu in gpus)

    inputs = EPPayload(
        atomic_counters=_counter_types(config),
        send_buf_ptrs=tables(),
        recv_buf_ptrs=tables(),
        recv_count_ptrs=tables(),
    )
    maps = [config.num_moe_layers, config.num_logical_experts]
    eplb = {
        "eplb_log2phy": per_device(DType.int32, [*maps, config.max_replicas]),
        "eplb_logcnt": per_device(DType.int32, maps),
    }
    n_local_experts = config.n_experts // (n_gpus * config.n_nodes)
    tallies = {
        "eplb_expert_load": per_device(
            DType.uint64, [config.num_moe_layers, n_local_experts]
        )
    }
    inputs.update(eplb if config.eplb_enabled else {})
    inputs.update(tallies if config.eplb_count_expert_load else {})
    return inputs


def payload_buffers(
    payload: Mapping[str, Sequence[Tensor | TensorValue | BufferValue]],
    mesh: DeviceMesh,
) -> EPBuffers:
    """Returns the buffers a decoder reads from its expert-parallel inputs.

    Each device's share of a per-device buffer joins the others in one
    tensor over ``mesh``: the counters split by device, the expert-load
    tally by expert, and the EPLB tables, which every device holds the same,
    replicated. Both directions synchronize on the first
    group's counters; the second group's stay graph inputs only because
    ``ep_init`` writes them.

    Args:
        payload: The graph's inputs, named as :func:`payload_types` names
            them.
        mesh: The decoder's mesh.

    Returns:
        The exchange's buffers.
    """

    def joined(name: str, mapping: DeviceMapping) -> Tensor:
        shards = payload[name][: mesh.num_devices]
        return Tensor.from_shard_values(shards, mapping)

    def tables(group: int) -> EPTables:
        send, recv, count = (
            Tensor.from_shard_values([payload[name][group]])
            for name in ("send_buf_ptrs", "recv_buf_ptrs", "recv_count_ptrs")
        )
        return EPTables(send=send, recv=recv, count=count)

    return EPBuffers(
        counters=joined("atomic_counters", _by_device(mesh)),
        dispatch=tables(0),
        combine=tables(1),
        eplb=(
            EPLBBuffers(
                log2phy=joined("eplb_log2phy", _everywhere(mesh)),
                logcnt=joined("eplb_logcnt", _everywhere(mesh)),
            )
            if "eplb_log2phy" in payload
            else None
        ),
        expert_load=(
            joined("eplb_expert_load", _by_expert(mesh))
            if "eplb_expert_load" in payload
            else None
        ),
    )


def flat_buffers(
    config: EPConfig | None,
    values: Iterable[Tensor | TensorValue | BufferValue],
    mesh: DeviceMesh,
) -> EPBuffers | None:
    """Returns the buffers of a graph that takes its EP inputs flat.

    Reads the inputs :func:`payload_types` declares, in its leaf order, from
    the front of ``values``. An iterator keeps any values after them.

    Args:
        config: The expert-parallel configuration, or ``None`` without
            expert parallelism.
        values: The graph's inputs, from the first expert-parallel one.
        mesh: The decoder's mesh.

    Returns:
        The exchange's buffers, or ``None`` without expert parallelism.
    """
    if config is None:
        return None
    _, payload = tree.flatten(payload_types(config))
    return payload_buffers(tree.unflatten(payload, values, exact=False), mesh)


@dataclass(frozen=True)
class EPInitGraph:
    """Builds the ``ep_init`` graph, as a pipeline's graph spec traces it.

    The graph takes each device's atomic counters for both buffer groups. It
    returns, per device, the device pointers to the SHMEM send, receive and
    receive count buffers, then every device's rank. With
    ``config.fused_moe``, each device then also returns its fused MoE
    workspace size and token block, and why the backend refuses the
    configuration, if it does.

    A raw graph, because the serving engine runs it as is: its inputs are
    the counters one per device in group-major order, and its outputs are
    positional.
    """

    config: EPConfig
    """The expert-parallel configuration."""

    def input_types(self) -> list[BufferType]:
        """Returns the atomic counter types, both groups' per device."""
        return list(_counter_types(self.config))

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph over ``input_types``.

        Args:
            *input_types: The atomic counter types, as
                :meth:`input_types` returns them.
            name: The graph's name. Defaults to ``"ep_init"``.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced graph.
        """
        return Graph(
            "ep_init" if name is None else name,
            self._forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )

    def state_dict(self) -> dict[str, Buffer]:
        """Returns no weights: the graph has none."""
        return {}

    def _forward(self, *counters: BufferValue) -> list[TensorValue]:
        n_gpus = self.config.n_gpus_per_node
        pointers: list[TensorValue] = []
        ranks: list[TensorValue] = []
        fused: list[TensorValue] = []
        for i in range(n_gpus):
            group_0 = counters[i]
            device_ptrs, rank = kernels.call_ep_init(
                group_0, counters[i + n_gpus], self.config
            )
            # The session may hold no CPU device, so host results go back to
            # the counters' device.
            pointers.append(device_ptrs.to(group_0.device))
            ranks.append(rank)
            fused += self._fused_moe_init(group_0, device_ptrs)
        all_ranks = ops.concat(ranks, axis=0).to(DeviceRef.GPU(0))
        return [*pointers, all_ranks, *fused]

    def _fused_moe_init(
        self, counters: BufferValue, device_ptrs: TensorValue
    ) -> list[TensorValue]:
        """Sets up one rank's fused MoE workspace, if the config asks."""
        if not self.config.fused_moe:
            return []
        workspace, refusal = kernels._call_mega_ffn_ep_fused_init(
            counters, device_ptrs, self.config
        )
        return [workspace.to(counters.device), refusal.to(counters.device)]


def _set_nvshmem_environment(config: EPConfig) -> None:
    """Sets NVSHMEM's IBGDA transport up for inter-node or no-P2P runs."""
    n_ranks = config.n_nodes * config.n_gpus_per_node
    max_rcs = _MAX_RCS_PER_PE.get(accelerator_api())
    if max_rcs is None:
        raise ValueError(f"Unsupported accelerator API: {accelerator_api()}")
    os.environ.update(
        NVSHMEM_IB_ENABLE_IBGDA="1",
        NVSHMEM_IBGDA_NIC_HANDLER="gpu",
        NVSHMEM_IBGDA_RC_MAP_BY="warp",
        NVSHMEM_IBGDA_NUM_RC_PER_PE=str(
            min(config.n_experts // n_ranks, max_rcs)
        ),
        NVSHMEM_ENABLE_NIC_PE_MAPPING="1",
    )


def init_graph(config: EPConfig) -> EPInitGraph:
    """Returns the ``ep_init`` graph, with this process ready to load it.

    NVSHMEM reads its transport from the environment when the graph loads,
    so this sets it for an inter-node or no-P2P run. It touches no device,
    so the graph compiles on virtual devices too.

    Args:
        config: The expert-parallel configuration.

    Returns:
        The graph that sets up SHMEM and allocates the buffers.
    """
    _logger.info(
        "Estimated EP memory usage per device: "
        f"{to_human_readable_bytes(config.estimate_memory_usage())}"
    )
    # A single node talks over P2P, unless P2P is disabled.
    if config.n_nodes > 1 or os.getenv("NVSHMEM_DISABLE_P2P") == "1":
        _set_nvshmem_environment(config)
    return EPInitGraph(config)


@dataclass(frozen=True)
class EPExchange:
    """The exchange that :func:`start` set up."""

    config: EPConfig
    """The configuration, with the node ID and fused MoE readiness that
    ``ep_init`` found."""

    buffers: tuple[Buffer, ...]
    """The buffers, in the leaf order of :func:`payload_types`."""

    def model_inputs(self) -> list[Buffer]:
        """Returns the buffers each step passes the decoder."""
        return list(self.buffers)


def start(
    config: EPConfig,
    model: Model,
    eplb: tuple[Sequence[Buffer], Sequence[Buffer]] | None = None,
) -> EPExchange:
    """Runs the compiled :func:`init_graph` on this node's devices.

    Allocates the atomic counters and expert-load tallies zeroed, and lets
    ``ep_init`` set SHMEM up and allocate the other buffers.

    Args:
        config: The configuration the graph was built for.
        model: The compiled and initialized graph.
        eplb: Each device's ``log2phy`` and ``logcnt`` tables, which EPLB
            needs.

    Returns:
        The exchange.

    Raises:
        ValueError: If ``eplb`` is given without EPLB or missing with it, or
            if the devices of this node report different node IDs.
    """
    if (eplb is not None) != config.eplb_enabled:
        raise ValueError(
            f"eplb_enabled={config.eplb_enabled} needs the EPLB tables given"
            " exactly when it is set"
        )
    n_gpus = config.n_gpus_per_node
    counters = [
        Buffer(
            DType.int32,
            [_counter_words(config)],
            device=Accelerator(i % n_gpus),
        )
        for i in range(NUM_GROUPS * n_gpus)
    ]
    n_local_experts = config.n_experts // (n_gpus * config.n_nodes)
    zeros = np.zeros((config.num_moe_layers, n_local_experts), np.uint64)
    expert_load = [
        Buffer.from_numpy(zeros).to(Accelerator(i))
        for i in range(n_gpus * config.eplb_count_expert_load)
    ]
    enable_all_peer_access()

    # The outputs mix uint64 pointers, int32 ranks and uint8 text.
    outputs: list[npt.NDArray[Any]] = []
    for output in model.execute(*counters):
        assert isinstance(output, Buffer)
        outputs.append(output.to_numpy())

    # Each device returns its [NUM_GROUPS, 3] pointers: per group, the send,
    # receive and receive count buffers. The tables go type-major.
    per_group = [
        np.stack([outputs[d][group] for d in range(n_gpus)])
        for group in range(NUM_GROUPS)
    ]
    tables = [Buffer.from_numpy(p[:, t]) for t in range(3) for p in per_group]

    node_ids = outputs[n_gpus] // n_gpus
    if not np.all(node_ids == node_ids[0]):
        raise ValueError(
            "All GPUs in the same node must have the same node ID."
        )
    node_id = int(node_ids[0])
    _logger.info(f"Initialized EP for node {node_id}")
    maps = [*eplb[0], *eplb[1]] if eplb is not None else []
    return EPExchange(
        dataclasses.replace(
            config,
            node_id=node_id,
            fused_moe_ready=config.fused_moe
            and _fused_moe_ready(config, outputs[n_gpus + 1 :]),
        ),
        (*counters, *tables, *maps, *expert_load),
    )


def _fused_moe_ready(
    config: EPConfig, outputs: Sequence[npt.NDArray[Any]]
) -> bool:
    """Returns whether ``ep_init`` set the one-launch fused MoE up.

    ``outputs`` holds, per device, the workspace bytes and token block, then
    the refusal text of ``mega_ffn.ep_fused_init``. Every device gets the
    same configuration, so they answer alike. Each op synchronizes its
    device before it answers, so once all have answered, every workspace
    holds its initial values and peers may write into it.
    """
    answers = {
        _fused_moe_answer(w, r)
        for w, r in zip(outputs[0::2], outputs[1::2], strict=True)
    }
    if len(answers) != 1:
        raise RuntimeError(
            f"EP devices disagree on the fused MoE workspace: {answers}"
        )
    ((size, token_block, refusal),) = answers
    if refusal:
        # The workspace has the EP capacity, which comes from
        # max_batch_input_tokens, so every step must fit it.
        _logger.warning(
            "The fused EP MoE does not serve this configuration (EP capacity"
            " %d tokens per rank): %s. Keeping dispatch, FFN and combine.",
            config.max_tokens_per_rank,
            refusal,
        )
        return False
    _logger.info(
        "Initialized the one-launch fused EP MoE workspace:"
        f" {to_human_readable_bytes(size)} per device, token block"
        f" {token_block}"
    )
    return True
