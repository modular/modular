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
"""The buffers an expert-parallel exchange reads and writes."""

from __future__ import annotations

import dataclasses

from max import tree
from max.nn.ep_config import EPConfig
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated, Sharded

from . import _formats

__all__ = ["EPBuffers", "EPLBBuffers", "EPTables", "check_supported"]


@tree.dataclass
class EPTables:
    """One direction's SHMEM pointer tables.

    Each table is a host tensor with one device pointer per GPU of the node.
    """

    send: Tensor
    """Every device's send buffer pointer."""

    recv: Tensor
    """Every device's receive buffer pointer."""

    count: Tensor
    """Every device's receive count buffer pointer."""


@tree.dataclass
class EPLBBuffers:
    """The expert-parallel load balancing (EPLB) buffers.

    Every device holds the same tables, so they are replicated over the
    mesh.
    """

    log2phy: Tensor
    """The map from logical to physical experts."""

    logcnt: Tensor
    """The count of physical replicas per logical expert."""


@tree.dataclass
class EPBuffers:
    """Every buffer of the expert-parallel exchange.

    A decoder receives it ready as ``DecoderInputs.ep`` and passes it to its
    MoE layers. Dispatch and combine each send through their own pointer
    tables, so that combine never writes to buffers that a dispatch still
    reads. Both directions synchronize on the same atomic counters.
    """

    counters: Tensor
    """Each device's atomic counters, sharded over the mesh."""

    dispatch: EPTables
    """The pointer tables dispatch uses."""

    combine: EPTables
    """The pointer tables combine uses."""

    eplb: EPLBBuffers | None = None
    """The EPLB buffers, or ``None`` when EPLB is off."""

    expert_load: Tensor | None = None
    """Each device's ``[num_moe_layers, n_local_experts]`` ``uint64`` tally
    of the tokens routed to its experts, sharded over the mesh by expert, or
    ``None`` when expert-load counting is off.

    Every dispatch adds to the row of its MoE layer. Counting is independent
    of EPLB, so that a static placement can be profiled too.
    """

    def on(self, mesh: DeviceMesh) -> EPBuffers:
        """Returns these buffers on ``mesh``, moving no data.

        The framework places the buffers on a mesh of one axis. A model with
        its own mesh of the same devices re-places them on it, so that
        :func:`~max.nn.ep.dispatch` takes them with that mesh's tokens. The
        host pointer tables stay as they are.

        Args:
            mesh: A mesh of the same devices, in the same order, of any
                shape.

        Returns:
            The same buffers, each placed on ``mesh`` as it is placed now.

        Raises:
            ValueError: If ``mesh`` holds other devices.
        """
        if mesh.devices != self.counters.mesh.devices:
            raise ValueError(
                "EPBuffers.on needs a mesh of the buffers' own devices in"
                f" their order: got {mesh.devices} for"
                f" {self.counters.mesh.devices}"
            )

        def placed(tensor: Tensor, mapping: DeviceMapping) -> Tensor:
            return Tensor.from_shard_values(list(tensor.local_shards), mapping)

        eplb = self.eplb
        load = self.expert_load
        return dataclasses.replace(
            self,
            counters=placed(self.counters, _by_device(mesh)),
            eplb=None
            if eplb is None
            else EPLBBuffers(
                log2phy=placed(eplb.log2phy, _everywhere(mesh)),
                logcnt=placed(eplb.logcnt, _everywhere(mesh)),
            ),
            expert_load=None
            if load is None
            else placed(load, _by_expert(mesh)),
        )


def _by_device(mesh: DeviceMesh) -> DeviceMapping:
    """Returns the mapping of a buffer that each device holds its own row of."""
    return DeviceMapping(mesh, (Sharded(0),) * mesh.ndim)


def _everywhere(mesh: DeviceMesh) -> DeviceMapping:
    """Returns the mapping of a buffer that every device holds the same."""
    return DeviceMapping(mesh, (Replicated(),) * mesh.ndim)


def _by_expert(mesh: DeviceMesh) -> DeviceMapping:
    """Returns the mapping of :attr:`EPBuffers.expert_load`.

    Its columns are each device's local experts.
    """
    return DeviceMapping(mesh, (Sharded(1),) * mesh.ndim)


def check_supported(config: EPConfig, mesh: DeviceMesh) -> None:
    """Raises if this exchange cannot run ``config`` on ``mesh``.

    The exchange dispatches BF16 tokens, or quantizes them in transit to
    FP8 (block-scaled, compressed-tensors or FBGEMM), MXFP4, MXFP6, MXFP8 or
    NVFP4. NVFP4 with per-token global scales dispatches in one launch only.
    With ``config.eplb_count_expert_load``, each dispatch also counts the
    tokens routed to each local expert.

    Args:
        config: The expert-parallel configuration.
        mesh: The mesh of the devices that the exchange runs on.

    Raises:
        ValueError: If ``config`` dispatches in another format, or in a
            format that this accelerator's kernels do not serve, or if
            ``mesh`` does not hold one device per GPU of the node.
    """
    _check_mesh(config, mesh)
    _check_dispatch_format(config)


def _check_mesh(config: EPConfig, mesh: DeviceMesh) -> None:
    # One device is allowed, as the legacy exchange allows it: a rank-local
    # simulation runs one rank's share of the exchange on one GPU.
    if mesh.num_devices != config.n_gpus_per_node:
        raise ValueError(
            "expert parallelism needs a mesh of one device per GPU of the"
            f" node: got {mesh.num_devices} devices for"
            f" n_gpus_per_node={config.n_gpus_per_node}"
        )


def _check_dispatch_format(config: EPConfig) -> None:
    if not _formats.is_supported(config):
        raise ValueError(
            "max.nn.ep dispatches BF16, FP8, MXFP4, MXFP6, MXFP8 or NVFP4"
            " tokens:"
            f" got dispatch_dtype {config.dispatch_dtype} with format"
            f" {_formats.dispatch_format(config)} and nvfp4_dyn_global_scales"
            f" {config.nvfp4_dyn_global_scales} on this accelerator"
        )
