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
"""Completes a registered module's config with its deployment.

A config built from the checkpoint doesn't know where it runs. After the
config is finalized, and before any module is constructed, the framework
hands it a :class:`ModuleRuntime`: the device mesh, the batch limits the
memory planner chose and, under expert parallelism, the initialized
:class:`~max.nn.legacy.comm.ep.EPConfig`.

A registered config takes part by subclassing :class:`DeploymentConfig`, or
:class:`ExpertParallelDeploymentConfig` when it serves its experts across
devices. Their fields declare what the deployment adds to the decoder's
inputs and how the checkpoint splits between roles.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, TypeVar

from max.driver import Buffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.legacy.graph_groups import EPPayload
from max.pipelines.context import TextContext
from max.tensor.sharding import DeviceMesh
from typing_extensions import Self

__all__ = [
    "BatchContextLengths",
    "DeploymentConfig",
    "ExpertParallelDeploymentConfig",
    "ModuleRuntime",
    "batch_context_length_types",
    "ep_input_tree",
    "initialize_ep_comm_buffers",
    "role_weights",
]

_WeightT = TypeVar("_WeightT")


@dataclass(frozen=True)
class ModuleRuntime:
    """The deployment a registered module's config is completed with."""

    mesh: DeviceMesh
    """The devices the decoder runs on, as the framework constructs it."""

    data_parallel_degree: int
    """The number of data-parallel replicas."""

    max_batch_input_tokens: int
    """The most tokens one batch feeds the decoder."""

    max_batch_total_tokens: int | None
    """The KV context one batch can hold, as the memory planner sized it, or
    ``None`` when the planner set no bound."""

    ep_size: int = 1
    """The number of devices the experts are spread over; one without
    expert parallelism."""

    ep_use_allreduce: bool = False
    """Whether expert parallelism combines through an allreduce."""

    ep: EPConfig | None = None
    """The initialized expert-parallel config, under expert parallelism."""


class DeploymentConfig:
    """A registered config that the deployment completes.

    A registered config that subclasses it gets :meth:`apply_runtime` called
    before any module is constructed, and declares through its class fields
    what else the framework does for it. A config that doesn't subclass it
    takes every default.
    """

    takes_batch_context_lengths: ClassVar[bool] = False
    """Whether the decoder's graph takes
    :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.batch_context_lengths`,
    as the MLA family does to size its prefill buffers without a
    device-to-host copy."""

    weight_prefixes: ClassVar[Mapping[str, str]] = MappingProxyType({})
    """The checkpoint prefix each role's weights sit under, keyed by
    :class:`~max.pipelines.lib.graph_specs.GraphRole`, when one checkpoint
    holds the weights of several roles. A role without one takes every
    weight (:func:`role_weights`)."""

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Returns the config completed with ``runtime``.

        The default needs nothing from the deployment.
        """
        del runtime
        return self


class ExpertParallelDeploymentConfig(DeploymentConfig, abc.ABC):
    """A registered config whose decoder serves its experts across devices.

    Under expert parallelism the framework sizes the communication buffers
    with :meth:`expert_parallel_config`, initializes them, and passes the
    result to :meth:`apply_runtime` as :attr:`ModuleRuntime.ep`.
    """

    ep_dispatch_token_layout: ClassVar[str | None] = None
    """How tokens reach the expert dispatch, when the decoder fixes it.
    Exporters read it off the pipeline model to pick the matching
    expert-parallel backend."""

    @abc.abstractmethod
    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        """Sizes the expert-parallel buffers from the checkpoint's experts."""


def role_weights(
    state_dict: Mapping[str, _WeightT], config: object, role: str
) -> dict[str, _WeightT]:
    """Returns the weights of one role, named as its module names them.

    A role with a prefix in :attr:`DeploymentConfig.weight_prefixes` takes
    the weights under it, with the prefix removed; one without takes them
    all.
    """
    prefix = (
        config.weight_prefixes.get(role, "")
        if isinstance(config, DeploymentConfig)
        else ""
    )
    if not prefix:
        return dict(state_dict)
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def initialize_ep_comm_buffers(
    config: ExpertParallelDeploymentConfig,
    runtime: ModuleRuntime,
    session: InferenceSession,
) -> tuple[ModuleRuntime, EPCommInitializer | None]:
    """Sizes and initializes the expert-parallel communication buffers.

    It sizes the buffers from the config, then compiles and runs the
    expert-parallel init graph, which sets the node ID each device picks its
    experts by. That is why it runs before the modules are constructed.

    Args:
        config: The finalized config, which sizes the buffers.
        runtime: The deployment, with ``ep_size`` above one and without
            ``ep``.
        session: The session that compiles the init graph.

    Returns:
        ``runtime`` with :attr:`ModuleRuntime.ep` set, and the initializer
        that owns the buffers. The initializer is ``None`` on virtual
        devices, which can't run the init graph.

    Raises:
        ValueError: If the expert-parallel size isn't a multiple of the
            device count.
    """
    n_devices = len(runtime.mesh.devices)
    if runtime.ep_size % n_devices != 0:
        raise ValueError(
            f"ep_size={runtime.ep_size} must be a multiple of the number of "
            f"devices on this node ({n_devices})"
        )
    ep = config.expert_parallel_config(runtime)
    initializer = None
    if not is_virtual_device_mode():
        initializer = EPCommInitializer(ep)
        initializer.ep_init(session)
        ep.node_id = initializer.config.node_id
    return dataclasses.replace(runtime, ep=ep), initializer


def ep_input_tree(ep: EPConfig) -> EPPayload:
    """Returns the decoder's ``ep`` input types for an initialized config."""
    return EPBatchManager(ep).input_tree()


def batch_context_length_types(
    data_parallel_degree: int,
) -> tuple[TensorType, ...]:
    """Returns the ``batch_context_lengths`` input types, one per replica."""
    return tuple(
        TensorType(DType.int32, shape=[1], device=DeviceRef.CPU())
        for _ in range(max(1, data_parallel_degree))
    )


class BatchContextLengths:
    """Each replica's page-aligned KV context length, refreshed per batch.

    The buffers persist across batches, and each batch rewrites them in
    place, so a captured device graph keeps reading the live values.
    """

    def __init__(self, data_parallel_degree: int, page_size: int) -> None:
        self._page_size = page_size
        self._buffers = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(max(1, data_parallel_degree))
        ]

    def update(
        self, replica_batches: Sequence[Sequence[TextContext]]
    ) -> list[Buffer]:
        """Writes each replica's context length and returns the buffers."""
        page = self._page_size
        for buffer, batch in zip(self._buffers, replica_batches, strict=False):
            buffer.to_numpy()[0] = sum(
                (ctx.tokens.current_position + page - 1) // page * page
                for ctx in batch
            )
        return self._buffers
