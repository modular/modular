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
from typing import ClassVar, Protocol, TypeVar

from max import tree
from max.driver import Buffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import WeightData
from max.nn.ep_config import EPConfig
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer
from max.pipelines.context import TextContext
from max.tensor.sharding import DeviceMesh
from typing_extensions import Self

from .interfaces.pipeline_model import ModelLoadContext

__all__ = [
    "BatchContextLengths",
    "ConstantInputs",
    "DeploymentConfig",
    "ExpertParallelDeploymentConfig",
    "ExpertParallelExchange",
    "ExpertParallelInputs",
    "ModuleRuntime",
    "batch_context_length_types",
    "ep_input_tree",
    "initialize_ep_comm_buffers",
    "prepare_weights",
    "role_weights",
    "uses_expert_parallel_exchange",
]

_WeightT = TypeVar("_WeightT")


@dataclass(frozen=True)
class ConstantInputs:
    """The declared graph types and lifetime-owned deployment buffers.

    The buffers stay alive with the batch processor and are passed unchanged
    every step. They are checkpoint-backed inputs, not mutable workspace.
    """

    input_types: tree.Tree[TensorType | BufferType]
    buffers: tree.Tree[Buffer]

    def __post_init__(self) -> None:
        _, structure = tree.flatten(self.input_types)
        structure.flatten_up_to(self.buffers)


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

    max_batch_size: int | None = None
    """The most requests a step holds, as the memory planner sized it, or
    ``None`` when the caller doesn't size batches."""


class ExpertParallelExchange(Protocol):
    """The initialized exchange's buffers, as each batch passes them."""

    def model_inputs(self) -> Sequence[Buffer]:
        """Returns the buffers, in the order of the graph's ``ep`` leaves."""
        ...


class ExpertParallelInputs:
    """How a decoder's graph takes the expert-parallel exchange.

    The default takes :class:`~max.nn.legacy.comm.ep.EPBatchManager`'s
    inputs: both buffer groups' atomic counters on every device, then their
    send, receive and count pointer tables. A config whose decoder takes
    them in another layout sets
    :attr:`ExpertParallelDeploymentConfig.expert_parallel_inputs` to a
    subclass.
    """

    def input_tree(self, ep: EPConfig) -> object:
        """Returns the decoder's ``ep`` input types for ``ep``."""
        return EPBatchManager(ep).input_tree()

    def exchange(
        self, initializer: EPCommInitializer
    ) -> ExpertParallelExchange:
        """Returns the buffers of ``initializer`` the graph takes."""
        return initializer


class DeploymentConfig:
    """A registered config that the deployment completes.

    A registered config that subclasses it gets :meth:`apply_runtime` called
    before any module is constructed, and declares through its class fields
    what else the framework does for it. A config that doesn't subclass it
    takes every default.
    """

    export_ep_init: ClassVar[bool] = False
    """Whether artifacts include the exchange initialization graph."""

    preserve_checkpoint_keys: ClassVar[bool] = False
    """Whether finalization reads raw keys before :meth:`prepare_weights`
    performs the architecture's declared weight transforms.
    """

    expert_parallel_communication: ClassVar[bool] = True
    """Whether expert parallelism uses a runtime exchange. A config that
    disables it validates local expert sharding in :meth:`apply_runtime`.
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

    def apply_model_load_context(self, context: ModelLoadContext) -> Self:
        """Returns the config completed with composition-only settings."""
        del context
        return self

    def prepare_weights(
        self, state_dict: Mapping[str, WeightData]
    ) -> dict[str, WeightData]:
        """Returns checkpoint weights prepared for the registered modules.

        Serving and composed decoders apply this before role prefixes and
        parameter dtype conversion. The default preserves every weight.
        """
        return dict(state_dict)

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Returns the config completed with ``runtime``.

        The default needs nothing from the deployment.
        """
        del runtime
        return self

    def export_records(self, runtime: ModuleRuntime) -> Mapping[str, object]:
        """Returns what an exporter records of the deployment, by name.

        The serving pipeline model keeps them as its ``export_records``,
        where an exporter reads them. The default records nothing.
        """
        del runtime
        return {}


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

    expert_parallel_inputs: ClassVar[ExpertParallelInputs] = (
        ExpertParallelInputs()
    )
    """How the decoder's graph takes the exchange's buffers."""

    @abc.abstractmethod
    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        """Sizes the expert-parallel buffers from the checkpoint's experts."""


def uses_expert_parallel_exchange(config: object) -> bool:
    """Returns whether expert sharding requires exchange initialization."""
    return (
        not isinstance(config, DeploymentConfig)
        or config.expert_parallel_communication
    )


def prepare_weights(
    state_dict: Mapping[str, WeightData], config: object
) -> dict[str, WeightData]:
    """Prepares a checkpoint before selecting registered role weights."""
    if isinstance(config, DeploymentConfig):
        return config.prepare_weights(state_dict)
    return dict(state_dict)


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
) -> tuple[ModuleRuntime, ExpertParallelExchange | None]:
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
        ``runtime`` with :attr:`ModuleRuntime.ep` set, and the buffers the
        decoder's graph takes, as the config's
        :attr:`~ExpertParallelDeploymentConfig.expert_parallel_inputs`
        selects them. They are ``None`` on virtual devices, which can't run
        the init graph.

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
    exchange = None
    if not is_virtual_device_mode():
        initializer = EPCommInitializer(ep)
        initializer.ep_init(session)
        ep.node_id = initializer.config.node_id
        exchange = config.expert_parallel_inputs.exchange(initializer)
    return dataclasses.replace(runtime, ep=ep), exchange


def ep_input_tree(
    config: ExpertParallelDeploymentConfig, ep: EPConfig
) -> object:
    """Returns the decoder's ``ep`` input types for an initialized config."""
    return config.expert_parallel_inputs.input_tree(ep)


def batch_context_length_types(
    data_parallel_degree: int,
    *,
    devices_per_replica: int = 1,
    dtype: DType = DType.int32,
) -> tuple[TensorType, ...]:
    """Returns the ``batch_context_lengths`` input types.

    There is one per replica, or with ``devices_per_replica``, one for each
    device of each replica, a replica's devices in a row.
    """
    return tuple(
        TensorType(dtype, shape=[1], device=DeviceRef.CPU())
        for _ in range(max(1, data_parallel_degree) * devices_per_replica)
    )


class BatchContextLengths:
    """Each replica's page-aligned KV context length, refreshed per batch.

    With ``devices_per_replica``, each replica's length fills that many
    buffers in a row, one for each of its devices. The buffers persist
    across batches, and each batch rewrites them in place, so a captured
    device graph keeps reading the live values.
    """

    def __init__(
        self,
        data_parallel_degree: int,
        page_size: int,
        *,
        devices_per_replica: int = 1,
        dtype: DType = DType.int32,
    ) -> None:
        self._page_size = page_size
        self._devices_per_replica = devices_per_replica
        self._buffers = [
            Buffer.zeros(shape=[1], dtype=dtype)
            for _ in range(max(1, data_parallel_degree) * devices_per_replica)
        ]

    def update(
        self, replica_batches: Sequence[Sequence[TextContext]]
    ) -> list[Buffer]:
        """Writes each replica's context length and returns the buffers."""
        page = self._page_size
        count = self._devices_per_replica
        for replica, batch in enumerate(replica_batches):
            length = sum(
                (ctx.tokens.current_position + page - 1) // page * page
                for ctx in batch
            )
            for buffer in self._buffers[replica * count :][:count]:
                buffer.to_numpy()[0] = length
        return self._buffers
