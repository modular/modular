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
:class:`~max.nn.ep_config.EPConfig`.

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
from typing import Any, ClassVar, Protocol, TypeVar

from max import tree
from max.driver import Buffer, Device, Usage, is_virtual_device_mode
from max.dtype import DType
from max.engine import CompilationStopped, InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import WeightData
from max.nn.ep import check_supported
from max.nn.ep import init as ep_init
from max.nn.ep_config import EPConfig, _ep_fused_moe_requested
from max.nn.legacy.graph_groups import EPPayload
from max.pipelines.context import TextContext
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import DeviceMesh
from typing_extensions import Self

from .ep_initializer import EPInitializerFactory, EPInitializerLike
from .graph_specs import GraphRole, GraphSpec, compile_and_init_graphs
from .interfaces.pipeline_model import (
    ModelLoadContext,
    compile_ep_init,
    init_ep_comm,
)
from .utils import CompilationTimer
from .weight_transform import WeightEntry

__all__ = [
    "BatchContextLengths",
    "ConstantInputs",
    "DeploymentConfig",
    "ExpertParallelDeploymentConfig",
    "ExpertParallelExchange",
    "ExpertParallelInputs",
    "ModuleRuntime",
    "allocate_static_buffers",
    "batch_context_length_types",
    "ep_input_tree",
    "initialize_ep_comm_buffers",
    "initialize_static_inputs",
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
    initializer: GraphSpec | None = None
    devices: tuple[Device, ...] = ()
    _initialized: bool = dataclasses.field(
        default=False, init=False, compare=False
    )

    def __del__(self) -> None:
        # Custom kernels dereference these host pointers directly. Their
        # allocation cannot retire until all consuming device streams drain.
        if getattr(self, "_initialized", False):
            for device in self.devices:
                device.synchronize()

    def initialize(
        self,
        compiled: CompiledCallable[..., Any],
        session: InferenceSession | None = None,
    ) -> None:
        """Fills the owned buffers once, before a decoder can read them."""
        if self._initialized or is_virtual_device_mode():
            return
        if self.initializer is None:
            raise ValueError("constant inputs declare no static initializer")
        object.__setattr__(self, "_execution_owner", (session, compiled))
        compiled.execute_raw(*tree.leaves(self.buffers))
        object.__setattr__(self, "_initialized", True)

    def __post_init__(self) -> None:
        if is_virtual_device_mode() and not self.buffers:
            # Compile-only: a virtual device cannot allocate, and only
            # execution reads the buffers. The declarations still stand.
            return
        _, structure = tree.flatten(self.input_types)
        structure.flatten_up_to(self.buffers)


def allocate_static_buffers(
    input_types: tree.Tree[TensorType | BufferType], devices: Sequence[Device]
) -> tree.Tree[Buffer]:
    """Allocates statically shaped CPU deployment buffers pinned to the GPU.

    A virtual export records the declarations without allocating the buffers.
    """
    if is_virtual_device_mode():
        return {}
    if not devices or devices[0].is_host:
        raise ValueError(
            "static buffers require an accelerator for host pinning"
        )

    def allocate(spec: TensorType | BufferType) -> Buffer:
        if (
            not isinstance(spec, TensorType | BufferType)
            or not spec.device.is_cpu()
        ):
            raise ValueError("static resources must be CPU tensors or buffers")
        try:
            shape = tuple(int(dim) for dim in spec.shape)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "static buffers require static dimensions"
            ) from error
        if any(dim <= 0 for dim in shape):
            raise ValueError(
                "static buffers require positive static dimensions"
            )
        buffer = Buffer(
            shape=shape,
            dtype=spec.dtype,
            device=devices[0],
            usage=Usage.STAGING,
        )
        if not buffer.pinned:
            raise ValueError("static buffer allocation is not pinned")
        return buffer

    return tree.map(allocate, input_types)


def initialize_static_inputs(
    session: InferenceSession,
    constants: ConstantInputs | None,
) -> dict[str, CompiledCallable[..., Any]]:
    """Compiles and runs deployment-buffer initialization, when declared."""
    if constants is None or constants.initializer is None:
        return {}
    modules = compile_and_init_graphs(
        session, {GraphRole.STATIC_INIT: constants.initializer}
    )
    constants.initialize(modules[GraphRole.STATIC_INIT], session)
    return modules


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

    def __post_init__(self) -> None:
        n_devices = len(self.mesh.devices)
        if self.ep_size > 1 and self.ep_size % n_devices:
            raise ValueError(
                f"ep_size={self.ep_size} must be a multiple of the number of "
                f"devices on this node ({n_devices})"
            )


class ExpertParallelExchange(Protocol):
    """The initialized exchange's buffers, as each batch passes them."""

    def model_inputs(self) -> Sequence[Buffer]:
        """Returns the buffers, in the order of the graph's ``ep`` leaves."""
        ...


class ExpertParallelInputs:
    """How a decoder's graph takes the expert-parallel exchange.

    The graph takes each buffer of each device as its own input, named and
    ordered as the serving engine binds them
    (:func:`~max.nn.ep.init.payload_types`). A ModuleV3 decoder reads them
    as :class:`~max.nn.ep.EPBuffers`.
    """

    def input_tree(self, ep: EPConfig) -> EPPayload:
        """Returns the decoder's ``ep`` input types for ``ep``."""
        return ep_init.payload_types(ep)

    def exchange(
        self, initializer: EPInitializerLike
    ) -> ExpertParallelExchange:
        """Returns the buffers of a graph-API ``initializer``."""
        return initializer

    def initialize(
        self, session: InferenceSession, ep: EPConfig, mesh: DeviceMesh
    ) -> tuple[
        EPConfig,
        ExpertParallelExchange | None,
        dict[str, CompiledCallable[..., Any]],
    ]:
        """Sets up the exchange of a ModuleV3 decoder.

        Compiles the ``ep_init`` graph, which an export records for the
        serving engine, and runs it unless the devices are virtual. It sets
        up the fused MoE when ``MODULAR_EP_FUSED_MOE=1`` asks for it.

        Args:
            session: The session that compiles and runs the init graph.
            ep: The sized expert-parallel config.
            mesh: The decoder's mesh.

        Returns:
            ``ep`` with the node ID and fused MoE readiness that init found,
            the buffers each step passes, and the init graph by role. On
            virtual devices init finds nothing and there are no buffers; a
            ``pre-jit`` stop there also leaves no init graph.

        Raises:
            ValueError: If the exchange cannot run ``ep`` on ``mesh``.
        """
        check_supported(ep, mesh)
        ep = dataclasses.replace(ep, fused_moe=_ep_fused_moe_requested())
        with CompilationTimer("ep_init graph") as timer:
            init = ep_init.init_graph(ep)
            specs: dict[str, GraphSpec] = {
                GraphRole.EP_INIT: GraphSpec(
                    init, "ep_init", init.input_types()
                )
            }
            timer.mark_build_complete()
            if is_virtual_device_mode():
                try:
                    return ep, None, compile_and_init_graphs(session, specs)
                except CompilationStopped:
                    # `pre-jit` stops every compile, and this graph compiles
                    # before the decoder's, which must still be emitted.
                    return ep, None, {}
            modules = compile_and_init_graphs(session, specs)
        exchange = ep_init.start(ep, modules[GraphRole.EP_INIT].engine_model)
        return exchange.config, exchange, modules


class DeploymentConfig:
    """A registered config that the deployment completes.

    A registered config that subclasses it gets :meth:`apply_runtime` called
    before any module is constructed, and declares through its class fields
    what else the framework does for it. A config that doesn't subclass it
    takes every default.
    """

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

    @classmethod
    def weight_entries(cls) -> Mapping[str, Sequence[WeightEntry]]:
        """Returns the entries that name each role's checkpoint weights.

        A config that names the weights itself, by keeping the checkpoint's
        keys or overriding :meth:`prepare_weights`, declares here the names
        its roles' weights end up with, so an exported artifact names them
        as serving does. Keyed by
        :class:`~max.pipelines.lib.graph_specs.GraphRole`; the default names
        none.
        """
        return {}

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
    Registrations copy it onto the pipeline model. Exporters ask
    :meth:`ep_dispatch_token_layout_for`, which returns it unless the
    decoder's layout follows the backend."""

    expert_parallel_inputs: ClassVar[ExpertParallelInputs] = (
        ExpertParallelInputs()
    )
    """How the decoder's graph takes the exchange's buffers."""

    @classmethod
    def ep_dispatch_token_layout_for(cls, *, use_allreduce: bool) -> str | None:
        """Returns how tokens reach the expert dispatch under a backend.

        Exporters ask before any config is built, with the backend they
        export for. The default is :attr:`ep_dispatch_token_layout` under
        both backends; a decoder whose layout follows the backend overrides
        this instead.

        Args:
            use_allreduce: Whether the export uses the allreduce backend.
        """
        del use_allreduce
        return cls.ep_dispatch_token_layout

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
    initializer_cls: EPInitializerFactory,
) -> tuple[
    ModuleRuntime,
    ExpertParallelExchange | None,
    dict[str, CompiledCallable[..., Any]],
]:
    """Sizes and initializes the expert-parallel communication buffers.

    It sizes the buffers from the config, then compiles and runs the
    expert-parallel init graph, which sets the node ID each device picks its
    experts by. That is why it runs before the modules are constructed.

    Args:
        config: The finalized config, which sizes the buffers.
        runtime: The deployment, with ``ep_size`` above one and without
            ``ep``.
        session: The session that compiles the init graph.
        initializer_cls: Makes the initializer: the one of the pipeline
            model's module API.

    Returns:
        ``runtime`` with :attr:`ModuleRuntime.ep` set, the buffers the
        decoder's graph takes, as the config's
        :attr:`~ExpertParallelDeploymentConfig.expert_parallel_inputs`
        selects them, and the init graph by role, for the pipeline's
        ``compiled_modules``. On virtual devices, which can't run the init
        graph, the graph is only compiled and the buffers are ``None``. A
        ``pre-jit`` stop leaves no init graph.
    """
    ep = config.expert_parallel_config(runtime)
    if is_virtual_device_mode():
        try:
            _, modules = compile_ep_init(session, ep, initializer_cls)
        except CompilationStopped:
            # `pre-jit` stops every compile, and this graph compiles before
            # the decoder's. Propagating would leave the decoder unemitted.
            modules = {}
        return dataclasses.replace(runtime, ep=ep), None, modules
    initializer, modules = init_ep_comm(session, ep, initializer_cls)
    ep = initializer.config
    exchange = config.expert_parallel_inputs.exchange(initializer)
    return dataclasses.replace(runtime, ep=ep), exchange, modules


def ep_input_tree(
    config: ExpertParallelDeploymentConfig, ep: EPConfig
) -> EPPayload:
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
