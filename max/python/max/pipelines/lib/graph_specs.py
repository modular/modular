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
"""Compiles a pipeline's graphs by role.

A pipeline describes each graph it runs as a :class:`GraphSpec` under a role
key, such as ``"decoder"`` or ``"sampler"``. The keys are whatever the
pipeline's task defines; :class:`GraphRole` holds MAX's built-in ones.
:func:`compile_graphs` compiles the graphs into :class:`CompiledArtifacts`,
which stay available after initialization, and :func:`init_graphs`
initializes them for execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from max.driver import DLPackArray
from max.engine import CompiledModel, InferenceSession, Model
from max.graph import Graph, Type
from max.graph import Module as GraphModule
from max.nn import Module


class GraphRole(str, Enum):
    """The roles of the graphs MAX's pipelines run."""

    DECODER = "decoder"
    """The forward pass: tokens and KV cache to logits."""

    SAMPLER = "sampler"
    """Samples the next token from the decoder's logits."""

    SAMPLER_BITMASK = "sampler_bitmask"
    """Samples the next token, masked by a structured-output bitmask."""

    VISION = "vision"
    """The vision encoder: image patches to soft-token embeddings."""

    ENCODER = "encoder"
    """The single forward pass of a model without a KV cache, such as an
    embedding model: inputs to embeddings or hidden states."""

    EP_INIT = "ep_init"
    """The one-time expert-parallel setup graph run at engine startup."""

    SPEC_DECODER = "spec_decoder"
    """One speculative-decoding step, the target and draft models fused:
    tokens, the previous drafts and KV cache to the accepted tokens and the
    next drafts."""


@dataclass(frozen=True)
class GraphSpec:
    """A graph a pipeline compiles, as the module that traces it.

    Args:
        module: The module whose ``__call__`` the graph traces.
        name: The graph's name. Graphs in one group need distinct names.
        input_types: The type of each graph input.
        weights: The weight values, keyed by name. Defaults to the module's
            :meth:`~max.nn.Module.state_dict` once traced.
        group: Roles with the same group compile into one artifact, which
            holds each role's graph as a named submodel. ``None`` compiles
            the role on its own.
        custom_extensions: Paths to custom kernel libraries the graph uses.
        is_device_graph: Whether to build a device graph.
    """

    module: Module
    name: str
    input_types: Sequence[Type[Any]]
    weights: Mapping[str, DLPackArray] | None = None
    group: str | None = None
    custom_extensions: Sequence[Path] = ()
    is_device_graph: bool = False

    def trace(self, module: GraphModule | None = None) -> Graph:
        """Traces the graph, into ``module`` if given."""
        return self.module.trace(
            *self.input_types,
            name=self.name,
            module=module,
            custom_extensions=self.custom_extensions,
            is_device_graph=self.is_device_graph,
        )

    def weights_registry(self) -> Mapping[str, DLPackArray]:
        """Returns the weight values the graph binds, keyed by name."""
        if self.weights is not None:
            return self.weights
        return self.module.state_dict()


@dataclass(frozen=True)
class CompiledArtifacts:
    """The compiled artifacts of a set of roles.

    Holds no device state: :func:`init_graphs` initializes the graphs for
    execution. Look up the artifact that holds a role's graph with
    :meth:`artifact`.

    Args:
        compiled: One artifact per compile group.
    """

    compiled: Sequence[CompiledModel] = ()
    # Each role's index into `compiled`.
    _roles: Mapping[str, int] = field(default_factory=dict)

    def artifact(self, role: str) -> CompiledModel:
        """Returns the artifact that holds a role's graph.

        Args:
            role: The role to look up.

        Raises:
            KeyError: If no graph is registered for ``role``.
        """
        if role not in self._roles:
            raise KeyError(f"no graph is registered for role {role!r}")
        return self.compiled[self._roles[role]]

    def __contains__(self, role: object) -> bool:
        return role in self._roles

    def __or__(self, other: CompiledArtifacts) -> CompiledArtifacts:
        overlap = self._roles.keys() & other._roles.keys()
        if overlap:
            raise ValueError(f"roles {sorted(overlap)} are registered twice")
        offset = len(self.compiled)
        return CompiledArtifacts(
            compiled=[*self.compiled, *other.compiled],
            _roles={
                **self._roles,
                **{role: i + offset for role, i in other._roles.items()},
            },
        )


def compile_graphs(
    session: InferenceSession, specs: Mapping[str, GraphSpec]
) -> CompiledArtifacts:
    """Compiles each role's graph.

    Roles that share a group trace into one :class:`~max.graph.Module` and
    compile once. Each artifact compiles through
    :meth:`~max.engine.InferenceSession.compile_reusing_mefs`, so an ungrouped
    role reuses the session's precompiled or exported MEFs. A group compiles
    every time, as any module of several graphs does.

    Args:
        session: The session that compiles the graphs.
        specs: The graphs to compile, keyed by role.

    Returns:
        The compiled artifacts, ready for :func:`init_graphs`.

    Raises:
        ValueError: If two roles in one group share a graph name.
    """
    groups: dict[str, list[str]] = {}
    for role, spec in specs.items():
        # Ungrouped roles compile alone, so each gets its own key.
        key = spec.group if spec.group is not None else f"\0{role}"
        groups.setdefault(key, []).append(role)

    compiled: list[CompiledModel] = []
    roles: dict[str, int] = {}
    for members in groups.values():
        names = [specs[role].name for role in members]
        if len(set(names)) != len(names):
            raise ValueError(
                f"roles {members} share a group but not distinct graph names:"
                f" {names}"
            )
        if len(members) == 1:
            artifact: Graph | GraphModule = specs[members[0]].trace()
            extensions = None
        else:
            artifact = GraphModule()
            for role in members:
                specs[role].trace(artifact)
            # A graph carries its own extensions; a module of graphs does not.
            extensions = [
                path
                for role in members
                for path in specs[role].custom_extensions
            ]
        for role in members:
            roles[role] = len(compiled)
        compiled.append(
            session.compile_reusing_mefs(artifact, custom_extensions=extensions)
        )
    return CompiledArtifacts(compiled=compiled, _roles=roles)


def init_graphs(
    session: InferenceSession,
    artifacts: CompiledArtifacts,
    specs: Mapping[str, GraphSpec],
) -> dict[str, Model]:
    """Initializes compiled graphs for execution, with their specs' weights.

    Args:
        session: The session that initializes the graphs.
        artifacts: The artifacts :func:`compile_graphs` compiled from
            ``specs``.
        specs: The graphs' specs, keyed by role.

    Returns:
        Each role's initialized model, ready to execute.

    Raises:
        KeyError: If a role in ``specs`` is not in ``artifacts``.
        ValueError: If two specs in one artifact bind different values to one
            weight name.
    """
    members: dict[int, list[str]] = {}
    for role in specs:
        if role not in artifacts:
            raise KeyError(f"role {role!r} was not compiled")
        members.setdefault(artifacts._roles[role], []).append(role)

    models: dict[str, Model] = {}
    for index, roles in members.items():
        weights: dict[str, DLPackArray] = {}
        for role in roles:
            for name, value in specs[role].weights_registry().items():
                if weights.setdefault(name, value) is not value:
                    raise ValueError(
                        f"weight {name!r} has two different values"
                    )
        by_name = session.init_all(
            artifacts.compiled[index], weights_registry=weights
        )
        for role in roles:
            models[role] = by_name[specs[role].name]
    return models
