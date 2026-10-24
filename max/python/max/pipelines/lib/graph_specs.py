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
:func:`compile_and_init_graphs` compiles and initializes the graphs,
returning one :class:`~max.experimental.compilation.CompiledCallable` per
role.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

from max import tree
from max._artifact.value_types import ArtifactError, treedef_to_json
from max.driver import DLPackArray
from max.engine import CompiledModel, InferenceSession, Model
from max.experimental.compilation import CompiledCallable, StagedGraph
from max.experimental.nn import Module as ModuleV3
from max.experimental.sharding import TensorLayout
from max.experimental.sharding.types import as_layout
from max.experimental.tensor import Tensor
from max.graph import Graph, Type
from max.graph import Module as GraphModule
from max.nn import Module
from max.nn.graph_groups import HiddenStates, Logits


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

    A :class:`~max.experimental.nn.Module` (ModuleV3) graph traces its
    ``forward`` through :meth:`~max.experimental.nn.Module.stage` and
    compiles on its own.

    Args:
        module: The module whose ``__call__`` (``forward`` for a ModuleV3
            module) the graph traces.
        name: The graph's name. Graphs in one group need distinct names.
        input_types: The graph's input types: a sequence, or a named tree
            such as a module's ``input_tree`` whose leaves are the inputs in
            order. A ModuleV3 graph takes a sequence, one type per argument
            of its ``forward``.
        weights: The weight values, keyed by name. Defaults to the module's
            :meth:`~max.nn.Module.state_dict` once traced, or a ModuleV3
            module's own parameters.
        group: Roles with the same group compile into one model, which
            holds each role's graph as a named submodel. ``None`` compiles
            the role on its own. A ModuleV3 graph takes no group.
        custom_extensions: Paths to custom kernel libraries the graph uses.
        is_device_graph: Whether to build a device graph.
        auto_cast: Whether a ModuleV3 graph casts a ``float32`` or
            ``bfloat16`` weight to the dtype its parameter declares.
    """

    module: Module | ModuleV3[..., Any]
    name: str
    input_types: Sequence[Type[Any]] | Mapping[str, object]
    weights: Mapping[str, DLPackArray] | None = None
    group: str | None = None
    custom_extensions: Sequence[Path] = ()
    is_device_graph: bool = False
    auto_cast: bool = False

    def trace(self, module: GraphModule | None = None) -> Graph:
        """Traces the graph, into ``module`` if given.

        Raises:
            ValueError: If ``module`` is given for a ModuleV3 graph.
        """
        if isinstance(self.module, ModuleV3):
            if module is not None:
                raise ValueError(
                    f"ModuleV3 graph {self.name!r} cannot trace into a shared"
                    " module; give its spec no group"
                )
            return self._staged[0].graph
        return self.module.trace(
            *tree.leaves(self.input_types),
            name=self.name,
            module=module,
            custom_extensions=self.custom_extensions,
            is_device_graph=self.is_device_graph,
        )

    def weights_registry(self) -> Mapping[str, DLPackArray]:
        """Returns the weight values the graph binds, keyed by name."""
        if isinstance(self.module, ModuleV3):
            return self._staged[1]
        if self.weights is not None:
            return self.weights
        return self.module.state_dict()

    @cached_property
    def _staged(self) -> tuple[StagedGraph[..., Any], dict[str, Tensor]]:
        # A ModuleV3 graph and its registry come from one trace, which
        # records the layout each checkpoint entry arrives in.
        assert isinstance(self.module, ModuleV3)
        assert isinstance(self.input_types, Sequence)
        return self.module.stage(
            *self.input_types,
            weights=self.weights,
            name=self.name,
            custom_extensions=self.custom_extensions,
            auto_cast=self.auto_cast,
            is_device_graph=self.is_device_graph,
        )


# A context variable rather than a config field: only the export entry point
# sets it, around the pipeline build, and no pipeline option should.
# TODO(MXSERV-765): Export builds and initializes the whole pipeline through
# PIPELINE_REGISTRY.retrieve only to compile its graphs; replace this with a
# compile-only export path that compiles each role directly.
_compile_roles_separately: ContextVar[bool] = ContextVar(
    "compile_roles_separately", default=False
)


@contextmanager
def compile_roles_separately() -> Iterator[None]:
    """Compiles each role's graph on its own while active, ignoring groups.

    A pipeline artifact stores one MEF per role, so ``max export`` builds
    the pipeline inside this. Serving keeps the groups.
    """
    token = _compile_roles_separately.set(True)
    try:
        yield
    finally:
        _compile_roles_separately.reset(token)


def compile_and_init_graphs(
    session: InferenceSession, specs: Mapping[str, GraphSpec]
) -> dict[str, CompiledCallable[..., Any]]:
    """Compiles and initializes each role's graph.

    Roles that share a group trace into one :class:`~max.graph.Module`,
    compile once, and initialize once with the weights of all of them; each
    role's callable then holds the same compiled model. Each compiles through
    :meth:`~max.engine.InferenceSession.compile_reusing_mefs`, so an ungrouped
    role reuses the session's precompiled or exported MEFs. A group compiles
    every time, as any module of several graphs does. Inside
    :func:`compile_roles_separately`, groups are ignored.

    Each callable is called with one argument per input type, positionally,
    or with the named input tree as its one argument, and returns the graph's
    outputs in the tree its ``forward`` returned. It keeps no weight registry,
    so the host weights can be released after initialization. A ModuleV3
    role's callable instead takes and returns what the module's ``forward``
    does, adds the signal buffers a multi-device graph takes, and keeps its
    weight registry.

    Args:
        session: The session that compiles and initializes the graphs.
        specs: The graphs to compile, keyed by role.

    Returns:
        Each role's compiled, initialized graph.

    Raises:
        ValueError: If two roles in one group share a graph name, or bind
            different values to one weight name.
    """
    separately = _compile_roles_separately.get()
    groups: dict[str, list[str]] = {}
    for role, spec in specs.items():
        # Ungrouped roles compile alone, so each gets its own key.
        key = (
            spec.group
            if spec.group is not None and not separately
            else f"\0{role}"
        )
        groups.setdefault(key, []).append(role)

    callables: dict[str, CompiledCallable[..., Any]] = {}
    for members in groups.values():
        names = [specs[role].name for role in members]
        if len(set(names)) != len(names):
            raise ValueError(
                f"roles {members} share a group but not distinct graph names:"
                f" {names}"
            )
        traced_graph: Graph | GraphModule
        if len(members) == 1:
            graphs = {members[0]: specs[members[0]].trace()}
            traced_graph = graphs[members[0]]
            extensions = None
        else:
            traced_graph = GraphModule()
            graphs = {role: specs[role].trace(traced_graph) for role in members}
            # A graph carries its own extensions; a module of graphs does not.
            extensions = [
                path
                for role in members
                for path in specs[role].custom_extensions
            ]
        output_trees = {
            role: graph.output_tree for role, graph in graphs.items()
        }
        # Not session.load_all, which drops the compiled model a callable
        # keeps to export its MEF.
        compiled = session.compile_reusing_mefs(
            traced_graph, custom_extensions=extensions
        )

        weights: dict[str, DLPackArray] = {}
        for role in members:
            for name, value in specs[role].weights_registry().items():
                if weights.setdefault(name, value) is not value:
                    raise ValueError(
                        f"weight {name!r} has two different values"
                    )
        by_name = session.init_all(compiled, weights_registry=weights)
        for role in members:
            if len(by_name) == 1:
                # The session can hand back a cached MEF compiled from an
                # identical graph under another name, so don't look it up by
                # this spec's name.
                (model,) = by_name.values()
            else:
                model = by_name[specs[role].name]
            spec = specs[role]
            if isinstance(spec.module, ModuleV3):
                # The trace's own signature, so the callable takes and
                # returns what ``forward`` does and adds any signal buffers.
                staged, registry = spec._staged
                callables[role] = CompiledCallable.from_engine_model(
                    staged, compiled, model, registry
                )
            else:
                callables[role] = _callable(
                    compiled, model, spec.input_types, output_trees[role]
                )
                _check_inputs(role, callables[role].in_tree)
                _check_outputs(role, callables[role].out_tree)
    return callables


def _check_inputs(role: str, in_tree: tree.TreeDef) -> None:
    """Fails a load whose decoder names inputs a serving engine can't place.

    A decoder that names its inputs, rather than passing them positionally,
    has a serving engine find each one by its group's kind, so every input
    must sit inside a kinded group.

    Raises:
        ValueError: If a named decoder input is outside every kinded group.
    """
    args = in_tree.children[0]
    if role != GraphRole.DECODER or all(
        arg.kind == "leaf" for arg in args.children
    ):
        return
    unkinded = [
        path
        for path, kinded in zip(
            args.leaf_paths, _leaves_in_kinded_groups(args), strict=True
        )
        if not kinded
    ]
    if unkinded:
        raise ValueError(
            "the decoder graph's inputs "
            + ", ".join(unkinded)
            + " are in no kinded group: build them with a"
            " max.nn.graph_groups group or a kinded KV cache input"
        )


def _leaves_in_kinded_groups(
    treedef: tree.TreeDef, inside: bool = False
) -> Iterator[bool]:
    """Yields, per leaf in order, whether a kinded node encloses it."""
    inside = inside or (
        treedef.kind == "node" and tree.kind_of(treedef.meta[0]) is not None
    )
    if treedef.kind == "leaf":
        yield inside
    for child in treedef.children:
        yield from _leaves_in_kinded_groups(child, inside)


def _check_outputs(role: str, out_tree: tree.TreeDef) -> None:
    """Fails a load whose outputs an exported artifact could not name.

    Raises:
        ArtifactError: If the output structure can't be serialized.
        ValueError: If a decoder has neither a ``logits`` group nor, as a
            pipeline stage before the last, ``hidden_states``.
    """
    try:
        treedef_to_json(out_tree)
    except ArtifactError as e:
        raise ArtifactError(f"{role} graph outputs: {e}") from e
    # A pipeline stage before the last hands on hidden states, not logits.
    # TODO(MXSERV-682): ModuleV3 decoders still return positional outputs.
    if role == GraphRole.DECODER and not (
        _has_kind(out_tree, Logits) or _has_kind(out_tree, HiddenStates)
    ):
        raise ValueError(
            "the decoder graph's outputs have no logits group: return"
            " max.nn.transformer.logits_outputs(...) or a DecoderOutputs"
            " with a Logits group"
        )


def _has_kind(treedef: tree.TreeDef, cls: type) -> bool:
    if treedef.kind == "node" and treedef.meta[0] is cls:
        return True
    return any(_has_kind(child, cls) for child in treedef.children)


def _input_signature(
    input_types: Sequence[Type[Any]] | Mapping[str, object],
) -> tuple[list[TensorLayout], tree.TreeDef]:
    """Returns the input layouts and call structure of a role's callable.

    A sequence of input types is called positionally. A named tree, such as
    a module's ``input_tree``, is the one argument, so its names and group
    kinds reach the artifact.

    Args:
        input_types: A :class:`GraphSpec`'s input types.

    Returns:
        The layouts in graph order, and the structure of ``(args, kwargs)``.
    """
    args = (
        (input_types,)
        if isinstance(input_types, Mapping)
        else tuple(input_types)
    )
    # TODO(MXSERV-766): as_layout opens each GPU DeviceRef as a device
    # (DeviceRef.to_device), so building a layout needs the physical GPU.
    return tree.flatten((tree.map(as_layout, args), {}), leaf=TensorLayout)


def _callable(
    compiled: CompiledModel,
    model: Model,
    input_types: Sequence[Type[Any]] | Mapping[str, object],
    output_types: tree.Tree[Type[Any]],
) -> CompiledCallable[..., Any]:
    in_layouts, in_tree = _input_signature(input_types)
    out_layouts, out_tree = tree.flatten(
        tree.map(as_layout, output_types), leaf=TensorLayout
    )
    return CompiledCallable.from_compiled(
        compiled, model, in_layouts, in_tree, out_layouts, out_tree
    )
