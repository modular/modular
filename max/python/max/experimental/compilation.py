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
"""Transforms over callables on tensors: stage, compile and as_subgraph.

:func:`stage` records a callable into a :class:`StagedGraph`, and
:func:`compile` goes one step further to a :class:`CompiledCallable`, which
initializes only when first called, so exporting a MEF never builds a model.
:func:`as_subgraph` lowers a callable to one shared subgraph body per distinct
stage. Options bind at the transform and specs at the call.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeAlias, TypeVar, get_args

from max.driver import Accelerator, Buffer, Device, DLPackArray
from max.engine import CompiledModel, Model
from max.experimental import tree_utils as tree
from max.experimental.realization_context import (
    GraphRealizationContext,
    _cached_signal_buffers,
    in_graph_context,
    open_subgraph,
    share_subgraph,
    subgraph_context,
)
from max.experimental.sharding import (
    DeviceMesh,
    DistributedBufferType,
    PlacementMapping,
    Replicated,
    TensorLayout,
)
from max.experimental.sharding.placements import local_shard_shape_from_global
from max.experimental.support import _session
from max.experimental.tensor import Tensor, realization_context
from max.experimental.tree_utils import TreeDef
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    StaticDim,
    TensorType,
    TensorValue,
    Value,
    ops,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")

Spec: TypeAlias = (
    TensorLayout | TensorType | Tensor | BufferType | DistributedBufferType
)
"""What one tensor argument may be staged as."""

NormalizedSpec: TypeAlias = TensorLayout | BufferType | DistributedBufferType
"""A :data:`Spec` that :func:`as_layout` has normalized, and so what the graph
reads."""

#: The types accepted as one input spec, and so the leaves of a spec tree.
SPEC_TYPES: tuple[type, ...] = get_args(Spec)
#: The types a graph boundary carries, and so the leaves of a staged result.
GRAPH_VALUE_TYPES = (BufferValue, TensorValue)


def _sanitized_graph_name(fn: Callable[..., object]) -> str:
    """Returns ``fn``'s name with non-identifier characters folded away."""
    raw = getattr(fn, "__name__", None) or type(fn).__name__
    return re.sub(r"\W+", "_", raw).strip("_") or "fn"


def as_layout(spec: Spec) -> NormalizedSpec:
    """Normalizes a tensor spec into a layout.

    Args:
        spec: A :class:`~max.experimental.sharding.TensorLayout`, a
            :class:`~max.graph.TensorType`, an example
            :class:`~max.experimental.tensor.Tensor`, or a buffer spec.

    Returns:
        The equivalent layout, or ``spec`` itself when it is a buffer spec.

    Raises:
        TypeError: If ``spec`` is none of the accepted forms.
    """
    if isinstance(spec, (TensorLayout, BufferType, DistributedBufferType)):
        return spec
    if isinstance(spec, Tensor):
        return TensorLayout(spec.dtype, spec.shape, spec.mapping)
    if isinstance(spec, TensorType):
        mesh = DeviceMesh.single(spec.device.to_device())
        return TensorLayout(
            spec.dtype, spec.shape, PlacementMapping(mesh, (Replicated(),))
        )
    raise TypeError(
        f"expected a TensorLayout, TensorType, Tensor, BufferType or "
        f"DistributedBufferType spec, got {type(spec).__name__}: {spec!r}"
    )


def _graph_input_types(
    spec: NormalizedSpec,
) -> list[TensorType | BufferType]:
    """The graph input types one layout contributes, in mesh order."""
    if isinstance(spec, BufferType):
        return [spec]
    if isinstance(spec, TensorLayout):
        shapes = local_shard_shape_from_global(
            spec.shape, spec.mesh, spec.placements
        )
        return [
            TensorType(spec.dtype, shape, DeviceRef.from_device(device))
            for shape, device in zip(shapes, spec.mesh.devices, strict=True)
        ]
    return list(spec.local_types)


def _fills_one_slot(value: object) -> bool:
    """Whether ``value`` fills one argument slot rather than nesting further."""
    # Deferring to ``tree.is_node`` keeps this agreeing with ``_record_graph``'s walk.
    return isinstance(value, SPEC_TYPES) or not tree.is_node(value)


@dataclasses.dataclass(frozen=True)
class Signature:
    """How a flat graph boundary maps onto a Python call.

    The trees here are :obj:`~typing.Any` because a call carries whatever the
    caller passed -- specs beside floats, strings and records -- and they cross
    this boundary rather than being inspected, so no union closes them and
    :class:`object` would not survive the round trip.
    """

    in_specs: Any
    """One layout per tensor argument, nested as the staged call passed them."""

    out_structure: TreeDef
    """How to rebuild the return value from the graph's flat results."""

    def flatten(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> list[Buffer]:
        """Checks a call against the specs and flattens it into graph order.

        Args:
            args: One :class:`~max.experimental.tensor.Tensor` per tensor
                argument, nested as the staged call passed them.
            kwargs: The staged signature's keyword arguments.

        Returns:
            One buffer per graph input, a distributed argument expanded into
            one per shard.

        Raises:
            TypeError: If an argument is not a
                :class:`~max.experimental.tensor.Tensor` where one belongs.
            ValueError: If the arguments do not match what was staged.
        """
        # Arity first: the path mismatch below reports it only as an absence.
        wanted_args, wanted_kwargs = self.in_specs
        if len(args) != len(wanted_args) or set(kwargs) != set(wanted_kwargs):
            raise ValueError(
                f"expected {len(wanted_args)} positional argument(s), got "
                f"{len(args)}, and keywords {sorted(wanted_kwargs)}, got "
                f"{sorted(kwargs)}."
            )

        # A call boundary is positional, so a walk across it duplicates.
        wanted = tree.paths(self.in_specs, leaf=_fills_one_slot)
        given = tree.paths((args, dict(kwargs)), leaf=_fills_one_slot)
        if list(wanted) != list(given):
            raise ValueError(
                f"tree structure mismatch: expected {list(wanted)}, "
                f"got {list(given)}"
            )

        buffers: list[Buffer] = []
        for path, layout in wanted.items():
            arg, name = given[path], path.partition(".")[2]
            if not isinstance(layout, SPEC_TYPES):
                # __eq__ may be elementwise, so only a clean True counts.
                try:
                    matches = arg is layout or bool(arg == layout)
                except Exception:
                    matches = False
                if not matches:
                    raise ValueError(
                        f"tree structure mismatch: argument {name} staged as "
                        f"{layout!r}, got {arg!r}"
                    )
                continue
            if not isinstance(arg, Tensor):
                raise TypeError(
                    f"argument {name}: expected a Tensor, got "
                    f"{type(arg).__name__}; use execute_raw()"
                )
            shards = arg.local_shards
            if isinstance(layout, (BufferType, DistributedBufferType)):
                # Dtype and shard count only: extent is whatever was allocated.
                expected = _graph_input_types(layout)
                if len(shards) != len(expected) or arg.dtype != layout.dtype:
                    raise ValueError(
                        f"argument {name}: expected {layout.dtype} across "
                        f"{len(expected)} device(s), got {arg.dtype} across "
                        f"{len(shards)}"
                    )
                buffers.extend(shard.driver_tensor for shard in shards)
                continue
            # Every non-buffer spec went through ``as_layout`` when staged.
            assert isinstance(layout, TensorLayout)
            devices = [shard.device for shard in shards]
            # A symbolic dim was staged to accept any size, so only match statics.
            if (
                (arg.dtype, len(arg.shape)) != (layout.dtype, len(layout.shape))
                or devices != list(layout.mesh.devices)
                or any(
                    isinstance(dim, StaticDim) and dim != size
                    for dim, size in zip(layout.shape, arg.shape, strict=True)
                )
            ):
                raise ValueError(
                    f"argument {name}: expected {layout.dtype} shape "
                    f"{list(layout.shape)} on {list(layout.mesh.devices)}, got "
                    f"{arg.dtype} shape {list(arg.shape)} on {devices}"
                )
            buffers.extend(shard.driver_tensor for shard in shards)
        return buffers

    def unflatten(self, buffers: Sequence[Buffer]) -> Any:
        """Rebuilds the staged callable's return value from flat results.

        The counterpart of :meth:`flatten`, on the way back out.

        Args:
            buffers: The graph's results, flat and in graph order.

        Returns:
            The staged callable's return value, nested as it returned it.
        """
        return tree.unflatten(self.out_structure, list(buffers))


@dataclasses.dataclass(frozen=True, repr=False)
class StagedGraph(Generic[_P, _R]):
    """A graph and the calling convention it was staged from.

    What :func:`stage` returns; printing one renders the MLIR of the whole
    module, subgraph bodies included. It cannot run: :meth:`compile` it first.

    .. code-block:: python

        print(compilation.stage(attend)(x_spec, mask=mask_spec))
    """

    graph: Graph
    """The recorded graph."""

    signature: Signature
    """How a call crosses the graph's boundary."""

    signal_device_ids: tuple[int, ...] = ()
    """The accelerators whose collectives need signal buffers."""

    def __str__(self) -> str:
        """Returns the whole module's MLIR, shared subgraph bodies included.

        The graph's own op names the bodies it calls but does not contain
        them, so rendering only that would hide them -- and would quietly
        weaken any check that an op is *absent*.
        """
        return str(self.graph._module)

    # The default repr would pass a test asserting an op is absent.
    __repr__ = __str__

    def compile(
        self, *, weights: Mapping[str, DLPackArray] | None = None
    ) -> CompiledCallable[_P, _R]:
        """Compiles the graph, binding neither weights nor device memory.

        Args:
            weights: Data for the external constants the graph declares, keyed
                as the graph names them, one entry per shard of a distributed
                weight. Bound when the model initializes, not here.

        Returns:
            The :class:`CompiledCallable`, which initializes on its first call.
        """
        return CompiledCallable(
            self.signature,
            _session().compile(self.graph),
            self.signal_device_ids,
            dict(weights or {}),
        )


@dataclasses.dataclass(frozen=True, eq=False)
class CompiledCallable(Generic[_P, _R]):
    """Calls a compiled graph, initializing its model once, on first call.

    What :func:`compile` returns. It holds no graph: :attr:`signature` is all
    that a call needs, and :meth:`export_mef` writes :attr:`artifact` without
    ever initializing a model.

    .. code-block:: python

        compiled = compilation.compile(scale)(spec)
        result = compiled(Tensor.ones([3]))
    """

    signature: Signature
    """How a call crosses the graph's boundary."""

    artifact: CompiledModel
    """The compiled graph, binding neither weights nor device memory."""

    signal_device_ids: tuple[int, ...] = ()
    """The accelerators whose collectives need signal buffers."""

    weights: Mapping[str, DLPackArray] = dataclasses.field(default_factory=dict)
    """Data for the external constants the graph declares, keyed as the graph
    names them, one entry per shard of a distributed weight."""

    @functools.cached_property
    def engine_model(self) -> Model:
        """The initialized model this calls, with :attr:`weights` bound in."""
        return _session().init(
            self.artifact, weights_registry=dict(self.weights)
        )

    @property
    def signal_buffers(self) -> list[Buffer]:
        """Buffers for multi-device collectives, allocated once, not per call."""
        ids = self.signal_device_ids
        return _cached_signal_buffers(ids)[0] if ids else []

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Runs the graph on ``args``, rebuilding the result's tree.

        Args:
            args: One :class:`~max.experimental.tensor.Tensor` per tensor
                argument of the staged signature, nested as it passed them.
            kwargs: The staged signature's keyword arguments.

        Returns:
            The staged callable's return value.

        Raises:
            TypeError: If an argument is not a
                :class:`~max.experimental.tensor.Tensor` where one belongs.
            ValueError: If the arguments do not match what was staged.
        """
        buffers = self.signature.flatten(args, kwargs)
        return self.signature.unflatten(self.execute_raw(*buffers))

    def execute_raw(self, *buffers: Buffer) -> list[Buffer]:
        """Executes the graph on raw buffers, appending the signal buffers.

        Args:
            buffers: One per graph input, a distributed argument expanded into
                one per shard.

        Returns:
            The result buffers, flat and in graph order.
        """
        return list(self.engine_model(*buffers, *self.signal_buffers))

    def export_mef(self, path: str | Path) -> None:
        """Writes the compiled graph to a MEF file at ``path``.

        Compiling is all this needs, so it never initializes the model.

        Args:
            path: Where to write the file.
        """
        self.artifact.export_mef(path)


def _signal_device_ids(
    layouts: Sequence[NormalizedSpec], signal_devices: Iterable[Device]
) -> tuple[int, ...]:
    """The accelerators whose collectives need signal buffers.

    Args:
        layouts: The input layouts, whose meshes span devices of their own.
        signal_devices: Devices taking part beyond what the inputs span.

    Returns:
        Their ids, or empty when fewer than two accelerators take part.
    """
    spanned = (
        device
        for layout in layouts
        if not isinstance(layout, BufferType) and layout.mesh.num_devices > 1
        for device in layout.mesh.devices
    )
    ids = tuple(
        dict.fromkeys(
            device.id
            for device in itertools.chain(spanned, signal_devices)
            if isinstance(device, Accelerator)
        )
    )
    return ids if len(ids) > 1 else ()


def _argument_tensor(
    ctx: GraphRealizationContext,
    values: Iterator[Value[Any]],
    spec: NormalizedSpec,
) -> Tensor:
    """One staged argument, rebuilt from the graph inputs its spec claims.

    Args:
        ctx: The context the rebuilt tensor records into.
        values: The graph's inputs, in spec order, drained by what ``spec``
            spans.
        spec: What the argument was staged as.

    Returns:
        The tensor to pass ``fn`` in that argument's place.
    """
    if isinstance(spec, BufferType):
        return Tensor.from_graph_value(next(values).buffer)
    shards = itertools.islice(values, spec.mesh.num_devices)
    if isinstance(spec, DistributedBufferType):
        return ctx.create_unrealized(
            tuple(value.buffer for value in shards),
            mapping=PlacementMapping(spec.mesh, spec.placements),
        )
    return ctx.create_unrealized(
        tuple(value.tensor for value in shards), mapping=spec.mapping
    )


def stage(
    fn: Callable[_P, _R],
    *,
    name: str | None = None,
    custom_extensions: Iterable[Path] = (),
    allow_subgraphs: bool = True,
    signal_devices: Iterable[Device] = (),
    is_device_graph: bool = False,
) -> Callable[..., StagedGraph[_P, _R]]:
    """Returns a stager that records ``fn`` into a graph, without compiling it.

    .. code-block:: python

        print(compilation.stage(attend)(x_spec, mask=mask_spec))

    Args:
        fn: The callable to record, over
            :class:`~max.experimental.tensor.Tensor` values or containers of
            them.
        name: The graph's name. Defaults to ``fn``'s own name.
        custom_extensions: Paths to custom Mojo kernel libraries.
        allow_subgraphs: Whether :func:`as_subgraph` bodies become shared
            subgraphs rather than inlining into the caller.
        signal_devices: Devices taking part in collectives beyond what the
            specs span.
        is_device_graph: Whether to record a device graph.

    Returns:
        A callable taking one spec per argument of ``fn``, as :func:`as_layout`
        accepts them, and returning the :class:`StagedGraph`.
    """

    def record(*args: Any, **kwargs: Any) -> StagedGraph[_P, _R]:
        # Positional, like Signature.flatten: one graph input per argument slot.
        in_specs = tree.map(as_layout, (args, dict(kwargs)), leaf=SPEC_TYPES)
        layouts, structure = tree.flatten(in_specs, leaf=SPEC_TYPES)
        types = [t for spec in layouts for t in _graph_input_types(spec)]
        ids = _signal_device_ids(layouts, signal_devices)
        graph = Graph(
            name or _sanitized_graph_name(fn),
            input_types=[
                *types,
                *(_cached_signal_buffers(ids)[1] if ids else []),
            ],
            custom_extensions=custom_extensions,
            is_device_graph=is_device_graph,
        )
        ctx = GraphRealizationContext(
            graph,
            signal_buffers=[i.buffer for i in graph.inputs[len(types) :]]
            or None,
        )
        if allow_subgraphs:
            ctx.subgraph_cache = {}

        values = iter(graph.inputs[: len(types)])
        with realization_context(ctx), ctx:
            in_args, in_kwargs = tree.unflatten(
                structure,
                [_argument_tensor(ctx, values, spec) for spec in layouts],
            )
            # Also positional on the way out: ``return x, x`` is two results.
            flat, out_structure = tree.flatten(
                fn(*in_args, **in_kwargs), leaf=GRAPH_VALUE_TYPES
            )
            graph.output(*flat)
        return StagedGraph(graph, Signature(in_specs, out_structure), ids)

    return record


def compile(
    fn: Callable[_P, _R],
    *,
    weights: Mapping[str, DLPackArray] | None = None,
    name: str | None = None,
    custom_extensions: Iterable[Path] = (),
    allow_subgraphs: bool = True,
    signal_devices: Iterable[Device] = (),
    is_device_graph: bool = False,
) -> Callable[..., CompiledCallable[_P, _R]]:
    """Returns a compiler that records ``fn`` and compiles it.

    Like :func:`stage`, plus the compile. Initializing a model is what waits
    for the first call, so exporting a MEF needs nothing more than this.

    .. code-block:: python

        compiled = compilation.compile(scale)(spec)

    Args:
        fn: The callable to compile.
        weights: As :meth:`StagedGraph.compile` takes them.
        name: As :func:`stage` takes it.
        custom_extensions: As :func:`stage` takes them.
        allow_subgraphs: As :func:`stage` takes it.
        signal_devices: As :func:`stage` takes them.
        is_device_graph: As :func:`stage` takes it.

    Returns:
        A callable taking one spec per argument of ``fn`` and returning the
        compiled :class:`CompiledCallable`.
    """

    def stage_and_compile(
        *args: Any, **kwargs: Any
    ) -> CompiledCallable[_P, _R]:
        return stage(
            fn,
            name=name,
            custom_extensions=custom_extensions,
            allow_subgraphs=allow_subgraphs,
            signal_devices=signal_devices,
            is_device_graph=is_device_graph,
        )(*args, **kwargs).compile(weights=weights)

    return stage_and_compile


class _InferKey:
    """Sentinel for "no key given", which a caller's own ``None`` is not."""


_INFER_KEY = _InferKey()


def _inferred_key(fn: Callable[..., Any]) -> str | None:
    """A dedup key for ``fn``, or ``None`` when it does not fix its body."""
    if getattr(fn, "__closure__", None) is not None:
        return None
    if hasattr(fn, "__self__"):
        return None
    # A default is captured state the argument structure never sees.
    if getattr(fn, "__defaults__", None) or getattr(fn, "__kwdefaults__", None):
        return None
    qualname = getattr(fn, "__qualname__", None)
    if qualname is None:
        return None
    return f"{getattr(fn, '__module__', '?')}.{qualname}"


def as_subgraph(
    fn: Callable[_P, _R],
    *,
    name: str | None = None,
    prefix: str = "",
    key: str | _InferKey | None = _INFER_KEY,
) -> Callable[_P, _R]:
    """Returns ``fn`` lowered to one shared subgraph body per distinct stage.

    Usable as a decorator or at the call site.

    .. code-block:: python

        @compilation.as_subgraph
        def block(x: Tensor, w: Tensor) -> Tensor:
            return F.relu(x * w)

        y = compilation.as_subgraph(block, prefix="layers.0")(x, w)

    Args:
        fn: The callable to lower.
        name: The subgraph's name. Defaults to ``fn``'s own name.
        prefix: Prepended to the relative weight names the body declares, so
            each call site resolves its own weights from a shared body.
        key: What identifies this body beyond its arguments, completed here
            with the argument structure and operand types. Pass :obj:`None` to
            compare the staged IR instead. Omitted, a key is derived from
            ``fn`` where that is sound.

    Returns:
        A callable with ``fn``'s signature that emits a call to the shared body.

    Raises:
        TypeError: If called outside a capture. Call ``fn`` directly to run
            eagerly.
    """
    name = name or _sanitized_graph_name(fn)
    key = _inferred_key(fn) if isinstance(key, _InferKey) else key

    @functools.wraps(fn)
    def emit_subgraph_call(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        if not in_graph_context():
            raise TypeError(
                f"as_subgraph({name}) needs a capture (compile() / stage() / "
                "F.lazy()); call it directly to run eagerly"
            )
        ctx = subgraph_context()
        if ctx is None:
            return fn(*args, **kwargs)
        # Positional, like Signature.flatten: one operand per argument slot.
        operands, structure = tree.flatten(
            (args, kwargs), leaf=GRAPH_VALUE_TYPES
        )
        # Completed here, where the operands are flat and the prefix is known.
        types = [operand.type for operand in operands]
        full_key = (
            None
            if key is None
            else f"{key}|{name}|{structure}|{types}|{bool(prefix)}"
        )

        cache = ctx.subgraph_cache
        assert cache is not None, "subgraph_context() returns armed contexts"
        # A None key is never stored: share_subgraph hashes the IR instead.
        if (found := cache.get(full_key)) is None:
            with open_subgraph(ctx, name, types, prefix=prefix) as body:
                in_args, in_kwargs = tree.unflatten(
                    structure, body.inputs[: len(operands)]
                )
                # Also positional out: ``return x, x`` is two results.
                outputs, out_structure = tree.flatten(
                    fn(*in_args, **in_kwargs), leaf=GRAPH_VALUE_TYPES
                )
                body.output(*outputs)
            found = share_subgraph(ctx, body, out_structure, key=full_key)

        subgraph, out_structure = found
        results = ops.call(
            subgraph, *operands, *(ctx.signal_buffers or []), prefix=prefix
        )
        return tree.unflatten(out_structure, list(results))

    return emit_subgraph_call
