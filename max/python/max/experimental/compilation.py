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
"""Traces and compiles Python functions over tensors.

:func:`compile` takes a function that operates on
:class:`~max.experimental.tensor.Tensor` values and returns a
:class:`CompiledCallable` that runs it as a compiled graph. :func:`stage`
performs the tracing step only and returns a :class:`StagedGraph`, which is
useful for inspecting the generated MLIR.

Both take the function first and its input types second. An input type
describes one tensor argument: pass a :class:`~max.graph.TensorType` or a
:class:`~max.experimental.sharding.TensorLayout` for an argument the function
reads, or a :class:`~max.graph.BufferType` or
:class:`~max.experimental.sharding.BufferLayout` for one it also writes to.
Arguments that are not tensors, such as a Python ``float`` or ``bool``, are
static: their values are fixed into the graph during tracing, so pass the
same values when calling the compiled function.

The following example compiles a function with one tensor argument and one
static argument:

.. code-block:: python

    from max.driver import CPU
    from max.dtype import DType
    from max.experimental import compilation
    from max.experimental.sharding import TensorLayout
    from max.experimental.tensor import Tensor

    def step(x: Tensor, *, gain: float) -> Tensor:
        return x * gain

    x_type = TensorLayout(DType.float32, ["batch", 2], CPU())

    run = compilation.compile(step)(x_type, gain=3.0)
    out = run(Tensor.ones([4, 2], device=CPU()), gain=3.0)

.. invisible-code-block: python

    import numpy as np

    np.testing.assert_allclose(out.to_numpy(), np.full((4, 2), 3.0))
"""

from __future__ import annotations

import functools
import itertools
import re
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, ParamSpec, TypeVar

from max import _validation_hooks, tree
from max.driver import Accelerator, Buffer, Device, DLPackArray
from max.engine import CompiledModel, Model
from max.experimental.realization_context import (
    GraphRealizationContext,
    _cached_signal_buffers,
    _signal_buffer_types,
    in_graph_context,
    subgraph_context,
)
from max.experimental.sharding import (
    BufferLayout,
    DeviceMesh,
    PlacementMapping,
    TensorLayout,
    as_layout,
)
from max.experimental.sharding.per_shard_dim import make_per_shard_dim
from max.experimental.support import _session
from max.experimental.tensor import (
    Tensor,
    current_realization_context,
    realization_context,
)
from max.graph import (
    BufferType,
    BufferValue,
    Graph,
    Shape,
    StaticDim,
    TensorType,
    TensorValue,
    Value,
    ops,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# ``Tensor`` is included so that ``as_layout`` reports a clear error when a
# tensor is passed as an input type.
_LAYOUT_TYPES: tuple[type, ...] = (
    TensorLayout,
    Tensor,
    TensorType,
    BufferType,
)
# Leaf types when flattening a traced function's return value.
_VALUE_TYPES: tuple[type, ...] = (Tensor, BufferValue, TensorValue)


def _sanitized_graph_name(fn: Callable[..., object]) -> str:
    """Returns the name of ``fn`` with non-identifier characters replaced."""
    raw = getattr(fn, "__name__", None) or type(fn).__name__
    return re.sub(r"\W+", "_", raw).strip("_") or "fn"


def _declared_layouts(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[list[TensorLayout], tree.TreeDef]:
    """Extracts the layouts from the input types passed to ``stage``.

    Returns the layouts in graph order and the tree structure of the
    arguments, which records the position of each layout and the value of
    each static argument. Raises ``TypeError`` if a tensor is passed where an
    input type is expected.
    """
    input_types, treedef = tree.flatten(
        (args, dict(kwargs)), leaf=_LAYOUT_TYPES
    )
    return [as_layout(t) for t in input_types], treedef


def _align_signature(
    layouts: Sequence[TensorLayout],
    treedef: tree.TreeDef,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    label: str = "",
) -> list[tuple[str, TensorLayout, Any]]:
    """Matches the arguments of a call to the layouts they were staged with.

    Returns one ``(name, layout, argument)`` triple per tensor argument, in
    graph order. The name is the argument's position or keyword followed by
    its path inside any container, for example ``"0"``, ``"1.a"`` or
    ``"gain"``. Raises ``ValueError`` if the call does not have the staged
    structure, or a static argument has a different value.
    """
    try:
        given = treedef.flatten_up_to((args, dict(kwargs)))
    except ValueError as e:
        raise ValueError(f"{label}tree structure mismatch at {e}") from None
    return [
        (path.partition(".")[2], layout, arg)
        for path, layout, arg in zip(
            treedef.leaf_paths, layouts, given, strict=True
        )
    ]


def _input_buffers(
    layouts: Sequence[TensorLayout],
    treedef: tree.TreeDef,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> list[Buffer]:
    """Converts the tensor arguments of a call to buffers for the engine.

    Each tensor is checked against its staged layout and contributes one
    buffer per shard. A single-device argument may be passed as its
    :class:`~max.driver.Buffer`, which is how the pipelines pass their
    inputs, or as a lazy tensor, which is realized here. Raises
    ``TypeError`` if an argument is neither and ``ValueError`` if it does not
    match its layout.
    """
    buffers: list[Buffer] = []
    for name, layout, arg in _align_signature(layouts, treedef, args, kwargs):
        if isinstance(arg, Buffer):
            arg = Tensor(storage=arg)
        if not isinstance(arg, Tensor):
            raise TypeError(
                f"argument {name}: expected a Tensor or Buffer, got "
                f"{type(arg).__name__}"
            )
        # The engine consumes buffers, so the call is what forces a lazy
        # argument's value. A distributed one still has to arrive realized.
        if not arg.is_distributed:
            arg._sync_realize()
        shards = arg.local_shards
        devices = [shard.device for shard in shards]
        if isinstance(layout, BufferLayout):
            # Buffers are checked by dtype and shard count only. Their extent
            # is whatever was allocated.
            if len(shards) != layout.mesh.num_devices or (
                arg.dtype != layout.dtype
            ):
                raise ValueError(
                    f"argument {name}: expected {layout.dtype} across "
                    f"{layout.mesh.num_devices} device(s), got {arg.dtype} "
                    f"across {len(shards)}"
                )
        # Symbolic dims accept any size, so only static dims are compared.
        elif (
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


class StagedGraph(Generic[_P, _R]):
    """A traced graph that is ready to inspect or compile.

    :func:`stage` returns one. Print it to see the MLIR of the whole module,
    including any subgraphs. Pass it to :class:`CompiledCallable` to compile
    it.

    The following example stages a function and checks its MLIR:

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.sharding import TensorLayout
        from max.experimental.tensor import Tensor
        from max.driver import CPU

        def scale(x: Tensor) -> Tensor:
            return x * 2

        x_type = TensorLayout(DType.float32, [4], CPU())
        staged = compilation.stage(scale)(x_type)
        print(staged)

    .. invisible-code-block: python

        assert "mo.mul" in str(staged)

    Args:
        fn: The function to trace.
        graph: The graph to trace into. Its inputs must be one per shard of
            each layout, in order, followed by the signal buffers if any.
        layouts: The layout of each tensor argument, in graph order.
        treedef: The tree structure of the arguments, as returned by
            ``_declared_layouts``.
        signal_device_ids: The ids of the accelerators whose collectives need
            signal buffers.
        prefix: The name prefix that weights created inside ``fn`` are
            relative to.
        subgraph_cache: The table that :func:`as_subgraph` stores traced
            bodies in. Pass ``None`` to inline every subgraph call.
    """

    graph: Graph
    """The traced graph."""

    in_layouts: list[TensorLayout]
    """The layout of each tensor argument, in graph order."""

    in_tree: tree.TreeDef
    """The tree structure of the arguments, including static arguments."""

    out_layouts: list[TensorLayout]
    """The layout of each tensor result, in graph order."""

    out_tree: tree.TreeDef
    """The tree structure of the return value."""

    _signal_device_ids: tuple[int, ...]

    def __init__(
        self,
        fn: Callable[..., Any],
        graph: Graph,
        layouts: Sequence[TensorLayout],
        treedef: tree.TreeDef,
        signal_device_ids: tuple[int, ...] = (),
        *,
        prefix: str = "",
        subgraph_cache: dict[Any, Any] | None = None,
    ) -> None:
        arity = sum(len(layout.local_types) for layout in layouts)
        ctx = GraphRealizationContext(
            graph,
            signal_buffers=[i.buffer for i in graph.inputs[arity:]] or None,
            prefix=prefix,
        )
        ctx.subgraph_cache = subgraph_cache
        rest = iter(graph.inputs[:arity])
        with realization_context(ctx), ctx:
            in_args, in_kwargs = tree.unflatten(
                treedef, [_argument_tensor(rest, layout) for layout in layouts]
            )
            # Flatten the rebuilt arguments rather than reuse ``treedef``: a
            # container of input types may unflatten to a different class
            # than the one passed at call time.
            _, in_tree = tree.flatten((in_args, in_kwargs), leaf=Tensor)
            # ``shared`` is off, so ``return x, x`` produces two results.
            results, out_tree = tree.flatten(
                fn(*in_args, **in_kwargs), leaf=_VALUE_TYPES
            )
            if not all(isinstance(r, Tensor) for r in results):
                raise TypeError("a staged callable must return Tensors")
            graph.output(*(v for r in results for v in r.graph_values))
        self.graph = graph
        self.in_layouts, self.in_tree = list(layouts), in_tree
        self.out_layouts = [r.layout for r in results]
        self.out_tree = out_tree
        self._signal_device_ids = signal_device_ids

    def __str__(self) -> str:
        """Returns the MLIR of the whole module, including subgraphs."""
        op = self.graph._mlir_op
        # Print the module op rather than its Python wrapper, whose repr
        # wraps the MLIR in ``ModuleOp(..)``.
        return str(op.block.owner if getattr(op, "block", None) else op)

    # The default repr does not include the MLIR, so a test asserting that an
    # op is absent would pass trivially.
    __repr__ = __str__


class CompiledCallable(Generic[_P, _R]):
    """A compiled function that runs on tensors.

    :func:`compile` returns one. Call it with the same arguments as the
    original function, passing a :class:`~max.experimental.tensor.Tensor`
    for each input type. The weights and the signal buffers are bound when
    the object is created, so the first call has no extra setup cost.

    The following example compiles a function, exports it, and runs it:

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.sharding import TensorLayout
        from max.experimental.tensor import Tensor

        def scale(x: Tensor) -> Tensor:
            return x * 2

        x_type = TensorLayout(DType.float32, [3], CPU())

        run = compilation.compile(scale)(x_type)
        run.export_mef("scale.mef")
        out = run(Tensor.ones([3], device=CPU()))

    .. invisible-code-block: python

        import numpy as np

        np.testing.assert_allclose(out.to_numpy(), [2.0, 2.0, 2.0])

    Args:
        staged: The traced graph to compile.
        weights: The weight registry: a mapping from the name of each weight
            the graph declares to its data, with one entry per shard for a
            distributed weight.
    """

    #: See :attr:`StagedGraph.in_layouts`.
    in_layouts: list[TensorLayout]
    #: See :attr:`StagedGraph.in_tree`.
    in_tree: tree.TreeDef
    #: See :attr:`StagedGraph.out_layouts`.
    out_layouts: list[TensorLayout]
    #: See :attr:`StagedGraph.out_tree`.
    out_tree: tree.TreeDef

    _weights: dict[str, DLPackArray]
    _artifact: CompiledModel
    _signal_device_ids: tuple[int, ...]
    _engine_model: Model
    _signal_buffers: list[Buffer]

    def __init__(
        self,
        staged: StagedGraph[_P, _R],
        weights: Mapping[str, DLPackArray] | None = None,
    ) -> None:
        self.in_layouts, self.in_tree = staged.in_layouts, staged.in_tree
        self.out_layouts, self.out_tree = staged.out_layouts, staged.out_tree
        self._signal_device_ids = staged._signal_device_ids
        # Use the session's MEF cache. Plain ``compile`` bypasses it.
        self._artifact = _session().compile_reusing_mefs(staged.graph)
        self._weights = dict(weights or {})
        # Bind the weights and allocate the signal buffers now rather than on
        # the first call, so that the first call is not slower than the rest.
        self._engine_model = _session().init(
            self._artifact, weights_registry=dict(self._weights)
        )
        ids = self._signal_device_ids
        self._signal_buffers = _cached_signal_buffers(ids)[0] if ids else []

    @property
    def weights(self) -> Mapping[str, DLPackArray]:
        """Returns the weight registry this callable was compiled with.

        The mapping is read-only. To run the same graph with different
        weights, create a new :class:`CompiledCallable` from the same
        :class:`StagedGraph`.
        """
        return MappingProxyType(self._weights)

    @property
    def engine_model(self) -> Model:
        """Returns the underlying :class:`~max.engine.Model`.

        Use it to drive the model directly, for example to capture and
        replay it:

        .. code-block:: text

            run.engine_model.capture(key, *buffers, *run.signal_buffers)
            run.engine_model.replay(key, *buffers, *run.signal_buffers)
        """
        return self._engine_model

    @property
    def signal_buffers(self) -> list[Buffer]:
        """Returns the signal buffers for multi-device collectives.

        The list is empty when the graph runs on a single device. The buffers
        are allocated once per set of devices and shared by every compiled
        callable that runs on them.
        """
        return self._signal_buffers

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Runs the compiled function.

        Args:
            args: The positional arguments. Pass a
                :class:`~max.experimental.tensor.Tensor` for each input type.
            kwargs: The keyword arguments, likewise.

        Returns:
            The return value of the original function, with a
            :class:`~max.experimental.tensor.Tensor` in place of each result.

        Raises:
            TypeError: If an argument that was staged as a tensor is not a
                :class:`~max.experimental.tensor.Tensor`.
            ValueError: If an argument does not match the input types the
                function was compiled with.
        """
        rest = iter(
            self.execute_raw(
                *_input_buffers(self.in_layouts, self.in_tree, args, kwargs)
            )
        )
        return tree.unflatten(
            self.out_tree,
            [
                Tensor._from_shards(
                    tuple(itertools.islice(rest, layout.mesh.num_devices)),
                    layout.mesh,
                    layout.placements,
                )
                for layout in self.out_layouts
            ],
        )

    def execute_raw(self, *buffers: Buffer) -> list[Buffer]:
        """Runs the compiled graph on buffers, skipping the argument checks.

        Args:
            buffers: One buffer per graph input, in graph order. A distributed
                argument takes one buffer per shard. Do not pass the signal
                buffers; they are appended automatically.

        Returns:
            One buffer per graph output, in graph order.
        """
        results = list(self.engine_model(*buffers, *self._signal_buffers))
        _validation_hooks.compiled_call(self.engine_model, buffers, results)
        return results

    def export_mef(self, path: str | Path) -> None:
        """Writes the compiled graph to a MEF file.

        Load the file with :func:`max.engine.read` to skip compilation the
        next time.

        Args:
            path: The path of the file to write.
        """
        self._artifact.export_mef(path)


def _argument_tensor(
    values: Iterator[Value[Any]], layout: TensorLayout
) -> Tensor:
    """Creates the tensor for ``layout`` from the next values in ``values``.

    Consumes one value per shard of the layout.
    """
    shards = itertools.islice(values, layout.mesh.num_devices)
    graph_values: tuple[BufferValue | TensorValue, ...] = (
        tuple(value.buffer for value in shards)
        if isinstance(layout, BufferLayout)
        else tuple(value.tensor for value in shards)
    )
    return current_realization_context().create_unrealized(
        graph_values, mapping=layout.mapping
    )


def _own_devices(fn: object) -> Iterator[Device]:
    """Yields the devices of the distributed tensors held by ``fn``.

    A model holds its weights as attributes rather than taking them as
    arguments, so this is how :func:`stage` learns which devices a
    tensor-parallel model needs signal buffers for.
    """
    # ``shared=True`` because a model may contain reference cycles.
    for value in tree.leaves(fn, leaf=Tensor, shared=True):
        if isinstance(value, Tensor) and value.is_distributed:
            yield from value.mesh.devices


def stage(
    fn: Callable[_P, _R],
    *,
    name: str | None = None,
    custom_extensions: Iterable[Path] = (),
    allow_subgraphs: bool = True,
    signal_devices: Iterable[Device] = (),
    is_device_graph: bool = False,
) -> Callable[..., StagedGraph[_P, _R]]:
    """Traces a function into a graph without compiling it.

    Returns a function that takes the input types of ``fn`` and returns a
    :class:`StagedGraph`. Use it to inspect the MLIR that :func:`compile`
    would compile.

    The following example stages a function whose tensor arguments are
    passed in a dictionary:

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.sharding import TensorLayout
        from max.experimental.tensor import Tensor
        from max.driver import CPU

        def combine(kv: dict[str, Tensor], alpha: float) -> Tensor:
            return (kv["a"] + kv["b"]) * alpha

        x_type = TensorLayout(DType.float32, [2], CPU())

        staged = compilation.stage(combine)({"a": x_type, "b": x_type}, 2.0)
        print(staged)

    .. invisible-code-block: python

        assert len(staged.graph.inputs) == 2

    Args:
        fn: The function to trace. It takes and returns
            :class:`~max.experimental.tensor.Tensor` values, possibly inside
            lists, tuples or dictionaries.
        name: The name of the graph. Defaults to the name of ``fn``.
        custom_extensions: Paths to custom Mojo kernel libraries.
        allow_subgraphs: Whether calls made through :func:`as_subgraph` are
            traced as subgraphs. If ``False``, they are inlined.
        signal_devices: Additional accelerators to allocate signal buffers
            for, beyond the devices of the input types and of the tensors
            held by ``fn``.
        is_device_graph: Whether to build a device graph.

    Returns:
        A function that takes one input type per tensor argument of ``fn``
        and returns the :class:`StagedGraph`.
    """

    def record(*args: Any, **kwargs: Any) -> StagedGraph[_P, _R]:
        layouts, treedef = _declared_layouts(args, kwargs)
        # Signal buffers are only needed when at least two accelerators take
        # part in the graph.
        ids = tuple(
            dict.fromkeys(
                device.id
                for device in itertools.chain(
                    (
                        device
                        for layout in layouts
                        if layout.mesh.num_devices > 1
                        for device in layout.mesh.devices
                    ),
                    _own_devices(fn),
                    signal_devices,
                )
                if isinstance(device, Accelerator)
            )
        )
        ids = ids if len(ids) > 1 else ()
        graph = Graph(
            name or _sanitized_graph_name(fn),
            input_types=[
                *(t for layout in layouts for t in layout.local_types),
                *_signal_buffer_types(ids),
            ],
            custom_extensions=custom_extensions,
            is_device_graph=is_device_graph,
        )
        return StagedGraph(
            fn,
            graph,
            layouts,
            treedef,
            ids,
            subgraph_cache={} if allow_subgraphs else None,
        )

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
    """Traces and compiles a function.

    Returns a function that takes the input types of ``fn`` and returns a
    :class:`CompiledCallable`. Call that on tensors to run ``fn``.

    A :class:`~max.experimental.tensor.Tensor` is not accepted as an input
    type, because its shape would fix every dimension of the graph to the
    tensor's current size. Pass ``tensor.layout`` to do this deliberately.

    The following example compiles a function that reads a weight, and
    supplies the weight's value through ``weights``:

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental import functional as F
        from max.experimental.sharding import TensorLayout
        from max.experimental.tensor import Tensor

        w_type = TensorLayout(DType.float32, [2], CPU())

        def layer(x: Tensor) -> Tensor:
            return x * F.constant_external("w", w_type)

        x_type = TensorLayout(DType.float32, ["batch", 2], CPU())
        w = Tensor.ones([2], device=CPU()) * 3

        run = compilation.compile(layer, weights={"w": w})(x_type)
        out = run(Tensor.ones([4, 2], device=CPU()))

    .. invisible-code-block: python

        import numpy as np

        np.testing.assert_allclose(out.to_numpy(), np.full((4, 2), 3.0))

    Args:
        fn: The function to compile. It takes and returns
            :class:`~max.experimental.tensor.Tensor` values, possibly inside
            lists, tuples or dictionaries.
        weights: The weight registry: a mapping from the name of each weight
            the graph declares to its data, with one entry per shard for a
            distributed weight.
        name: The name of the graph. Defaults to the name of ``fn``.
        custom_extensions: Paths to custom Mojo kernel libraries.
        allow_subgraphs: Whether calls made through :func:`as_subgraph` are
            traced as subgraphs. If ``False``, they are inlined.
        signal_devices: Additional accelerators to allocate signal buffers
            for, beyond the devices of the input types and of the tensors
            held by ``fn``.
        is_device_graph: Whether to build a device graph.

    Returns:
        A function that takes one input type per tensor argument of ``fn``
        and returns the :class:`CompiledCallable`.
    """

    def stage_and_compile(
        *args: Any, **kwargs: Any
    ) -> CompiledCallable[_P, _R]:
        staged = stage(
            fn,
            name=name,
            custom_extensions=custom_extensions,
            allow_subgraphs=allow_subgraphs,
            signal_devices=signal_devices,
            is_device_graph=is_device_graph,
        )(*args, **kwargs)
        return CompiledCallable(staged, weights)

    return stage_and_compile


def _boundary_layout(tensor: Tensor) -> TensorLayout:
    """Returns the layout of a tensor passed to a subgraph.

    The layout is built from the tensor's graph values so that the subgraph's
    input types match the call exactly: the devices are those of the values,
    the dims are kept per shard, and a tensor backed by buffers gets a
    ``BufferLayout`` so the subgraph can write to it.
    """
    values = tensor.graph_values
    if not values:
        return tensor.layout
    mapping = tensor._mapping
    devices = tuple(value.type.device.to_device() for value in values)
    if tuple(mapping.mesh.devices) != devices:
        # A tensor that was moved to the host keeps its original mesh. The
        # graph values are on the devices it is actually on.
        mapping = PlacementMapping(
            DeviceMesh(
                devices, mapping.mesh.mesh_shape, mapping.mesh.axis_names
            ),
            mapping.to_placements(),
        )
    shape = Shape(
        make_per_shard_dim(cells, force_wrap=True)
        for cells in zip(
            *(tuple(value.type.shape) for value in values), strict=True
        )
    )
    if all(isinstance(value, BufferValue) for value in values):
        return BufferLayout(tensor.dtype, shape, mapping)
    return TensorLayout(tensor.dtype, shape, mapping)


def _stage_body(
    ctx: GraphRealizationContext,
    fn: Callable[..., Any],
    symbol: str,
    layouts: Sequence[TensorLayout],
    treedef: tree.TreeDef,
    prefix: str,
) -> StagedGraph[..., Any]:
    """Traces ``fn`` into a new subgraph of the graph ``ctx`` is building.

    The subgraph is named ``symbol``, with a numeric suffix if that name is
    already taken. The parent graph's own name counts as taken, because
    ``mo.call`` resolves names against the parent.
    """
    taken = {*ctx.graph._subgraphs, ctx.graph.name}
    sym, collisions = symbol, 0
    while sym in taken:
        collisions += 1
        sym = f"{symbol}_{collisions}"
    signals = ctx.signal_buffers or []
    return StagedGraph(
        fn,
        ctx.graph.add_subgraph(
            sym,
            input_types=[
                *(t for layout in layouts for t in layout.local_types),
                *(buffer.type for buffer in signals),
            ],
            custom_extensions=ctx.graph.kernel_libraries_paths,
            devices=list(ctx.graph.device_chains),
        ),
        layouts,
        treedef,
        prefix=prefix,
    )


def _position_of(fn: object) -> str:
    """Returns the weight name prefix of ``fn`` within its model.

    Weights are named by their path from the root of the model, so a weight
    named ``layers.3.mlp.w`` that ``fn`` holds at ``mlp.w`` puts ``fn`` at
    ``layers.3.``. Returns ``""`` if ``fn`` holds no named weight. Raises
    ``TypeError`` if a weight's name does not end with its path in ``fn``.
    """
    for path, leaf in tree.paths(fn, leaf=Tensor, shared=True).items():
        if not isinstance(leaf, Tensor) or (name := leaf.external_name) is None:
            continue
        if not name.endswith(path):
            raise TypeError(
                f"weight {name!r} is stored at {path!r} in this callable, "
                "but its name does not end with that path."
            )
        return name[: len(name) - len(path)]
    return ""


def as_subgraph(
    fn: Callable[_P, _R], *, name: str | None = None, prefix: str | None = None
) -> Callable[_P, _R]:
    """Traces a function into a subgraph once and calls it from every call site.

    Use this for a block of computation that repeats, such as a transformer
    layer, so that the compiler processes the definition once instead of
    once per repetition. The subgraph's input types are taken from the
    arguments of the first call, so there is nothing to declare besides
    ``fn``. Outside a trace, or when the graph was staged with
    ``allow_subgraphs=False``, the returned function simply calls ``fn``.

    Usable as a decorator or at the call site. The following example shares
    one subgraph between three calls:

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.sharding import TensorLayout
        from max.experimental.tensor import Tensor

        @compilation.as_subgraph
        def block(x: Tensor) -> Tensor:
            return x * 2

        x_type = TensorLayout(DType.float32, [4], CPU())
        staged = compilation.stage(lambda x: block(block(block(x))))(x_type)

    .. invisible-code-block: python

        assert str(staged).count("mo.graph @block") == 1
        assert str(staged).count("mo.call @block") == 3

    Each distinct callable gets its own subgraph, so two closures over
    different values are traced separately. The layers of a model are
    distinct objects that differ only in their weights. To share one
    subgraph between them, give every call site the same ``name``; each call
    then resolves its own weights under its ``prefix``, which defaults to the
    layer's position in the model:

    .. skip: next

    .. code-block:: python

        # Illustrative: ``layers`` holds modules whose weights are named
        # ``layers.<i>.<...>``, as ``Module.compile`` names them.
        def transformer(x: Tensor, layers: list[Module]) -> Tensor:
            for layer in layers:
                x = compilation.as_subgraph(layer, name="layer")(x)
            return x

    Args:
        fn: The function to trace.
        name: The name of the subgraph. Defaults to the name of ``fn``.
        prefix: The name prefix that weights used inside ``fn`` are relative
            to. Defaults to the position of ``fn`` in its model, read from
            the names of the weights it holds, or ``""`` if it holds none.

    Returns:
        A function with the signature of ``fn`` that emits one call to the
        subgraph each time it is called.
    """
    symbol = name or _sanitized_graph_name(fn)
    # Read once. The weights are already named when ``fn`` is wrapped.
    at = _position_of(fn) if prefix is None else prefix

    @functools.wraps(fn)
    def emit_subgraph_call(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        # Outside a trace, or with subgraphs disabled, call ``fn`` directly.
        ctx = subgraph_context() if in_graph_context() else None
        if ctx is None:
            return fn(*args, **kwargs)
        # A lazy context has no graph yet to add a subgraph to.
        assert isinstance(ctx, GraphRealizationContext)

        values, treedef = tree.flatten((args, dict(kwargs)), leaf=Tensor)
        layouts = [_boundary_layout(value) for value in values]
        # The cache key identifies the body by the callable, so two closures
        # over different values get two subgraphs. A named body with a
        # prefix is identified by its name instead, so that call sites that
        # differ only in their weights share one subgraph. Layouts contain
        # unhashable shapes, so the key holds their graph types.
        types = tuple(t for layout in layouts for t in layout.local_types)
        identity = symbol if (name is not None and at) else id(fn)
        key = (identity, symbol, str(treedef), types, bool(at))

        cache = ctx.subgraph_cache
        assert cache is not None, (
            "subgraph_context() only returns contexts with a cache"
        )
        if (entry := cache.get(key)) is None:
            # Keep ``fn`` alive so that its id is not reused while the key
            # depends on it.
            cache[key] = entry = (
                fn,
                _stage_body(ctx, fn, symbol, layouts, treedef, at),
            )
        body = entry[1]

        operands = [
            value
            for _, _, arg in _align_signature(
                body.in_layouts, body.in_tree, args, kwargs, f"{symbol} "
            )
            for value in (
                arg.graph_values if isinstance(arg, Tensor) else (arg,)
            )
        ]
        results = ops.call(
            body.graph, *operands, *(ctx.signal_buffers or []), prefix=at
        )
        rest = iter(results)
        return tree.unflatten(
            body.out_tree,
            [_argument_tensor(rest, layout) for layout in body.out_layouts],
        )

    return emit_subgraph_call
