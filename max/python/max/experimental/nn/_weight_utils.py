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
"""Declares, names and loads the weights of a module.

A weight is a parameter as the graph sees it: an ``_ExternalWeight`` with a
dtype, shape and device placement but no data, named by its path in the
module. When a graph uses a weight, the weight adds an external constant of
that name to the graph, one per shard. The weight registry maps those names
to data when the graph is compiled. Using a weight outside a graph raises.

The constant is added to whichever graph is being traced when the weight is
first used, under that graph's name prefix. This is what allows a subgraph
shared between several layers to resolve a different set of weights at each
call site.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Mapping
from typing import Any, NoReturn

import numpy.typing as npt
from max import tree
from max.driver import CPU, Device, DLPackArray
from max.dtype import DType
from max.experimental.realization_context import (
    EagerRealizationContext,
    GraphRealizationContext,
    LazyRealizationContext,
)
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    as_device_mapping,
)
from max.experimental.tensor import (
    GraphValue,
    Tensor,
    current_realization_context,
    defaults,
    external_shard_names,
)
from max.graph import Graph, Shape, ShapeLike

_logger = logging.getLogger(__name__)

#: Error message for a weight used before ``name_weights`` named it.
_UNNAMED = (
    "{!r} has no name. Weights are named by ``name_weights`` on the model that "
    "holds them."
)

#: Descriptions of the contexts that cannot use a weight, for error messages.
_NOT_A_TRACE: dict[type, str] = {
    LazyRealizationContext: "a lazy context",
    EagerRealizationContext: "an eager context",
}


#: Dtypes that ``auto_cast`` may convert between. Intentionally narrow;
#: widen as new pairs are validated.
_SAFE_CAST_DTYPES: frozenset[DType] = frozenset({DType.float32, DType.bfloat16})


#: The external constants added to each graph, keyed by weight name. Weakly
#: keyed so that a graph's entries are released with the graph.
_RECORDED: weakref.WeakKeyDictionary[Graph, dict[str, tuple[GraphValue, ...]]]
_RECORDED = weakref.WeakKeyDictionary()


class _ExternalWeight(Tensor):
    """A parameter whose value is loaded from the weight registry.

    Holds the dtype, shape and device placement of the parameter and no
    data. Using it in a graph adds an external constant with its name.

    Args:
        shape: The global shape of the weight.
        dtype: The data type of the weight. Defaults to the default dtype.
        device: The device or device mapping the weight is placed on.
            Defaults to the default device.
        name: The name of the weight, or ``None`` until ``name_weights``
            assigns one.
        source: The layout the weight has in the checkpoint, if it differs
            from ``device``.
    """

    name: str | None

    #: The layout the weight has in the checkpoint, set by ``shard_checkpoint``.
    #: The external constant is declared with this layout and then transferred
    #: to ``_mapping`` inside the graph, so no resharding happens on the host.
    source: _ExternalWeight | None

    _dtype: DType
    _shape: Shape

    def __new__(cls, *args: Any, **kwargs: Any) -> _ExternalWeight:
        # Skip ``Tensor.__new__``, which creates a constant from data.
        return object.__new__(cls)

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DType | None = None,
        device: Device | DeviceMapping | DeviceMesh | None = None,
        *,
        name: str | None = None,
        source: _ExternalWeight | None = None,
    ) -> None:
        self.name = name
        self.source = source
        self._dtype = dtype if dtype is not None else defaults(None, None)[0]
        self._shape = Shape(shape)
        self._mapping = as_device_mapping(
            device if device is not None else defaults(None, None)[1]
        )
        self._storages = None
        # A weight is neither realized nor unrealized. It has no value.
        self._state = None

    @classmethod
    def like(
        cls, tensor: Tensor, *, name: str | None = None
    ) -> _ExternalWeight:
        """Creates a weight with the shape, dtype and placement of ``tensor``.

        Returns ``tensor`` itself if it is already a weight.
        """
        if isinstance(tensor, _ExternalWeight):
            return tensor
        return cls(tensor.shape, tensor.dtype, tensor._mapping, name=name)

    @property
    def real(self) -> bool:
        """Returns ``False``. A weight holds no data."""
        return False

    @property
    def external_name(self) -> str | None:
        """Returns the name of the weight, which is its key in the registry."""
        return self.name

    @property
    def dtype(self) -> DType:
        """Returns the data type of the weight."""
        return self._dtype

    @property
    def shape(self) -> Shape:
        """Returns the global shape of the weight."""
        return self._shape

    @property
    def num_shards(self) -> int:
        """Returns the number of devices the weight is placed on."""
        return self._mapping.mesh.num_devices

    @property
    def local_shards(self) -> tuple[Tensor, ...]:
        """Returns one tensor per device, backed by the graph values."""
        if not self.is_distributed:
            return (self,)
        values = self.graph_values
        ctx = current_realization_context(None)
        assert isinstance(ctx, GraphRealizationContext)
        return tuple(ctx.create_unrealized((value,)) for value in values)

    @property
    def graph_values(self) -> tuple[GraphValue, ...]:
        """Returns the graph values of the weight, one per shard.

        The external constant is added to the current graph the first time
        the weight is used in it and reused afterwards. Raises ``TypeError``
        if no graph is being traced or the weight has no name.
        """
        # Check the context here rather than in ``Graph.current`` to give a
        # clear error message.
        ctx = current_realization_context(None)
        if not isinstance(ctx, GraphRealizationContext):
            ctx_kind = _NOT_A_TRACE.get(type(ctx), "no realization context")
            raise TypeError(
                f"{self.name!r} holds no data, so it can only be used while a "
                f"graph is being traced, not in {ctx_kind}. Trace the model "
                "that holds it, or supply its data instead."
            )
        if self.name is None:
            raise TypeError(_UNNAMED.format(self))
        recorded = _RECORDED.setdefault(ctx.graph, {})
        if (values := recorded.get(self.name)) is None:
            # Strip the graph's prefix so that a shared subgraph resolves a
            # different weight at each call site.
            declared = self.source if self.source is not None else self
            recorded[self.name] = values = (
                Tensor._as_constant_external(
                    declared,
                    self.name.removeprefix(ctx.prefix),
                    align=1,
                    is_placeholder=bool(ctx.prefix),
                )
                .to(self._mapping)
                .graph_values
            )
        return values

    @property
    def _graph_value(self) -> GraphValue:
        self._check_not_distributed("_graph_value")
        (value,) = self.graph_values
        return value

    def __tensorvalue__(self) -> Any:
        """Returns the graph value of the weight."""
        return self._graph_value

    def _no_data(self, verb: str = "read") -> NoReturn:
        """Raises ``TypeError``. A weight has no data to ``verb``."""
        raise TypeError(
            f"weight {self.name!r} holds no data to {verb}. Supply its value "
            "through the weight registry when compiling."
        )

    # Every operation that needs the data goes through one of these.
    def __buffervalue__(self) -> Any:
        self._no_data("store into")

    def __await__(self) -> Any:
        self._no_data("await")

    def _sync_realize(self) -> None:
        self._no_data()

    def to_numpy(self) -> npt.NDArray[Any]:
        self._no_data()

    def __repr__(self) -> str:
        named = f", name={self.name!r}" if self.name is not None else ""
        return (
            f"_ExternalWeight({self._dtype}, {list(self._shape)}, "
            f"{self._mapping}{named})"
        )


def name_weights(root: Any) -> None:
    """Names each unnamed weight in ``root`` after its path from ``root``."""
    for path, parameter in tree.paths(root, leaf=Tensor).items():
        if isinstance(parameter, _ExternalWeight) and parameter.name is None:
            parameter.name = path


CastRecord = tuple[DType, DType]


def _entry(arrays: Mapping[str, Any], path: str) -> Tensor:
    """Returns the entry ``path`` of ``arrays`` as a tensor.

    Raises ``KeyError`` if there is no such entry.
    """
    try:
        array = arrays[path]
    except KeyError:
        raise KeyError(
            f"Weight {path!r} is missing from the provided weights mapping."
        ) from None
    return array if isinstance(array, Tensor) else Tensor.from_dlpack(array)


def check_fits(
    path: str, array: Tensor, parameter: Tensor, auto_cast: bool
) -> CastRecord | None:
    """Checks that ``array`` can be loaded into ``parameter``.

    Returns the ``(from, to)`` dtype pair if the load needs a cast that
    ``auto_cast`` permits, and ``None`` if no cast is needed. Raises
    ``ValueError`` if the shapes differ or the dtypes differ and the cast is
    not permitted.
    """
    dtype = array.dtype
    # The engine reads the registry data with the declared dtype and shape,
    # so a mismatch with the same byte count would be reinterpreted silently.
    shapes_agree = tuple(int(d) for d in array.shape) == tuple(
        int(d) for d in parameter.shape
    )
    if (
        auto_cast
        and shapes_agree
        and dtype != parameter.dtype
        and {dtype, parameter.dtype} <= _SAFE_CAST_DTYPES
    ):
        return (dtype, parameter.dtype)
    if not shapes_agree or dtype != parameter.dtype:
        raise ValueError(
            f"{path!r}: Loaded tensor (shape={list(array.shape)}, "
            f"dtype={dtype}) not assignable to parameter "
            f"(shape={[int(d) for d in parameter.shape]}, "
            f"dtype={parameter.dtype})."
        )
    return None


def log_casts(casts: Mapping[CastRecord, int]) -> None:
    """Logs one warning that summarizes all the casts."""
    if not casts:
        return
    parts = []
    for (loaded, declared), count in casts.items():
        narrowing = (
            " (precision loss)"
            if declared.size_in_bytes < loaded.size_in_bytes
            else ""
        )
        parts.append(
            f"{count} parameter(s) from {loaded} to {declared}{narrowing}"
        )
    _logger.warning("load_state_dict auto-cast: %s.", "; ".join(parts))


def load_checkpoint(
    root: Any,
    state: Mapping[str, Any],
    *,
    strict: bool = True,
    auto_cast: bool = False,
) -> None:
    """Sets the values of the parameters of ``root`` from ``state``.

    See :meth:`~max.experimental.nn.Module.load_state_dict` for the
    arguments.
    """
    parameters = tree.paths(root, leaf=Tensor)
    if strict and (unused := state.keys() - parameters.keys()):
        raise ValueError(f"load_state_dict did not use some weights: {unused}")
    casts: dict[CastRecord, int] = {}
    loaded: dict[str, Tensor] = {}
    for path, parameter in parameters.items():
        weight = _entry(state, path)
        if cast := check_fits(path, weight, parameter, auto_cast):
            casts[cast] = casts.get(cast, 0) + 1
        if weight.dtype != parameter.dtype:
            weight = weight.cast(parameter.dtype)
        if weight._mapping != parameter._mapping:
            weight = weight.to(parameter._mapping)
        loaded[path] = weight
    tree.update(root, loaded, leaf=Tensor)
    log_casts(casts)


def shard_checkpoint(
    root: Any,
    arrays: Mapping[str, DLPackArray | Tensor],
    *,
    auto_cast: bool = False,
) -> dict[str, Tensor]:
    """Builds the weight registry for a model.

    Each entry of ``arrays`` is checked against the weight it loads into,
    cast if ``auto_cast`` permits, moved to the host, and split into one
    registry entry per shard. The weight records the layout of the entry as
    its ``source``, so that the graph declares the external constant in that
    layout and transfers it to the weight's placement. No resharding happens
    on the host.

    Args:
        root: The model, with its parameters replaced by weights as
            :meth:`~max.experimental.nn.Module.trace` does.
        arrays: The values of the parameters, keyed by the names that
            :attr:`~max.experimental.nn.Module.parameters` yields.
        auto_cast: Whether to permit a cast between ``float32`` and
            ``bfloat16``.

    Returns:
        A mapping from the name of each external constant to its data.

    Raises:
        KeyError: If a parameter has no entry in ``arrays``.
        ValueError: If an entry does not match the shape or dtype of its
            parameter.
    """
    casts: dict[CastRecord, int] = {}
    cpu = CPU()
    registry: dict[str, Tensor] = {}
    for path, declaration in tree.paths(root, leaf=Tensor).items():
        array = _entry(arrays, path)
        if cast := check_fits(path, array, declaration, auto_cast):
            casts[cast] = casts.get(cast, 0) + 1
            array = array.cast(declaration.dtype)
        shards = array.local_shards
        names = external_shard_names(path, len(shards))
        for name, shard in zip(names, shards, strict=True):
            if shard.real and shard.device != cpu:
                shard = shard.to(cpu)
            registry[name] = shard
        if isinstance(declaration, _ExternalWeight):
            declaration.source = _ExternalWeight(
                array.shape, declaration.dtype, array._mapping, name=path
            )
    log_casts(casts)
    return registry
