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
"""User-facing definitions of custom ops for eager and graph use.

:class:`CustomOp` stages a ``mo.custom`` op from a declared
``inputs``/``outputs`` signature; works both eagerly (``Tensor``) and inside
a :class:`~max.graph.Graph` build (``TensorValue``).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

from max.dtype import DType
from max.experimental.functional import _load_custom_extensions
from max.experimental.realization_context import ensure_context
from max.experimental.tensor import Tensor
from max.graph import (
    DeviceRef,
    Dim,
    DimLike,
    Graph,
    StaticDim,
    SymbolicDim,
    TensorType,
    TensorValue,
    Type,
    ops,
)
from mojo.paths import _build_mojo_source_package, is_mojo_source_package_path

__all__ = [
    "CustomOp",
    "DTypeVar",
    "Symbol",
    "Symbols",
    "TemplateType",
    "declare",
]

_ResultT = TypeVar("_ResultT")

#: What a call returns: eager in, eager out; graph in, graph out. Also the
#: return annotation `inspect.signature` and autodoc report for an op.
_Result = Tensor | list[Tensor] | TensorValue | list[TensorValue]


def _single_or_list(results: list[_ResultT]) -> _ResultT | list[_ResultT]:
    """Unwraps a single-output result; keeps a list for multi-output ops."""
    return results[0] if len(results) == 1 else results


@dataclass(frozen=True, init=False)
class TemplateType:
    """One entry in a :class:`CustomOp` signature, input or output.

    ``dtype`` is a concrete :class:`~max.dtype.DType` or a :class:`DTypeVar`;
    entries sharing a variable must agree on the actual dtype at call time.
    ``shape`` holds :class:`Symbol`s, statics, or expressions over them; its
    length is the entry's rank. There is no device: the op takes its device
    from its operands at each call.

    Instances are immutable: :class:`CustomOp` validates a signature once
    at declaration, so an entry must not change underneath it afterwards.
    """

    dtype: DType | DTypeVar
    shape: tuple[Dim, ...]

    def __init__(
        self, dtype: DType | DTypeVar, shape: Sequence[DimLike]
    ) -> None:
        if not isinstance(dtype, DType | DTypeVar):
            raise TypeError(
                f"TemplateType dtype must be a DType or DTypeVar, got "
                f"{type(dtype).__name__}"
            )
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", tuple(Dim(dim) for dim in shape))


#: Namespace a signature's own dims live under, keeping them disjoint from
#: the dim names a caller writes. A legal `SymbolicDim` name that no caller
#: picks.
_SYMBOL_PREFIX = "__co_"


class Symbol(SymbolicDim):
    """A dim belonging to one `CustomOp` signature.

    Subclassing :class:`~max.graph.SymbolicDim` lets a signature dim take
    part in ordinary ``Dim`` arithmetic (``k // 2``). The namespaced name is
    what survives that arithmetic: dim expressions fold through MLIR and
    reconstruct operands as plain symbols, so the type alone cannot carry
    the marker.
    """

    def __init__(self, name: str | Symbol) -> None:
        # `Dim(x)` hands back the same instance and Python re-runs this on
        # it, so prefixing has to be idempotent.
        if isinstance(name, Symbol):
            name = name.symbol_name
        super().__init__(f"{_SYMBOL_PREFIX}{name}")

    @property
    def symbol_name(self) -> str:
        """The declared name, without the namespace prefix."""
        return self.name.removeprefix(_SYMBOL_PREFIX)


def Symbols(*names: str) -> tuple[Symbol, ...]:
    """Returns one :class:`Symbol` per name, for tuple unpacking."""
    return tuple(Symbol(name) for name in names)


@dataclass(frozen=True)
class DTypeVar:
    """A dtype variable: inputs sharing one must agree at call time."""

    name: str


def _resolve_extension(path: Path) -> Path:
    """Compiles a Mojo source package once, at declaration.

    A call then only registers the resulting binary. Resolving per call
    would rerun ``mojo precompile`` every time the op is invoked.
    """
    if is_mojo_source_package_path(path):
        return _build_mojo_source_package(path)
    return path


@dataclass(frozen=True)
class CustomOp:
    """A declared custom op; call it eagerly or in a graph.

    Build one with :func:`declare`, which validates the signature and
    compiles the kernel packages once. Instances are immutable and compare
    by declaration.

    The same op accepts eager :class:`~max.experimental.tensor.Tensor`s,
    running the kernel immediately, or :class:`~max.graph.TensorValue`s
    inside a :class:`~max.graph.Graph` build, staging a ``mo.custom`` op.
    Output types come from the declared signature, so nothing is computed
    per call beyond binding the signature's symbols to the operands' dims.

    .. Skipped: the example needs a Mojo package registering a `downsample`
       kernel, which this library does not ship.
    .. skip: next

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import Tensor, custom
        from max.graph import DeviceRef, Graph, TensorType

        n, w, c = custom.Symbols("n", "w", "c")
        downsample = custom.declare(
            "downsample",
            inputs={"x": custom.TemplateType(DType.float32, [n, w, c])},
            outputs=[custom.TemplateType(DType.float32, [n, w // 2, c])],
            custom_extensions=["path/to/kernels"],
        )

        # Eager: runs now, `y.shape == (2, 4, 3)`.
        y = downsample(Tensor.ones((2, 8, 3)))

        # Graph: staged, output type `[n, w // 2, c]` -> `["n", 4, "c"]`.
        x_type = TensorType(DType.float32, ["n", 8, "c"], DeviceRef.CPU())
        with Graph("g", input_types=[x_type]) as graph:
            graph.output(downsample(graph.inputs[0]))
    """

    #: The registered kernel symbol.
    name: str
    #: One entry per operand, keyed by operand name, in operand order. A
    #: dict is not hashable, so the hash covers the other fields; equality
    #: still covers all of them.
    inputs: Mapping[str, TemplateType] = field(hash=False)
    #: One entry per result.
    outputs: tuple[TemplateType, ...]
    #: Kernel packages, already compiled to binaries.
    extensions: tuple[Path, ...]

    @cached_property
    def __signature__(self) -> inspect.Signature:
        """The declared operand names as the call signature.

        :func:`inspect.signature` reads this (PEP 362) instead of
        :meth:`__call__`'s bare ``*args``, so the declared ``inputs`` keys are
        what a keyword call binds through and what tooling displays.
        """
        return inspect.Signature(
            [
                inspect.Parameter(
                    key,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Tensor | TensorValue,
                )
                for key in self.inputs
            ],
            return_annotation=_Result,
        )

    def _check_signature(self) -> None:
        """Rejects any dim a signature is not allowed to name.

        Every dim must reduce to statics or :class:`Symbol`s. Because dim
        arithmetic erases the Python subclass, the test is on the namespaced
        name, which is exactly what a :class:`Symbol` contributes and nothing
        else can.

        An input dim must additionally be a bare :class:`Symbol` or a static,
        never an expression over one: ``Symbol("m") // 2`` is unresolvable,
        since :meth:`_unify` cannot solve it for ``m``. An output dim keeps
        the looser rule, since it only ever gets substituted into, never
        solved.
        """
        entries = [
            (f"input {key!r}", spec) for key, spec in self.inputs.items()
        ] + [(f"output {i}", spec) for i, spec in enumerate(self.outputs)]
        for where, spec in entries:
            for j, dim in enumerate(spec.shape):
                for symbol in dim.parameters:
                    if symbol.name.startswith(_SYMBOL_PREFIX):
                        continue
                    raise TypeError(
                        f"custom op {self.name!r}: {where} dim {j} uses "
                        f"{symbol.name!r}, which is not a signature dim; use "
                        "custom.Symbol or a static"
                    )
        for key, spec in self.inputs.items():
            for j, dim in enumerate(spec.shape):
                if isinstance(dim, Symbol | StaticDim):
                    continue
                raise TypeError(
                    f"custom op {self.name!r}: input {key!r} dim {j} "
                    f"({dim}) is not directly bindable; an input dim "
                    "must be a plain custom.Symbol or a static, not an "
                    "algebraic expression"
                )

    def _check_reserved_dim_names(self, in_types: Sequence[Type[Any]]) -> None:
        """Rejects an incoming dim inside the signature's own namespace.

        A caller-supplied dim named like a signature symbol would be
        indistinguishable from one once bound, silently aliasing the
        caller's dim to a signature symbol it never declared. Refusing it
        at the boundary is simpler than trying to resolve the ambiguity.
        """
        for key, actual in zip(self.inputs, in_types, strict=False):
            assert isinstance(actual, TensorType)
            for j, dim in enumerate(actual.shape):
                for symbol in dim.parameters:
                    if symbol.name.startswith(_SYMBOL_PREFIX):
                        raise TypeError(
                            f"custom op {self.name!r}: input {key!r} dim "
                            f"{j} is named {symbol.name!r}, which is "
                            "reserved for custom op signature dims; rename "
                            "the dim"
                        )

    def _unify(self, in_types: Sequence[Type[Any]]) -> dict[str, Dim]:
        """Binds signature symbols to the actual dims of *in_types*.

        A symbol binds on first occurrence and must agree afterwards, which is
        how a repeated symbol declares cross-input equality. Dtype variables
        bind the same way. Returns the bindings keyed by namespaced name,
        ready for :meth:`Dim.substitute`.
        """
        if len(in_types) != len(self.inputs):
            raise TypeError(
                f"custom op {self.name!r} expects {len(self.inputs)} inputs, "
                f"got {len(in_types)}"
            )
        self._check_reserved_dim_names(in_types)
        dims: dict[str, Dim] = {}
        dtypes: dict[str, DType] = {}
        origin: dict[str, str] = {}
        for (key, spec), actual in zip(
            self.inputs.items(), in_types, strict=False
        ):
            assert isinstance(actual, TensorType)
            if len(spec.shape) != len(actual.shape):
                raise ValueError(
                    f"custom op {self.name!r}: input {key!r} rank mismatch: "
                    f"signature declares {len(spec.shape)}, got "
                    f"{len(actual.shape)}"
                )
            if isinstance(spec.dtype, DTypeVar):
                bound = dtypes.setdefault(spec.dtype.name, actual.dtype)
                if bound != actual.dtype:
                    raise ValueError(
                        f"custom op {self.name!r}: input {key!r} dtype must "
                        f"equal dtype variable {spec.dtype.name!r}'s binding "
                        f"{bound}: got {actual.dtype}"
                    )
            elif spec.dtype != actual.dtype:
                raise ValueError(
                    f"custom op {self.name!r}: input {key!r} dtype mismatch: "
                    f"expected {spec.dtype}, got {actual.dtype}"
                )
            for j, (declared, got) in enumerate(
                zip(spec.shape, actual.shape, strict=False)
            ):
                if isinstance(declared, StaticDim):
                    if declared != got:
                        raise ValueError(
                            f"custom op {self.name!r}: input {key!r} dim {j} "
                            f"must equal static {int(declared)}: got {got}"
                        )
                    continue
                assert isinstance(declared, Symbol)
                previous = dims.setdefault(declared.name, got)
                if previous != got:
                    raise ValueError(
                        f"custom op {self.name!r}: input {key!r} dim {j} must "
                        f"equal {origin[declared.name]}: expected {previous}, "
                        f"got {got}"
                    )
                origin.setdefault(declared.name, f"input {key!r} dim {j}")
        return dims

    def _op_device(self, values: Sequence[TensorValue]) -> DeviceRef:
        """Returns the single device the op runs on.

        One `mo.custom` carries one device and the kernel gets one
        `DeviceContext`, so mixed operands are refused rather than silently
        taking the first one's device.
        """
        devices = {value.type.device for value in values}
        if len(devices) != 1:
            listed = ", ".join(sorted(str(d) for d in devices))
            raise ValueError(
                f"custom op {self.name!r}: all inputs must be on one device, "
                f"got {listed}; insert an explicit transfer"
            )
        return values[0].type.device

    def _stage(self, values: Sequence[TensorValue]) -> list[TensorValue]:
        """Stages the op into the current graph and returns its outputs."""
        bindings = self._unify([value.type for value in values])
        device = self._op_device(values)
        out_types = [
            TensorType(
                self._result_dtype(spec, values),
                [dim.substitute(bindings) for dim in spec.shape],
                device,
            )
            for spec in self.outputs
        ]
        results = ops.custom(
            self.name, device, list(values), out_types=out_types
        )
        return [result.tensor for result in results]

    def _result_dtype(
        self, spec: TemplateType, values: Sequence[TensorValue]
    ) -> DType:
        """Resolves an output's dtype, following a dtype variable if used.

        Args:
            spec: The output's declared :class:`TemplateType`.
            values: The call's actual operands, in declared order, to read a
                dtype variable's bound value from.

        Returns:
            The concrete :class:`~max.dtype.DType` this output stages with.

        Raises:
            TypeError: If *spec*'s dtype is a :class:`DTypeVar` bound by no
                input.
        """
        if not isinstance(spec.dtype, DTypeVar):
            return spec.dtype
        for (_, declared), value in zip(
            self.inputs.items(), values, strict=False
        ):
            if declared.dtype == spec.dtype:
                return value.type.dtype
        raise TypeError(
            f"custom op {self.name!r}: output dtype variable "
            f"{spec.dtype.name!r} is bound by no input"
        )

    def _bind(
        self,
        args: tuple[Tensor | TensorValue, ...],
        kwargs: Mapping[str, Tensor | TensorValue],
    ) -> tuple[Tensor | TensorValue, ...]:
        """Orders a keyword call's operands the way ``inputs`` declared them.

        Binding through :attr:`__signature__` itself keeps the advertised
        contract and the accepted one from drifting apart.
        """
        try:
            bound = self.__signature__.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"custom op {self.name!r}: {e}") from None
        return tuple(bound.args)

    def __call__(
        self,
        *args: Tensor | TensorValue,
        **kwargs: Tensor | TensorValue,
    ) -> _Result:
        """Calls the op: ``Tensor``s in eagerly, ``TensorValue``s in a graph.

        Operands may be named, since :attr:`__signature__` advertises the
        declared ``inputs`` keys as keyword parameters. ``F.functional``'s
        distributed path binds a keyword call through that same signature
        before dispatching per shard, so refusing keywords here would make an
        otherwise identical call succeed or fail depending on whether its
        arguments happen to be sharded.
        """
        if kwargs:
            args = self._bind(args, kwargs)
        graph_values = [a for a in args if isinstance(a, TensorValue)]
        if graph_values and len(graph_values) != len(args):
            # The two paths stage into different graphs, so there is no
            # meaningful reading of a call that mixes them.
            raise TypeError(
                f"custom op {self.name!r}: arguments must be either all eager "
                "Tensors or all graph TensorValues, got "
                f"({', '.join(type(a).__name__ for a in args)})"
            )
        if graph_values:
            Graph.current._import_kernels(self.extensions)
            return _single_or_list(self._stage(graph_values))

        with ensure_context():
            _load_custom_extensions(self.extensions)
            values = [TensorValue(a) for a in args]
            results = self._stage(values)
            outputs = [Tensor.from_graph_value(v) for v in results]
            return _single_or_list(outputs)


def declare(
    name: str,
    *,
    inputs: Mapping[str, TemplateType],
    outputs: Sequence[TemplateType],
    custom_extensions: Sequence[Path | str] = (),
) -> CustomOp:
    """Declares a custom op from its signature.

    Validates the signature and compiles any Mojo source package once, here,
    rather than on every call.

    Args:
        name: The registered kernel symbol.
        inputs: One :class:`TemplateType` per operand, keyed by a name used
            in diagnostics and accepted as a keyword at call time. Iteration
            order is operand order.
        outputs: One :class:`TemplateType` per result. Dims may only be
            statics, :class:`Symbol`s, or expressions over them.
        custom_extensions: Paths to the Mojo packages defining the kernel.

    Returns:
        An immutable :class:`CustomOp`.

    Raises:
        ValueError: If ``inputs`` or ``outputs`` is empty.
        TypeError: If a dim names a symbol that is not a :class:`Symbol`.
        MojoCompilationError: If a source package fails to compile.
    """
    if not inputs:
        raise ValueError(
            f"custom op {name!r}: a signature needs at least one input"
        )
    if not outputs:
        raise ValueError(
            f"custom op {name!r}: a signature needs at least one output"
        )
    op = CustomOp(
        name,
        dict(inputs),
        tuple(outputs),
        tuple(_resolve_extension(Path(p)) for p in custom_extensions),
    )
    op._check_signature()
    return op
