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
a :class:`~max.graph.Graph` build (``TensorValue``). Interpreter binding
logic lives in ``custom_gc``.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Container, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

from max._core import Operation
from max._core.dialects import builtin
from max._mlir_context import in_default_mlir_context
from max.dtype import DType
from max.experimental.functional import _load_custom_extensions
from max.experimental.realization_context import ensure_context
from max.experimental.tensor import Tensor
from max.graph import (
    DeviceRef,
    Dim,
    DimLike,
    Graph,
    KernelLibrary,
    StaticDim,
    SymbolicDim,
    TensorType,
    TensorValue,
    Type,
    ops,
)
from max.graph.graph import _resolved_custom_extensions
from mojo.paths import _build_mojo_source_package, is_mojo_source_package_path

__all__ = [
    "CustomOp",
    "DTypeVar",
    "Symbol",
    "Symbols",
    "TemplateType",
    "declare",
]

#: Attribute key stamped on each `mo.custom` op so the interpreter can look
#: up its `CustomOp`; survives `graph.copy()` and RMO->MO lowering.
_SPEC_ATTR_KEY = "max.custom_op_def"

#: Keyed by content token (see `CustomOp.token`), not kernel
#: name: two ops can share a name with different params or signatures,
#: and name-keying would let one overwrite another's already-staged ops.
#: Content-keying also means identical redeclarations share one entry.
#: Holds strong references and never shrinks.
#: TODO(pprovins): a program constructing genuinely distinct declarations in
#: a loop grows this registry without limit. Revisit if a workload hits it;
#: weak values are not the fix, since entry lifetime would then decide which
#: executor runs a staged op.
_SPEC_REGISTRY: dict[str, CustomOp] = {}

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
    ``shape`` holds :class:`Symbol`s, :class:`Param`s, statics, or expressions
    over them; its length is the entry's rank. There is no device: the op takes
    its device from its operands at each call.

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


#: Namespace `Param`'s parameter symbols live under, disjoint from graph dims.
_PARAM_DIM_PREFIX = "__param_"


class Param(SymbolicDim):
    """A named kernel parameter, usable directly in dim expressions.

    Subclassing :class:`~max.graph.SymbolicDim` lets a parameter take part
    in ordinary ``Dim`` arithmetic (``Param("g") // 2``). Its symbol is
    namespaced, so ``Param("g")`` and a graph dim named ``g`` are different
    symbols and nothing but a ``Param`` can be resolved to a parameter's
    value by accident. The type alone can't carry that guarantee, since dim
    arithmetic folds through MLIR and reconstructs operands as plain symbols.

    Means something different by position: in an input template it's a
    static pin (the actual dim must equal ``self.parameters[name]``); in
    an output expression it's substituted to the parameter's concrete
    value.
    """

    def __init__(self, name: str | Param) -> None:
        # `Dim(param)` hands back the same instance and Python re-runs this
        # on it, so prefixing has to be idempotent.
        if isinstance(name, Param):
            name = name.param_name
        super().__init__(f"{_PARAM_DIM_PREFIX}{name}")

    @property
    def param_name(self) -> str:
        """The declared parameter name, without the namespace prefix."""
        return self.name.removeprefix(_PARAM_DIM_PREFIX)


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


#: Canonicalized, hashable, order-independent param-dict key; see
#: `_frozen_params`.
_ParamKey = tuple[tuple[str, str, object], ...]


def _frozen_params(
    params: Mapping[str, bool | int | str | DType | None],
) -> _ParamKey:
    """Canonicalizes a param dict into a hashable, order-independent key.

    Each entry carries its value's type name, because ``ops.custom`` lowers
    ``bool``, ``int``, ``str`` and ``DType`` to four different MLIR
    attributes (see ``_parameter_attribute``) while a key over the values
    alone collapses two of those pairs: ``DType.float32`` and ``"float32"``
    share a spelling, and ``True`` and ``1`` compare and hash equal.
    """
    return tuple(
        sorted(
            (
                name,
                type(value).__name__,
                value.name if isinstance(value, DType) else value,
            )
            for name, value in params.items()
        )
    )


def _dtype_key(dtype: DType | DTypeVar) -> str:
    """Spells a template dtype for a key, keeping the two kinds apart.

    A ``DTypeVar`` named ``float32`` means something entirely different from
    :attr:`DType.float32` and the two share a spelling, so the kind is part of
    the key.
    """
    kind = "var" if isinstance(dtype, DTypeVar) else "dtype"
    return f"{kind}:{dtype.name}"


def _signature_key(
    inputs: Mapping[str, TemplateType], outputs: Sequence[TemplateType]
) -> str:
    """Canonicalizes a signature into one string, for `_digest`.

    Operand names are part of it: they are what a keyword call binds through
    (see :attr:`CustomOp.__signature__`). Each dim contributes its own
    ``repr``, so ``w // 2`` and ``w // 3`` stay distinct.
    """
    return repr(
        (
            [
                (key, _dtype_key(spec.dtype), [repr(d) for d in spec.shape])
                for key, spec in inputs.items()
            ],
            [
                (_dtype_key(spec.dtype), [repr(d) for d in spec.shape])
                for spec in outputs
            ],
        )
    )


#: Namespace for allocated data-dependent dims; operands may carry one.
_DYN_PREFIX = "__dyn_"


def _dim_name_fragment(name: str) -> str:
    """Makes *name* embeddable in a dim symbol; lossy, the digest is unique."""
    return re.sub(r"\W", "_", name)


def _digest(*fields: str) -> str:
    """Digests *fields*, distinctly for every distinct field tuple.

    Each field is length-prefixed, so the concatenation is decodable and no
    two tuples share one digest input: without that, a value containing the
    separator would let a field boundary move (``("a_b", "c")`` and
    ``("a", "b_c")``) without changing what is hashed.

    Uses ``hashlib``, never the builtin ``hash()``: ``hash()`` is salted per
    process for ``str``, and callers here need names that are identical run to
    run (see :meth:`Graph._allocate_data_dependent_ordinal` on IR cache
    hits).
    """
    encoded = "".join(f"{len(field)}:{field}" for field in fields)
    return hashlib.blake2b(encoded.encode(), digest_size=8).hexdigest()


def _allocate_unbound_dim(
    dim: Dim, bound: Container[str], graph: Graph, kernel: str
) -> Dim:
    """Allocates a unique per-graph dim for a symbol no input bound."""
    if not isinstance(dim, Symbol) or dim.name in bound:
        return dim
    index = graph._allocate_data_dependent_ordinal()
    graph_fragment = _dim_name_fragment(graph.name)
    kernel_fragment = _dim_name_fragment(kernel)
    symbol_name = dim.symbol_name
    name = (
        f"{_DYN_PREFIX}{graph_fragment}_{kernel_fragment}_{index}_"
        f"{symbol_name}_{_digest(graph.name, kernel, str(index), symbol_name)}"
    )
    if name in graph._params:
        raise TypeError(
            f"custom op {kernel!r}: graph {graph.name!r} already holds a dim "
            f"named {name!r}, which is the name this staging allocates for "
            f"data-dependent dim {symbol_name!r}; rename the dim"
        )
    return SymbolicDim(name)


def _resolve_extension(path: Path) -> Path:
    """Compiles a Mojo source package once, at declaration.

    A call then only registers the resulting binary. Resolving per call
    would rerun ``mojo precompile`` every time the op is invoked.
    """
    if is_mojo_source_package_path(path):
        return _build_mojo_source_package(path)
    return path


def _resolve_parameters(
    inputs: Mapping[str, TemplateType],
    outputs: Sequence[TemplateType],
    parameters: Mapping[str, bool | int | str | DType | None],
) -> tuple[
    dict[str, TemplateType],
    tuple[TemplateType, ...],
    dict[tuple[str, int], str],
]:
    """Substitutes valued parameters into a declared signature.

    Parameters are compile-time, so a valued one can always be folded away
    here, in an input template (pinning a dim) as well as an output
    expression. What remains unresolved afterward is exactly what makes an
    op uncallable (see :attr:`CustomOp.is_complete`).

    Returns:
        The resolved inputs and outputs, plus which input dims a
        :class:`Param` pinned, recorded before the fold erases it so
        :meth:`CustomOp._unify` can still name the parameter in a mismatch
        error.
    """
    substitutions = {
        f"{_PARAM_DIM_PREFIX}{name}": value
        for name, value in parameters.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    if not substitutions:
        return dict(inputs), tuple(outputs), {}

    def resolved(spec: TemplateType) -> TemplateType:
        return TemplateType(
            spec.dtype, [d.substitute(substitutions) for d in spec.shape]
        )

    pinned = {
        (key, j): dim.param_name
        for key, spec in inputs.items()
        for j, dim in enumerate(spec.shape)
        if isinstance(dim, Param) and dim.name in substitutions
    }
    return (
        {key: resolved(spec) for key, spec in inputs.items()},
        tuple(resolved(spec) for spec in outputs),
        pinned,
    )


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
    #: Compile-time parameters passed to the kernel. ``None`` declares one
    #: without giving it a value yet (see :attr:`is_complete`).
    parameters: Mapping[str, bool | int | str | DType | None] = field(
        hash=False
    )
    #: The signature as declared, before valued parameters were folded
    #: into ``inputs``/``outputs``, so :meth:`__getitem__` can re-specialize
    #: a :class:`Param`. Derived from the fields above, so not compared.
    declared_inputs: Mapping[str, TemplateType] = field(
        hash=False, compare=False
    )
    declared_outputs: tuple[TemplateType, ...] = field(compare=False)
    #: Input dims a :class:`Param` pinned before folding, by (operand, dim
    #: index), so a static-mismatch error in :meth:`_unify` can still name
    #: the parameter -- by then the dim is a bare static.
    pinned_params: Mapping[tuple[str, int], str] = field(
        hash=False, compare=False
    )

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

    @cached_property
    def token(self) -> str:
        """The registry token for this op, a digest of its declaration.

        Content-derived, so identical declarations share one token and so
        one compile: the token is stamped into a discardable attribute,
        which prints into the module ASM that
        ``executor._eager_model_cache_key`` hashes.

        Covers exactly what a resolved op is read for: kernel symbol,
        parameters, resolved signature, and the extension paths
        ``custom_gc.binding_for`` links against. Kernel library CONTENT is
        deliberately absent: both consumers already hash it themselves --
        ``BindingKey.lib_hashes`` on the interpreter path, and
        ``_eager_model_cache_key``'s resolved-library component alongside the
        ASM hash on the compile path -- so a kernel edit is covered without
        reading every library file at every declaration.
        """
        digest = _digest(
            self.name,
            repr(_frozen_params(self.parameters)),
            _signature_key(self.inputs, self.outputs),
            repr([str(path) for path in self.extensions]),
        )
        return f"{self.name}#{digest}"

    def _check_signature(self) -> None:
        """Rejects any dim a signature is not allowed to name.

        Every dim must reduce to statics, :class:`Symbol`s, or :class:`Param`s.
        Because dim arithmetic erases the Python subclass, the test is on the
        namespaced name, which is exactly what a :class:`Symbol` or
        :class:`Param` contributes and nothing else can.

        An input dim must additionally be a bare :class:`Symbol` or a static,
        never an expression over one: an int-valued :class:`Param` is already
        folded to a static by :func:`_resolve_parameters`, so what remains is
        either unresolvable (``Symbol("m") // 2``, which :meth:`_unify` cannot
        solve for ``m``) or a ``Param`` :meth:`_unify` has no value to bind.
        An output dim keeps the looser rule, since it only ever gets
        substituted into, never solved -- but a ``Param`` naming a value that
        can never fold (see :meth:`_check_dim_param_value`) is refused on
        both sides.
        """
        entries = [
            (f"input {key!r}", spec) for key, spec in self.inputs.items()
        ] + [(f"output {i}", spec) for i, spec in enumerate(self.outputs)]
        for where, spec in entries:
            for j, dim in enumerate(spec.shape):
                for symbol in dim.parameters:
                    if symbol.name.startswith(_PARAM_DIM_PREFIX):
                        self._check_dim_param_value(
                            where,
                            j,
                            symbol.name.removeprefix(_PARAM_DIM_PREFIX),
                        )
                        continue
                    if symbol.name.startswith(_SYMBOL_PREFIX):
                        continue
                    raise TypeError(
                        f"custom op {self.name!r}: {where} dim {j} uses "
                        f"{symbol.name!r}, which is not a signature dim; use "
                        "custom.Symbol, a Param, or a static"
                    )
        for key, spec in self.inputs.items():
            for j, dim in enumerate(spec.shape):
                if isinstance(dim, Symbol | StaticDim):
                    continue
                raise TypeError(
                    f"custom op {self.name!r}: input {key!r} dim {j} "
                    f"({dim}) is not directly bindable; an input dim "
                    "must be a plain custom.Symbol or a static, not a "
                    "Param or an algebraic expression"
                )

    def _check_dim_param_value(self, where: str, j: int, name: str) -> None:
        """Refuses a dim naming a parameter whose value can never fold.

        :func:`_resolve_parameters` only folds an int (never a bool), so a
        ``str``- or ``DType``-valued parameter named by a dim is a dead end:
        the dim keeps its ``Param`` symbol forever, and
        :meth:`_unresolved_parameters` then reports a parameter that has a
        value as having none. An unvalued parameter is left alone: it is the
        ordinary incomplete case :meth:`__getitem__` can still resolve.
        """
        value = self.parameters.get(name)
        if value is None or (
            isinstance(value, int) and not isinstance(value, bool)
        ):
            return
        raise TypeError(
            f"custom op {self.name!r}: {where} dim {j}: Param({name!r}) is "
            f"{type(value).__name__}-valued ({value!r}); only int-valued "
            "parameters can appear in a dim"
        )

    def _check_unbound_outputs(self) -> None:
        """Requires a shape function for any output dim no input binds."""
        has_shape_function: bool | None = None
        first_seen: dict[str, tuple[int, int]] = {}
        for i, j, dim, symbol in self._unbound_output_symbols():
            if not isinstance(dim, Symbol):
                raise ValueError(
                    f"custom op {self.name!r}: output {i} dim {j} "
                    f"symbol {symbol.name!r} is data-dependent and "
                    "must appear bare, not inside an expression"
                )
            # GEX-2198: lift once the compiler shares one shape call.
            self._check_symbol_used_once(first_seen, i, j, dim)
            if has_shape_function is None:
                has_shape_function = self._has_shape_function()
            if not has_shape_function:
                raise ValueError(
                    f"kernel {self.name!r} declares data-dependent "
                    f"output {i} dim {symbol.name!r} but registers "
                    "no shape function; register one with "
                    "@extensibility.register_shape_function (see "
                    "'Declare the output shape' at "
                    "docs.modular.com/max/develop/build-custom-ops)"
                )

    def _unbound_output_symbols(
        self,
    ) -> Iterator[tuple[int, int, Dim, SymbolicDim]]:
        """Yields ``(i, j, dim, symbol)`` per unbound data-dependent dim."""
        bound = {
            symbol.name
            for spec in self.inputs.values()
            for dim in spec.shape
            for symbol in dim.parameters
        }
        for i, spec in enumerate(self.outputs):
            for j, dim in enumerate(spec.shape):
                for symbol in dim.parameters:
                    if not symbol.name.startswith(_SYMBOL_PREFIX):
                        continue
                    if symbol.name in bound:
                        continue
                    yield i, j, dim, symbol

    def _check_symbol_used_once(
        self,
        first_seen: dict[str, tuple[int, int]],
        i: int,
        j: int,
        dim: Symbol,
    ) -> None:
        """Refuses a data-dependent symbol that names a second output dim."""
        earlier_i, earlier_j = first_seen.setdefault(dim.name, (i, j))
        if (earlier_i, earlier_j) == (i, j):
            return
        if earlier_i == i:
            raise ValueError(
                f"custom op {self.name!r}: data-dependent "
                f"symbol {dim.symbol_name!r} appears in "
                f"more than one dim of output {i} (dim "
                f"{earlier_j} and dim {j}); use a "
                "distinct symbol per dim, and "
                "ops.rebind downstream if the two must "
                "be equal"
            )
        raise ValueError(
            f"custom op {self.name!r}: data-dependent "
            f"symbol {dim.symbol_name!r} appears in more "
            f"than one output (output {earlier_i} and "
            f"output {i}); use a distinct symbol per "
            f"output, and ops.rebind downstream if the "
            "two must be equal"
        )

    @property
    def is_complete(self) -> bool:
        """Whether the signature has no unresolved :class:`Param` left.

        ``False`` means at least one parameter a dim names, or one declared
        bare, has no value, so the def cannot be staged (see :meth:`_stage`)
        until it is given one, either at declaration or through
        :meth:`__getitem__`. A parameter whose value could never fold into a
        dim is refused at declaration (see :meth:`_check_dim_param_value`)
        rather than stranding a def here with no remedy.
        """
        return not self._unresolved_parameters()

    def _unresolved_parameters(self) -> list[str]:
        """Parameter names with no value: as a dim symbol, or declared bare.

        A dim-symbol scan alone would miss a declared parameter that names no
        dim at all -- an expert id or a kernel flag the kernel body reads
        directly -- so every ``None``-valued entry in ``parameters`` is
        reported too.
        """
        names = {
            name for name, value in self.parameters.items() if value is None
        }
        for spec in [*self.inputs.values(), *self.outputs]:
            for dim in spec.shape:
                for symbol in dim.parameters:
                    if symbol.name.startswith(_PARAM_DIM_PREFIX):
                        names.add(symbol.name.removeprefix(_PARAM_DIM_PREFIX))
        return sorted(names)

    @in_default_mlir_context
    def _has_shape_function(self) -> bool:
        """Whether the kernel (overlay included) registers a shape function."""
        library = KernelLibrary()
        library.load_paths(_resolved_custom_extensions(self.extensions))
        return library.has_shape_function(self.name)

    def __getitem__(
        self, params: Mapping[str, bool | int | str | DType | None]
    ) -> CustomOp:
        """Returns a derived op specialized with per-call *params*.

        Merges *params* into this op's own parameters and declares the
        result afresh, so the merged params are re-validated and the derived
        op gets its own token: a later specialization cannot overwrite this
        one's registry entry. Repeated subscripts with the same merged params
        return equal ops sharing one token, and so one compiled binding.

        Declares from the *declared* (pre-resolution) signature: this op's
        ``inputs``/``outputs`` may already have folded a :class:`Param` into
        a static value, leaving nothing for the derived op to re-resolve to a
        different one.
        """
        return declare(
            self.name,
            inputs=self.declared_inputs,
            outputs=self.declared_outputs,
            parameters={**self.parameters, **params},
            custom_extensions=self.extensions,
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
                        pinned = self.pinned_params.get((key, j))
                        if pinned is not None:
                            raise ValueError(
                                f"custom op {self.name!r}: input {key!r} "
                                f"dim {j}: Param({pinned!r}) pins this dim "
                                f"to {int(declared)}, got {got}"
                            )
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
        if unresolved := self._unresolved_parameters():
            raise ValueError(
                f"custom op {self.name!r}: parameter(s) "
                f"{', '.join(unresolved)} have no value; supply them at "
                "definition or with op[{...}]"
            )
        bindings = self._unify([value.type for value in values])
        graph = Graph.current
        device = self._op_device(values)
        out_types = [
            TensorType(
                self._result_dtype(spec, values),
                [
                    _allocate_unbound_dim(
                        dim.substitute(bindings), bindings, graph, self.name
                    )
                    for dim in spec.shape
                ],
                device,
            )
            for spec in self.outputs
        ]
        # Drops nothing at runtime (the guard above refused any `None`); it
        # narrows `_parameters` to the value union `ops.custom` declares.
        params = {
            key: value
            for key, value in self.parameters.items()
            if value is not None
        }
        results = ops.custom(
            self.name,
            device,
            list(values),
            out_types=out_types,
            parameters=params or None,
        )
        if results:
            op = results[0]._mlir_value.owner
            assert isinstance(op, Operation)
            op.discardable_attributes[_SPEC_ATTR_KEY] = builtin.StringAttr(
                self.token
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
    parameters: Mapping[str, bool | int | str | DType | None] | None = None,
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
            statics, :class:`Symbol`s, :class:`Param`s, or expressions over
            them. A :class:`Symbol` no input binds is data-dependent: the
            kernel must register a shape function with
            ``@extensibility.register_shape_function`` (see the "Declare the
            output shape" section of
            https://docs.modular.com/max/develop/build-custom-ops).
        parameters: Compile-time parameters passed to the kernel. A value of
            ``None`` declares the parameter without giving it one yet,
            leaving any :class:`Param` naming it unresolved (see
            :attr:`CustomOp.is_complete`).
        custom_extensions: Paths to the Mojo packages defining the kernel.

    Returns:
        An immutable :class:`CustomOp`.

    Raises:
        ValueError: If ``inputs`` or ``outputs`` is empty.
        TypeError: If a dim names a symbol that is neither a :class:`Symbol`
            nor a :class:`Param`.
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
    declared_inputs = dict(inputs)
    declared_outputs = tuple(outputs)
    params = dict(parameters or {})
    resolved_inputs, resolved_outputs, pinned = _resolve_parameters(
        declared_inputs, declared_outputs, params
    )
    op = CustomOp(
        name,
        resolved_inputs,
        resolved_outputs,
        tuple(_resolve_extension(Path(p)) for p in custom_extensions),
        params,
        declared_inputs,
        declared_outputs,
        pinned,
    )
    op._check_signature()
    op._check_unbound_outputs()
    # Anything sharing a content token declares the same kernel, parameters
    # and signature, so it is interchangeable with this op for every purpose
    # a resolved op is read for.
    _SPEC_REGISTRY[op.token] = op
    return op
