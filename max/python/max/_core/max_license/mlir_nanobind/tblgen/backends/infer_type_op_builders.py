# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Codegen for the InferTypeOp builder variants exposed to Python.

For each MLIR op with the `InferTypeOpAdaptor` trait, `mlir-tblgen`
emits up to four builders that infer result types from operands and
attributes. We mirror that set on the Python side as two pairs:

  separate args, wrapped attributes
  separate args, unwrapped (raw) attributes
  collective `(operands, attributes)`
  collective `(operands, properties, discardable_attributes)`

The corresponding C++ helpers live in
``_core/Graph/InferTypeOpBuilders.h``. The per-op codegen here is small
because the boilerplate (cast the builder handle, construct
``OperationState``, call ``inferReturnTypes``, raise on failure) is
encapsulated in those templates. The separate-arg variants contribute
only the state-population body via the ``Variant`` strategy below; the
collective variants are op-agnostic one-liners.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..types import (
    Arg,
    ArgList,
    Attr,
    DefaultValuedAttr,
    Op,
    OpBuilder,
    TypeConstraint,
    Variadic,
    python_variable_name,
)

# ---------------------------------------------------------------------- #
# Tablegen-parity helpers for unwrapped attribute conversion
# ---------------------------------------------------------------------- #


def _can_unwrap(attr: Attr) -> bool:
    """Port of ``mlir-tblgen``'s ``canUseUnwrappedRawValue``: true iff the
    attr has a return type distinct from its storage type AND a
    ``constBuilderCall`` template that tells us how to wrap a raw value
    into the storage ``Attribute``."""
    data = attr.data
    return bool(
        (data.get("returnType") or "")
        and (data.get("storageType") or "")
        and data.get("returnType") != data.get("storageType")
        and (data.get("constBuilderCall") or "")
    )


def _unwrapped_return_type(attr: Attr) -> str:
    """The raw scalar/enum return type used in the unwrapped variant
    (such as ``bool`` for ``BoolAttr``, ``int64_t`` for ``I64Attr``)."""
    return attr.data["returnType"]


def _const_builder_expr(attr: Attr, value_expr: str) -> str:
    """Substitute the attr's ``constBuilderCall`` template, mapping
    ``$_builder`` → a local ``odsBuilder`` and ``$0`` → ``value_expr``.
    Mirrors ``constBuildAttrFromParam`` in ``mlir-tblgen``."""
    return (
        attr.data["constBuilderCall"]
        .replace("$_builder", "odsBuilder")
        .replace("$0", value_expr)
    )


# ---------------------------------------------------------------------- #
# Variant strategy — per-arg behaviour for the separate-arg variants
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class Variant:
    """How a given InferTypeOp-builder flavour handles each builder arg.

    Subclasses differ only in ``param_type`` (how an attribute is typed
    in the Python-facing signature) and ``populate`` (how that attribute
    is dropped into ``OperationState``). The lambda boilerplate around
    them is handed off to the ``bind_infer_type_op{,_unwrapped}<Op>`` C++
    template named by ``bind_helper``.
    """

    bind_helper: str
    populator_extra_param: str

    def param_type(self, arg: Arg) -> str:
        return arg.type.cpp

    def populate(self, arg: Arg, op: Op) -> str:
        py_name = python_variable_name(arg.name)
        if isinstance(arg.entry, (TypeConstraint, Variadic)):
            # Both single Value (``TypedValue<T>``) and ``ValueRange``
            # (variadic) use the same ``state.addOperands(...)`` overload.
            return f"state.addOperands({py_name});"
        if isinstance(arg.entry, Attr):
            # Direct field assignment into the op's Properties struct,
            # the same pattern ODS uses for wrapped-attr builders. The
            # ``if (attr)`` guards null Attributes (optional / defaulted
            # attrs); the ODS builder emits the same guard.
            return (
                f"if ({py_name}) "
                f"state.getOrAddProperties<{op.type.cpp}::Properties>()"
                f".{arg.name} = {py_name};"
            )
        raise NotImplementedError(
            f"unsupported arg kind {type(arg.entry).__name__}"
            f" for {op.type.cpp}::{arg.name}"
        )


class _UnwrappedVariant(Variant):
    """Unwrapped-attr flavour: attributes with an unwrapped representation
    arrive as their ``returnType`` (such as ``bool``, ``int64_t``) and are
    wrapped to the storage ``Attribute`` via the attr's
    ``constBuilderCall``.

    Mirrors ``genCodeForAddingArgAndRegionForBuilder(..., isRawValueAttr=true)``.
    """

    def param_type(self, arg: Arg) -> str:
        if isinstance(arg.entry, Attr) and _can_unwrap(arg.entry):
            return _unwrapped_return_type(arg.entry)
        return super().param_type(arg)

    def populate(self, arg: Arg, op: Op) -> str:
        if isinstance(arg.entry, Attr) and _can_unwrap(arg.entry):
            py_name = python_variable_name(arg.name)
            field = (
                f"state.getOrAddProperties<{op.type.cpp}::Properties>()"
                f".{arg.name}"
            )
            return f"{field} = {_const_builder_expr(arg.entry, py_name)};"
        return super().populate(arg, op)


WRAPPED = Variant(bind_helper="bind_infer_type_op", populator_extra_param="")
UNWRAPPED = _UnwrappedVariant(
    bind_helper="bind_infer_type_op_unwrapped",
    populator_extra_param=", mlir::Builder& odsBuilder",
)


# ---------------------------------------------------------------------- #
# Emitters
# ---------------------------------------------------------------------- #


def _arg_annotation(arg: Arg, variant: Variant | None = None) -> str:
    """``nb::arg("name")`` for one Python parameter.

    For ``DefaultValuedAttr`` args in the unwrapped variant, emit the ODS
    default expression directly — it's typed as the attr's ``returnType``,
    which matches the unwrapped param type. (mlir-tblgen does the same in
    its own generated ``build`` overloads.) The wrapped variant doesn't
    carry a default because the ODS expression doesn't construct a wrapped
    attr without an MLIRContext, which isn't available at binding
    registration time; callers either pass the wrapped attr explicitly or
    omit the arg so the unwrapped overload's default kicks in.
    """
    name_part = f'nb::arg("{python_variable_name(arg.name)}")'
    if variant is UNWRAPPED and isinstance(arg.entry, DefaultValuedAttr):
        return f"{name_part} = {arg.entry.default}"
    return name_part


def _separate_arg_def(
    op: Op, builder: OpBuilder, args: ArgList, variant: Variant
) -> str:
    """One ``.def`` line for the separate-arg InferTypeOp builder, in the
    flavour described by ``variant``. The per-op codegen here is just
    the state-populator body — wrapping (state ctor, OpBuilder cast,
    ``finalize_checked`` tail) lives in
    ``bind_infer_type_op{,_unwrapped}<Op>``.
    """
    param_decls = ", ".join(
        f"{variant.param_type(a)} {python_variable_name(a.name)}"
        for a in builder.args
    )
    body = "\n      ".join(variant.populate(a, op) for a in builder.args)
    populator = (
        f"+[](mlir::OperationState& state{variant.populator_extra_param}"
        f", {param_decls}) {{\n"
        f"      {body}\n"
        f"    }}"
    )
    annotations = ", ".join(_arg_annotation(a, variant) for a in args)
    return (
        f".def(nb::new_(::M::Graph::Python::{variant.bind_helper}"
        f"<{op.type.cpp}>({populator})), "
        f'nb::arg("builder"), {annotations})'
    )


def _collective_attrdict_def(op: Op) -> str:
    """``.def`` for the collective ``(operands, attributes)`` form,
    pointing at the generic ``create_collective_attrdict<Op>`` template
    — body is fully op-agnostic, no per-op codegen."""
    return (
        f".def(nb::new_(&::M::Graph::Python::create_collective_attrdict"
        f"<{op.type.cpp}>), "
        f'nb::arg("builder"), nb::arg("location"), nb::arg("operands"), '
        f'nb::arg("attributes") = ::mlir::DictionaryAttr())'
    )


def _collective_propstruct_def(op: Op) -> str:
    """``.def`` for the collective
    ``(operands, properties, discardable_attributes)`` form, pointing at
    the generic ``create_collective_propstruct<Op>`` template — same
    pattern as the attrdict form with the extra ``properties`` dict."""
    return (
        f".def(nb::new_(&::M::Graph::Python::create_collective_propstruct"
        f"<{op.type.cpp}>), "
        f'nb::arg("builder"), nb::arg("location"), nb::arg("operands"), '
        f'nb::arg("properties") = ::mlir::DictionaryAttr(), '
        f'nb::arg("discardable_attributes") = ::mlir::DictionaryAttr())'
    )


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #


def overload_defs(op: Op, builder: OpBuilder, args: ArgList) -> Iterable[str]:
    """Yield ``.def`` lines for every InferTypeOp-builder variant
    applicable to ``op``.

    Decides which variants to emit, mirroring the set ``mlir-tblgen``
    generates for an op with ``InferTypeOpAdaptor`` (see ``genBuilder``
    in ``OpDefinitionsGen.cpp``). The caller — ``NanobindBackend.op_builder``
    — handles the upstream "is this even an InferTypeOp builder?" gate.
    """
    yield _separate_arg_def(op, builder, args, WRAPPED)

    # The unwrapped flavour is emitted only when at least one attr has a
    # raw representation, matching `canGenerateUnwrappedBuilder` in
    # `mlir-tblgen`.
    if any(
        isinstance(a.entry, Attr) and _can_unwrap(a.entry) for a in builder.args
    ):
        yield _separate_arg_def(op, builder, args, UNWRAPPED)

    yield _collective_attrdict_def(op)

    # The propstruct collective is emitted only when the op has a
    # non-empty Properties struct (any attribute arg or explicit
    # Property), matching `emitHelper.hasNonEmptyPropertiesStruct()`.
    if any(isinstance(a.entry, Attr) for a in op.arguments) or op.data.get(
        "properties"
    ):
        yield _collective_propstruct_def(op)
