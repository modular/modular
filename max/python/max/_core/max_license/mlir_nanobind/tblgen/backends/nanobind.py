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
"""Nanobind C++ backend."""

from __future__ import annotations

from collections.abc import Iterable

from ..backend import Backend, BackendTypeRegistry, indent, indent_all
from ..casing import snake_case, upper_camel_case
from ..types import (
    Arg,
    ArgList,
    Attr,
    AttrBuilder,
    AttrDef,
    AttrInterface,
    BuilderCall,
    CArg,
    DefMixin,
    Dialect,
    Entry,
    EnumAttrInfo,
    Interface,
    InterfaceMethod,
    Name,
    Op,
    OpBuilder,
    OpInterface,
    Parameter,
    Pass,
    Signature,
    TypeBuilder,
    TypeConstraint,
    TypeDef,
    TypeInterface,
    Variadic,
    python_variable_name,
)
from . import infer_type_op_builders


def annotation(arg: Arg):  # noqa: ANN201
    annotation = f"""nb::arg("{python_variable_name(arg.name)}")"""
    if isinstance(arg.entry, CArg):
        default = arg.entry.default
        default_value = (
            f"{arg.type.cpp}()"
            if default == "{}"
            else f"{arg.type.cpp}({default})"
        )
        return f"{annotation} = {default_value}"
    return annotation


def annotations_(
    args: Iterable[Arg],
) -> str:  # from __future__ import annotations
    return ", ".join(annotation(arg) for arg in args)


def docstring(doc: str) -> str:
    return f'R"nbdoc({doc})nbdoc"'


def not_implemented(signature: Signature) -> str:
    types = ", ".join(
        (
            signature.return_type.cpp,
            "nb::handle",  # self
            *(type.cpp for type in signature.args.types),
        )
    )
    return f"optional_pimpl(&not_implemented<{types}>)"


def cpp_call_args(args: Iterable[str | None]) -> str:
    return ", ".join(str(arg) for arg in args if arg)


def properties_and_methods(entry: Interface | DefMixin):  # noqa: ANN201
    properties = {param.name: param for param in entry.parameters}

    # De-duplicate interface methods without re-ordering them.
    methods = {method: None for method in entry.methods}

    # Match up getter+setter-like interface methods into pythonic properties
    interface_properties = {}

    for method in list(methods):  # concurrent modification
        if name := method.getter_for:
            interface_properties[name] = Parameter(entry, name, method)
            del methods[method]

    for method in list(methods):  # concurrent modification
        if name := method.setter_for:
            # If we have an associated getter
            # - default to properties, which override interface_properties
            if prop := properties.get(name, interface_properties.get(name)):
                # - if there's multiple setters, just use the first one we find
                # - we can try to deal with setter overloads at some point, maybe
                if not prop.setter:
                    prop.setter = method
                # only delete if we found a property to use it for
                del methods[method]

    # parameter fields override interface properties
    properties = {**interface_properties, **properties}
    return properties, methods


class NanobindBackend(Backend):
    TYPES_WITHOUT_TYPEINFO = (
        "mlir::RewriterBase",
        "llvm::raw_ostream",
        "AsmParser",
        "AsmPrinter",
        "IRRewriter",
        "llvm::fltSemantics",
    )

    registry = BackendTypeRegistry()

    def comment(self, comment: str) -> str:
        return f"/// {comment}"

    def missing_typeinfo(self, name: Name) -> bool:
        cpp = name.cpp
        return any(type in cpp for type in self.TYPES_WITHOUT_TYPEINFO)

    def interface_method(self, method: InterfaceMethod, entry: Entry):  # noqa: ANN201
        name = python_variable_name(method.function_name).removeprefix("get_")
        signature = method.signature(entry)
        arg_types = (*method.args.types, method.return_type)
        if unsupported_types := {
            type.cpp for type in arg_types if self.missing_typeinfo(type)
        }:
            yield f""".def_static("{name}", []() {{ throw nb::type_error("No type info available for inputs {unsupported_types}"); }})"""
        else:
            # There's some problem with adding annotations here.
            # Nanobind seems to count the number of function arguments oddly.
            yield f""".def("{name}", optional_pimpl(checked({signature.cpp_reference_overload_cast})), {docstring(method["description"])})"""

    def builder(  # noqa: ANN201
        self, builder: AttrBuilder | TypeBuilder | BuilderCall, entry: Entry
    ):
        if isinstance(builder, BuilderCall):
            yield f""".def(nb::new_([]() {{
                return {builder.call.replace("$_builder", "mlir::OpBuilder(current_context())")};
            }}))"""
        else:
            signature = builder.signature(entry)
            types = [type.cpp for type in signature.args.types]

            if entry["genVerifyDecl"]:
                fp = f"&get_checked<{entry.type.cpp}, {', '.join(types)}>"
            else:
                fp = signature.cpp_reference_overload_cast

            if builder.infer_context:
                args = signature.args
            else:
                fp = f"infer_context({fp})"
                args = signature.args[1:]

            if annotations := annotations_(args):
                yield f""".def(nb::new_({fp}), {annotations})"""
            else:
                yield f""".def(nb::new_({fp}))"""

    def op_builder(self, builder: OpBuilder, op: Op):  # noqa: ANN201
        """Bind one ODS-emitted builder for `op`.

        For non-InferTypeOp builders we point straight at
        `create_op<Op, ...>`. For InferTypeOp builders we hand off to
        `infer_type_op_builders.overload_defs`, which yields up to four
        `.def`s — the same set `mlir-tblgen` emits for an
        `InferTypeOpAdaptor` op (see `genBuilder` in
        `OpDefinitionsGen.cpp`). The default ODS InferTypeOp overload
        calls `report_fatal_error` on inference failure, which aborts
        the Python interpreter; our variants route through
        `finalize_checked` and raise a `ValueError` instead.
        """
        location = ArgList([Arg("::mlir::Location", "location")])
        args = location + builder.args

        op_arg_names = tuple(a.name for a in op.arguments)
        builder_arg_names = tuple(a.name for a in builder.args)
        # The InferTypeOp codegen only knows how to marshal operands and
        # wrapped attributes. Anything else (e.g. a raw `unsigned` the
        # op-specific builder converts to an attribute internally) needs
        # op-specific logic — fall back to `create_op` for those, losing
        # inference-failure handling but keeping the build correct.
        marshallable_args = all(
            isinstance(a.entry, (TypeConstraint, Variadic, Attr))
            for a in builder.args
        )
        is_infer_type_op_overload = (
            op.has_infer_type_op_adaptor
            and marshallable_args
            and builder_arg_names == op_arg_names
        )

        if not is_infer_type_op_overload:
            def_args = [
                f"nb::new_(&create_op<{op.type.cpp}, {', '.join(a.type.cpp for a in args)}>)",
                'nb::arg("builder")',
                annotations_(args),
            ]
            yield f".def({cpp_call_args(def_args)})"
            return

        yield from infer_type_op_builders.overload_defs(op, builder, args)

    @registry.generator(EnumAttrInfo)
    def enum(self, enum: EnumAttrInfo):  # noqa: ANN201
        yield f"""nb::enum_<{enum.enum_type.cpp}>(m, "{enum.enum_type.name}")"""
        for value in enum["enumerants"]:
            value_name = python_variable_name(value["str"])
            yield indent(
                f""".value("{value_name}", {enum.enum_type.cpp}::{value["symbol"]})""",
                1,
            )
        yield ";"
        if enum["genSpecializedAttr"]:
            yield f"""nb::class_<{enum.type.cpp}, ::mlir::Attribute>(m, "{enum.type.name}")"""
            yield indent(f""".def(nb::new_(&{enum.type.cpp}::get))""", 1)
            yield indent(
                f""".def_prop_ro("value", &{enum.type.cpp}::getValue);""", 1
            )

    @registry.generator(Dialect)
    def dialect(self, dialect: Dialect):  # noqa: ANN201
        yield f"""m.doc() = {docstring(dialect["description"])};"""

    @registry.generator(Pass)
    def pass_definition(self, entry: Pass):  # noqa: ANN201
        args = ", ".join(f"{o.type.cpp} {o.cpp_name}" for o in entry.options)
        yield f"""m.def("{entry.name}", []({args}) {{"""
        if not (constructor := entry.constructor):
            arg_names = ", ".join(f"{o.cpp_name}" for o in entry.options)
            options = (
                f"{entry.options_type}{{{arg_names}}}" if entry.options else ""
            )
            constructor = f"create{entry.name}({options})"
        yield indent(
            f"return std::make_unique<M::Graph::Python::Pass>({constructor});",
            2,
        )
        annotations = (
            f"""nb::arg("{snake_case(o.cpp_name)}") = {o.type.cpp}({o.default})"""
            for o in entry.options
        )
        yield indent(
            "}, " + ", ".join((*annotations, docstring(entry["description"]))),
            1,
        )
        yield ");"

    @registry.generator(Op)
    def op_definition(self, entry: Op):  # noqa: ANN201
        name = f"{upper_camel_case(entry['opName'].replace('.', '_'))}Op"
        yield f"""nb::class_<{entry.type.cpp}, ::mlir::OpState>(m, "{name}")"""
        for builder in entry.builders:
            yield from indent_all(self.op_builder(builder, entry), 1)
        yield from indent_all(self.definition(entry), 1)

    @registry.generator(AttrDef)
    def attr_definition(self, entry: AttrDef):  # noqa: ANN201
        yield f"""nb::class_<{entry.type.cpp}, ::mlir::Attribute>(m, "{entry.type.name}")"""
        for builder in sorted(entry.builders, key=lambda b: len(b.args)):
            yield from indent_all(self.builder(builder, entry), 1)
        yield from indent_all(self.definition(entry), 1)

    @registry.generator(TypeDef)
    def type_definition(self, entry: TypeDef):  # noqa: ANN201
        yield f"""nb::class_<{entry.type.cpp}, ::mlir::Type>(m, "{entry.type.name}")"""
        for builder in sorted(entry.builders, key=lambda b: len(b.args)):
            yield from indent_all(self.builder(builder, entry), 1)
        yield from indent_all(self.definition(entry), 1)

    @registry.generator(AttrInterface)
    def attr_interface_definition(self, entry: AttrInterface):  # noqa: ANN201
        if entry.anonymous:
            return

        yield f"""nb::class_<Protocol<{entry.type.cpp}>>(m, "{entry.type.name}", nb::sig("class {entry.type.name}(Protocol)"))"""
        yield from indent_all(self.interface_definition(entry), 1)

    @registry.generator(TypeInterface)
    def type_interface_definition(self, entry: TypeInterface):  # noqa: ANN201
        if entry.anonymous:
            return

        yield f"""nb::class_<Protocol<{entry.type.cpp}>>(m, "{entry.type.name}", nb::sig("class {entry.type.name}(Protocol)"))"""
        yield from indent_all(self.interface_definition(entry), 1)

    @registry.generator(OpInterface)
    def op_interface_definition(self, entry: OpInterface):  # noqa: ANN201
        if entry.anonymous:
            return

        yield f"""nb::class_<Protocol<{entry.type.cpp}>>(m, "{entry.type.name}", nb::sig("class {entry.type.name}(Protocol)"))"""
        yield from indent_all(self.interface_definition(entry), 1)

    def interface_definition(self, entry: Interface):  # noqa: ANN201
        properties, methods = properties_and_methods(entry)

        # generate all properties
        for name, prop in properties.items():
            name = python_variable_name(name)
            if self.missing_typeinfo(prop.getter.return_type):
                continue
            if prop.setter:
                yield f""".def_prop_rw("{name}", {not_implemented(prop.getter)}, {not_implemented(prop.setter)})"""
            else:
                yield f""".def_prop_ro("{name}", {not_implemented(prop.getter)})"""

        for method in methods:
            signature = method.signature(entry)
            if any(self.missing_typeinfo(t) for t in signature.args.types):
                continue
            yield f""".def("{python_variable_name(method.function_name)}", {not_implemented(signature)})"""

        yield f""".doc() = R"nbdoc({entry[("description", str)]})nbdoc";"""

    def definition(self, entry: DefMixin):  # noqa: ANN201
        properties, methods = properties_and_methods(entry)

        # generate all properties
        for name, prop in properties.items():
            name = python_variable_name(name)
            getter = prop.getter.signature(entry)
            if self.missing_typeinfo(prop.getter.return_type):
                yield f""".def_prop_ro("{name}", []() {{ throw nb::type_error("No typeinfo available for '{prop.getter.return_type.cpp}'"); }})"""
                continue
            if prop.setter:
                setter = prop.setter.signature(entry)
                yield f""".def_prop_rw("{name}", optional_pimpl({getter.cpp_reference}), {setter.cpp_reference})"""
            else:
                yield f""".def_prop_ro("{name}", optional_pimpl({getter.cpp_reference}))"""

        for method in methods:
            yield from self.interface_method(method, entry)

        yield f""".doc() = R"nbdoc({entry[("description", str)]})nbdoc";"""
        yield f"::M::Graph::Python::registerTypeID(::mlir::TypeID::get<{entry.type.cpp}>(), &typeid({entry.type.cpp}));"
        for builder in entry.builders:
            if len(builder.args) == 1:
                (type,) = builder.args.types
                yield f"nb::implicitly_convertible<{type.cpp}, {entry.type.cpp}>();"
