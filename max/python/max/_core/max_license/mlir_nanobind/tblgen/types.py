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
"""Type helpers for working with TableGen generated json."""

from __future__ import annotations

import enum
import keyword
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Final, TypeVar, overload

from .casing import camel_case, snake_case, upper_camel_case

T = TypeVar("T")
D = TypeVar("D")


def as_t(v: Any, T: type[T]) -> T:
    assert isinstance(v, T), f"Expected {T}, got {v}"
    return v


@dataclass
class Table:
    data: dict  # type: ignore[type-arg]

    def __getitem__(self, name) -> Entry:  # noqa: ANN001
        data = self.data[name]
        classes = [Entry] + [
            Entry.TYPES[classname]
            for classname in data["!superclasses"]
            if classname in Entry.TYPES
        ]
        return classes[-1](self, name, data)

    def get(self, name, default=None) -> Entry | None:  # noqa: ANN001
        try:
            return self[name]
        except KeyError:
            return default

    @overload
    def resolve(self, data: list[T]) -> list[Entry | T]: ...

    @overload
    def resolve(self, data: dict) -> Entry | ArgList: ...  # type: ignore[type-arg]

    @overload
    def resolve(self, data: T) -> T: ...

    def resolve(self, data):
        if isinstance(data, dict):
            if data.get("kind") == "def":
                return self[data["def"]]
            elif data.get("kind") == "dag":
                return ArgList(
                    Arg(self.resolve(arg), name) for arg, name in data["args"]
                )
        elif isinstance(data, list):
            return [self.resolve(item) for item in data]
        elif isinstance(data, tuple):
            return ArgList(data)
        return data

    def __iter__(self):
        for name in self.data:
            if name.startswith("!"):
                continue
            yield self[name]


def _format_field(field):  # noqa: ANN001, ANN202
    if isinstance(field, Entry):
        return field.format_simple()
    elif isinstance(field, list):
        return f"[{', '.join(_format_field(i) for i in field)}]"
    else:
        return repr(field)


# Defining this as an enum allows it to be used in a `Literal` type
class Missing(enum.Enum):
    MISSING = "MISSING"


MISSING: Final = Missing.MISSING


def python_variable_name(name: str) -> str:
    snake_name = snake_case(name.replace("-", "_"))
    u = "_" if keyword.iskeyword(snake_name) else ""
    return f"{snake_name}{u}"


def python_type_name(name: str) -> str:
    return upper_camel_case(name)


@dataclass
class NameSpace:
    names: list[str]

    @property
    def cpp(self) -> str:
        return "::".join(self.names)

    @property
    def python(self) -> str:
        return ".".join(python_variable_name(name) for name in self.names)

    def __bool__(self) -> bool:
        return bool(self.names)


@dataclass
class Name:
    name: str
    namespace: NameSpace

    @classmethod
    def from_entry(cls, entry: Entry):  # noqa: ANN206
        return entry.type

    @classmethod
    def from_cpp_qualified_name(cls, name: str):  # noqa: ANN206
        *namespace, name = name.removeprefix("::").split("::")
        return cls(name, NameSpace(namespace))

    @property
    def cpp(self) -> str:
        return "::".join((*self.namespace.names, self.name))

    @property
    def python(self) -> str:
        return (
            f"{self.namespace.python}.{self.name}"
            if self.namespace
            else self.name
        )

    def __repr__(self) -> str:
        return f'Name("{self.cpp}")'


@dataclass
class Signature:
    entry: Entry
    method: InterfaceMethod | AttrOrTypeBuilder | OpBuilder | Function
    function_name: Name

    @property
    def args(self) -> ArgList:
        return self.method.args

    @property
    def return_type(self) -> Name:
        return self.method.return_type or self.entry.type

    @property
    def cpp_reference(self) -> str:
        return f"&{self.function_name.cpp}"

    @property
    def cpp_reference_overload_cast(self) -> str:
        types = ", ".join(type.cpp for type in self.args.types)
        return f"nb::overload_cast<{types}>({self.cpp_reference})"

    @property
    def cpp(self) -> str:
        function_pointer_scope = (
            "" if self.method.is_static else f"{self.entry.type.cpp}::"
        )
        arg_types = ", ".join(type.cpp for type in self.args.types)
        return (
            f"{self.return_type.cpp} ({function_pointer_scope}*)({arg_types})"
        )


@dataclass
class Parameter:
    """MLIR doesn't have an official notion of a "parameter", but fields on Attributes, Types, and Ops
    are generally annotated as "parameters"."""

    entry: Entry
    name: str
    getter: InterfaceMethod
    setter: InterfaceMethod | None = None

    @classmethod
    def from_parameter_field(
        cls, entry: Entry, parameter: Entry | str, name: str
    ) -> Parameter:
        upper_name = upper_camel_case(name)
        getter = InterfaceMethod(
            entry.table,
            name,
            dict(
                arguments=ArgList(),
                name=f"get{upper_name}",
                returnType=parameter.type.cpp
                if isinstance(parameter, Entry)
                else parameter,
            ),
        )
        op_attr = isinstance(entry, Op) and isinstance(parameter, Attr)
        if op_attr:
            setter = InterfaceMethod(
                entry.table,
                name,
                dict(
                    arguments=ArgList([Arg(parameter, name)]),
                    name=f"set{upper_name}Attr",
                    returnType="void",
                ),
            )
        else:
            setter = None
        return cls(entry, name, getter, setter)


@dataclass(frozen=True)
class Entry:
    table: Table
    name: str
    data: dict  # type: ignore[type-arg]

    TYPES: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls) -> None:
        cls.TYPES[cls.__name__] = cls

    @overload
    def __getitem__(self, name_type: tuple[str, type[T]]) -> T: ...

    @overload
    def __getitem__(self, name_type: str) -> Any: ...

    def __getitem__(self, name_type):
        if isinstance(name_type, tuple):
            name, type = name_type
        else:
            name, type = name_type, object

        try:
            field = self.data[name]
        except KeyError:
            raise ValueError(f"{name} not found in {self}")  # noqa: B904
        return as_t(self.table.resolve(field), type)

    @overload
    def get(self, field: str, *, type: type[T]) -> T: ...

    @overload
    def get(self, field: str, default: D | None = None) -> D | object: ...

    @overload
    def get(self, field: str, default: D | None, type: type[T]) -> D | T: ...

    def get(
        self,
        field: str,
        default: Any | None = None,
        type: type[Any] = object,
    ) -> Any:
        return self.resolve(self.data.get(field, default))

    @overload
    def one_of(
        self,
        fields: Iterable[str],
        default: Missing = MISSING,
    ) -> object: ...

    @overload
    def one_of(self, fields: Iterable[str], default: D) -> object | D: ...

    @overload
    def one_of(self, fields: Iterable[str], *, type: type[T]) -> T: ...

    @overload
    def one_of(
        self, fields: Iterable[str], default: D, type: type[T]
    ) -> T | D: ...

    def one_of(self, fields, default=MISSING, type=object):
        for field_name in fields:
            if (field := self.get(field_name, type=type)) is not None:
                return field
        if default is not MISSING:
            return default
        raise ValueError(f"None of {fields} found in {self}")

    def resolve(self, data):  # noqa: ANN001, ANN201
        return self.table.resolve(data)

    @property
    def dialect(self) -> Dialect | None:
        if "_" in self.name and not self.name.startswith("_"):
            inferred_dialect_name = self.name.split("_", 1)[0]
            inferred_dialect = self.table.get(f"{inferred_dialect_name}Dialect")
        else:
            inferred_dialect = None
        assert isinstance(inferred_dialect, Dialect | type(None))
        return self.one_of(
            ("dialect", "opDialect"), default=inferred_dialect, type=Dialect
        )

    @property
    def anonymous(self):  # noqa: ANN201
        return self["!anonymous", bool]

    @property
    def traits(self):  # noqa: ANN201
        # Can't use type=Iterable[Trait] due to MyPy restriction:
        #     error: Only concrete class can be given where
        #     "type[Iterable[Trait]]" is expected  [type-abstract]
        yield from self.get("traits", [], list[Trait])

    @property
    def methods(self) -> Iterable[InterfaceMethod]:
        for interface in self.base_interfaces:
            yield from interface.methods
        yield from self.get("methods", [], list[InterfaceMethod])

    @property
    def base_interfaces(self) -> Iterable[Trait]:
        yield from self.get("baseInterfaces", [], list[Trait])

    def format_simple(self) -> str:
        return f"{type(self).__name__}[{self.name}]"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:  # noqa: ANN001
        return self.table is other.table and self.name == other.name

    def __repr__(self) -> str:
        fieldlist = "\n".join(
            f"  {key} = {_format_field(self[key])}" for key in self.data
        )
        return f"{self.format_simple()}(\n{fieldlist}\n)"

    @property
    def type(self) -> Name:
        qualified_name = self.one_of(
            (
                "cppType",
                "cppClassName",
                "cppInterfaceName",
                "storageType",
                "type",
                "returnType",
            ),
            type=str,
        )
        return Name.from_cpp_qualified_name(qualified_name)


@dataclass(frozen=True)
class Arg:
    entry: Entry | str
    name: str

    @property
    def type(self) -> Name:
        if isinstance(self.entry, str):
            return Name.from_cpp_qualified_name(self.entry)
        try:
            return self.entry.type
        except Exception:
            # Occasionally fails for some weird types like Res
            return Name.from_cpp_qualified_name("::mlir::Type")

    def __repr__(self):
        entry = (
            self.entry
            if isinstance(self.entry, str)
            else self.entry.format_simple()
        )
        return f"{type(self).__qualname__}({self.name}, {entry})"


class OpArg(Arg):
    @classmethod
    def from_arg(cls, arg: Arg) -> OpArg:
        return cls(arg.entry, arg.name)

    @property
    def type(self) -> Name:
        if isinstance(self.entry, TypeConstraint):
            return Name.from_cpp_qualified_name(
                f"::mlir::TypedValue<{super().type.cpp}>"
            )
        elif isinstance(self.entry, Variadic):
            return Name.from_cpp_qualified_name("::mlir::ValueRange")
        return super().type


class ArgList(tuple[Arg | OpArg]):
    @property
    def names(self):  # noqa: ANN201
        for arg in self:
            yield arg.name

    @property
    def types(self):  # noqa: ANN201
        for arg in self:
            yield arg.type

    def cpp_signature(self) -> str:
        return ", ".join(f"{arg.type.cpp} {arg.name}" for arg in self)

    def format_simple(self):  # noqa: ANN201
        return repr(self)

    def __add__(self, other: tuple):  # type: ignore[type-arg]
        return ArgList(super().__add__(other))

    def __repr__(self):
        return f"{type(self).__qualname__}{super().__repr__()}"


class Dialect(Entry):
    def __eq__(self, other):  # noqa: ANN001
        return isinstance(other, Dialect) and self.name == other.name


class Constraint(Entry):
    pass


class TypeConstraint(Entry):
    pass


class Type(TypeConstraint):
    pass


class Attr(Constraint):
    pass


class DefaultValuedAttr(Attr):
    @property
    def default(self) -> str:
        return self["defaultValue", str]


class EnumAttrInfo(Attr):
    # These types masquerade as their cpp types a lot for some reason?
    # But aren't actually implicitly convertible to/from
    @property
    def type(self) -> Name:
        namespace = self["cppNamespace", str]
        name = self["specializedAttrClassName", str]
        return Name.from_cpp_qualified_name(f"{namespace}::{name}")

    @property
    def enum_type(self) -> Name:
        return super().type


@dataclass
class BuilderCall:
    table: Table
    call: str

    @property
    def args(self) -> ArgList:
        return ArgList()


class DefMixin(Entry):
    @property
    def default_builders(self):  # noqa: ANN201
        ctx = ArgList([Arg("::mlir::MLIRContext*", "ctx")])
        name = Name.from_cpp_qualified_name(f"{self.type.cpp}::get")
        yield Function(name, ctx + self.arguments, None, infer_context=False)

    @property
    def arguments(self) -> ArgList:
        return self["parameters", ArgList]

    @property
    def builders(self):  # noqa: ANN201
        if builder_call := self.get("builderCall", None, str):
            yield BuilderCall(self.table, builder_call)
        elif not self.get("skipDefaultBuilders"):
            yield from self.default_builders
        yield from self.get("builders", type=list) or ()

    @property
    def parameters(self):  # noqa: ANN201
        if not self.get("genAccessors", True):
            return
        for arg in self.arguments:
            if isinstance(arg.entry, Property):
                continue  # Properties work differently
            yield Parameter.from_parameter_field(self, arg.entry, arg.name)


class AttrOrTypeDef(Constraint, DefMixin):
    pass


class AttrDef(Attr, AttrOrTypeDef):
    pass


class TypeDef(Type, AttrOrTypeDef):
    pass


class Op(DefMixin):
    @property
    def default_builders(self):  # noqa: ANN201
        name = Name.from_cpp_qualified_name(f"{self.type.cpp}::create")
        yield Function(name, self.results + self.arguments, self.type)
        if self.has_infer_type_op_adaptor:
            yield Function(name, self.arguments, self.type)

    @property
    def has_infer_type_op_adaptor(self) -> bool:
        """Whether the op infers its result types, with either adaptor trait."""
        traits = [
            trait
            for name in (
                "InferTypeOpAdaptor",
                "InferTypeOpAdaptorWithIsCompatible",
            )
            if (trait := self.table.get(name, None)) is not None
        ]
        return any(trait in self.traits for trait in traits)

    @property
    def arguments(self) -> ArgList:
        return ArgList(
            OpArg.from_arg(arg) for arg in self["arguments", ArgList]
        )

    @property
    def results(self) -> ArgList:
        return self["results", ArgList]

    @property
    def type(self) -> Name:
        # The first `_` in the op's TableGen def name is treated as separating the
        # dialect prefix and the op class name. The dialect prefix will be ignored if
        # not empty. Otherwise, if def name starts with a `_`, the `_` is considered
        # as part of the class name.
        name = self["!name", str]
        name = name.split("_", 1)[1] if "_" in name else name
        namespace = NameSpace(
            self["cppNamespace", str].removeprefix("::").split("::")
        )
        return Name(name, namespace)


class AttrOrTypeBuilder(Entry):
    is_static = True

    @property
    def args(self):  # noqa: ANN201
        args = self["dagParams", ArgList]
        ctx = ArgList([Arg("::mlir::MLIRContext*", "context")])
        return args if self.infer_context else (ctx + args)

    @property
    def infer_context(self) -> bool:
        return self["hasInferredContextParam"]

    def signature(self, entry: Entry):  # noqa: ANN201
        return Signature(
            entry, self, Name.from_cpp_qualified_name(f"{entry.type.cpp}::get")
        )

    @property
    def return_type(self) -> Name | None:
        return_type = self.get("returnType", None, str)
        return (
            Name.from_cpp_qualified_name(return_type) if return_type else None
        )


class AttrBuilder(AttrOrTypeBuilder):
    pass


class TypeBuilder(AttrOrTypeBuilder):
    pass


@dataclass
class Function:
    function_name: Name
    args: ArgList
    return_type: Name | None
    is_static: bool = True
    infer_context: bool = True

    def signature(self, entry: Entry) -> Signature:
        return Signature(entry, self, self.function_name)


class OpBuilder(Entry):
    is_static = True

    @property
    def args(self) -> ArgList:
        dag_params = self["dagParams", ArgList]
        return ArgList(OpArg.from_arg(arg) for arg in dag_params)

    @property
    def return_type(self) -> Name | None:
        return_type = self.get("returnType", None, str)
        return (
            Name.from_cpp_qualified_name(return_type) if return_type else None
        )


class Property(Entry):
    pass


class Variadic(Entry):
    @property
    def type(self):  # noqa: ANN201
        return Name.from_cpp_qualified_name("mlir::TypeRange")


class Trait(Entry):
    pass


class DeclareInterfaceMethods(Entry):
    pass


class DeclareAttrInterfaceMethods(Entry):
    pass


class DeclareTypeInterfaceMethods(Entry):
    pass


class Interface(Entry):
    parameters: list[Parameter] = []


class AttrInterface(Attr, Interface):
    pass


class TypeInterface(Type, Interface):
    pass


class OpInterface(Interface):
    @property
    def arguments(self):  # noqa: ANN201
        return self.get("arguments", default=ArgList(), type=ArgList)

    @property
    def type(self):  # noqa: ANN201
        namespace = self["cppNamespace", str]
        type_name = self["cppInterfaceName", str]
        return Name.from_cpp_qualified_name(f"{namespace}::{type_name}")


class InterfaceMethod(Entry):
    is_static = False

    def __eq__(self, other):  # noqa: ANN001
        return (
            self.function_name == other.function_name
            and self.args == other.args
        )

    def __hash__(self):
        return hash((self.name, self.args))

    @property
    def args(self) -> ArgList:
        return self["arguments", ArgList]

    def signature(self, entry: Entry) -> Signature:
        return Signature(
            entry,
            self,
            Name.from_cpp_qualified_name(
                f"{entry.type.cpp}::{self.function_name}"
            ),
        )

    @property
    def function_name(self) -> str:
        return self["name", str]

    @property
    def cpp_reference(self) -> str:
        return f"&{Name.from_cpp_qualified_name(self.function_name).cpp}"

    @property
    def return_type(self) -> Name:
        return Name.from_cpp_qualified_name(self["returnType", str])

    @property
    def getter_for(self) -> str | None:
        if self.args:
            return None
        if match := re.match("^(?:get|is)([A-Z].*$)", self.function_name):
            return camel_case(match.group(1))
        return None

    @property
    def setter_for(self) -> str | None:
        if len(self.args) > 1:  # bool setters have 0 args
            return None
        if match := re.match("^set([A-Z].*$)", self.function_name):
            return camel_case(match.group(1))
        return None


class StaticInterfaceMethod(InterfaceMethod):
    is_static = True


class Region(Entry):
    @property
    def type(self) -> Name:
        return Name.from_cpp_qualified_name("::mlir::Region")


class CArg(Entry):
    @property
    def default(self) -> str:
        return self["defaultValue", str]


class AttrOrTypeParameter(Entry):
    pass


class Pass(Entry):
    @property
    def options(self) -> list[Option]:
        return self["options"]

    @property
    def options_type(self) -> str:
        return f"{self.name}Options"

    @property
    def constructor(self) -> str | None:
        return self["constructor", str] or None


class Option(Entry):
    @property
    def cpp_name(self) -> str:
        return self["cppName", str]

    @property
    def default(self) -> str:
        return self["defaultValue", str]


class ListOption(Option):
    @property
    def type(self) -> Name:
        return Name.from_cpp_qualified_name(
            f"""::llvm::SmallVector<{super().type.cpp}>"""
        )
