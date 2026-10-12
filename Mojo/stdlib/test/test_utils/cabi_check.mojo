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
"""Checks Mojo mirrors of C declarations against the platform headers.

A suite is a manifest package defining `CABI_INCLUDES`,
`CABI_STRUCTS`, `CABI_TYPEDEFS`, and `CABI_CONSTANTS`, built by the
`mojo_cabi_test` macro (`Mojo/stdlib/test/cabi_test.bzl`), which:

1. Runs a generator that prints `emit_cabi_checks_for` of the manifest:
   a C file that contains only names.
2. Compiles that file against the real platform headers, so every size,
   offset, alignment, and constant value comes from the C compiler.
3. Runs a test that calls `assert_cabi_checks_for` on the manifest,
   comparing the mirrors against the generated `mojo_check_cabi_*`
   reporters. A failure means the mirror no longer matches the C ABI.
"""

from std.ffi import external_call
from std.reflection.type_info import _unqualified_type_name
from std.sys import align_of, size_of
from std.testing import assert_equal


comptime _MACROS: StaticString = """\
#define MOJO_CHECK_CABI_LAYOUT(name, type)                                    \\
  int64_t mojo_check_cabi_sizeof_##name(void) {                               \\
    return (int64_t)sizeof(type);                                             \\
  }                                                                           \\
  int64_t mojo_check_cabi_alignof_##name(void) {                              \\
    return (int64_t)_Alignof(type);                                           \\
  }

#define MOJO_CHECK_CABI_FIELD(name, type, field)                              \\
  int64_t mojo_check_cabi_offsetof_##name##_##field(void) {                   \\
    return (int64_t)offsetof(type, field);                                    \\
  }                                                                           \\
  int64_t mojo_check_cabi_fieldsizeof_##name##_##field(void) {                \\
    return (int64_t)sizeof(((type *)0)->field);                               \\
  }

// `decl` declares a pointer named `ptr` at the field's expected C type
// (for example `uint16_t *ptr` or `unsigned char (*ptr)[16]`). Assigning
// `&field` to it only compiles if the real field is pointer-compatible
// with that type, so a mismatched field type fails the build.
#define MOJO_CHECK_CABI_FIELD_TYPE(name, type, field, decl)                   \\
  void mojo_check_cabi_fieldtype_##name##_##field(void* input) {              \\
    decl;                                                                     \\
    ptr = &(((type *)input)->field);                                          \\
    (void)ptr;                                                                \\
  }

// An expression's type is signed exactly when -1 cast to it stays negative.
#define MOJO_CHECK_CABI_IS_SIGNED(e) ((__typeof__(e))-1 < 0)

#define MOJO_CHECK_CABI_CONST(name)                                           \\
  int64_t mojo_check_cabi_const_##name(void) { return (int64_t)(name); }      \\
  int64_t mojo_check_cabi_constsizeof_##name(void) {                          \\
    return (int64_t)sizeof(name);                                             \\
  }                                                                           \\
  int64_t mojo_check_cabi_constsigned_##name(void) {                          \\
    return (int64_t)MOJO_CHECK_CABI_IS_SIGNED(name);                          \\
  }"""


def preamble(headers: ImmSpan[StaticString, _]) -> String:
    """Returns the generated file's includes and macros.

    Args:
        headers: The headers to include with the preamble.

    Returns:
        The preamble's C source.
    """
    var out = String("// Generated ABI reference; do not edit.\n\n")
    for header in headers:
        out += String(t"#include {header}\n")
    out += "#include <stddef.h>\n"
    out += "#include <stdint.h>\n\n"
    out += _MACROS
    return out^


trait _SIMDMarker:
    """Marks `SIMD[dtype, length]`."""

    pass


__extension SIMD(_SIMDMarker):
    pass


trait _ArrayMarker:
    """Marks `Array[T, length]`, exposing its generic parameters."""

    comptime _cabi_element_type: AnyType
    comptime _cabi_length: Int


__extension Array(_ArrayMarker):
    pass


def _c_scalar_name[T: AnyType]() -> Optional[StaticString]:
    """Returns the `<stdint.h>` name for a scalar type, or None if T isn't one.
    """
    comptime if T == Int8:
        return StaticString("int8_t")
    elif T == UInt8:
        return StaticString("uint8_t")
    elif T == Int16:
        return StaticString("int16_t")
    elif T == UInt16:
        return StaticString("uint16_t")
    elif T == Int32:
        return StaticString("int32_t")
    elif T == UInt32:
        return StaticString("uint32_t")
    elif T == Int64:
        return StaticString("int64_t")
    elif T == UInt64:
        return StaticString("uint64_t")
    else:
        return None


def _c_field_decl[T: AnyType]() raises -> Optional[String]:
    """Returns a declarator for a pointer to a field's expected C type.

    Parameters:
        T: The Mojo mirror field's type.

    Returns:
        A declarator such as `uint16_t *ptr` or `struct in_addr *ptr`, or
        `None` for a 1-byte integer or an array of them, which gets no type
        check.
    """
    comptime TName = reflect[T].name()

    comptime if conforms_to(T, _SIMDMarker):
        # TODO(https://github.com/modular/modular/issues/7262)
        # Enable field-type checks for 8-bit types
        comptime if T == UInt8 or T == Int8:
            return None
        __match _c_scalar_name[T]():
            case .Some(scalar_name):
                return String(t"{scalar_name} *ptr")
            case .None:
                raise Error(t"unexpected SIMD type: {TName}")

    comptime if conforms_to(T, _ArrayMarker):
        comptime ElementType = T._cabi_element_type
        # TODO(https://github.com/modular/modular/issues/7262)
        # Enable field-type checks for 8-bit types
        comptime if ElementType == UInt8 or ElementType == Int8:
            return None
        __match _c_scalar_name[ElementType]():
            case .Some(scalar_name):
                return String(t"{scalar_name} (*ptr)[{T._cabi_length}]")
            case .None:
                raise Error(t"unsupported array type: {TName}")

    # Falls back to a mirrored struct, named verbatim after its C
    # counterpart (the same convention `struct_checks` uses).
    return String(t"struct {_unqualified_type_name[T]()} *ptr")


def struct_checks[T: AnyType]() raises -> String:
    """Returns the reporters for a struct and all of its fields.

    Parameters:
        T: The Mojo mirror struct.

    Returns:
        One `MOJO_CHECK_CABI_LAYOUT` line, one `MOJO_CHECK_CABI_FIELD` line
        per field, and a `MOJO_CHECK_CABI_FIELD_TYPE` line per field that
        isn't a 1-byte integer or an array of them. The latter checks the
        field's C type by assigning its address to a pointer declared at the
        expected type, so an incompatible field type fails to compile rather
        than failing an assertion.
    """
    comptime name = _unqualified_type_name[T]()
    var out = String(t"MOJO_CHECK_CABI_LAYOUT({name}, struct {name})")
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    comptime for i in range(names.length):
        comptime field = names[i]
        out += String(
            t"\nMOJO_CHECK_CABI_FIELD({name}, struct {name}, {field})"
        )
        __match _c_field_decl[types[i]]():
            case .Some(decl):
                out += String(
                    t"\nMOJO_CHECK_CABI_FIELD_TYPE({name}, struct {name},"
                    t" {field}, {decl})"
                )
            case .None:
                pass
    return out^


trait AbiTypedefLike:
    """A C type alias to check: its C name paired with the Mojo mirror."""

    comptime name: StaticString
    """The alias's name as spelled in C."""

    comptime Type: AnyType
    """The Mojo mirror of the alias."""


struct AbiTypedef[c_name: StaticString, T: AnyType](AbiTypedefLike):
    """A C type alias to check, as a `TypeList` entry.

    Parameters:
        c_name: The alias's name as spelled in C.
        T: The Mojo mirror of the alias.
    """

    comptime name = Self.c_name
    """The alias's name as spelled in C."""

    comptime Type = Self.T
    """The Mojo mirror of the alias."""


def typedef_checks[T: AbiTypedefLike]() -> String:
    """Returns the size and alignment reporters for a C type alias.

    Parameters:
        T: The typedef to report.

    Returns:
        The alias's `MOJO_CHECK_CABI_LAYOUT` line.
    """
    comptime name = T.name
    return String(t"MOJO_CHECK_CABI_LAYOUT({name}, {name})")


def constant_checks(name: StaticString) -> String:
    """Returns the value, size, and signedness reporters for a constant.

    Args:
        name: The constant's name as spelled in C.

    Returns:
        The constant's `MOJO_CHECK_CABI_CONST` line.
    """
    return String(t"MOJO_CHECK_CABI_CONST({name})")


def emit_cabi_checks_for(
    *,
    includes: ImmSpan[StaticString, _],
    structs: TypeList[Trait=AnyType, ...],
    typedefs: TypeList[Trait=AbiTypedefLike, ...],
    constants: ImmSpan[AbiConstant, _],
) raises -> String:
    """Returns a complete C ABI reference file for a set of mirrors.

    Args:
        includes: The headers that declare the C side, each spelled as it
            is included (for example `<netinet/in.h>`).
        structs: The struct mirrors to check.
        typedefs: The type aliases to check.
        constants: The constants to check.

    Returns:
        The preamble followed by the checks for every declaration.
    """
    var out = preamble(includes)
    comptime for i in range(structs.length):
        out += "\n\n" + struct_checks[structs[i]]()
    comptime for i in range(typedefs.length):
        out += "\n\n" + typedef_checks[typedefs[i]]()
    for constant in constants:
        out += "\n\n" + constant_checks(constant.name)
    return out^


def _check[sym: String](mojo_value: Int64) raises:
    assert_equal(mojo_value, external_call[sym, Int64](), sym)


struct AbiConstant(TrivialRegisterPassable):
    """A constant to check: its C name paired with the mirror's value."""

    var name: StaticString
    """The constant's name."""
    var value: Int64
    """The Mojo mirror's value for the constant."""
    var size: Int64
    """The size of the mirror value's type, in bytes."""
    var signed: Bool
    """Whether the mirror value's type is signed."""

    def __init__[
        dtype: DType, //
    ](out self, name: StaticString, value: Scalar[dtype]):
        """Records a constant and its type from the mirror's typed value.

        Parameters:
            dtype: The mirror value's type, inferred from `value`.

        Args:
            name: The constant's name as spelled in C.
            value: The mirror's value, at the mirror's declared type.
        """
        self.name = name
        self.value = value.cast[.int64]()
        self.size = Int64(size_of[Scalar[dtype]]())
        self.signed = dtype.is_signed()


def assert_cabi_constant[entry: AbiConstant]() raises:
    """Compares a constant's Mojo-side value and type against C.

    Checks the value, size, and signedness against the
    `mojo_check_cabi_const*_<name>` reporters.

    Parameters:
        entry: The constant to check.

    Raises:
        When the value, size, or signedness differs.
    """
    _check["mojo_check_cabi_const_" + entry.name](entry.value)
    _check["mojo_check_cabi_constsizeof_" + entry.name](entry.size)
    _check["mojo_check_cabi_constsigned_" + entry.name](
        Int64(1) if entry.signed else 0
    )


def assert_cabi_layout[name: StaticString, T: AnyType]() raises:
    """Compares a mirror type's size and alignment against C.

    Parameters:
        name: The type's name as spelled in C; the reporters called
            are `mojo_check_cabi_sizeof_<name>` and
            `mojo_check_cabi_alignof_<name>`.
        T: The Mojo mirror type.

    Raises:
        When the size or alignment differs.
    """
    _check["mojo_check_cabi_sizeof_" + name](Int64(size_of[T]()))
    _check["mojo_check_cabi_alignof_" + name](Int64(align_of[T]()))


def assert_cabi_layout[T: AbiTypedefLike]() raises:
    """Compares a typedef's size and alignment against C.

    Parameters:
        T: The typedef to check.

    Raises:
        When the size or alignment differs.
    """
    assert_cabi_layout[T.name, T.Type]()


def assert_cabi_struct[T: AnyType]() raises:
    """Compares a mirror struct's layout against C, field by field.

    Checks the size and alignment of `T`, but also walks through the
    fields of `T` and checks its existence, size, alignment and offset.

    For example, calling this method with the struct:
    ```mojo
    def Foo:
        var fieldA: Array[UInt8, 2]
        var fieldB: Int64
    ```
    will produce the following checks:

    ```
    // check the struct itself
    mojo_check_cabi_sizeof_Foo(...)
    mojo_check_cabi_alignof_Foo(...)

    // check each field of the struct
    mojo_check_cabi_offsetof_Foo_fieldA(...)
    mojo_check_cabi_fieldsizeof_Foo_fieldA(...)

    mojo_check_cabi_offsetof_Foo_fieldB(...)
    mojo_check_cabi_fieldsizeof_Foo_fieldB(...)
    ```

    Parameters:
        T: The Mojo mirror struct.

    Raises:
        When the size, alignment, or any field's offset or size differs.
    """
    comptime name = _unqualified_type_name[T]()
    assert_cabi_layout[name, T]()
    comptime r = reflect[T]
    comptime names = r.field_names()
    comptime types = r.field_types()
    comptime for i in range(names.length):
        _check["mojo_check_cabi_offsetof_" + name + "_" + names[i]](
            Int64(r.field_offset[index=i]())
        )
        _check["mojo_check_cabi_fieldsizeof_" + name + "_" + names[i]](
            Int64(size_of[types[i]]())
        )


def assert_cabi_checks_for[
    structs: TypeList[Trait=AnyType, ...],
    typedefs: TypeList[Trait=AbiTypedefLike, ...],
    constants: ImmSpan[AbiConstant, _],
]() raises:
    """Compares every declaration in a set of mirrors against C.

    The test-side twin of `emit_cabi_checks_for`: the reporters it calls
    are the ones that function generates for the same declarations.

    Parameters:
        structs: The struct mirrors to check.
        typedefs: The type aliases to check.
        constants: The constants to check.

    Raises:
        When any size, alignment, field offset or size, or constant
        value, size, or signedness differs.
    """
    comptime for i in range(structs.length):
        assert_cabi_struct[structs[i]]()
    comptime for i in range(typedefs.length):
        assert_cabi_layout[typedefs[i]]()
    comptime for constant in constants:
        assert_cabi_constant[constant]()
