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
# test_reflection.mojo
# Tests for reflection.mdx code examples.
# Skip: show_type (print-only), describe/explore_type (print-only,
#        uses __mlir_type), show_layout (print-only),
#        require/call_location (raises on purpose),
#        source_location log (print-only, path-dependent).
from std.testing import assert_equal, assert_true
from std.reflection import (
    source_location,
    call_location,
    get_function_name,
    get_linkage_name,
    SourceLocation,
)


# --- Inspecting a type: name, field_count, field_names, field_types ---


@fieldwise_init
struct Point:
    var x: Int
    var y: Float64


def test_reflect_name() raises:
    """Returns the compiler-resolved type name."""
    comptime name = reflect[Point].name()
    assert_true(name.find("Point") >= 0)


def test_reflect_field_count() raises:
    """Returns the number of fields."""
    assert_equal(reflect[Point].field_count(), 2)


def test_reflect_field_names() raises:
    """Returns field names in order."""
    comptime names = reflect[Point].field_names()
    assert_equal(String(names[0]), "x")
    assert_equal(String(names[1]), "y")


def test_reflect_field_types() raises:
    """Returns field types iterable with reflect."""
    comptime types = reflect[Point].field_types()
    comptime first_type_name = reflect[types[0]].name()
    assert_equal(first_type_name, "Int")


# --- base_name ---


def test_base_name_parameterized() raises:
    """Strips parameters and module path."""
    assert_equal(reflect[List[Int]].base_name(), "List")
    assert_equal(reflect[Dict[String, Int]].base_name(), "Dict")


def test_base_name_simple() raises:
    """Returns simple name for non-parameterized type."""
    assert_equal(reflect[Int].base_name(), "Int")


# --- is_struct ---


def test_is_struct_on_struct() raises:
    """Returns True for Mojo struct types."""
    assert_true(reflect[Point].is_struct())


def test_is_struct_on_int() raises:
    """Returns True for Int, a struct wrapping an MLIR primitive."""
    assert_true(reflect[Int].is_struct())


# --- Example: Finding what changed (diff_fields) ---


@fieldwise_init
struct Config(Equatable):
    var host: String
    var port: Int
    var verbose: Bool
    var timeout: Float64


def diff_fields[T: AnyType](a: T, b: T) -> List[String]:
    """From the page: returns names of fields that differ."""
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    var diffs = List[String]()

    comptime for idx in range(reflect[T].field_count()):
        comptime if not conforms_to(types[idx], Equatable):
            continue

        ref a_val = reflect[T].field_ref[idx](a)
        ref b_val = reflect[T].field_ref[idx](b)

        if trait_downcast[Equatable](a_val) != trait_downcast[Equatable](b_val):
            diffs.append(String(names[idx]))

    return diffs^


def test_diff_fields_detects_changes() raises:
    """Returns names of changed fields."""
    var old = Config("localhost", 8080, False, 30.0)
    var new = Config("localhost", 9090, True, 30.0)
    var changes = diff_fields(old, new)
    assert_equal(len(changes), 2)
    assert_equal(changes[0], "port")
    assert_equal(changes[1], "verbose")


def test_diff_fields_identical() raises:
    """Returns empty list for identical instances."""
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("localhost", 8080, False, 30.0)
    var changes = diff_fields(a, b)
    assert_equal(len(changes), 0)


def test_diff_fields_all_different() raises:
    """Returns all field differences."""
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("remote", 9090, True, 60.0)
    var changes = diff_fields(a, b)
    assert_equal(len(changes), 4)


def test_diff_fields_single_field() raises:
    """Returns a single field difference."""
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("localhost", 8080, False, 99.0)
    var changes = diff_fields(a, b)
    assert_equal(len(changes), 1)
    assert_equal(changes[0], "timeout")


# --- conforms_to and trait_downcast ---


def test_conforms_to_positive() raises:
    """Returns True for a conforming type."""
    assert_true(conforms_to(Int, Equatable))


def test_trait_downcast_equality() raises:
    """Enables trait operations on reflected fields."""
    var p1 = Point(x=1, y=2.0)
    var p2 = Point(x=1, y=2.0)
    ref lhs = reflect[Point].field_ref[0](p1)
    ref rhs = reflect[Point].field_ref[0](p2)
    var equal = trait_downcast[Equatable](lhs) == trait_downcast[Equatable](rhs)
    assert_true(equal)


def test_trait_downcast_inequality() raises:
    """Detects differing field values."""
    var p1 = Point(x=1, y=2.0)
    var p2 = Point(x=1, y=9.0)
    ref lhs = reflect[Point].field_ref[1](p1)
    ref rhs = reflect[Point].field_ref[1](p2)
    var equal = trait_downcast[Equatable](lhs) == trait_downcast[Equatable](rhs)
    assert_true(not equal)


struct ConditionalCopyableWrapper[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T

    # Standard initializer
    def __init__(out self, var value: Self.T):
        self.value = value^

    # Copy initializer
    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.value = rebind_var[Self.T](
            trait_downcast[Copyable](copy.value).copy()
        )


# All structs are inherently `ImplicitlyDestructible`
@fieldwise_init
struct NotCopyable(Movable):
    pass


def test_trait_downcast_copy_constructor() raises:
    """Rebind for conditional conformance."""
    var i = ConditionalCopyableWrapper(42)
    var i_copy = ConditionalCopyableWrapper(copy=i)
    assert_equal(i.value, i_copy.value)


# --- field_ref: access and mutation ---


@fieldwise_init
struct Container:
    var id: Int
    var name: String


def test_field_ref_read() raises:
    """Returns a reference to the field."""
    var c = Container(id=42, name="test")
    ref id_ref = reflect[Container].field_ref[0](c)
    assert_equal(id_ref, 42)


def test_field_ref_mutate() raises:
    """Supports mutation through the reference."""
    var c = Container(id=1, name="test")
    reflect[Container].field_ref[0](c) = 99
    assert_equal(c.id, 99)


# --- Example: MakeCopyable trait ---


trait MakeCopyable:
    def copy_to(self, mut other: Self):
        comptime field_count = reflect[Self].field_count()
        comptime field_types = reflect[Self].field_types()

        comptime for idx in range(field_count):
            comptime field_type = field_types[idx]
            comptime if not conforms_to(field_type, Copyable):
                continue

            ref p_value = reflect[Self].field_ref[idx](self)
            trait_downcast[Copyable & ImplicitlyDestructible](
                reflect[Self].field_ref[idx](other)
            ) = trait_downcast[Copyable & ImplicitlyDestructible](
                p_value
            ).copy()


@fieldwise_init
struct MultiType(MakeCopyable, Writable):
    var w: String
    var x: Int
    var y: Bool
    var z: Float64

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z))


def test_copy_to_transfers_values() raises:
    """Copies all Copyable fields from source to target."""
    var original = MultiType("Hello", 1, True, 2.5)
    var target = MultiType("", 0, False, 0.0)
    original.copy_to(target)
    assert_equal(target.w, "Hello")
    assert_equal(target.x, 1)
    assert_equal(target.y, True)
    assert_equal(target.z, 2.5)


def test_copy_to_independent() raises:
    """After copy_to, modifying original doesn't affect target."""
    var original = MultiType("Hello", 1, True, 2.5)
    var target = MultiType("", 0, False, 0.0)
    original.copy_to(target)
    original.x = 999
    assert_equal(target.x, 1)


# --- Accessing fields by name ---


def test_field_type_by_name() raises:
    """Returns a Reflected handle for the field; .T is usable."""
    comptime host_handle = reflect[Config].field_type["host"]
    var default_host: host_handle.T = "localhost"
    assert_equal(default_host, "localhost")


def test_field_index_by_name() raises:
    """Returns the zero-based index for a field name."""
    comptime x_idx = reflect[Point].field_index["x"]()
    comptime y_idx = reflect[Point].field_index["y"]()
    assert_equal(x_idx, 0)
    assert_equal(y_idx, 1)


# --- Field layout ---


struct Padded:
    var a: UInt8
    var b: UInt32
    var c: UInt64


def test_field_offset_first_field() raises:
    """First field is always at offset 0."""
    comptime off = reflect[Padded].field_offset[index=0]()
    assert_equal(off, 0)


def test_field_offset_alignment() raises:
    """Returns the offset of a field, accounting for alignment padding."""
    comptime off_b = reflect[Padded].field_offset[index=1]()
    assert_true(off_b >= 4)


def test_field_offset_by_name() raises:
    """Returns the offset of a field by name, matching the index-based offset.
    """
    comptime by_name = reflect[Point].field_offset[name="y"]()
    comptime by_index = reflect[Point].field_offset[index=1]()
    assert_equal(by_name, by_index)


# --- Capturing types and origins: type_of ---


def test_type_of() raises:
    """Captures the type of an expression."""
    var x = 42
    var y: type_of(x) = 0
    assert_equal(y, 0)
    _ = x


# --- Capturing types and origins: origin_of ---


def first_ref[T: Copyable](ref list: List[T]) -> ref[origin_of(list)] T:
    return list[0]


def test_origin_of_first_ref() raises:
    """Ties the returned reference to the input's lifetime."""
    var l = [1, 2, 3]
    var x = first_ref(l)
    assert_equal(x, 1)


# --- Source locations ---


def test_source_location_line() raises:
    """Returns a positive line number."""
    var loc = source_location()
    assert_true(loc.line() > 0)


def test_source_location_column() raises:
    """Returns a positive column number."""
    var loc = source_location()
    assert_true(loc.column() > 0)


def test_source_location_file() raises:
    """Returns a non-empty file name."""
    var loc = source_location()
    assert_true(loc.file_name())


def test_source_location_prefix() raises:
    """Formats 'At file:line:col: msg'."""
    var loc = source_location()
    var msg = loc.prefix("test message")
    assert_true(msg.find("At ") == 0)
    assert_true(msg.find("test message") >= 0)


@always_inline
def get_caller_line() -> Int:
    return call_location().line()


def test_call_location() raises:
    """Captures the caller's line, not its own."""
    var line = get_caller_line()
    assert_true(line > 0)


# --- Function names ---


def my_named_function():
    pass


def test_get_function_name() raises:
    """Returns the source-level function name."""
    comptime name = get_function_name[my_named_function]()
    assert_equal(name, "my_named_function")


def test_get_linkage_name() raises:
    """Returns a non-empty mangled symbol."""
    comptime linkage = get_linkage_name[my_named_function]()
    assert_true(linkage)


def test_linkage_name_differs_from_source_name() raises:
    """Linkage name includes mangling beyond the source name."""
    comptime source = get_function_name[my_named_function]()
    comptime linkage = get_linkage_name[my_named_function]()
    assert_true(source != linkage)


# --- Iteration pattern: comptime for + conforms_to ---


def count_equatable_fields[T: AnyType]() -> Int:
    comptime types = reflect[T].field_types()
    var count = 0
    comptime for idx in range(reflect[T].field_count()):
        comptime if conforms_to(types[idx], Equatable):
            count += 1
    return count


def test_count_equatable_fields_multi() raises:
    """All four MultiType fields conform to Equatable."""
    assert_equal(count_equatable_fields[MultiType](), 4)


def test_count_equatable_fields_point() raises:
    """Both Point fields conform to Equatable."""
    assert_equal(count_equatable_fields[Point](), 2)


def main() raises:
    # Inspecting a type
    test_reflect_name()
    test_reflect_field_count()
    test_reflect_field_names()
    test_reflect_field_types()

    # base_name
    test_base_name_parameterized()
    test_base_name_simple()

    # is_struct
    test_is_struct_on_struct()
    test_is_struct_on_int()

    # diff_fields
    test_diff_fields_detects_changes()
    test_diff_fields_identical()
    test_diff_fields_all_different()
    test_diff_fields_single_field()

    # conforms_to and trait_downcast
    test_conforms_to_positive()
    test_trait_downcast_equality()
    test_trait_downcast_inequality()
    test_trait_downcast_copy_constructor()

    # field_ref
    test_field_ref_read()
    test_field_ref_mutate()

    # MakeCopyable
    test_copy_to_transfers_values()
    test_copy_to_independent()

    # Accessing fields by name
    test_field_type_by_name()
    test_field_index_by_name()

    # Field layout
    test_field_offset_first_field()
    test_field_offset_alignment()
    test_field_offset_by_name()

    # type_of
    test_type_of()

    # origin_of
    test_origin_of_first_ref()

    # Source locations
    test_source_location_line()
    test_source_location_column()
    test_source_location_file()
    test_source_location_prefix()
    test_call_location()

    # Function names
    test_get_function_name()
    test_get_linkage_name()
    test_linkage_name_differs_from_source_name()

    # Iteration patterns
    test_count_equatable_fields_multi()
    test_count_equatable_fields_point()
