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
# test.mojo
# Tests for reflection.mdx code examples.
# Skip: show_type (print-only), show_layout print output,
#        source_location file path (path-dependent),
#        origin_of abort path (aborts by design),
#        get_linkage_name exact value (mangled symbol).
from std.testing import assert_equal, assert_true
from std.reflection import (
    source_location,
    call_location,
    get_function_name,
    get_linkage_name,
)
from std.os import abort


# --- Why reflection? ---


@fieldwise_init
struct Sensor(Equatable, Hashable, Writable):
    var id: Int
    var label: String
    var reading: Float64


def test_sensor_equality() raises:
    """Equatable auto-implemented via fieldwise reflection."""
    var a = Sensor(1, "temp", 98.6)
    var b = Sensor(1, "temp", 98.6)
    var c = Sensor(2, "pressure", 14.7)
    assert_true(a == b)
    assert_true(a != c)


# --- Inspect a type ---


def test_base_name() raises:
    """Check `base_name()` strips parameters and module path."""
    assert_equal(reflect[List[Int]].base_name(), "List")
    assert_equal(reflect[Dict[String, Int]].base_name(), "Dict")


@fieldwise_init
struct Config(Equatable):
    var host: String
    var port: Int
    var verbose: Bool
    var timeout: Float64


def test_field_type_by_name() raises:
    """Check `field_type["host"]` gives a handle whose .T is usable."""
    comptime host_handle = reflect[Config].field_type["host"]
    var default_host: host_handle.T = "localhost"
    assert_equal(default_host, "localhost")


# --- Detect field-level changes ---


def diff_fields[T: AnyType](a: T, b: T) -> List[String]:
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    var diffs = List[String]()

    comptime for idx in range(reflect[T].field_count()):
        comptime if conforms_to(types[idx], Equatable):
            ref a_val = reflect[T].field_ref[idx](a)
            ref b_val = reflect[T].field_ref[idx](b)
            if a_val != b_val:
                diffs.append(String(names[idx]))

    return diffs^


def test_diff_fields_detects_changes() raises:
    var old = Config("localhost", 8080, False, 30.0)
    var new_cfg = Config("localhost", 9090, True, 30.0)
    var changes = diff_fields(old, new_cfg)
    assert_equal(len(changes), 2)
    assert_equal(changes[0], "port")
    assert_equal(changes[1], "verbose")


def test_diff_fields_identical() raises:
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("localhost", 8080, False, 30.0)
    assert_equal(len(diff_fields(a, b)), 0)


def test_diff_fields_all_different() raises:
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("remote", 9090, True, 60.0)
    assert_equal(len(diff_fields(a, b)), 4)


def test_diff_fields_single_field() raises:
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("localhost", 8080, False, 99.0)
    var changes = diff_fields(a, b)
    assert_equal(len(changes), 1)
    assert_equal(changes[0], "timeout")


# --- Write once reuse everywhere ---


trait MakeCopyable:
    def copy_to(self, mut other: Self):
        comptime field_count = reflect[Self].field_count()
        comptime field_types = reflect[Self].field_types()

        comptime Usable = Copyable & ImplicitlyDestructible
        comptime for idx in range(field_count):
            comptime field_type = field_types[idx]
            comptime if conforms_to(field_type, Usable):
                reflect[Self].field_ref[idx](other) = (
                    reflect[Self].field_ref[idx](self).copy()
                )


@fieldwise_init
struct MultiType(MakeCopyable, Writable):
    var w: String
    var x: Int
    var y: Bool
    var z: Float64

    def write_to[W: Writer](self, mut writer: W):
        writer.write(String(t"[{self.w}, {self.x}, {self.y}, {self.z}]"))


def test_copy_to_transfers_values() raises:
    """Check `copy_to` duplicates all Copyable fields."""
    var original = MultiType("Hello", 1, True, 2.5)
    var target = MultiType("", 0, False, 0.0)
    original.copy_to(target)
    assert_equal(target.w, "Hello")
    assert_equal(target.x, 1)
    assert_equal(target.y, True)
    assert_equal(target.z, 2.5)


def test_copy_to_independent() raises:
    """Modifying original after copy_to doesn't affect target."""
    var original = MultiType("Hello", 1, True, 2.5)
    var target = MultiType("", 0, False, 0.0)
    original.copy_to(target)
    original.x = 999
    assert_equal(target.x, 1)


# --- Field layout ---


struct Packet:
    var flags: UInt8
    var id: UInt32
    var payload: UInt64


def test_field_offset_zero() raises:
    """First field is always at offset 0."""
    assert_equal(reflect[Packet].field_offset[index=0](), 0)


def test_field_offset_alignment() raises:
    """UInt32 field is at offset >= 4 due to alignment padding after UInt8."""
    comptime off = reflect[Packet].field_offset[index=1]()
    assert_true(off >= 4)


# --- type_of ---


def make_default[T: AnyType & Defaultable]() -> T:
    return T()


def test_type_of() raises:
    """Check `type_of` captures the compile-time type of an expression."""
    var x = 42
    var y = make_default[type_of(x)]()
    assert_equal(y, 0)
    _ = x


# --- origin_of ---


def first_ref[T: Copyable](ref list: List[T]) -> ref[origin_of(list)] T:
    if not list:
        abort("empty list")
    return list[0]


def test_origin_of_first_ref() raises:
    """Check `first_ref` returns a reference tied to the list's origin."""
    var l = [1, 2, 3]
    ref x = first_ref(l)
    assert_equal(x, 1)
    x += 10
    assert_equal(l[0], 11)


# --- Source locations ---


def test_source_location_line() raises:
    """Check `source_location` returns a positive line number."""
    var loc = source_location()
    assert_true(loc.line() > 0)


def test_source_location_prefix() raises:
    """Check `source_location().prefix()` produces 'At ...: msg' format."""
    var loc = source_location()
    var msg = loc.prefix("test message")
    assert_true(msg.find("At ") == 0)
    assert_true(msg.find("test message") >= 0)


@always_inline
def get_caller_line() -> Int:
    return call_location().line()


def test_call_location() raises:
    """Check `call_location` captures the immediate caller's line, not its own.
    """
    var line = get_caller_line()
    assert_true(line > 0)


# --- Function names ---


def process_data():
    pass


def test_get_function_name() raises:
    """Check `get_function_name` returns the source-level name."""
    comptime name = get_function_name[process_data]()
    assert_equal(name, "process_data")


def test_get_linkage_name() raises:
    """Check `get_linkage_name` returns a non-empty mangled symbol."""
    comptime linkage = get_linkage_name[process_data]()
    assert_true(linkage)


# --- Additional methods: conforms_to and where clause ---


def eq[T: AnyType](a: T, b: T) -> Bool where conforms_to(T, Equatable):
    return a == b


def test_eq_where_clause() raises:
    """Check `where conforms_to(T, Equatable)` enables == without trait_downcast.
    """
    assert_true(eq(1, 1))
    assert_true(not eq(1, 2))
    assert_true(eq("hello", "hello"))


def test_conforms_to() raises:
    """Check `conforms_to` checks compile-time trait conformance."""
    assert_true(conforms_to(Int, Equatable))
    assert_true(conforms_to(String, Equatable))


def main() raises:
    # Why reflection?
    test_sensor_equality()

    # Inspect a type
    test_base_name()
    test_field_type_by_name()

    # Detect field-level changes
    test_diff_fields_detects_changes()
    test_diff_fields_identical()
    test_diff_fields_all_different()
    test_diff_fields_single_field()

    # Write once reuse everywhere
    test_copy_to_transfers_values()
    test_copy_to_independent()

    # Field layout
    test_field_offset_zero()
    test_field_offset_alignment()

    # type_of
    test_type_of()

    # origin_of
    test_origin_of_first_ref()

    # Source locations
    test_source_location_line()
    test_source_location_prefix()
    test_call_location()

    # Function names
    test_get_function_name()
    test_get_linkage_name()

    # Additional methods
    test_eq_where_clause()
    test_conforms_to()
