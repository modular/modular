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
from testing import *
from test_utils.reflection import SimplePoint, NestedStruct, EmptyStruct
from benchmark import keep
from compile import compile_info
from collections.string.format import _FormatUtils
from format._utils import write_sequence_to


@fieldwise_init
struct TestWritable(Writable):
    var x: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("write_to: ", self.x)

    fn write_repr_to(self, mut writer: Some[Writer]):
        writer.write("write_repr_to: ", self.x)


def test_repr():
    var t = TestWritable(42)
    assert_equal(repr(t), "write_repr_to: 42")


def test_string_constructor():
    var s = String(TestWritable(42))
    assert_equal(s, "write_to: 42")


def test_format_string():
    assert_equal("{}".format(TestWritable(42)), "write_to: 42")
    assert_equal(String("{}").format(TestWritable(42)), "write_to: 42")
    assert_equal(StringSlice("{}").format(TestWritable(42)), "write_to: 42")

    assert_equal("{!r}".format(TestWritable(42)), "write_repr_to: 42")
    assert_equal(String("{!r}").format(TestWritable(42)), "write_repr_to: 42")
    assert_equal(
        StringSlice("{!r}").format(TestWritable(42)), "write_repr_to: 42"
    )


def test_default_write_to_simple():
    """Test the reflection-based default write_to with a simple struct."""
    var p = SimplePoint(1, 2)
    # Note: get_type_name returns module-qualified names
    assert_equal(String(p), "SimplePoint(x=1, y=2)")
    assert_equal(repr(p), "SimplePoint(x=Int(1), y=Int(2))")


def test_default_write_to_nested():
    """Test the reflection-based default write_to with nested structs."""
    var s = NestedStruct(SimplePoint(3, 4), "test")
    # Note: String's write_repr_to doesn't add quotes (write_to is same as write_repr_to for String)
    assert_equal(
        String(s),
        "NestedStruct(point=SimplePoint(x=3, y=4), name=test)",
    )
    assert_equal(
        repr(s),
        "NestedStruct(point=SimplePoint(x=Int(3), y=Int(4)), name='test')",
    )


def test_default_write_to_empty():
    """Test the reflection-based default write_to with an empty struct."""
    var e = EmptyStruct()
    assert_equal(String(e), "EmptyStruct()")
    assert_equal(repr(e), "EmptyStruct()")


def test_write_sequence_to_with_element_fn_counter():
    """Test write_sequence_to with ElementFn using a simple counter.

    This demonstrates the basic usage of ElementFn: a closure that writes
    elements and raises StopIteration when done.
    """
    var output = String()

    var count = 0

    @parameter
    fn write_numbers[T: Writer](mut w: T) raises StopIteration:
        if count >= 3:
            raise StopIteration()
        w.write(count)
        count += 1

    write_sequence_to[ElementFn=write_numbers](output)
    assert_equal(output, "[0, 1, 2]")

    _ = count


def test_write_sequence_to_empty_sequence():
    """Test write_sequence_to with ElementFn that immediately raises StopIteration.
    """
    var output = String()

    @parameter
    fn write_nothing[T: Writer](mut w: T) raises StopIteration:
        raise StopIteration()

    write_sequence_to[ElementFn=write_nothing](output)
    assert_equal(output, "[]")


def test_write_sequence_to_single_element():
    """Test write_sequence_to with ElementFn that writes one element."""
    var output = String()

    var written = False

    @parameter
    fn write_once[T: Writer](mut w: T) raises StopIteration:
        if written:
            raise StopIteration()
        w.write("only")
        written = True

    write_sequence_to[ElementFn=write_once](output)
    assert_equal(output, "[only]")

    _ = written


def test_write_sequence_to_custom_delimiters():
    """Test write_sequence_to with custom opening, closing, and separator."""
    var output = String()

    var index = 0

    @parameter
    fn write_items[T: Writer](mut w: T) raises StopIteration:
        if index >= 3:
            raise StopIteration()
        w.write("item", index)
        index += 1

    write_sequence_to[ElementFn=write_items](
        output, start="{", end="}", sep="; "
    )
    assert_equal(output, "{item0; item1; item2}")

    _ = index


struct NullWriter(Writer):
    fn write_string(mut self, string: StringSlice):
        keep(string)


comptime ALLOC_FUNC = "KGEN_CompilerRT_AlignedAlloc"


def test_format_runtime_does_allocate():
    def runtime_format[
        *Ts: Writable,
    ](mut writer: NullWriter, *args: *Ts):
        _FormatUtils.format_to_runtime(writer, "Hello, {}, {}, {}", args)

    var info = compile_info[
        runtime_format[Int, String, List[Float32]],
        emission_kind="llvm-opt",
    ]()
    assert_true(ALLOC_FUNC in info, info.asm)


def test_format_comptime_does_not_allocate():
    fn comptime_format[
        *Ts: Writable,
    ](mut writer: NullWriter, *args: *Ts):
        _FormatUtils.format_to_comptime["Hello, {}, {}, {}"](writer, args)

    var info = compile_info[
        comptime_format[Int, String, List[Float32]],
        emission_kind="llvm-opt",
    ]()
    assert_false(ALLOC_FUNC in info)


def test_format_long_strings_simd_boundaries():
    """Test format strings with braces at SIMD vector boundaries.

    SIMD widths are typically 16 (SSE), 32 (AVX2), or 64 (AVX512) bytes.
    These tests place braces and escaped braces at and around those boundaries
    to exercise the vectorized scan and scalar tail fallback.
    """

    # Helper: build a string of 'x' repeated `n` times.
    fn pad(n: Int) -> String:
        return String("x") * n

    # --- Braces at boundary positions (15, 16, 31, 32, 63, 64) ---

    # Brace at position 15 (last byte of first 16-byte SIMD chunk).
    assert_equal((pad(15) + "{}").format("A"), pad(15) + "A")

    # Brace at position 16 (first byte of second 16-byte chunk).
    assert_equal((pad(16) + "{}").format("B"), pad(16) + "B")

    # Brace at position 31 (last byte of second 16-byte chunk / first 32-byte
    # AVX2 chunk).
    assert_equal((pad(31) + "{}").format("C"), pad(31) + "C")

    # Brace at position 32.
    assert_equal((pad(32) + "{}").format("D"), pad(32) + "D")

    # Brace at position 63 (last byte of first 64-byte AVX512 chunk).
    assert_equal((pad(63) + "{}").format("E"), pad(63) + "E")

    # Brace at position 64.
    assert_equal((pad(64) + "{}").format("F"), pad(64) + "F")

    # --- Escaped braces {{/}} straddling boundaries ---

    # Escaped {{ with first { at position 15, second { at position 16.
    assert_equal((pad(15) + "{{}}").format(), pad(15) + "{}")

    # Escaped {{ at position 31-32.
    assert_equal((pad(31) + "{{}}").format(), pad(31) + "{}")

    # Escaped }} at position 15-16.
    assert_equal((pad(14) + "{}{{}}").format("V"), pad(14) + "V{}")

    # --- Multiple fields across chunks ---

    # Fields at positions 14, 30 (spanning two 16-byte chunks).
    assert_equal(
        (pad(14) + "{}" + pad(14) + "{}").format("X", "Y"),
        pad(14) + "X" + pad(14) + "Y",
    )

    # Fields spanning three 16-byte chunks.
    assert_equal(
        (pad(14) + "{}" + pad(14) + "{}" + pad(14) + "{}").format(
            "A", "B", "C"
        ),
        pad(14) + "A" + pad(14) + "B" + pad(14) + "C",
    )

    # --- Brace in scalar tail (string length not aligned to SIMD width) ---

    # 17 bytes total: 16-byte vectorized chunk + 1 byte scalar tail with {}.
    assert_equal((pad(15) + "{}" + "z").format("T"), pad(15) + "T" + "z")

    # 33 bytes: two 16-byte chunks + 1 byte tail.
    assert_equal((pad(31) + "{}" + "z").format("T"), pad(31) + "T" + "z")

    # --- Long string with many fields ---
    assert_equal(
        (pad(100) + "{}" + pad(100) + "{}" + pad(100)).format("M", "N"),
        pad(100) + "M" + pad(100) + "N" + pad(100),
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
