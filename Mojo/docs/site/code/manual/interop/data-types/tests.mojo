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
# tests.mojo
# Tests for interop/data-types.mdx. Banners follow the page's headings.
#
# Page claims, and where each is verified:
#    1. `c_int` is 32 bits, not Mojo's 64-bit `Int`  -> test_int_sizes
#    2. `c_long` is 64 bits on macOS and Linux; `c_ulong` matches it
#                                                     -> test_int_sizes
#    3. `c_char` is unsigned on Linux ARM64, signed elsewhere
#                                                     -> test_char_signedness
#    4. `c_size_t` is `UInt`, `c_ssize_t` is `Int`     -> test_size_types
#    4a. Fixed aliases: `c_uchar` `UInt8`, `c_ushort` `UInt16`, `c_uint`
#        `UInt32`, `c_long_long` `Int64`, `c_ulong_long` `UInt64`, `c_pid_t`
#        `Int32`                                       -> test_fixed_aliases
#    5. `Bool` is one byte; `c_float`/`c_double` are 32/64 bits
#                                                     -> test_bool_and_floats
#    6. `OpaquePointer` is `Pointer[NoneType]`         -> test_opaque_pointer
#    7. `platform_map()` gives `CLOCK_MONOTONIC` 1 on Linux, 6 on macOS
#                                                     -> test_platform_map
#    8. The `FILE` alias declares a nullable handle    -> test_file_alias
#    9. Matching fields in C order get C's padding  -> test_struct_layout
#   10. `external_call()` needs a `RegisterPassable` return type, so
#       `DivT` conforms (`div()`)                    -> test_struct_layout
#   10a. An exported `abi("C")` function can return a struct that isn't
#        `RegisterPassable`, such as one with an `Array` field
#                        -> passing-data-to-mojo/exports_c_host.mojo (2b)
#   11. A struct you only pass to C needs no particular trait
#                                                     -> test_pass_any_struct
#   12. Nested structs and `Array[c_char, 16]` fields are C-compatible
#                                                     -> test_nested_fields
#   13. `SIMD[DType.float32, 4]` aligns to 16 where C's `float[4]` aligns to
#       4: with a `UInt8` before it, 32 bytes versus 20; `Array` matches C
#                                                     -> test_field_alignment
#   14. `@align` adds alignment to a struct            -> test_align
#   15. A union's size is its largest member's size, rounded up to the
#       strictest member alignment                -> test_union_size
#   16. `as_c_string_span()` returns a `CStringSpan` ending in a zero byte
#                                                     -> test_c_string_span
#   17. String literals already end in a zero byte    -> test_literal_span
#   18. A second `as_c_string_span()` call reuses the buffer, and the newest
#       span is the one to use                        -> test_span_twice
#   19. `CStringSpan(unsafe_from_ptr=)` stops at the zero byte, and
#       `String(from_utf8=)` copies its bytes; `String(unsafe_from_utf8_ptr=)`
#       makes the same copy unchecked                 -> test_copy_to_zero
#   20. `strlen()` measures a C string; `byte_length()` measures Mojo's
#                                                     -> test_strlen
#   21. `from_utf8` raises on invalid bytes, `from_utf8_lossy` substitutes
#       U+FFFD, `unsafe_from_utf8` keeps valid bytes  -> test_utf8_choices
#   22. A thin `abi("C")` function type, with a `void*` context, can be stored
#       and called                                    -> test_function_types
#   23. Different callbacks passed directly share one `external_call()`
#       signature                              -> test_callbacks_share_signature
#   24. An `Optional` of a callback type isn't the size of a pointer
#                                              -> test_optional_callback_size
#
# Not tested:
#   - Ownership and lifetime prose (who frees, ASAP destruction, stored
#     callbacks), the `List` storage move, and `getenv()` values changing:
#     they describe when memory becomes invalid, which no stable assertion
#     can observe. passing-data-to-c/tests.mojo covers the calls.
#   - Pointers can't be null without `Optional`, and foreign code must not
#     return a null where Mojo expects a `Pointer`: undefined behavior.
#   - Enum values passed as `c_int`: passing-data-to-c covers
#     `CLOCK_REALTIME`.
#   - Bitfields and packed layouts: Mojo has none to test.
#   - `external_call()` and `OwnedDLHandle.get_function()` rejecting a
#     return type that isn't `RegisterPassable`, and so a returned struct
#     with an `Array` field: a compile error. Passing an array-field struct by
#     value is checked in the other direction by
#     passing-data-to-mojo/exports_c_host.mojo.
#   - The callback rules: a capturing or mismatched function at a typed
#     `abi("C")` parameter is a compile error, and `external_call()`
#     accepting a capturing closure ends in undefined behavior when C calls
#     it.
#   - Struct by-value limits for floating-point array fields and nested
#     floating-point structs: undefined behavior, covered by the safety page.
#   - One `external_call()` signature per C function: mixing argument types
#     for the same function, such as a function and an `Int` address, is a
#     compile error.
#   - Passing a union through a pointer versus by value, and the distinct
#     member-type rule: the first needs a C library that takes a union, and
#     the second is a compile error.
#   - Callback rules (`abi("C")` required, no capturing lambdas, no
#     `raises`): undefined behavior or compile errors.
#   - The "Ownership at a glance" table summarizes the sections above.
from std.ffi import (
    UnsafeUnion,
    c_char,
    c_double,
    c_float,
    c_int,
    c_long,
    c_long_long,
    c_pid_t,
    c_size_t,
    c_ssize_t,
    c_uchar,
    c_uint,
    c_ulong_long,
    c_ushort,
    c_ulong,
    external_call,
)
from std.ffi.cstring import CStringSpan
from std.sys import align_of, size_of
from std.sys.info import CompilationTarget, platform_map
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)


# --- C type aliases ---


def test_int_sizes() raises:
    assert_equal(size_of[c_int](), 4)
    assert_equal(size_of[Int](), 8)
    assert_equal(size_of[c_long](), 8)
    assert_equal(size_of[c_ulong](), size_of[c_long]())


def test_char_signedness() raises:
    assert_equal(size_of[c_char](), 1)
    comptime linux_arm = CompilationTarget.is_linux() and (
        CompilationTarget.is_arm()
    )
    assert_equal(c_char.dtype.is_signed(), not linux_arm)


def test_size_types() raises:
    assert_true(c_size_t.dtype == DType.uint)
    assert_true(c_ssize_t.dtype == DType.int)


def test_fixed_aliases() raises:
    assert_true(c_uchar.dtype == DType.uint8)
    assert_true(c_ushort.dtype == DType.uint16)
    assert_true(c_uint.dtype == DType.uint32)
    assert_true(c_long_long.dtype == DType.int64)
    assert_true(c_ulong_long.dtype == DType.uint64)
    assert_true(c_pid_t.dtype == DType.int32)


def test_bool_and_floats() raises:
    assert_equal(size_of[Bool](), 1)
    assert_equal(size_of[c_float](), 4)
    assert_equal(size_of[c_double](), 8)


def test_opaque_pointer() raises:
    var value = 7
    var opaque: OpaquePointer[origin_of(value)] = Pointer(
        to=value
    ).unsafe_bitcast[NoneType]()
    var typed: Pointer[NoneType, origin_of(value)] = opaque
    assert_equal(typed.unsafe_bitcast[Int]()[], 7)


comptime CLOCK_MONOTONIC = c_int(
    platform_map["CLOCK_MONOTONIC", linux=1, macos=6]()
)


def test_platform_map() raises:
    comptime if CompilationTarget.is_linux():
        assert_equal(CLOCK_MONOTONIC, 1)
    else:
        assert_equal(CLOCK_MONOTONIC, 6)


# --- Opaque handles ---

comptime FILE = Optional[MutOpaquePointer[MutUntrackedOrigin]]


def test_file_alias() raises:
    var missing: FILE = None
    assert_false(Bool(missing))


# --- Structs ---


@fieldwise_init
struct DivT(RegisterPassable):  # C's div_t
    var quot: c_int
    var rem: c_int


@fieldwise_init
struct Tagged(RegisterPassable):  # struct { char tag; int value; }
    var tag: c_char
    var value: c_int


@fieldwise_init
struct Sample(RegisterPassable):  # struct { char tag; double value; }
    var tag: c_char
    var value: c_double


def test_struct_layout() raises:
    assert_equal(size_of[DivT](), 8)
    # C pads `tag` to the next field's alignment, and so does Mojo.
    assert_equal(size_of[Tagged](), 8)
    assert_equal(size_of[Sample](), 16)
    # div() fills the fields in C order: quot, then rem.
    var r = external_call["div", DivT](c_int(17), c_int(5))
    assert_equal(r.quot, 3)
    assert_equal(r.rem, 2)


@fieldwise_init
struct Device(Movable):  # struct { char name[16]; div_t pair; }
    var name: Array[c_char, 16]
    var pair: DivT


def test_nested_fields() raises:
    assert_equal(size_of[Device](), 24)
    var device = Device(name=Array[c_char, 16](fill=0), pair=DivT(0, 0))
    # char *strcpy(char *dst, const char *src);
    _ = external_call["strcpy", Pointer[c_char, MutUntrackedOrigin]](
        device.name.unsafe_ptr(), "sensor0".as_c_string_span()
    )
    var name = CStringSpan(unsafe_from_ptr=device.name.unsafe_ptr())
    assert_equal(String(from_utf8=name.as_bytes()), "sensor0")


# struct in_addr { in_addr_t s_addr; }, deliberately not RegisterPassable.
@fieldwise_init
struct InAddr(Movable):
    var s_addr: UInt32


def test_pass_any_struct() raises:
    # 127.0.0.1 in network byte order, read as a little-endian UInt32.
    var addr = InAddr(s_addr=0x0100007F)
    # char *inet_ntoa(struct in_addr in);
    var text = external_call["inet_ntoa", Pointer[c_char, ImmUntrackedOrigin]](
        addr
    )
    var span = CStringSpan(unsafe_from_ptr=text)
    assert_equal(String(from_utf8=span.as_bytes()), "127.0.0.1")


struct SimdField:  # struct { uint8_t tag; float v[4]; }, mismatched
    var tag: UInt8
    var v: SIMD[DType.float32, 4]


struct ArrayField:  # struct { uint8_t tag; float v[4]; }, matching
    var tag: UInt8
    var v: Array[Float32, 4]


def test_field_alignment() raises:
    assert_equal(align_of[SIMD[DType.float32, 4]](), 16)
    assert_equal(size_of[SimdField](), 32)
    assert_equal(align_of[Array[Float32, 4]](), 4)
    assert_equal(size_of[ArrayField](), 20)
    assert_equal(align_of[ArrayField](), 4)


@align(16)
@fieldwise_init
struct Aligned(RegisterPassable):  # struct { _Alignas(16) int value; }
    var value: c_int


def test_align() raises:
    assert_equal(align_of[Aligned](), 16)
    assert_equal(size_of[Aligned](), 16)


# --- Unions ---


def test_union_size() raises:
    assert_equal(size_of[UnsafeUnion[c_int, c_double]](), 8)
    assert_equal(size_of[UnsafeUnion[UInt8, Array[UInt8, 24]]](), 24)
    # Five bytes of `char`, rounded up to `int`'s 4-byte alignment.
    assert_equal(size_of[UnsafeUnion[Array[c_char, 5], c_int]](), 8)


# --- Strings ---


# Accepts only a `CStringSpan`, so a call proves the type without naming the
# span's origin, which follows the string's internals.
def ends_in_zero(span: CStringSpan[_]) -> Bool:
    var bytes = span.as_bytes_with_nul()
    return bytes[len(bytes) - 1] == 0


def test_c_string_span() raises:
    var name = String("OMP_NUM_THREADS")
    var span = name.as_c_string_span()
    assert_true(ends_in_zero(span))
    # strlen() stops at the zero byte that as_c_string_span() guarantees.
    var length = external_call["strlen", c_size_t](span)
    assert_equal(Int(length), name.byte_length())


def test_literal_span() raises:
    var length = external_call["strlen", c_size_t]("HOME".as_c_string_span())
    assert_equal(length, 4)


def test_span_twice() raises:
    var name = String("OMP_NUM_THREADS")
    # A second call ends the first span, so record only its address.
    var first_address = Int(name.as_c_string_span().ptr())
    var span = name.as_c_string_span()
    assert_equal(Int(span.ptr()), first_address)
    assert_equal(external_call["strlen", c_size_t](span), 15)


def test_copy_to_zero() raises:
    var bytes = Array[c_char, 6](fill=0)
    bytes[0] = c_char(ord("o"))
    bytes[1] = c_char(ord("k"))
    bytes[3] = c_char(ord("x"))  # after the zero, so never copied
    var span = CStringSpan(unsafe_from_ptr=bytes.unsafe_ptr())
    assert_equal(len(span.as_bytes()), 2)
    assert_equal(String(from_utf8=span.as_bytes()), "ok")
    assert_equal(String(unsafe_from_utf8_ptr=bytes.unsafe_ptr()), "ok")


def test_strlen() raises:
    var buffer = Array[c_char, 32](fill=0)
    for i in range(5):
        buffer[i] = c_char(ord("a") + i)
    # The buffer holds 32 bytes, but the C string in it is 5 bytes long.
    assert_equal(external_call["strlen", c_size_t](buffer.unsafe_ptr()), 5)
    var span = CStringSpan(unsafe_from_ptr=buffer.unsafe_ptr())
    assert_equal(String(from_utf8=span.as_bytes()).byte_length(), 5)


def test_utf8_choices() raises:
    var bad: List[Byte] = [0x6F, 0x6B, 0xFF]
    with assert_raises():
        _ = String(from_utf8=Span(bad))
    assert_equal(String(from_utf8_lossy=Span(bad)), "ok�")
    var good: List[Byte] = [0x6F, 0x6B]
    assert_equal(String(unsafe_from_utf8=Span(good)), "ok")
    assert_equal(String(from_utf8=Span(good)), "ok")


# --- Function pointers ---

comptime Context = MutOpaquePointer[MutUntrackedOrigin]
comptime Progress = def(c_double, Context) thin abi("C")


def record(fraction: c_double, context: Context) abi("C"):
    context.unsafe_bitcast[c_double]()[] = fraction


comptime ConstIntPointer = Pointer[c_int, ImmUntrackedOrigin]


def ascending(left: ConstIntPointer, right: ConstIntPointer) abi("C") -> c_int:
    return c_int(left[] > right[]) - c_int(left[] < right[])


def descending(left: ConstIntPointer, right: ConstIntPointer) abi("C") -> c_int:
    return c_int(left[] < right[]) - c_int(left[] > right[])


def test_callbacks_share_signature() raises:
    var readings: List[c_int] = [42, 7, 19, 3]
    # void qsort(void *base, size_t nel, size_t width,
    #            int (*compar)(const void *, const void *));
    # Both calls pass a comparator directly, so they share one signature
    # even though the comparators differ.
    external_call["qsort", NoneType](
        readings.unsafe_ptr(),
        c_size_t(4),
        c_size_t(size_of[c_int]()),
        ascending,
    )
    assert_equal(readings[0], 3)
    assert_equal(readings[3], 42)
    external_call["qsort", NoneType](
        readings.unsafe_ptr(),
        c_size_t(4),
        c_size_t(size_of[c_int]()),
        descending,
    )
    assert_equal(readings[0], 42)
    assert_equal(readings[3], 3)


def test_optional_callback_size() raises:
    assert_true(size_of[Optional[Progress]]() != size_of[Context]())


def test_function_types() raises:
    var latest: c_double = 0
    var callback: Progress = record
    callback(
        0.5,
        Pointer(to=latest)
        .unsafe_bitcast[NoneType]()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin](),
    )
    assert_equal(latest, 0.5)


def main() raises:
    test_int_sizes()
    test_char_signedness()
    test_size_types()
    test_fixed_aliases()
    test_bool_and_floats()
    test_opaque_pointer()
    test_platform_map()
    test_file_alias()
    test_struct_layout()
    test_pass_any_struct()
    test_nested_fields()
    test_field_alignment()
    test_align()
    test_union_size()
    test_c_string_span()
    test_literal_span()
    test_span_twice()
    test_copy_to_zero()
    test_strlen()
    test_utf8_choices()
    test_function_types()
    test_callbacks_share_signature()
    test_optional_callback_size()
