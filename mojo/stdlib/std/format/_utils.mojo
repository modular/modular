# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from io.io import _printf
from os import abort
from sys import align_of, size_of
from sys.info import is_gpu
from sys.param_env import env_get_int

from bit import byte_swap
from memory import Span, bitcast, memcpy

comptime HEAP_BUFFER_BYTES = env_get_int["HEAP_BUFFER_BYTES", 2048]()
"""How much memory to pre-allocate for the heap buffer, will abort if exceeded."""

comptime STACK_BUFFER_BYTES = UInt(env_get_int["STACK_BUFFER_BYTES", 4096]())
"""The size of the stack buffer for IO operations from CPU."""


struct _WriteBufferHeap(Writable, Writer):
    var _data: UnsafePointer[Byte, MutExternalOrigin]
    var _pos: Int

    fn __init__(out self):
        comptime alignment: Int = align_of[Byte]()
        self._data = __mlir_op.`pop.stack_allocation`[
            count = HEAP_BUFFER_BYTES._mlir_value,
            _type = type_of(self._data)._mlir_type,
            alignment = alignment._mlir_value,
        ]()
        self._pos = 0

    fn write_list[
        T: Copyable & Writable, //
    ](mut self, values: List[T, ...], *, sep: StaticString = StaticString()):
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    # TODO: Removing @always_inline causes some AMD tests to fail.
    # This is likely because not inlining causes _WriteBufferHeap to
    # add a conditional allocation branch which is not supported on AMD.
    # However, when its inlined, the branch (and allocation) are removed.
    # We should consider uses _WriteBufferStack on AMD instead.
    @always_inline
    fn write_string(mut self, string: StringSlice):
        var len_bytes = len(string)
        if len_bytes + self._pos > HEAP_BUFFER_BYTES:
            _printf[
                "HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D"
                " HEAP_BUFFER_BYTES=4096`\n"
            ]()
            abort()
        memcpy(
            dest=self._data + self._pos,
            src=string.unsafe_ptr(),
            count=len_bytes,
        )
        self._pos += len_bytes

    fn write_to(self, mut writer: Some[Writer]):
        writer.write_string(
            StringSlice(unsafe_from_utf8=Span(ptr=self._data, length=self._pos))
        )

    fn nul_terminate(mut self):
        if self._pos + 1 > HEAP_BUFFER_BYTES:
            _printf[
                "HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D"
                " HEAP_BUFFER_BYTES=4096`\n"
            ]()
            abort()
        self._data[self._pos] = 0
        self._pos += 1

    fn as_string_slice[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref [origin]self) -> StringSlice[origin]:
        return StringSlice(
            unsafe_from_utf8=Span(
                ptr=self._data.mut_cast[mut]().unsafe_origin_cast[origin](),
                length=self._pos,
            )
        )


struct _WriteBufferStack[
    origin: MutOrigin,
    W: Writer,
    //,
    stack_buffer_bytes: UInt = STACK_BUFFER_BYTES,
](Writer):
    var data: InlineArray[UInt8, Int(Self.stack_buffer_bytes)]
    var pos: Int
    var writer: Pointer[Self.W, Self.origin]

    fn __init__(out self, ref [Self.origin]writer: Self.W):
        self.data = InlineArray[UInt8, Int(Self.stack_buffer_bytes)](
            uninitialized=True
        )
        self.pos = 0
        self.writer = Pointer(to=writer)

    fn write_list[
        T: Copyable & Writable, //
    ](mut self, values: List[T, ...], *, sep: String = String()):
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    fn flush(mut self):
        self.writer[].write_string(
            StringSlice(
                unsafe_from_utf8=Span(
                    ptr=self.data.unsafe_ptr(), length=self.pos
                )
            )
        )
        self.pos = 0

    fn write_string(mut self, string: StringSlice):
        len_bytes = len(string)
        # If span is too large to fit in buffer, write directly and return
        if len_bytes > Int(Self.stack_buffer_bytes):
            self.flush()
            self.writer[].write_string(string)
            return
        # If buffer would overflow, flush writer and reset pos to 0.
        elif self.pos + len_bytes > Int(Self.stack_buffer_bytes):
            self.flush()
        # Continue writing to buffer
        memcpy(
            dest=self.data.unsafe_ptr() + self.pos,
            src=string.unsafe_ptr(),
            count=len_bytes,
        )
        self.pos += len_bytes


struct _TotalWritableBytes(Writer):
    var size: Int

    fn __init__(out self):
        self.size = 0

    fn __init__[
        T: Copyable & Writable,
        //,
        origin: ImmutOrigin = StaticConstantOrigin,
    ](
        out self,
        values: Span[T, ...],
        sep: StringSlice[origin] = StringSlice[origin](),
    ):
        self.size = 0
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    fn write_string(mut self, string: StringSlice):
        self.size += len(string)


# fmt: off
comptime _hex_table = SIMD[DType.uint8, 16](
    ord("0"), ord("1"), ord("2"), ord("3"), ord("4"),
    ord("5"), ord("6"), ord("7"), ord("8"), ord("9"),
    ord("a"), ord("b"), ord("c"), ord("d"), ord("e"), ord("f"),
)
# fmt: on


@always_inline
fn _hex_digits_to_hex_chars(
    ptr: UnsafePointer[mut=True, Byte], decimal: Scalar
):
    """Write a fixed width hexadecimal value into an uninitialized pointer
    location, assumed to be large enough for the value to be written.

    Examples:

    ```mojo
    %# from memory import memset_zero
    %# from testing import assert_equal
    %# from utils import StringSlice
    %# from io.write import _hex_digits_to_hex_chars
    items: List[Byte] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    comptime S = StringSlice[origin_of(items)]
    ptr = items.unsafe_ptr()
    _hex_digits_to_hex_chars(ptr, UInt32(ord("🔥")))
    assert_equal("0001f525", S(ptr=ptr, length=8))
    memset_zero(ptr, len(items))
    _hex_digits_to_hex_chars(ptr, UInt16(ord("你")))
    assert_equal("4f60", S(ptr=ptr, length=4))
    memset_zero(ptr, len(items))
    _hex_digits_to_hex_chars(ptr, UInt8(ord("Ö")))
    assert_equal("d6", S(ptr=ptr, length=2))
    ```
    """
    comptime size = size_of[decimal.dtype]()
    var bytes = bitcast[DType.uint8, size](byte_swap(decimal))
    var nibbles = (bytes >> 4).interleave(bytes & 0xF)
    ptr.store(_hex_table._dynamic_shuffle(nibbles))


@always_inline
fn _write_hex[
    amnt_hex_bytes: Int
](p: UnsafePointer[mut=True, Byte], decimal: Int):
    """Write a python compliant hexadecimal value into an uninitialized pointer
    location, assumed to be large enough for the value to be written.

    Examples:

    ```mojo
    %# from memory import memset_zero
    %# from testing import assert_equal
    %# from utils import StringSlice
    %# from io.write import _write_hex
    items: List[Byte] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    comptime S = StringSlice[origin_of(items)]
    ptr = items.unsafe_ptr()
    _write_hex[8](ptr, ord("🔥"))
    assert_equal(r"\\U0001f525", S(ptr=ptr, length=10))
    memset_zero(ptr, len(items))
    _write_hex[4](ptr, ord("你"))
    assert_equal(r"\\u4f60", S(ptr=ptr, length=6))
    memset_zero(ptr, len(items))
    _write_hex[2](ptr, ord("Ö"))
    assert_equal(r"\\xd6", S(ptr=ptr, length=4))
    ```
    """

    __comptime_assert amnt_hex_bytes in (2, 4, 8), "only 2 or 4 or 8 sequences"

    comptime `\\` = Byte(ord("\\"))
    comptime `x` = Byte(ord("x"))
    comptime `u` = Byte(ord("u"))
    comptime `U` = Byte(ord("U"))

    p.init_pointee_move(`\\`)

    @parameter
    if amnt_hex_bytes == 2:
        (p + 1).init_pointee_move(`x`)
        _hex_digits_to_hex_chars(p + 2, UInt8(decimal))
    elif amnt_hex_bytes == 4:
        (p + 1).init_pointee_move(`u`)
        _hex_digits_to_hex_chars(p + 2, UInt16(decimal))
    else:
        (p + 1).init_pointee_move(`U`)
        _hex_digits_to_hex_chars(p + 2, UInt32(decimal))
