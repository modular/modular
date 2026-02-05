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
"""TVM FFI types and functions using the Stable C ABI.

See https://tvm.apache.org/ffi/get_started/stable_c_abi.html.

All implementations are based on that reference material.
"""

import format

from . import dlpack


struct Types:
    """Enum values for TVMFFITypeIndex."""

    # https://tvm.apache.org/ffi/reference/cpp/generated/enum_c__api_8h_1a1925bb5d568a3f5c92a6c28934c9bcc2.html#_CPPv415TVMFFITypeIndex
    comptime INT: Int32 = 1
    comptime TENSOR_POINTER: Int32 = 7
    comptime ERROR: Int32 = 67


@fieldwise_init
struct TVMFFIAny(Copyable, Movable):
    """Tagged union for passing arguments to the safe calling convention."""

    var type_index: Int32
    var zero_padding: UInt32
    var data: Int64

    def __init__(out self, tensor: dlpack.DLTensor):
        self.type_index = Types.TENSOR_POINTER
        self.zero_padding = 0
        self.data = Int(UnsafePointer(to=tensor))

    def __init__(out self, value: Int):
        self.type_index = Types.INT
        self.zero_padding = 0
        self.data = value


# ABI for TVMFFISafeCallType
# https://tvm.apache.org/ffi/concepts/func_module.html#sec-function-calling-convention
comptime SafeFunction = fn(
    module: UnsafePointer[NoneType, MutAnyOrigin],
    args: Pointer[TVMFFIAny, MutAnyOrigin],
    nargs: Int32,
    result: Pointer[TVMFFIAny, MutAnyOrigin],
) -> Int32

comptime TVMFFIByteArray = Span[Byte, MutAnyOrigin]


trait TVMFFIType:
    """Trait for types which can appear as a TVMFFIObject."""

    comptime type_index: Int32


struct TVMFFIObject:
    """TVM FFI Object header (precedes all heap-allocated objects)."""

    var combined_ref_count: UInt64
    var type_index: Int32
    var _padding: UInt32
    var _deleter: Int64  # function pointer stored as Int64
    # Object data lives here following the header

    fn __getitem__[T: TVMFFIType](ref self) -> ref[self] T:
        if not self.type_index == T.type_index:
            # TODO(MOCO-3215): raise instead
            abort(
                "Invalid type: {} != {}".format(self.type_index, T.type_index)
            )
        return (UnsafePointer(to=self) + 1).bitcast[T]()[]


struct TVMFFIErrorCell(
    ImplicitlyCopyable, Movable, TVMFFIType, format.Writable
):
    comptime type_index: Int32 = Types.ERROR

    var kind: TVMFFIByteArray
    var message: TVMFFIByteArray
    var backtrace: TVMFFIByteArray
    # Unused fields omitted (update_backtrace, cause_chain, extra_context)

    fn write_to(self, mut writer: Some[format.Writer]):
        writer.write(StringSlice(unsafe_from_utf8=self.kind))
        writer.write(": ")
        writer.write(StringSlice(unsafe_from_utf8=self.message))

    fn write_repr_to(self, mut writer: Some[format.Writer]):
        writer.write("TVMFFIErrorCell('")
        self.write_to(writer)
        writer.write("')")


fn _tvm_ffi_error_move_from_raised(
    mut result: UnsafePointer[TVMFFIObject, MutAnyOrigin]
) raises:
    """Wraps TVMFFIErrorMoveFromRaised."""
    # Expects that libtvm_ffi.so is available, for instance
    # loaded by python importing `tvm_ffi`.
    lib = OwnedDLHandle(path="libtvm_ffi.so")
    comptime FnType = fn(
        UnsafePointer[UnsafePointer[TVMFFIObject, MutAnyOrigin], MutAnyOrigin]
    ) -> None
    fn_ptr = lib.get_function[FnType]("TVMFFIErrorMoveFromRaised")
    fn_ptr(UnsafePointer(to=result))


fn take_latest_error() raises -> TVMFFIErrorCell:
    """Retrieves the last TVM FFI error message."""
    error_ptr = UnsafePointer[TVMFFIObject, MutAnyOrigin]()
    _tvm_ffi_error_move_from_raised(error_ptr)
    if not error_ptr:
        raise Error("TVM FFI: No error.")
    return error_ptr[][TVMFFIErrorCell]
