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
"""AMD-only GPU intrinsics. `max.gpu.intrinsics` re-exports all but `ballot`,
which only the stdlib's AMD hostcall support uses."""

from std.sys.info import is_amd_gpu, size_of
from std.sys.intrinsics import llvm_intrinsic


@inline(.always)
def implicitarg_ptr(
    out result: Pointer[UInt8, MutUntrackedOrigin, address_space=.CONSTANT]
):
    """Gets a pointer to AMD's implicit kernel arguments table.

    Returns:
        A pointer to LLVM's implicit arguments table.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    result = llvm_intrinsic[
        "llvm.amdgcn.implicitarg.ptr",
        type_of(result),
    ]()


@inline(.always)
def readfirstlane[T: AnyType, //](value: Pointer[T, ...]) -> type_of(value):
    """Gets the value in the lowest active lane of the input operand.

    Parameters:
        T: The type the pointer points to.

    Args:
        value: The input pointer.

    Returns:
        The value in the lowest active lane of the input operand.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    return llvm_intrinsic[
        "llvm.amdgcn.readfirstlane", type_of(value), type_of(value)
    ](value)


@inline(.always)
def readfirstlane[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """Gets the value in the lowest active lane of the input operand.

    Constraints:
        The scalar type must be 2, 4, or 8 bytes wide.

    Parameters:
        dtype: The element type.

    Args:
        value: The input scalar value.

    Returns:
        The value in the lowest active lane of the input operand.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    comptime assert (
        size_of[Scalar[dtype]]() >= 2
    ), "readfirstlane requires a scalar type of at least 16 bits"
    return llvm_intrinsic[
        "llvm.amdgcn.readfirstlane", Scalar[dtype], Scalar[dtype]
    ](value)


@inline(.always)
def sendmsg(opcode: Int32, msg: Int32):
    """Sends a message to fixed-function hardware.

    See the AMD ISA manual for the target for the opcodes and messages.

    Args:
        opcode: The operation to perform.
        msg: The message to send.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    _ = llvm_intrinsic["llvm.amdgcn.s.sendmsg", NoneType, Int32, Int32](
        opcode, msg
    )


@inline(.always)
def ballot[dtype: DType](value: Bool) -> Scalar[dtype]:
    """Returns a bitmask of `value` across the wavefront's lanes.

    Each active lane contributes its `value` and each inactive lane
    contributes zero, so `ballot(True)` returns the EXEC mask.

    Parameters:
        dtype: The DType of the return type.

    Args:
        value: The value to place across the mask.

    Returns:
        The `int32` or `int64` bitmask of `value` across the lanes.
    """
    comptime assert is_amd_gpu(), "This intrinsic is only defined for AMD GPUs"
    comptime assert (
        dtype == .int32 or dtype == .int64
    ), "This intrinsic is only defined for i32 or i64"
    return llvm_intrinsic["llvm.amdgcn.ballot", Scalar[dtype]](value)
