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


from .memory import AddressSpace as GPUAddressSpace

# ===-----------------------------------------------------------------------===#
# MLIR type conversion utils
# ===-----------------------------------------------------------------------===#


@always_inline
fn to_llvm_shared_mem_ptr[
    type: AnyType
](
    ptr: UnsafePointer[type, address_space = GPUAddressSpace.SHARED, **_]
) -> __mlir_type.`!llvm.ptr<3>`:
    """Cast shared memory pointer to LLVMPointer Type.

    Args:
        ptr: Shared memory pointer.

    Returns:
        A pointer of type !llvm.ptr<3>.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr<3>`
    ](ptr)


@always_inline
fn to_llvm_ptr[
    type: AnyType
](ptr: UnsafePointer[type]) -> __mlir_type.`!llvm.ptr`:
    """Cast a pointer to LLVMPointer Type.

    Args:
        ptr: A pointer.

    Returns:
        A pointer of type !llvm.ptr.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr`
    ](ptr)


@always_inline
fn to_i32(val: Int32) -> __mlir_type.i32:
    """Cast Scalar I32 value into MLIR i32.

    Args:
        val: Scalar I32 value.

    Returns:
       Input casted to MLIR i32 value.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.`i32`](val.value)


@always_inline
fn to_i16(val: UInt16) -> __mlir_type.i16:
    """Cast a scalar UInt16 value into MLIR i16.

    Args:
        val: Scalar I16 value.

    Returns:
       The input value cast to an MLIR i16.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.`i16`](val.value)


@always_inline
fn to_i64(val: Int64) -> __mlir_type.i64:
    """Cast Scalar I64 value into MLIR i64.

    Args:
        val: Scalar I64 value.

    Returns:
       Input casted to MLIR i64 value.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.`i64`](val.value)
