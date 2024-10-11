# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------===#
# MLIR type conversion utils
# ===----------------------------------------------------------------------===#


@always_inline
fn to_llvm_shared_mem_ptr[
    type: AnyType
](
    ptr: UnsafePointer[type, GPUAddressSpace.SHARED, *_]
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
