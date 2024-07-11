# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys._assembly import inlined_assembly

from memory.unsafe import DTypePointer, Pointer
from memory import UnsafePointer

# ===----------------------------------------------------------------------===#
# keep
# ===----------------------------------------------------------------------===#


@always_inline
fn keep(val: Bool):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Args:
      val: The value to not optimize away.
    """
    keep(UInt8(val))


@always_inline
fn keep(val: Int):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Args:
      val: The value to not optimize away.
    """
    keep(Scalar[DType.index](val))


@always_inline
fn keep[type: DType, simd_width: Int](val: SIMD[type, simd_width]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
      val: The value to not optimize away.
    """

    @parameter
    if simd_width > 1:
        for i in range(simd_width):
            # TODO(#27998): Remove the temporary variable.
            var tmp = val[i]
            keep(tmp)
        return

    var tmp = val
    var tmp_ptr = UnsafePointer.address_of(tmp)

    @parameter
    if sizeof[type]() <= sizeof[Pointer[SIMD[type, simd_width]]._mlir_type]():
        inlined_assembly[
            "",
            NoneType,
            constraints="+m,r,~{memory}",
            has_side_effect=True,
        ](tmp_ptr, val)
    else:
        inlined_assembly[
            "",
            NoneType,
            constraints="+m,~{memory}",
            has_side_effect=True,
        ](tmp_ptr, tmp_ptr)
    _ = tmp


@always_inline
fn keep[type: DType](val: DTypePointer[type]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    keep(val.address)


@always_inline
fn keep[type: AnyType](val: UnsafePointer[type]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    var tmp_ptr = UnsafePointer.address_of(tmp)
    inlined_assembly[
        "",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](tmp_ptr)
    _ = tmp


@always_inline
fn keep[type: AnyTrivialRegType](inout val: type):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    var tmp_ptr = UnsafePointer.address_of(tmp)
    inlined_assembly[
        "",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](tmp_ptr)
    _ = tmp
