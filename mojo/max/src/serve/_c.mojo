# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for working with C FFI."""


from sys.ffi import DLHandle
from memory.unsafe import DTypePointer, Pointer

from max.engine._utils import call_dylib_func


@value
@register_passable
struct TensorView:
    """Corresponds to the M_TensorView C type."""

    var name: StringRef
    var dtype: StringRef
    var shape: Pointer[Int64]
    var shapeSize: Int
    var contents: Pointer[UInt8]
    var contentsSize: Int

    alias _FreeValueFnName = "M_freeTensorView"

    @staticmethod
    fn free(borrowed lib: DLHandle, ptr: Pointer[TensorView]):
        call_dylib_func(lib, Self._FreeValueFnName, ptr)
