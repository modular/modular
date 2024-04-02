# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for working with C FFI."""


from sys.ffi import DLHandle
from memory.unsafe import DTypePointer, Pointer


@register_passable("trivial")
struct Int64ArrayRef(Boolable):
    """Represent a constant reference to an Int64 array."""

    var data: Pointer[Int64]
    var length: Int

    fn __init__(inout self, vec: List[Int64]):
        self.data = Pointer[Int64](vec.data.value)
        self.length = len(vec)

    fn __bool__(self) -> Bool:
        return self.length != 0

    fn _as_int_vector(self) -> List[Int]:
        var vec = List[Int]()
        for i in range(self.length):
            vec.append(self.data[i].to_int())
        return vec^


@value
@register_passable
struct TensorView:
    """Corresponds to the M_tensorView C type."""

    var name: StringRef
    var dtype: StringRef
    var shape: Int64ArrayRef
    var contents: StringRef
