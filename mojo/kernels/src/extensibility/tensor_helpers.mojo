# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os.atomic import Atomic


@value
@register_passable("trivial")
struct InnerStride:
    var val: Int
    alias Broadcast = InnerStride(0)
    alias Contiguous = InnerStride(1)
    alias Strided = InnerStride(2)

    @staticmethod
    fn from_stride(inner_stride: Int) -> Self:
        if inner_stride == 0:
            return InnerStride.Broadcast
        elif inner_stride == 1:
            return InnerStride.Contiguous
        else:
            return InnerStride.Strided

    @always_inline
    fn __eq__(self, other: InnerStride) -> Bool:
        return self.val == other.val


@value
@register_passable
struct UnsafeRefCounter[type: DType]:
    """
    Implements an atomic which can be sharable as a copyable reference.

    By design the counter memory must be managed manually, this allows us to
    copy this around by ref and have multiple references to the same pointer
    under the hood.
    """

    var _underlying_value: UnsafePointer[Scalar[type]]

    fn deallocate(owned self):
        self._underlying_value.free()

    fn increment(self) -> Scalar[type]:
        return Atomic[type]._fetch_add(self._underlying_value, 1)

    fn decrement(self) -> Scalar[type]:
        return Atomic[type]._fetch_add(self._underlying_value, -1)

    fn _value(inout self) -> Scalar[type]:
        return Atomic[type]._fetch_add(self._underlying_value, 0)
