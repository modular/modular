# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@no_inline
def takeIt[F: def[width: Int](idx: Int) -> Scalar[DType.int]](impl: F):
    print(impl.__call__[1](0))


def emitLoad(x: SIMD[DType.int, 1]):
    var ptr = UnsafePointer(
        alloc[SIMD[DType.int, 1]]({count = 1}).unsafe_leak()
    )
    ptr.store(x)
    var count = Scalar[DType.int](0)

    # TODO: MOCO-4037
    var memory: String = "memoryOnly"

    @no_inline
    def foo[
        width: Int
    ](idx: Int) {mut count, read ptr, var memory} -> Scalar[DType.int]:
        var vec = ptr.load[width=width](idx).cast[DType.int]()
        count = count + rebind[type_of(count)](vec)
        return count

    takeIt(foo)
