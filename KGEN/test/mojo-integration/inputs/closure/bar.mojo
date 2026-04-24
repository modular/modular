# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@no_inline
def takeIt[F: def[width: Int](idx: Int) -> Scalar[DType.int]](impl: F):
    print(impl.__call__[1](0))


def emitLoad(x: SIMD[DType.int, 1]):
    var ptr = alloc[SIMD[DType.int, 1]](1)
    ptr.store(x)
    var count = Scalar[DType.int](0)

    @no_inline
    def foo[width: Int](idx: Int) {mut count, read ptr} -> Scalar[DType.int]:
        var vec = ptr.load[width=width](idx).cast[DType.int]()
        count = count + rebind[type_of(count)](vec)
        return count

    takeIt(foo)
