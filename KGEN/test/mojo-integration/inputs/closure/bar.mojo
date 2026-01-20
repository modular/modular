# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@no_inline
fn takeIt[F: fn[width: Int] (idx: Int) unified -> Scalar[DType.int]](impl: F):
    print(impl.__call__[1](0))


fn emitLoad(x: SIMD[DType.int, 1]):
    var ptr = alloc[SIMD[DType.int, 1]](1)
    ptr.store(x)
    var count = Scalar[DType.int](0)

    @no_inline
    fn foo[
        width: Int
    ](idx: Int) unified {mut count, read ptr} -> Scalar[DType.int]:
        var vec = ptr.load[width=width](idx).cast[DType.int]()
        count = count + rebind[type_of(count)](vec)
        return count

    takeIt(foo)
