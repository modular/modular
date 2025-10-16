# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn printIt[kernel: fn (x: Int) unified -> Int](func: kernel, y: Int):
    print(func(y))


@no_inline
fn defineIt(y: Int):
    fn fallback(x: Int) unified {} -> Int:
        return x + x

    printIt[type_of(fallback)](fallback, y)
