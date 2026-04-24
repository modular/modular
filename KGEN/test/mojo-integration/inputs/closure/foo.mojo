# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def printIt[kernel: def(x: Int) -> Int](func: kernel, y: Int):
    print(func(y))


@no_inline
def defineIt(y: Int):
    def fallback(x: Int) {} -> Int:
        return x + x

    printIt(fallback, y)
