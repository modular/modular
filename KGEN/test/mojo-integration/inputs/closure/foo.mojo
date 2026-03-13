# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def printIt[kernel: def(x: Int) unified -> Int](func: kernel, y: Int):
    print(func(y))


@no_inline
def defineIt(y: Int):
    def fallback(x: Int) unified {} -> Int:
        return x + x

    printIt[type_of(fallback)](fallback, y)
