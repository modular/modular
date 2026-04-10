# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def consume[F: def() unified -> String](func: F):
    print(func())


@no_inline
def package_anchor():
    def identical() unified {var} -> String:
        return "hello"

    consume(identical)
