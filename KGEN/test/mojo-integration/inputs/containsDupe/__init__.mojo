# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def consume[F: def() -> String](func: F):
    print(func())


@no_inline
def package_anchor():
    def identical() {var} -> String:
        return "hello"

    consume(identical)
