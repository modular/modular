# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def pass_int(x: Int) -> def () escaping -> Int:
    def closure() -> Int:
        return x

    return closure
