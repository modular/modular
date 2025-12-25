# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn pass_int(x: Int) -> fn () escaping -> Int:
    fn closure() -> Int:
        return x

    return closure
