# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s
from testing import assert_true
from internal_utils import env_get_shape, parse_shape


fn print_static_shape[x: List[Int]]():
    @parameter
    for i in range(len(x)):
        print("dim", i, "=", x[i])


fn main() raises:
    alias shape_mnk = parse_shape["10x20x30"]()
    print_static_shape[shape_mnk]()
    assert_true(shape_mnk[0] == 10)
    assert_true(shape_mnk[1] == 20)
    assert_true(shape_mnk[2] == 30)

    alias shape = env_get_shape["shape", "1x2x3"]()
    print_static_shape[shape]()

    assert_true(shape[0] == 1)
    assert_true(shape[1] == 2)
    assert_true(shape[2] == 3)
