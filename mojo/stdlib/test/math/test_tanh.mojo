# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# RUN: %mojo %s

from math import tanh
from random import randn, seed

from buffer import Buffer
from test_utils import compare, libm_call
from testing import assert_almost_equal


def test_tanh_tfvals_fp32():
    alias dtype = DType.float32

    # The following input values for x are taken from
    # https://github.com/modularml/modular/issues/28981#issuecomment-1890182667
    var x = Buffer[dtype, 4].stack_allocation()
    x.store[width=4](
        0,
        SIMD[dtype, 4](
            -1.2583316564559937,
            -8.081921577453613,
            -8.626264572143555,
            -0.7127348184585571,
        ),
    )

    var y = Buffer[dtype, 4].stack_allocation()
    for i in range(4):
        y[i] = tanh(x[i])

    #################################################
    # TF results
    # use `tf.print(tf.math.tanh(numpy.float32(x)))`
    var tfvals_fp32 = Buffer[dtype, 4].stack_allocation()
    tfvals_fp32.store[width=4](
        0, SIMD[dtype, 4](-0.850603521, -1, -1, -0.612388909)
    )

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[dtype, 4](
        0.0, 1.1920928955078125e-07, 0.0, 1.1920928955078125e-07
    )
    var err = compare[dtype](
        y.data, tfvals_fp32.data, 4, msg="Compare Mojo vs. Tensorflow FP32"
    )
    assert_almost_equal(err, abs_rel_err)


def test_tanh_tfvals_fp64():
    alias dtype = DType.float64

    # The following input values for x are taken from
    # https://github.com/modularml/modular/issues/28981#issuecomment-1890182667
    var x = Buffer[dtype, 4].stack_allocation()
    x.store[width=4](
        0,
        SIMD[dtype, 4](
            -1.2583316564559937,
            -8.081921577453613,
            -8.626264572143555,
            -0.7127348184585571,
        ),
    )

    var y = Buffer[dtype, 4].stack_allocation()
    for i in range(4):
        y[i] = tanh(x[i])

    #################################################
    # TF results
    # use `tf.print(tf.math.tanh(numpy.float64(x)))`
    var tfvals_fp64 = Buffer[dtype, 4].stack_allocation()
    tfvals_fp64.store[width=4](
        0,
        SIMD[dtype, 4](
            -0.85060351067231821,
            -0.99999980894339091,
            -0.99999993567914991,
            -0.61238890225714893,
        ),
    )

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[dtype, 4](
        7.2062200651146213e-09,
        1.2149700800989649e-08,
        8.3577847290501252e-09,
        1.4283624095774667e-08,
    )

    var err = compare[dtype](
        y.data, tfvals_fp64.data, 4, msg="Compare Mojo vs. Tensorflow FP64"
    )
    assert_almost_equal(err, abs_rel_err)


def main():
    test_tanh_tfvals_fp32()
    test_tanh_tfvals_fp64()
