# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# COM: We only care if this file compiles or not
# RUN: %mojo-no-debug %s

from math import *
from pathlib import Path

from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from testing import assert_true


fn run_func[
    type: DType,
    kernel_fn: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
](val: Scalar[type] = 0) raises:
    @parameter
    fn kernel(out: DTypePointer[type], input: Scalar[type]):
        out[0] = kernel_fn(input)

    var func = Function[kernel]()

    var out = _malloc[type](1)
    func(out, val, grid_dim=1, block_dim=1)
    synchronize()

    _free(out)


fn hypot_fn(val: SIMD) -> __type_of(val):
    return hypot(val, val)


fn remainder_fn(val: SIMD) -> __type_of(val):
    return remainder(val, val)


fn scalb_fn(val: SIMD) -> __type_of(val):
    return scalb(val, val)


fn gcd_fn(val: SIMD) -> __type_of(val):
    return gcd(int(val), int(val))


fn lcm_fn(val: SIMD) -> __type_of(val):
    return lcm(int(val), int(val))


fn sqrt_fn(val: SIMD) -> __type_of(val):
    return sqrt(val)


fn ldexp_fn(val: SIMD) -> __type_of(val):
    return ldexp(val, 1)


fn frexp_fn(val: SIMD) -> __type_of(val):
    return frexp(val)[0]


fn floor_fn(val: SIMD) -> __type_of(val):
    return floor(val)


fn ceil_fn(val: SIMD) -> __type_of(val):
    return floor(val)


fn pow_fn(val: SIMD) -> __type_of(val):
    return val**val


fn powi_fn(val: SIMD) -> __type_of(val):
    return val**9


fn powf_fn(val: SIMD) -> __type_of(val):
    return val**3.2


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:

            @parameter
            fn test[
                *kernel_fns: fn[type: DType, width: Int] (
                    SIMD[type, width]
                ) -> SIMD[type, width]
            ]() raises:
                @parameter
                fn variadic_len[
                    *kernel_fns: fn[type: DType, width: Int] (
                        SIMD[type, width]
                    ) -> SIMD[type, width]
                ]() -> Int:
                    return __mlir_op.`pop.variadic.size`(kernel_fns)

                alias ls = variadic_len[kernel_fns]()

                @parameter
                for idx in range(ls):
                    alias kernel_fn = kernel_fns[idx]
                    run_func[DType.float32, kernel_fn[]]()
                    run_func[DType.float16, kernel_fn[]]()

            # Anything that's commented does not work atm and needs to be
            # implemented. This list is also not exhastive and needs to be
            # expanded.
            test[
                sqrt_fn,
                rsqrt,
                ldexp_fn,
                frexp_fn,
                log,
                log2,
                # log10,
                # log1p,
                # logb,
                # cbrt,
                # hypot_fn,
                # erfc,
                # lgamma,
                # gamma,
                # remainder_fn,
                # j0,
                # j1,
                # y0,
                # y1,
                # scalb_fn,
                gcd_fn,
                sin,
                # asin,
                cos,
                # acos,
                tanh,
                # atanh,
                exp,
                erf,
                floor_fn,
                ceil_fn,
                pow_fn,
                powi_fn,
                powf_fn,
            ]()
    except e:
        print("CUDA_ERROR:", e)
