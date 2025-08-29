# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import *

from gpu.host import DeviceContext


fn run_func[
    dtype: DType,
    kernel_fn: fn[dtype: DType, width: Int] (SIMD[dtype, width]) -> SIMD[
        dtype, width
    ],
](ctx: DeviceContext, val: Scalar[dtype] = 0) raises:
    @parameter
    fn kernel(output: UnsafePointer[Scalar[dtype]], input: Scalar[dtype]):
        output[0] = kernel_fn(input)

    var out = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_function_experimental[kernel](out, val, grid_dim=1, block_dim=1)
    ctx.synchronize()

    _ = out


fn hypot_fn(val: SIMD) -> __type_of(val):
    return hypot(val, val)


fn remainder_fn(val: SIMD) -> __type_of(val):
    return remainder(val, val)


fn scalb_fn(val: SIMD) -> __type_of(val):
    return scalb(val, val)


fn gcd_fn(val: SIMD) -> __type_of(val):
    return gcd(Int(val), Int(val))


fn lcm_fn(val: SIMD) -> __type_of(val):
    return lcm(Int(val), Int(val))


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


def main():
    with DeviceContext() as ctx:

        @parameter
        fn test[
            *kernel_fns: fn[dtype: DType, width: Int] (
                SIMD[dtype, width]
            ) -> SIMD[dtype, width]
        ](ctx: DeviceContext) raises:
            alias ls = stdlib.builtin.variadic_size(kernel_fns)

            @parameter
            for idx in range(ls):
                alias kernel_fn = kernel_fns[idx]
                run_func[DType.float32, kernel_fn[]](ctx)
                run_func[DType.float16, kernel_fn[]](ctx)

        # Anything that's commented does not work atm and needs to be
        # implemented. This list is also not exhastive and needs to be
        # expanded.
        test[
            sqrt_fn,
            isqrt,
            ldexp_fn,
            frexp_fn,
            log,
            log2,
            log10,
            log1p,
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
            recip,
        ](ctx)
