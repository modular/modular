# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from collections import Optional
from math import align_up
from sys import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    has_nvidia_gpu_accelerator,
    size_of,
    align_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import arg_parse
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils._utils import (
    InitializationType,
    ValOrDim,
    dynamic,
    init_vector_launch,
    static,
)
from linalg.matmul.gpu import _matmul_gpu
from linalg.utils import elementwise_compute_lambda_type
from utils import IndexList
from linalg.matmul.gpu.amd.pingpong_kernel import ping_pong_matmul
from layout._ndbuffer_stub import from_ndbuffer_row_major


fn _get_run_name[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
](
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) -> String:
    var vendor_str = "vendor_matmul" if use_vendor_blas else "matmul"
    var type_str = String("(", dtype, ") : ")
    # M
    var m_str = String(shape_c_dim[0], "_dynamic")
    # N
    var n_str = String(
        shape_c_dim[1],
        "_dynamic" if shape_c.at[1]().is_dynamic() else "",
    )
    # K
    var k_str = String(
        shape_a_dim[1],
        "_dynamic" if shape_a.at[1]().is_dynamic() else "",
    )

    var transpose_b_str = String(
        "/transpose_b=", "True" if transpose_b else "False"
    )
    var cache_busting_str = String(
        "/cache_busting=", "True" if cache_busting else "False"
    )
    return String(
        vendor_str,
        type_str,
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        transpose_b_str,
        cache_busting_str,
    )


fn bench_matmul[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    *,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    transpose_b: Bool = False,
    epilogue: Bool = False,
    register_based_epilogue: Bool = False,
](
    ctx: DeviceContext,
    mut b: Bench,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
    init_type: InitializationType,
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @always_inline
    fn get_size(shape: IndexList[2]) -> Int:
        return shape[0] * shape[1]

    comptime simd_size = 4
    var stride_a = align_up(get_size(shape_a_dim), simd_size)
    var stride_b = align_up(get_size(shape_b_dim), simd_size)
    var stride_c = align_up(get_size(shape_c_dim), simd_size)

    # Benchmark with the same data type for C as A and B
    comptime c_dtype = dtype

    comptime k128 = 512 * 1024 * 1024
    var cache_a = (
        align_up(k128, stride_a * size_of[dtype]()) // size_of[dtype]()
    )
    var cache_b = (
        align_up(k128, stride_b * size_of[dtype]()) // size_of[dtype]()
    )
    var cache_c = (
        align_up(k128, stride_c * size_of[c_dtype]()) // size_of[c_dtype]()
    )

    var buffer_a = ctx.enqueue_create_buffer[dtype](cache_a)
    var buffer_b = ctx.enqueue_create_buffer[dtype](cache_b)
    var buffer_c = ctx.enqueue_create_buffer[c_dtype](cache_c)

    init_vector_launch[dtype](buffer_a, cache_a, init_type, ctx)
    init_vector_launch[dtype](buffer_b, cache_b, init_type, ctx)

    @parameter
    @__copy_capture(cache_a, cache_b, cache_c, stride_a, stride_b, stride_c)
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var offset_a = 0
            var offset_b = 0
            var offset_c = 0

            comptime if cache_busting:
                offset_a = (iteration * stride_a) % cache_a
                offset_b = (iteration * stride_b) % cache_b
                offset_c = (iteration * stride_c) % cache_c
            var tensor_a = NDBuffer[dtype, 2, MutAnyOrigin, shape_a](
                buffer_a.unsafe_ptr() + offset_a, shape_a_dim
            )
            var tensor_b = NDBuffer[dtype, 2, MutAnyOrigin, shape_b](
                buffer_b.unsafe_ptr() + offset_b, shape_b_dim
            )
            var tensor_c = NDBuffer[c_dtype, 2, MutAnyOrigin, shape_c](
                buffer_c.unsafe_ptr() + offset_c, shape_c_dim
            )

            @parameter
            @always_inline
            @__copy_capture(tensor_c)
            fn test_lambda_add_coords_prod[
                _dtype: DType,
                width: Int,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var x = tensor_c.load[width=width](idx).cast[_dtype]()
                var y = val * x
                return y

            comptime optional_lambda_fn = Optional[
                elementwise_compute_lambda_type
            ](test_lambda_add_coords_prod) if epilogue else None

            comptime if use_vendor_blas:
                vendor_blas.matmul[use_tf32=True](
                    ctx,
                    tensor_c,
                    tensor_a,
                    tensor_b,
                    c_row_major=True,
                    transpose_b=transpose_b,
                )
            else:
                comptime use_ping_pong_matmul = env_get_bool[
                    "use_ping_pong_matmul", True
                ]()
                comptime enable_swizzle = env_get_bool["enable_swizzle", True]()

                comptime if use_ping_pong_matmul:
                    ping_pong_matmul[enable_swizzle=enable_swizzle](
                        from_ndbuffer_row_major(tensor_a),
                        from_ndbuffer_row_major(tensor_b),
                        from_ndbuffer_row_major(tensor_c),
                        ctx,
                    )
                else:
                    _matmul_gpu[
                        use_tensor_core=True,
                        transpose_b=transpose_b,
                        elementwise_compute_lambda_fn=optional_lambda_fn,
                        register_based_epilogue=register_based_epilogue,
                    ](tensor_c, tensor_a, tensor_b, ctx)

        b.iter_custom[kernel_launch](ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
        2 * shape_c_dim[0] * shape_c_dim[1] * shape_a_dim[1],
    )
    b.bench_function[bench_func](
        BenchId(
            _get_run_name[
                dtype,
                shape_c,
                shape_a,
                shape_b,
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                use_vendor_blas=use_vendor_blas,
            ](shape_c_dim, shape_a_dim, shape_b_dim)
        ),
        # TODO: Pick relevant benchmetric
        [flops],
    )

    # Consume device buffers
    _ = buffer_a^
    _ = buffer_b^
    _ = buffer_c^


fn create_matmul_bench[
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    epilogue: Bool,
    register_based_epilogue: Bool,
](
    ctx: DeviceContext,
    mut b: Bench,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    init_type: InitializationType,
) raises:
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    var dynamic_b_shape = (n.value, k.value) if transpose_b else (
        k.value,
        n.value,
    )

    bench_matmul[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        static_b_shape,
        transpose_b=transpose_b,
        cache_busting=cache_busting,
        use_vendor_blas=use_vendor_blas,
        epilogue=epilogue,
        register_based_epilogue=register_based_epilogue,
    ](
        ctx,
        b,
        (m.value, n.value),
        (m.value, k.value),
        dynamic_b_shape,
        init_type,
    )


def main():
    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()

    var M = Int(arg_parse("M", 1))
    comptime N = env_get_int["N", 1]()
    comptime K = env_get_int["K", 1]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    comptime cache_busting = True
    comptime transpose_b = True
    comptime use_vendor_blas = env_get_bool["use_vendor_blas", False]()
    comptime epilogue = env_get_bool["epilogue", False]()
    comptime register_based_epilogue = env_get_bool[
        "register_based_epilogue", True
    ]()

    var m = Bench()
    with DeviceContext() as ctx:
        create_matmul_bench[
            dtype,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            use_vendor_blas=use_vendor_blas,
            epilogue=epilogue,
            register_based_epilogue=register_based_epilogue,
        ](
            ctx,
            m,
            dynamic(M),
            static[N](),
            static[K](),
            init_type,
        )

    m.dump_report()
