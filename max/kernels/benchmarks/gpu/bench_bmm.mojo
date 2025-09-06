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

from math import align_up
from sys import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    has_nvidia_gpu_accelerator,
    size_of,
    simd_width_of,
)

import linalg.vendor_blas
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import DimList, NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from internal_utils import HostNDBuffer, arg_parse
from internal_utils._utils import (
    InitializationType,
    ValOrDim,
    dynamic,
    initialize,
    random,
    static,
    init_vector_launch,
)
from linalg.bmm import _batched_matmul_gpu

from utils import Index, IndexList

from algorithm.functional import elementwise


fn _get_run_name[
    dtype: DType,
    *,
    transpose_b: Bool,
    use_vendor_blas: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](b: ValOrDim, m: ValOrDim, n: ValOrDim, k: ValOrDim,) -> String:
    var vendor_str = "vendor_bmm" if use_vendor_blas else "bmm"
    var type_str = String("(", dtype, ") : ")
    # B
    var b_str = String(b.value, "" if b.dim else "_dynamic")
    # M
    var m_str = String(m.value, "" if m.dim else "_dynamic")
    # N
    var n_str = String(n.value, "" if n.dim else "_dynamic")
    # K
    var k_str = String(k.value, "" if k.dim else "_dynamic")

    var transpose_b_str = String("/transpose_b=", transpose_b)

    return String(
        vendor_str,
        type_str,
        b_str,
        " x ",
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        transpose_b_str,
    )


alias epilogue_func_type = fn[dtype: DType, width: Int, *, alignment: Int = 1] (
    SIMD[dtype, width]
) capturing -> SIMD[dtype, width]


@always_inline
@parameter
fn elementwise_epilogue_fn[
    dtype: DType,
    width: Int,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


fn bench_bmm[
    dtype: DType,
    /,
    *,
    use_vendor_blas: Bool = False,
    transpose_b: Bool = False,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    mut bench: Bench,
    b: ValOrDim,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    init_type: InitializationType,
) raises:
    var M = m.value
    var N = n.value
    var K = k.value
    var B = b.value

    alias batch_static_a_shape = DimList(b.dim, m.dim, k.dim)
    alias batch_static_b_shape = DimList(
        b.dim, n.dim, k.dim
    ) if transpose_b else DimList(b.dim, k.dim, n.dim)
    alias batch_static_c_shape = DimList(b.dim, m.dim, n.dim)

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)

    var batch_dynamic_a_shape = IndexList[3](b.value, m.value, k.value)
    var batch_dynamic_b_shape = IndexList[3](
        b.value, n.value, k.value
    ) if transpose_b else IndexList[3](b.value, k.value, n.value)

    var batch_dynamic_c_shape = IndexList[3](b.value, m.value, n.value)

    var dynamic_a_shape = IndexList[2](m.value, k.value)
    var dynamic_b_shape = IndexList[2](
        n.value, k.value
    ) if transpose_b else IndexList[2](k.value, n.value)

    var dynamic_c_shape = IndexList[2](m.value, n.value)

    var a_host = HostNDBuffer[dtype, 3, batch_static_a_shape](
        batch_dynamic_a_shape
    )
    var b_host = HostNDBuffer[dtype, 3, batch_static_b_shape](
        batch_dynamic_b_shape
    )
    var c_host = HostNDBuffer[dtype, 3, batch_static_c_shape](
        batch_dynamic_c_shape
    )

    var a_device_buffer = ctx.enqueue_create_buffer[dtype](
        a_host.tensor.num_elements()
    )
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](
        b_host.tensor.num_elements()
    )
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](
        c_host.tensor.num_elements()
    )

    var a_device = NDBuffer[
        dtype, 3, MutableAnyOrigin, batch_static_a_shape, _
    ](a_device_buffer._unsafe_ptr(), batch_dynamic_a_shape)
    var b_device = NDBuffer[
        dtype, 3, MutableAnyOrigin, batch_static_b_shape, _
    ](b_device_buffer._unsafe_ptr(), batch_dynamic_b_shape)
    var c_device = NDBuffer[
        dtype, 3, MutableAnyOrigin, batch_static_c_shape, _
    ](c_device_buffer._unsafe_ptr(), batch_dynamic_c_shape)

    # Initialize data on the device

    init_vector_launch[dtype](
        a_device_buffer, a_host.tensor.num_elements(), init_type, ctx
    )
    init_vector_launch[dtype](
        b_device_buffer, b_host.tensor.num_elements(), init_type, ctx
    )

    @parameter
    @always_inline
    @__copy_capture(c_device)
    fn epilogue_fn[
        dtype: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        alias func = lambda_fn.value()
        var update_val = func(val)
        c_device.store(
            Index(idx[0], idx[1], idx[2]), update_val.cast[c_device.type]()
        )

    alias pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    @always_inline
    @__copy_capture(c_device, B, M, N)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[3]](idx0)
        var val = c_device.load[width=simd_width](idx)
        alias element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device.store(
            idx,
            update_val,
        )

    @parameter
    @__copy_capture(a_device, b_device, c_device)
    @always_inline
    fn bench_func(mut bench: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            @parameter
            if use_vendor_blas:
                # Vendor BMM
                for i in range(B):
                    var c_ptr = c_device.data + (i * M * N)
                    var a_ptr = a_device.data + (i * M * K)
                    var b_ptr = b_device.data + (i * K * N)

                    var c_buffer = NDBuffer[dtype, 2, _, static_c_shape](
                        c_ptr, dynamic_c_shape
                    )
                    var a_buffer = NDBuffer[dtype, 2, _, static_a_shape](
                        a_ptr, dynamic_a_shape
                    )
                    var b_buffer = NDBuffer[dtype, 2, _, static_b_shape](
                        b_ptr, dynamic_b_shape
                    )

                    vendor_blas.matmul(
                        ctx,
                        c_buffer,
                        a_buffer,
                        b_buffer,
                        c_row_major=True,
                        transpose_b=transpose_b,
                    )
                ctx.synchronize()

                # Epilogue
                @parameter
                if lambda_fn:
                    elementwise[func, pack_size, target="gpu"](
                        IndexList[3](B, M, Int(N)),
                        ctx,
                    )
            else:

                @parameter
                if lambda_fn:
                    _batched_matmul_gpu[
                        transpose_b=transpose_b,
                        elementwise_epilogue_fn=epilogue_fn,
                    ](c_device, a_device, b_device, ctx)
                else:
                    _batched_matmul_gpu[transpose_b=transpose_b](
                        c_device, a_device, b_device, ctx
                    )

        bench.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func](
        BenchId(
            _get_run_name[
                dtype,
                transpose_b=transpose_b,
                use_vendor_blas=use_vendor_blas,
                lambda_fn=lambda_fn,
            ](b, m, n, k)
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.flops,
            2 * B * M * N * K,
        ),
    )

    # Retain our buffers till the end.
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^

    _ = a_host^
    _ = b_host^
    _ = c_host^


fn create_bmm_bench[
    dtype: DType,
    *,
    transpose_b: Bool,
    use_vendor_blas: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    mut bench: Bench,
    b: ValOrDim,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    init_type: InitializationType,
) raises:
    bench_bmm[
        dtype,
        transpose_b=transpose_b,
        use_vendor_blas=use_vendor_blas,
        lambda_fn=lambda_fn,
    ](
        ctx,
        bench,
        b,
        m,
        n,
        k,
        init_type,
    )


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    var B = Int(arg_parse("B", 1))
    var M = Int(arg_parse("M", 1))
    alias N = env_get_int["N", 1]()
    alias K = env_get_int["K", 1]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    alias transpose_b = False
    alias use_vendor_blas = env_get_bool["use_vendor_blas", False]()

    var m = Bench()
    with DeviceContext() as ctx:
        create_bmm_bench[
            dtype,
            transpose_b=transpose_b,
            use_vendor_blas=use_vendor_blas,
        ](
            ctx,
            m,
            dynamic(B),
            dynamic(M),
            static[N](),
            static[K](),
            init_type,
        )

    m.dump_report()
