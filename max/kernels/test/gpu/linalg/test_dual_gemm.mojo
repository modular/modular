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

from collections import OptionalReg
from math import exp2
from os import abort
from random import rand, randn
from sys import argv, simdwidthof

import benchmark
from algorithm.functional import elementwise
from gpu.host import DeviceContext, FuncAttribute
from gpu.host import get_gpu_target
from layout import Layout
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import UNKNOWN_VALUE, IntTuple
from layout.layout_tensor import LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.dual_gemm import binary_fn_type, multistage_dual_gemm
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig, _bk_base
from testing import assert_almost_equal

from utils import StaticTuple
from utils.index import Index, IndexList
from utils.numerics import FPUtils


fn binary_sub[
    type: DType, width: Int
](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
    return x - y


fn multistage_gemm_simple[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout, //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    binary_lambda_fn: binary_fn_type,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    ctx: DeviceContext,
) raises:
    var M = c.dim[0]()
    var N = c.dim[1]()

    # Dispatch w/o split K
    alias kernel = multistage_gemm_kernel[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b,
        c_layout_int_type = c.layout_int_type,
        c_linear_idx_type = c.linear_idx_type,
        a_layout_int_type = a.layout_int_type,
        a_linear_idx_type = a.linear_idx_type,
        b_layout_int_type = b.layout_int_type,
        b_linear_idx_type = b.linear_idx_type,
        config=config,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        UnsafePointer[Int32](),
        grid_dim=config.grid_dim(M, N),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )


fn naive_dual_gemm[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout, //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    binary_lambda_fn: binary_fn_type,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c01: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b01: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    ctx: DeviceContext,
):
    try:
        multistage_gemm_simple[
            transpose_b=transpose_b,
            config=config,
            binary_lambda_fn=binary_sub,
        ](c01, a, b01, ctx)

        alias simd_width = simdwidthof[
            c_type, target = get_gpu_target["sm_80"]()
        ]()
        alias align = alignof[SIMD[c_type, simd_width]]()

        var M = c01.dim[0]()
        var N = c01.dim[1]() // 2

        @always_inline
        @__copy_capture(c01, N)
        @parameter
        fn binary[simd_width: Int, rank: Int](idx0: IndexList[rank]):
            var m: Int = idx0[0]
            var n: Int = idx0[1]
            c01.vectorize[1, simd_width]()[
                m, n // simd_width
            ] = binary_lambda_fn(
                c01.vectorize[1, simd_width]()[m, n // simd_width],
                c01.vectorize[1, simd_width]()[m, (N + n) // simd_width],
            )

        ctx.synchronize()
        elementwise[binary, simd_width, target="gpu"](IndexList[2](M, N), ctx)
        ctx.synchronize()
    except e:
        abort(String(e))


fn runtime_row_major[
    cols: Int
](
    rows: Int,
    out res: RuntimeLayout[
        Layout(IntTuple(UNKNOWN_VALUE, cols), IntTuple(cols, 1))
    ],
):
    return __type_of(res).row_major(
        IndexList[2, element_type = res.element_type](rows, cols)
    )


fn test_dual_matmul[
    transpose_b: Bool, N: Int = 512, K: Int = 512
](ctx: DeviceContext, M: Int = 512, do_benchmark: Bool = False) raises:
    alias dst_type = DType.float32
    alias src_type = DType.bfloat16
    alias warp_shape = Index(64, 64, _bk_base[src_type]())
    alias config = MatmulConfig[src_type, src_type, dst_type, transpose_b]()
    alias M128_N28672_K4096_config = MatmulConfig[
        src_type, src_type, dst_type, transpose_b
    ](
        block_tile_shape=Index(128, 128, 32),
        warp_tile_shape=warp_shape,
        num_pipeline_stages=4,
        num_k_partitions=1,
    )
    alias M256_N28672_K4096_config_a100 = MatmulConfig[
        src_type, src_type, dst_type, transpose_b
    ](
        block_tile_shape=Index(64, 256, 64),
        warp_tile_shape=warp_shape,
        num_pipeline_stages=4,
        num_k_partitions=1,
    )
    alias M256_N28672_K4096_config = M256_N28672_K4096_config_a100 if M256_N28672_K4096_config_a100.shared_mem_usage() <= ctx.device_info.shared_memory_per_multiprocessor else config
    alias M512_N28672_K4096_config = MatmulConfig[
        src_type, src_type, dst_type, transpose_b
    ](
        block_tile_shape=Index(128, 128, 32),
        warp_tile_shape=warp_shape,
        num_pipeline_stages=4,
        num_k_partitions=1,
    )

    var layout_a = runtime_row_major[K](M)
    alias layout_b = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var layout_c = runtime_row_major[N](M)

    var mat_a = ManagedLayoutTensor[src_type](layout_a, ctx)
    randn(mat_a.tensor().ptr, layout_a.size())
    var mat_b0 = ManagedLayoutTensor[src_type, layout_b](ctx)
    var mat_b1 = ManagedLayoutTensor[src_type, layout_b](ctx)

    rand(mat_b0.tensor().ptr, layout_b.size(), min=0.0, max=1.0)
    rand(mat_b1.tensor().ptr, layout_b.size(), min=-1.0, max=0.0)
    var mat_c = ManagedLayoutTensor[dst_type](layout_c, ctx)

    var mat_c_device_tensor = mat_c.device_tensor()
    var mat_a_device_tensor = mat_a.device_tensor()
    var mat_b0_device_tensor = mat_b0.device_tensor()
    var mat_b1_device_tensor = mat_b1.device_tensor()

    @always_inline
    @parameter
    fn run_dual_gemm() raises:
        if M <= 128:
            multistage_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M128_N28672_K4096_config,
            ](
                mat_c_device_tensor,
                mat_a_device_tensor,
                mat_b0_device_tensor,
                mat_b1_device_tensor,
                ctx,
            )
        elif M <= 256:
            multistage_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M256_N28672_K4096_config,
            ](
                mat_c_device_tensor,
                mat_a_device_tensor,
                mat_b0_device_tensor,
                mat_b1_device_tensor,
                ctx,
            )
        else:
            multistage_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M512_N28672_K4096_config,
            ](
                mat_c_device_tensor,
                mat_a_device_tensor,
                mat_b0_device_tensor,
                mat_b1_device_tensor,
                ctx,
            )

    var dual_gemm_time: Float64 = 0.0
    if do_benchmark:
        dual_gemm_time = benchmark.run[run_dual_gemm](
            max_runtime_secs=5.0
        ).mean()
        print(
            "     DualGEMM[M=",
            M,
            ",N=",
            N,
            ",K=",
            K,
            "] took ",
            1e3 * dual_gemm_time,
            " ms",
            sep="",
        )
    else:
        run_dual_gemm()

    alias layout_b01 = Layout.row_major(
        2 * N, K
    ) if transpose_b else Layout.row_major(K, 2 * N)
    var mat_b01 = ManagedLayoutTensor[src_type, layout_b01](ctx)
    alias src_simd_width = simdwidthof[src_type]()

    var mat_b01v = mat_b01.tensor().vectorize[1, src_simd_width]()
    var mat_b0_tensor = mat_b0.tensor()
    var mat_b1_tensor = mat_b1.tensor()

    @parameter
    if transpose_b:
        for n in range(N):
            for k in range(K // src_simd_width):
                mat_b01v[n, k] = rebind[SIMD[src_type, mat_b01v.element_size]](
                    mat_b0_tensor.vectorize[1, src_simd_width]()[n, k]
                )
                mat_b01v[N + n, k] = rebind[
                    SIMD[src_type, mat_b01v.element_size]
                ](mat_b1_tensor.vectorize[1, src_simd_width]()[n, k])
    else:
        alias Niter = N // src_simd_width
        for k in range(K):
            for n in range(Niter):
                mat_b01v[k, n] = rebind[SIMD[src_type, mat_b01v.element_size]](
                    mat_b0_tensor.vectorize[1, src_simd_width]()[k, n]
                )
                mat_b01v[k, Niter + n] = rebind[
                    SIMD[src_type, mat_b01v.element_size]
                ](mat_b1_tensor.vectorize[1, src_simd_width]()[k, n])

    _ = mat_b0^
    _ = mat_b1^

    var layout_c01 = runtime_row_major[2 * N](M)
    var mat_c01 = ManagedLayoutTensor[dst_type](layout_c01, ctx)
    var mat_c01_device_tensor = mat_c01.device_tensor()
    var mat_b01_device_tensor = mat_b01.device_tensor()

    @always_inline
    @parameter
    fn run_naive_dual_gemm() raises:
        if M <= 128:
            naive_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M128_N28672_K4096_config,
            ](
                mat_c01_device_tensor,
                mat_a_device_tensor,
                mat_b01_device_tensor,
                ctx,
            )
        elif M <= 256:
            naive_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M256_N28672_K4096_config,
            ](
                mat_c01_device_tensor,
                mat_a_device_tensor,
                mat_b01_device_tensor,
                ctx,
            )
        else:
            naive_dual_gemm[
                transpose_b=transpose_b,
                binary_lambda_fn=binary_sub,
                config=M512_N28672_K4096_config,
            ](
                mat_c01_device_tensor,
                mat_a_device_tensor,
                mat_b01_device_tensor,
                ctx,
            )

    if do_benchmark:
        var dgs = benchmark.run[run_naive_dual_gemm](
            max_runtime_secs=5.0
        ).mean()
        print(
            "NaiveDualGEMM[M=",
            M,
            ",N=",
            N,
            ",K=",
            K,
            "] took ",
            1e3 * dgs,
            " ms (",
            round(100 * dgs / dual_gemm_time, ndigits=5),
            "%)",
            sep="",
        )
    else:
        run_naive_dual_gemm()
    var mat_c_ref = mat_c01.tensor().split[axis=1](count=2, idx=0)
    var mat_c_tensor = mat_c.tensor()
    _ = mat_a^
    _ = mat_b01^

    alias cbrt_eps = exp2(FPUtils[dst_type].mantissa_width() / -3)
    alias dst_simd_width = simdwidthof[dst_type]()
    # elementwise
    for m in range(M):
        for n in range(N // dst_simd_width):
            assert_almost_equal(
                rebind[SIMD[dst_type, dst_simd_width]](
                    mat_c_tensor.vectorize[1, dst_simd_width]()[m, n]
                ),
                rebind[SIMD[dst_type, dst_simd_width]](
                    mat_c_ref.vectorize[1, dst_simd_width]()[m, n]
                ),
                atol=cbrt_eps,
                rtol=cbrt_eps,
            )

    _ = mat_c^
    _ = mat_c01^


fn main() raises:
    var do_benchmark: Bool = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--benchmark" or args[i] == "--benchmark=yes":
            do_benchmark = True
    with DeviceContext() as ctx:
        test_dual_matmul[transpose_b=False](ctx, do_benchmark=do_benchmark)
        test_dual_matmul[transpose_b=True](ctx, do_benchmark=do_benchmark)
        alias Ms = StaticTuple[Int, 3](128, 256, 1024)
        alias Ms_transpose = StaticTuple[Int, 4](1, 128, 256, 1024)
        alias N = 14336
        alias K = 4096
        for m_idx in range(len(Ms)):
            var M = Ms[m_idx]
            test_dual_matmul[transpose_b=False, N=N, K=K](
                ctx, M=M, do_benchmark=do_benchmark
            )

        for m_idx in range(len(Ms_transpose)):
            var M = Ms_transpose[m_idx]
            test_dual_matmul[transpose_b=True, N=N, K=K](
                ctx, M=M, do_benchmark=do_benchmark
            )
