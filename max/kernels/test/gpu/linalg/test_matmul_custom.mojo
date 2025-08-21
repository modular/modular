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

from math import ceildiv, isclose
from random import random_float64

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.info import A100
from linalg.bmm import _batched_matmul_gpu
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive, multistage_gemm
from linalg.utils_gpu import MatmulConfig, MatmulKernels, select_config
from testing import assert_almost_equal

from utils import Index, IndexList
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout


fn run_matmul_naive(ctx: DeviceContext, M: Int, N: Int, K: Int) raises:
    print("== run_matmul naive kernel")

    var a_host = UnsafePointer[BFloat16].alloc(M * K)
    var b_host = UnsafePointer[BFloat16].alloc(K * N)
    var c_host = UnsafePointer[BFloat16].alloc(M * N)
    var a_host_n = UnsafePointer[Float32].alloc(M * K)
    var b_host_n = UnsafePointer[Float32].alloc(K * N)
    var c_host_n = UnsafePointer[Float32].alloc(M * N)

    var rand_min = -1.0
    var rand_max = 1.0

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        a_host[i] = val.cast[DType.bfloat16]()
        a_host_n[i] = a_host[i].cast[DType.float32]()

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        b_host[i] = val.cast[DType.bfloat16]()
        b_host_n[i] = b_host[i].cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_n[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var a_device_n = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_n = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_n = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias BLOCK_DIM = 16

    # Create layout tensors for bf16 kernel
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var c_tensor_bf16 = LayoutTensor[DType.bfloat16, layout, MutableAnyOrigin](
        c_device._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor_bf16 = LayoutTensor[DType.bfloat16, layout, MutableAnyOrigin](
        a_device._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor_bf16 = LayoutTensor[DType.bfloat16, layout, MutableAnyOrigin](
        b_device._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    @always_inline
    @parameter
    fn run_func_bf16() raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                c_tensor_bf16.layout,
                a_tensor_bf16.layout,
                b_tensor_bf16.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor_bf16,
            a_tensor_bf16,
            b_tensor_bf16,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_bf16()

    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    # Create layout tensors for fp32 kernel
    var c_tensor_fp32 = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor_fp32 = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        a_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor_fp32 = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        b_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    @always_inline
    @parameter
    fn run_func_fp32() raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.float32,
                DType.float32,
                DType.float32,
                c_tensor_fp32.layout,
                a_tensor_fp32.layout,
                b_tensor_fp32.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor_fp32,
            a_tensor_fp32,
            b_tensor_fp32,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_fp32()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i].cast[DType.bfloat16]()
        assert_almost_equal(out_val, out_ref)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


fn run_matmul[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    rtol: Float64 = 1e-05,
    atol: Float64 = 0.1,
    rng_width: Float64 = Float64(100.0),
    debug: Bool = True,
) raises:
    print("== run_matmul kernel => ", dtype, M, N, K)

    var a_host = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[dtype]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[dtype]].alloc(M * N)
    var a_host_n = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var b_host_n = UnsafePointer[Scalar[dtype]].alloc(K * N)
    var c_host_n = UnsafePointer[Scalar[dtype]].alloc(M * N)

    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        a_host[i] = val.cast[dtype]()
        a_host_n[i] = a_host[i]

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        b_host[i] = val.cast[dtype]()
        b_host_n[i] = b_host[i]

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[dtype]()
        c_host_n[i] = c_host[i]

    alias a_shape = DimList(M, K)
    alias b_shape = DimList(K, N)
    alias c_shape = DimList(M, N)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var a_buf = NDBuffer[dtype, 2, _, a_shape](
        a_device._unsafe_ptr(), Index(M, K)
    )
    var b_buf = NDBuffer[dtype, 2, _, b_shape](
        b_device._unsafe_ptr(), Index(K, N)
    )
    var c_buf = NDBuffer[dtype, 2, _, c_shape](
        c_device._unsafe_ptr(), Index(M, N)
    )

    var a_device_n = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device_n = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device_n = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    _matmul_gpu(c_buf, a_buf, b_buf, ctx)
    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    alias BLOCK_DIM = 16

    # Create layout tensors for naive kernel
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var c_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        c_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        a_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        b_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    @always_inline
    @parameter
    fn run_func_naive() raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                dtype,
                dtype,
                dtype,
                c_tensor.layout,
                a_tensor.layout,
                b_tensor.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_naive()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        if debug:
            if not isclose(out_val, out_ref, rtol=rtol, atol=atol):
                print(i, out_val, out_ref)
        assert_almost_equal(out_val, out_ref, rtol=rtol, atol=atol)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


fn run_matmul_split_k[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, False],
](
    ctx: DeviceContext,
    rtol: Float64 = 1e-05,
    atol: Float64 = 0.1,
    rng_width: Float64 = Float64(100.0),
    debug: Bool = True,
) raises:
    print(
        "== run_matmul kernel split_k => ",
        String(dtype),
        M,
        N,
        K,
    )

    var a_host = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[dtype]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[dtype]].alloc(M * N)
    var c_host_n = UnsafePointer[Scalar[dtype]].alloc(M * N)

    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        a_host[i] = val.cast[dtype]()

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        b_host[i] = val.cast[dtype]()

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[dtype]()
        c_host_n[i] = c_host[i]

    alias a_shape = DimList(M, K)
    alias b_shape = DimList(K, N)
    alias c_shape = DimList(M, N)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var a_buf = NDBuffer[dtype, 2, _, a_shape](
        a_device._unsafe_ptr(), Index(M, K)
    )
    var b_buf = NDBuffer[dtype, 2, _, b_shape](
        b_device._unsafe_ptr(), Index(K, N)
    )
    var c_buf = NDBuffer[dtype, 2, _, c_shape](
        c_device._unsafe_ptr(), Index(M, N)
    )

    var a_device_n = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device_n = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device_n = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var best_config = select_config[dtype, dtype, dtype, False](M, N, K, ctx)

    multistage_gemm[transpose_b=False, config=config](
        rebind[NDBuffer[dtype, 2, c_buf.origin, c_shape]](c_buf),
        rebind[NDBuffer[dtype, 2, a_buf.origin, a_shape]](a_buf),
        rebind[NDBuffer[dtype, 2, b_buf.origin, b_shape]](b_buf),
        best_config,
        ctx,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    # running naive
    ctx.enqueue_copy(a_device_n, a_host)
    ctx.enqueue_copy(b_device_n, b_host)

    alias BLOCK_DIM = 16

    # Create layout tensors for naive kernel
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var c_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        c_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        a_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        b_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    ctx.enqueue_function[
        matmul_kernel_naive[
            dtype,
            dtype,
            dtype,
            c_tensor.layout,
            a_tensor.layout,
            b_tensor.layout,
            BLOCK_DIM,
        ]
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        if debug:
            if not isclose(out_val, out_ref, rtol=rtol, atol=atol):
                print(i, out_val, out_ref)
        assert_almost_equal(out_val, out_ref, rtol=rtol, atol=atol)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_n


fn run_matmul_transpose[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    rtol: Float64 = 1e-05,
    atol: Float64 = 0.1,
    rng_width: Float64 = Float64(100.0),
    debug: Bool = True,
) raises:
    print("== run_matmul kernel transpose => ", String(dtype), M, N, K)

    alias transpose_b = True
    var a_host = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[dtype]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[dtype]].alloc(M * N)
    var a_host_n = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var b_host_n = UnsafePointer[Scalar[dtype]].alloc(K * N)
    var c_host_n = UnsafePointer[Scalar[dtype]].alloc(M * N)

    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        a_host[i] = val.cast[dtype]()
        a_host_n[i] = a_host[i]

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[DType.float32]()
        b_host[i] = val.cast[dtype]()
        b_host_n[i] = b_host[i]

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[dtype]()
        c_host_n[i] = c_host[i]

    alias a_shape = DimList(M, K)
    alias b_shape = DimList(N, K)
    alias c_shape = DimList(M, N)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var a_buf = NDBuffer[dtype, 2, _, a_shape](
        a_device._unsafe_ptr(), Index(M, K)
    )
    var b_buf = NDBuffer[dtype, 2, _, b_shape](
        b_device._unsafe_ptr(), Index(N, K)
    )
    var c_buf = NDBuffer[dtype, 2, _, c_shape](
        c_device._unsafe_ptr(), Index(M, N)
    )

    var a_device_n = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device_n = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device_n = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    _matmul_gpu[transpose_b=transpose_b, use_tensor_core=True](
        c_buf, a_buf, b_buf, ctx
    )
    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    alias BLOCK_DIM = 16

    # Create layout tensors for naive kernel
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var c_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        c_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        a_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
        b_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](N, K)),
    )

    @always_inline
    @parameter
    fn run_func_naive() raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                dtype,
                dtype,
                dtype,
                c_tensor.layout,
                a_tensor.layout,
                b_tensor.layout,
                BLOCK_DIM,
                transpose_b,
            ]
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_naive()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        if debug:
            if not isclose(out_val, out_ref, rtol=rtol, atol=atol):
                print(i, out_val, out_ref)
        assert_almost_equal(out_val, out_ref, rtol=rtol, atol=atol)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


fn run_batched_matmul(
    ctx: DeviceContext, B: Int, M: Int, N: Int, K: Int
) raises:
    print("== test_batched_matmul")

    var a_host = UnsafePointer[BFloat16].alloc(B * M * K)
    var b_host = UnsafePointer[BFloat16].alloc(B * K * N)
    var c_host = UnsafePointer[BFloat16].alloc(B * M * N)
    var a_host_n = UnsafePointer[Float32].alloc(B * M * K)
    var b_host_n = UnsafePointer[Float32].alloc(B * K * N)
    var c_host_n = UnsafePointer[Float32].alloc(B * M * N)

    var rand_min = -100.0
    var rand_max = 100.0

    for i in range(B * M * K):
        var val = random_float64(rand_min, rand_max)
        a_host[i] = val.cast[DType.bfloat16]()
        a_host_n[i] = a_host[i].cast[DType.float32]()

    for i in range(B * K * N):
        var val = random_float64(rand_min, rand_max)
        b_host[i] = val.cast[DType.bfloat16]()
        b_host_n[i] = b_host[i].cast[DType.float32]()

    for i in range(B * M * N):
        c_host[i] = 0
        c_host_n[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](B * M * K)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](B * K * N)
    var c_device = ctx.enqueue_create_buffer[DType.bfloat16](B * M * N)
    var a_buf = NDBuffer[DType.bfloat16, 3](
        a_device._unsafe_ptr(), Index(B, M, K)
    )
    var b_buf = NDBuffer[DType.bfloat16, 3](
        b_device._unsafe_ptr(), Index(B, K, N)
    )
    var c_buf = NDBuffer[DType.bfloat16, 3](
        c_device._unsafe_ptr(), Index(B, M, N)
    )

    var a_device_n = ctx.enqueue_create_buffer[DType.float32](B * M * K)
    var b_device_n = ctx.enqueue_create_buffer[DType.float32](B * K * N)
    var c_device_n = ctx.enqueue_create_buffer[DType.float32](B * M * N)
    var a_buf_n = NDBuffer[DType.float32, 3](
        a_device_n._unsafe_ptr(), Index(B, M, K)
    )
    var b_buf_n = NDBuffer[DType.float32, 3](
        b_device_n._unsafe_ptr(), Index(B, K, N)
    )
    var c_buf_n = NDBuffer[DType.float32, 3](
        c_device_n._unsafe_ptr(), Index(B, M, N)
    )

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    @always_inline
    @__copy_capture(c_buf)
    @parameter
    fn elementwise_epilogue_fn1[
        c_type: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[c_type, width]) -> None:
        c_buf.store(Index(idx[0], idx[1], idx[2]), val.cast[c_buf.type]() + 2)

    _batched_matmul_gpu[elementwise_epilogue_fn=elementwise_epilogue_fn1](
        c_buf, a_buf, b_buf, ctx
    )

    ctx.enqueue_copy(c_host, c_device)

    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    @always_inline
    @__copy_capture(c_buf_n)
    @parameter
    fn elementwise_epilogue_fn2[
        c_type: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[c_type, width]) -> None:
        c_buf_n.store(
            Index(idx[0], idx[1], idx[2]), val.cast[c_buf_n.type]() + 2
        )

    _batched_matmul_gpu[elementwise_epilogue_fn=elementwise_epilogue_fn2](
        c_buf_n, a_buf_n, b_buf_n, ctx
    )

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(B * M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i].cast[DType.bfloat16]()
        assert_almost_equal(out_val, out_ref, rtol=1e-02)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


def main():
    with DeviceContext() as ctx:
        alias kernels = MatmulKernels[
            DType.bfloat16, DType.bfloat16, DType.bfloat16, False
        ]()
        alias config = kernels.ampere_256x128_3 if ctx.default_device_info is A100 else kernels.ampere_128x128_4
        run_matmul_split_k[DType.bfloat16, 512, 4096, 14336, config](
            ctx, atol=1.0, rng_width=1.0
        )

        run_matmul_split_k[
            DType.bfloat16, 128, 128, 4096, kernels.ampere_128x128_4
        ](ctx, atol=0.5, rng_width=1.0)

        run_matmul_transpose[DType.bfloat16, 1, 200, 300](
            ctx, atol=0.25, rng_width=1.0
        )
        run_matmul_transpose[DType.bfloat16, 1, 300, 200](
            ctx, atol=0.25, rng_width=1.0
        )
        run_matmul_transpose[DType.bfloat16, 1, 5120, 3072](
            ctx, atol=0.25, rng_width=1.0
        )
        run_matmul_transpose[DType.bfloat16, 1, 12288, 3072](
            ctx, atol=0.5, rng_width=1.0
        )
        run_matmul_transpose[DType.bfloat16, 1, 5120, 12288](
            ctx, atol=0.5, rng_width=1.0
        )
        run_matmul_transpose[DType.bfloat16, 1, 131072, 5120](
            ctx, atol=0.5, rng_width=1.0
        )
        run_matmul_transpose[DType.bfloat16, 1, 3072, 12288](
            ctx, atol=0.5, rng_width=1.0
        )

        run_matmul[DType.bfloat16, 128, 128, 128](ctx)
        run_matmul[DType.bfloat16, 32, 32, 32](ctx)
        run_matmul[DType.bfloat16, 1024, 1, 1024](ctx, atol=0.2, rng_width=1.0)
        run_matmul[DType.bfloat16, 1, 1024, 1024](ctx)

        # KERN-1807 We need to systematically test the float16 kernels.
        # run_matmul[DType.float16, 128, 128, 128](ctx, rng_width=10.0)
        # run_matmul[DType.float16, 32, 32, 32](ctx, rng_width=10.0)
        # run_matmul[DType.float16, 1024, 1, 1024](ctx, 1e-03, rng_width=10.0)
        # run_matmul[DType.float16, 1, 1024, 1024](ctx, 1e-01, rng_width=10.0)

        run_batched_matmul(ctx, 1, 32, 32, 32)
        run_batched_matmul(ctx, 3, 32, 32, 32)
