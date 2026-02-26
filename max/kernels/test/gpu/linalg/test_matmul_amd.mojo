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
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from random import random_si64

import linalg.matmul.vendor.blas as vendor_blas
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from gpu.host.info import MI355X
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from linalg.matmul.gpu import (
    _amdgpu_matmul_config_from_block_shape,
    _matmul_gpu,
)
from linalg.utils_gpu import MatmulConfig
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from testing import assert_equal

from utils import Index, IndexList

comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


fn test[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    config: Optional[MatmulConfig[a_type, b_type, c_type, transpose_b]] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(m, "x", n, "x", k)

    comptime static_a_shape = DimList(to_dim[M], to_dim[K])
    comptime static_b_shape = DimList(
        to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[K], to_dim[N])
    comptime static_c_shape = DimList(to_dim[M], to_dim[N])

    var dynamic_a_shape = IndexList[2](M.or_else(m), K.or_else(k))
    var dynamic_b_shape = IndexList[2](
        N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[2](K.or_else(k), N.or_else(n))
    var dynamic_c_shape = IndexList[2](M.or_else(m), N.or_else(n))

    var a_size = m * k
    var b_size = n * k if transpose_b else k * n
    var c_size = m * n

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    var a_device = NDBuffer[a_type, 2, _, static_a_shape](
        a_device_buffer.unsafe_ptr(),
        DimList(m, k),
    )
    var b_device = NDBuffer[b_type, 2, _, static_b_shape](
        b_device_buffer.unsafe_ptr(),
        DimList(n, k) if transpose_b else DimList(k, n),
    )
    var c_device = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_buffer.unsafe_ptr(),
        DimList(m, n),
    )
    var c_device_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref_buffer.unsafe_ptr(),
        DimList(m, n),
    )

    comptime rand_min = -100
    comptime rand_max = 100

    for i in range(m * k):
        var val = random_si64(rand_min, rand_max)
        a_host_ptr[i] = val.cast[a_type]()

    for i in range(k * n):
        var val = random_si64(rand_min, rand_max)
        b_host_ptr[i] = val.cast[b_type]()

    for i in range(m * n):
        c_host_ptr[i] = 0
        c_host_ref_ptr[i] = 0

    # Move operands to the Device
    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b, config=config](
        c_device, a_device, b_device, ctx
    )

    vendor_blas.matmul(
        ctx,
        c_device_ref,
        a_device,
        b_device,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var errors = 0
    for i in range(m * n):
        if c_host_ptr[i] != c_host_ref_ptr[i]:
            errors += 1

    assert_equal(errors, 0)

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = c_device_ref_buffer^


fn test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    return test[in_type, in_type, out_type, transpose_b, M=M, N=N, K=K](
        ctx, m, n, k
    )


fn test[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    //,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    return test[a_type, b_type, c_type, transpose_b, config, M=M, N=N, K=K](
        ctx, m, n, k
    )


def test_bf16(ctx: DeviceContext):
    print("=== test_bf16")

    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=False,
        N = Int(256),
        K = Int(128),
    ](ctx, 256, 256, 128)
    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=True,
        N = Int(256),
        K = Int(128),
    ](ctx, 256, 256, 128)
    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=False,
        N = Int(256),
        K = Int(128),
    ](ctx, 256, 256, 128)
    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=True,
        N = Int(256),
        K = Int(128),
    ](ctx, 256, 256, 128)

    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=False,
        N = Int(256),
        K = Int(128),
    ](ctx, 1024, 256, 128)
    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=False,
        N = Int(256),
        K = Int(256),
    ](ctx, 1024, 256, 256)
    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=True,
        N = Int(256),
        K = Int(1024),
    ](ctx, 1024, 256, 1024)
    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=True,
        N = Int(1024),
        K = Int(1024),
    ](ctx, 1024, 1024, 1024)

    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=True,
        N = Int(284),
        K = Int(256),
    ](ctx, 256, 284, 256)


def test_float8[in_type: DType](ctx: DeviceContext):
    print("=== test_float8", in_type)

    test[
        in_type=in_type,
        out_type = DType.bfloat16,
        transpose_b=True,
        N = Int(512),
        K = Int(640),
    ](ctx, 480, 512, 640)


def test_block_k(ctx: DeviceContext):
    print("=== test_block_k")

    @parameter
    def test_block_k[
        in_type: DType,
        out_type: DType,
        block_k: Int,
        N: Int,
        K: Int,
    ](m: Int, n: Int, k: Int):
        comptime config = MatmulConfig[in_type, in_type, out_type, True](
            block_tile_shape=Index(64, 64, block_k),
            warp_tile_shape=Index(32, 32, block_k),
        )
        test[config, N = Int(N), K = Int(K)](ctx, m, n, k)

    comptime block_ks: List[Int] = [32, 64, 128, 256]

    comptime for i in range(len(block_ks)):
        test_block_k[DType.bfloat16, DType.bfloat16, block_ks[i], 1024, 1024](
            192, 1024, 1024
        )


def test_warp_k_partitions(ctx: DeviceContext):
    print("=== test_warp_k_partitions")

    @parameter
    def test_warp_k_partitions[
        in_type: DType,
        out_type: DType,
        N: Int,
        K: Int,
    ](m: Int, n: Int, k: Int):
        comptime config_type = MatmulConfig[in_type, in_type, out_type, True]
        comptime configs: List[config_type] = [
            # TEST: num_warps=(1, 4, 1).
            config_type(
                block_tile_shape=Index(16, 128, 128),
                warp_tile_shape=Index(16, 32, 128),
            ),
            # TEST: num_warps=(1, 1, 4).
            config_type(
                block_tile_shape=Index(16, 16, 64),
                warp_tile_shape=Index(16, 16, 64),
                num_warp_k_partitions=4,
            ),
            config_type(
                block_tile_shape=Index(16, 16, 128),
                warp_tile_shape=Index(16, 16, 128),
                num_warp_k_partitions=4,
            ),
            # TEST: num_warps=(1, 2, 2).
            config_type(
                block_tile_shape=Index(16, 128, 64),
                warp_tile_shape=Index(16, 64, 64),
                num_warp_k_partitions=2,
            ),
        ]

        comptime for i in range(len(configs)):
            test[configs[i], N = Int(N), K = Int(K)](ctx, m, n, k)

    test_warp_k_partitions[DType.bfloat16, DType.bfloat16, 2048, 2048](
        16, 2048, 2048
    )


def test_matmul_config_from_block_shape(ctx: DeviceContext):
    # This test takes too long to execute for CI, but is maintained here as a useful
    # unit test for verifying changes to parts of the matmul dispatcher.
    print("=== test_matmul_config_from_block_shape")

    comptime in_type = DType.bfloat16
    comptime out_type = DType.float32
    comptime transpose_b = True

    # The test is intended to cover partial and complete blocks.
    comptime m_val = 1012
    comptime n_val = 1016

    comptime block_sizes = [16, 32, 64, 96, 128, 160, 192, 224, 256]

    comptime for block_m in block_sizes:
        comptime for block_n in block_sizes:

            @parameter
            def test_block_shape[block_m: Int, block_n: Int, k: Int]():
                comptime config = _amdgpu_matmul_config_from_block_shape[
                    out_type, in_type, in_type, transpose_b, k
                ](Index(block_m, block_n))
                print(
                    block_m,
                    block_n,
                    config.block_tile_shape,
                    config.warp_tile_shape,
                    config.num_warp_k_partitions,
                )
                test[config, M = Int(m_val), N = Int(n_val), K = Int(k)](
                    ctx, m_val, n_val, k
                )

            comptime if block_m <= 32 and block_n <= 32:
                # Exercise the warp_k partitioning where the number of partitions
                # depends on breaking K into even chunks.
                comptime for k in [256, 384, 512, 768, 1024]:
                    test_block_shape[block_m, block_n, k]()
            else:
                # Exercise the logic where block_k is increased, but only if K is
                # multiple of the increased block size.
                comptime for k in [320, 768]:
                    test_block_shape[block_m, block_n, k]()


def main():
    with DeviceContext() as ctx:
        test_bf16(ctx)

        comptime if ctx.default_device_info == MI355X:
            test_float8[DType.float8_e4m3fn](ctx)
            test_float8[DType.float8_e5m2](ctx)
        else:
            test_float8[DType.float8_e4m3fnuz](ctx)
            test_float8[DType.float8_e5m2fnuz](ctx)

        test_block_k(ctx)
        test_warp_k_partitions(ctx)
