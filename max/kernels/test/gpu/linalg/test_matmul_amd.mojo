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
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from collections.optional import OptionalReg
from random import random_si64

import linalg.vendor_blas
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.matmul_gpu import (
    _matmul_gpu,
    _amdgpu_matmul_config_from_block_shape,
)
from linalg.utils_gpu import MatmulConfig
from testing import assert_equal

from utils import Index


fn test[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    constrained[
        Int(n.dim) > 0 and Int(k.dim) > 0,
        "This test currently requires static N and K.",
    ]()

    var M = m.value
    var N = n.value
    var K = k.value
    print(M, "x", N, "x", K)

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )

    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    alias rand_min = -100
    alias rand_max = 100

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host.tensor.data[i] = val.cast[a_type]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host.tensor.data[i] = val.cast[b_type]()

    for i in range(M * N):
        c_host.tensor.data[i] = 0
        c_host_ref.tensor.data[i] = 0

    # Move operands to the Device

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b, config=config](
        c_device.tensor, a_device.tensor, b_device.tensor, ctx
    )

    vendor_blas.matmul(
        ctx,
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    var errors = 0
    for i in range(M * N):
        # print(i // N, i % N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
        if c_host.tensor.data[i] != c_host_ref.tensor.data[i]:
            # print(i//N, i%N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
            errors += 1

    assert_equal(errors, 0)

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


fn test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    return test[in_type, in_type, out_type, transpose_b](ctx, m, n, k)


fn test[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool, //,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    return test[a_type, b_type, c_type, transpose_b, config](ctx, m, n, k)


def test_bf16(ctx: DeviceContext):
    print("=== test_bf16")

    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=False,
    ](ctx, dynamic(256), static[256](), static[128]())
    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=True,
    ](ctx, dynamic(256), static[256](), static[128]())
    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=False,
    ](ctx, dynamic(256), static[256](), static[128]())
    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=True,
    ](ctx, dynamic(256), static[256](), static[128]())

    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=False,
    ](ctx, dynamic(1024), static[256](), static[128]())
    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=False,
    ](ctx, dynamic(1024), static[256](), static[256]())
    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=True,
    ](ctx, dynamic(1024), static[256](), static[1024]())
    test[
        in_type = DType.bfloat16,
        out_type = DType.float32,
        transpose_b=True,
    ](ctx, dynamic(1024), static[1024](), static[1024]())

    test[
        in_type = DType.bfloat16,
        out_type = DType.bfloat16,
        transpose_b=True,
    ](ctx, dynamic(256), static[284](), static[256]())


def test_float8[in_type: DType](ctx: DeviceContext):
    print("=== test_float8", in_type)

    test[
        in_type=in_type,
        out_type = DType.bfloat16,
        transpose_b=True,
    ](ctx, dynamic(480), static[512](), static[640]())


def test_block_k(ctx: DeviceContext):
    print("=== test_block_k")

    @parameter
    def test_block_k[
        in_type: DType, out_type: DType, block_k: Int
    ](m: ValOrDim, n: ValOrDim, k: ValOrDim):
        alias config = MatmulConfig[in_type, in_type, out_type, True](
            block_tile_shape=Index(64, 64, block_k),
            warp_tile_shape=Index(32, 32, block_k),
        )
        test[config](ctx, m, n, k)

    alias block_ks = List[Int](32, 64, 128, 256)

    @parameter
    for i in range(len(block_ks)):
        test_block_k[DType.bfloat16, DType.bfloat16, block_ks[i]](
            dynamic(192), static[1024](), static[1024]()
        )


def test_warp_k_partitions(ctx: DeviceContext):
    print("=== test_warp_k_partitions")

    @parameter
    def test_warp_k_partitions[
        in_type: DType, out_type: DType
    ](m: ValOrDim, n: ValOrDim, k: ValOrDim):
        alias config_type = MatmulConfig[in_type, in_type, out_type, True]
        alias configs = List[config_type](
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
        )

        @parameter
        for i in range(len(configs)):
            test[configs[i]](ctx, m, n, k)

    test_warp_k_partitions[DType.bfloat16, DType.bfloat16](
        dynamic(16), static[2048](), static[2048]()
    )


def test_matmul_config_from_block_shape(ctx: DeviceContext):
    # This test takes too long to execute for CI, but is maintained here as a useful
    # unit test for verifying changes to parts of the matmul dispatcher.
    print("=== test_matmul_config_from_block_shape")

    alias in_type = DType.bfloat16
    alias out_type = DType.float32
    alias transpose_b = True

    # The test is intended to cover partial and complete blocks.
    var m = static[1012]()
    var n = static[1016]()

    alias block_sizes = [16, 32, 64, 96, 128, 160, 192, 224, 256]

    @parameter
    for block_m in block_sizes:

        @parameter
        for block_n in block_sizes:

            @parameter
            def test_block_shape[block_m: Int, block_n: Int, k: Int]():
                alias config = _amdgpu_matmul_config_from_block_shape[
                    out_type, in_type, in_type, transpose_b, k
                ](block_m, block_n)
                print(
                    block_m,
                    block_n,
                    config.block_tile_shape,
                    config.warp_tile_shape,
                    config.num_warp_k_partitions,
                )
                test[config](ctx, m, n, static[k]())

            @parameter
            if block_m <= 32 and block_n <= 32:
                # Exercise the warp_k partitioning where the number of partitions
                # depends on breaking K into even chunks.
                @parameter
                for k in [256, 384, 512, 768, 1024]:
                    test_block_shape[block_m, block_n, k]()
            else:
                # Exercise the logic where block_k is increased, but only if K is
                # multiple of the increased block size.
                @parameter
                for k in [320, 768]:
                    test_block_shape[block_m, block_n, k]()


def main():
    with DeviceContext() as ctx:
        test_bf16(ctx)
        test_float8[DType.float8_e4m3fnuz](ctx)
        test_float8[DType.float8_e5m2fnuz](ctx)
        test_block_k(ctx)
        test_warp_k_partitions(ctx)
