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

from std.testing import assert_equal
from max.gpu.host import DeviceContext, FuncAttribute
from layout import TileTensor, row_major
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import MatmulConfig
from std.utils.index import Index


def test_epilogue_owners[
    dtype: DType, PARTITIONS: Int
](ctx: DeviceContext,) raises:
    comptime M = 32
    comptime N = 32
    comptime K = 512
    comptime GUARD = 32
    comptime config = MatmulConfig[dtype, dtype, dtype, True](
        block_tile_shape=Index(32, 32, 32),
        warp_tile_shape=Index(16, 16, 32),
        num_pipeline_stages=3,
        num_warp_k_partitions=PARTITIONS,
    )
    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N + GUARD)
    with a_device.map_to_host() as a, b_device.map_to_host() as b, c_device.map_to_host() as c:
        for row in range(M):
            for kk in range(K):
                a[row * K + kk] = Scalar[dtype]((row + kk) % 7 - 3) / 8
        for col in range(N):
            for kk in range(K):
                b[col * K + kk] = Scalar[dtype]((col + 3 * kk) % 11 - 5) / 8
        for i in range(M * N + GUARD):
            c[i] = Scalar[dtype](-31)
    var a = TileTensor(a_device, row_major[M, K]()).as_imm()
    var b = TileTensor(b_device, row_major[N, K]()).as_imm()
    var c = TileTensor(c_device, row_major[M, N]())
    # Call the legacy multistage kernel itself, not the AMD facade that selects
    # AMDMatmul and would bypass the local cast owner being exercised.
    comptime kernel = multistage_gemm_kernel[
        dtype,
        c.LayoutType,
        dtype,
        a.LayoutType,
        dtype,
        b.LayoutType,
        True,
        c_linear_idx_type=c.linear_idx_type,
        a_linear_idx_type=a.linear_idx_type,
        b_linear_idx_type=b.linear_idx_type,
        config=config,
    ]
    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        grid_dim=config.grid_dim(M, N),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )
    with a_device.map_to_host() as a, b_device.map_to_host() as b, c_device.map_to_host() as c:
        for row in range(M):
            for col in range(N):
                var expected = Float64(0)
                for kk in range(K):
                    expected += Float64(a[row * K + kk]) * Float64(
                        b[col * K + kk]
                    )
                assert_equal(c[row * N + col], expected.cast[dtype]())
        for i in range(M * N, M * N + GUARD):
            assert_equal(c[i], Scalar[dtype](-31))


def main() raises:
    var ctx = DeviceContext()
    test_epilogue_owners[.bfloat16, 1](ctx)
    test_epilogue_owners[.float16, 1](ctx)
    test_epilogue_owners[.bfloat16, 2](ctx)
    test_epilogue_owners[.bfloat16, 4](ctx)
