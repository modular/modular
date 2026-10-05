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

from std.math import isnan, nan
from std.sys import align_of
from std.testing import assert_almost_equal, assert_true
from std.utils import Index, IndexList

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from layout.tensor_core import get_mma_shape
from linalg.matmul.gpu import _amdgpu_get_mma_shape, multistage_gemm
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig


def run_split_k[
    N: Int,
    K: Int,
    P: Int,
    transpose_b: Bool,
    use_epilogue: Bool = False,
](ctx: DeviceContext, m: Int) raises:
    comptime dtype = DType.float32
    comptime guard_size = 32
    comptime config = MatmulConfig[dtype, dtype, dtype, transpose_b](
        block_tile_shape=Index(
            16, 16, 64
        ) if ctx.target.is_amd_gpu() else Index(128, 128, 32),
        warp_tile_shape=Index(16, 16, 64) if ctx.target.is_amd_gpu() else Index(
            64, 64, 32
        ),
        mma_shape=_amdgpu_get_mma_shape[
            dtype, True
        ]() if ctx.target.is_amd_gpu() else get_mma_shape[dtype, dtype](),
        num_pipeline_stages=1 if ctx.target.is_amd_gpu() else 4,
        num_k_partitions=P,
    )
    print("split-K:", m, N, K, P, transpose_b, use_epilogue)

    var a_device = ctx.enqueue_create_buffer[dtype](m * K)
    var b_device = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device = ctx.enqueue_create_buffer[dtype](m * N + guard_size)
    c_device.enqueue_fill(nan[dtype]())

    with a_device.map_to_host() as a_host, b_device.map_to_host() as b_host:
        for row in range(m):
            for kk in range(K):
                a_host[row * K + kk] = Float32((row + kk) % 7 - 3) / 8
        for col in range(N):
            for kk in range(K):
                var idx = col * K + kk if transpose_b else kk * N + col
                b_host[idx] = Float32((col + 3 * kk) % 11 - 5) / 8

    var a = TileTensor(a_device, row_major(m, Idx[K]))
    var b = TileTensor(
        b_device,
        row_major(Idx[N if transpose_b else K], Idx[K if transpose_b else N]),
    )
    var c = TileTensor(c_device, row_major(m, Idx[N]))

    @inline(.always)
    @__copy_capture(c)
    def epilogue[
        value_type: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[value_type, width]](),
    ](idx: IndexList[2], value: SIMD[value_type, width]) capturing -> None:
        c.store[alignment=alignment](Coord(idx), value.cast[dtype]() + 1)

    multistage_gemm[
        transpose_b=transpose_b,
        config=config,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            epilogue
        ) if use_epilogue else None,
    ](c, a.as_imm(), b.as_imm(), config, ctx)

    with a_device.map_to_host() as a_host, b_device.map_to_host() as b_host, c_device.map_to_host() as c_host:
        for row in range(m):
            for col in range(N):
                var expected = Float64(1 if use_epilogue else 0)
                for kk in range(K):
                    var b_idx = col * K + kk if transpose_b else kk * N + col
                    expected += Float64(a_host[row * K + kk]) * Float64(
                        b_host[b_idx]
                    )
                assert_almost_equal(
                    c_host[row * N + col],
                    Float32(expected),
                    rtol=1e-3,
                    atol=1e-2,
                )
        for idx in range(m * N, m * N + guard_size):
            assert_true(isnan(c_host[idx]), "split-K wrote past the output")


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_amd_gpu():
            run_split_k[32, 512, 4, True](ctx, 1)
            run_split_k[32, 512, 4, True](ctx, 17)
            run_split_k[32, 768, 3, True, True](ctx, 33)
        else:
            run_split_k[128, 512, 4, False](ctx, 17)
            run_split_k[128, 512, 4, True](ctx, 17)
            run_split_k[128, 640, 3, False](ctx, 129)
            run_split_k[128, 640, 3, True, True](ctx, 129)
