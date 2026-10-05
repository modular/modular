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

from std.random import rand

from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import (
    _amdgpu_get_mma_shape,
    _amdgpu_matmul_config_from_block_shape,
    multistage_gemm,
)
from linalg.utils_gpu import MatmulConfig
from std.testing import assert_equal

from std.utils import Index, IndexList


def test_epilogue_fn[
    a_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    config: MatmulConfig[a_type, a_type, c_type, True],
](ctx: DeviceContext, m: Int) raises:
    """Checks that the closure-value overloads store exactly what the
    legacy path computes, passed through the closure."""
    print(a_type, "->", c_type, m, "x", N, "x", K, config.num_k_partitions)

    var a_dev = ctx.enqueue_create_buffer[a_type](m * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_out_dev = ctx.enqueue_create_buffer[c_type](m * N)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        rand(ha.unsafe_ptr(), m * K, min=-1.0, max=1.0)
        rand(hb.unsafe_ptr(), N * K, min=-1.0, max=1.0)
    ctx.enqueue_memset(c_ref_dev, 0)
    ctx.enqueue_memset(c_dev, 0)
    ctx.enqueue_memset(c_out_dev, 0)

    var a = TileTensor(a_dev, row_major(m, Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_ref = TileTensor(c_ref_dev, row_major(m, Idx[N]))
    var c = TileTensor(c_dev, row_major(m, Idx[N]))
    var c_out = TileTensor(c_out_dev, row_major(m, Idx[N]))

    def store_doubled[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: IndexList[2], val: SIMD[dtype, width]) {var c_out}:
        c_out.store(Coord(idx[0], idx[1]), rebind[SIMD[c_type, width]](val * 2))

    comptime if config.num_k_partitions > 1:
        multistage_gemm[transpose_b=True, config=config](
            c_ref, a, b, config, ctx
        )
        multistage_gemm[transpose_b=True, config=config](
            c, a, b, config, store_doubled, ctx
        )
    else:
        multistage_gemm[transpose_b=True, config=config](c_ref, a, b, ctx)
        multistage_gemm[transpose_b=True, config=config](
            c, a, b, store_doubled, ctx
        )

    with c_ref_dev.map_to_host() as h_ref, c_dev.map_to_host() as h_c, c_out_dev.map_to_host() as h_out:
        for i in range(m * N):
            assert_equal(h_out[i], h_ref[i] * 2)
            assert_equal(h_c[i], 0)

    _ = a_dev^
    _ = b_dev^
    _ = c_ref_dev^
    _ = c_dev^
    _ = c_out_dev^


def main() raises:
    with DeviceContext() as ctx:
        comptime bf16_config = _amdgpu_matmul_config_from_block_shape[
            .float32, .bfloat16, .bfloat16, True, 512
        ](Index(128, 128))
        test_epilogue_fn[.bfloat16, .float32, 256, 512, bf16_config](ctx, 130)

        comptime fp8_type = (
            DType.float8_e4m3fn if ctx.default_device_info
            == MI355X else DType.float8_e4m3fnuz
        )
        comptime fp8_config = _amdgpu_matmul_config_from_block_shape[
            .bfloat16, fp8_type, fp8_type, True, 512
        ](Index(128, 128))
        # Standard, skinny, and ping-pong branches of the FP8 dispatch.
        for m in [64, 300, 640]:
            test_epilogue_fn[fp8_type, .bfloat16, 4096, 512, fp8_config](ctx, m)

        comptime split_k_config = MatmulConfig[
            .float32, .float32, .float32, True
        ](
            block_tile_shape=Index(16, 16, 64),
            warp_tile_shape=Index(16, 16, 64),
            mma_shape=_amdgpu_get_mma_shape[.float32, True](),
            num_pipeline_stages=1,
            num_k_partitions=4,
        )
        test_epilogue_fn[.float32, .float32, 128, 6144, split_k_config](ctx, 16)
