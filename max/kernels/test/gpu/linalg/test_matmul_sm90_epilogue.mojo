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

from std.sys import size_of

from max.gpu.host import DeviceContext
from linalg.matmul.gpu.sm90.testbed import test_matmul_sm90
from linalg.matmul.gpu.tile_scheduler import MatmulSchedule

from std.utils.index import Index
from layout import Coord, Idx

comptime block_tile_shape[wgmma_n: Int, a_dtype: DType] = Index(
    128, wgmma_n, 128 // size_of[a_dtype]()
)
comptime wgmma_shape[wgmma_n: Int, a_dtype: DType] = Index(
    64, wgmma_n, 32 // size_of[a_dtype]()
)


def main() raises:
    with DeviceContext() as ctx:
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[80, .bfloat16],
            wgmma_shape[80, .bfloat16],
            num_consumer=2,
            num_pipeline_stages=8,
            partitioned_multicast=False,
            grid_shape=Index(32, 4),
            schedule=MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](ctx, Int(512), Idx[2560], Idx[8192])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[144, .bfloat16],
            wgmma_shape[144, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(277), Idx[2560], Idx[128])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[232, .bfloat16],
            wgmma_shape[232, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(277), Idx[2560], Idx[128])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(277), Idx[2560], Idx[128])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, .bfloat16],
            wgmma_shape[64, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(393), Idx[8192], Idx[2048])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(532), Idx[8192], Idx[7168])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[64, .bfloat16],
            wgmma_shape[64, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(604), Idx[14336], Idx[8192])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            default_epilogue=True,
        ](ctx, Int(2021), Idx[512], Idx[128])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](
            ctx,
            Idx[8192],
            Idx[2560],
            Idx[8192],
        )

        # Odd N dim
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(100), Idx[331], Idx[1024])

        # Odd N dim and K not multiple of 16B
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[128, .bfloat16],
            wgmma_shape[128, .bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(91), Idx[111], Idx[588])

        @inline(.always)
        def test_lambda_fn_square[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
            comptime assert (
                idx.rank == 2
            ), "the matmul epilogue contract is rank 2"
            return val * val

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            has_compute_fn=True,
        ](ctx, Int(277), Idx[2560], Idx[128], test_lambda_fn_square)

        @inline(.always)
        def test_lambda_add_coords[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
            comptime assert (
                idx.rank == 2
            ), "the matmul epilogue contract is rank 2"
            # Cast indices between 0-1 to avoid accuracy issues
            var col = Int(idx[1])
            var i = Float32(idx[0].value()) / 277.0
            var j = Float32(col - col % 8) / 2560.0
            return val + i.cast[_dtype]() + 2 * j.cast[_dtype]()

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            has_compute_fn=True,
        ](ctx, Int(277), Idx[2560], Idx[128], test_lambda_add_coords)

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[56, .bfloat16],
            wgmma_shape[56, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](
            ctx,
            Idx[1024],
            Idx[168],
            Idx[128],
        )

        # FP32-TF32
        test_matmul_sm90[
            .float32,
            .float32,
            .float32,
            Index(1, 1, 1),
            block_tile_shape[128, .float32],
            wgmma_shape[128, .float32],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, Int(277), Idx[2560], Idx[128])

        test_matmul_sm90[
            .float32,
            .float32,
            .float32,
            Index(2, 2, 1),
            block_tile_shape[256, .float32],
            wgmma_shape[256, .float32],
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](
            ctx,
            Int(1024),
            Idx[256 * 6],
            Idx[128],
        )
