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
from sys import alignof, sizeof
import linalg.vendor_blas
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils._utils import dynamic, static
from linalg.matmul_sm90_testbed import test_matmul_sm90
from linalg.matmul_tile_scheduler import MatmulSchedule
from linalg.utils import elementwise_compute_lambda_type
from utils.index import Index, IndexList

alias block_tile_shape[wgmma_n: Int, a_dtype: DType] = Index(
    128, wgmma_n, 128 // sizeof[a_dtype]()
)
alias wgmma_shape[wgmma_n: Int, a_dtype: DType] = Index(
    64, wgmma_n, 32 // sizeof[a_dtype]()
)


def main():
    with DeviceContext() as ctx:
        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[80, DType.bfloat16],
            wgmma_shape[80, DType.bfloat16],
            num_consumer=2,
            num_pipeline_stages=8,
            partitioned_multicast=False,
            grid_shape = Index(32, 4),
            schedule = MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](ctx, dynamic(512), static[2560](), static[8192]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[144, DType.bfloat16],
            wgmma_shape[144, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(277), static[2560](), static[128]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[232, DType.bfloat16],
            wgmma_shape[232, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(277), static[2560](), static[128]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(277), static[2560](), static[128]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, DType.bfloat16],
            wgmma_shape[64, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(393), static[8192](), static[2048]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(532), static[8192](), static[7168]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[64, DType.bfloat16],
            wgmma_shape[64, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(604), static[14336](), static[8192]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            default_epilogue=True,
        ](ctx, dynamic(2021), static[512](), static[128]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](
            ctx,
            static[8192](),
            static[2560](),
            static[8192](),
        )

        # Odd N dim
        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(100), static[331](), static[1024]())

        # Odd N dim and K not multiple of 16B
        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[128, DType.bfloat16],
            wgmma_shape[128, DType.bfloat16],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(91), static[111](), static[588]())

        @parameter
        @always_inline
        fn test_lambda_fn_square[
            _dtype: DType,
            width: Int,
            *,
            alignment: Int = alignof[SIMD[_dtype, width]](),
        ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
            _dtype, width
        ]:
            return val * val

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            elementwise_compute_lambda_fn = OptionalReg[
                elementwise_compute_lambda_type
            ](test_lambda_fn_square),
            default_epilogue=True,
        ](ctx, dynamic(277), static[2560](), static[128]())

        @parameter
        @always_inline
        fn test_lambda_add_coords[
            _dtype: DType,
            width: Int,
            *,
            alignment: Int = alignof[SIMD[_dtype, width]](),
        ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
            _dtype, width
        ]:
            # Cast indices between 0-1 to avoid accuracy issues
            var i = Float32(idx[0]) / 277.0
            var j = Float32(idx[1] - idx[1] % 8) / 2560.0
            return val + i.cast[_dtype]() + 2 * j.cast[_dtype]()

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            elementwise_compute_lambda_fn = OptionalReg[
                elementwise_compute_lambda_type
            ](test_lambda_add_coords),
            default_epilogue=True,
        ](ctx, dynamic(277), static[2560](), static[128]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[56, DType.bfloat16],
            wgmma_shape[56, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](
            ctx,
            static[1024](),
            static[168](),
            static[128](),
        )

        # FP32-TF32
        test_matmul_sm90[
            DType.float32,
            DType.float32,
            DType.float32,
            Index(1, 1, 1),
            block_tile_shape[128, DType.float32],
            wgmma_shape[128, DType.float32],
            num_consumer=2,
            default_epilogue=True,
        ](ctx, dynamic(277), static[2560](), static[128]())

        test_matmul_sm90[
            DType.float32,
            DType.float32,
            DType.float32,
            Index(2, 2, 1),
            block_tile_shape[256, DType.float32],
            wgmma_shape[256, DType.float32],
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            default_epilogue=True,
        ](
            ctx,
            dynamic(1024),
            static[256 * 6](),
            static[128](),
        )
