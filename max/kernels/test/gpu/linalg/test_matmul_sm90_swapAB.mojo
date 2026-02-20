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
"""Tests for swapAB matmul optimization comparing normal vs swapAB execution."""

from collections import Optional
from sys import align_of

from gpu.host import DeviceContext
from internal_utils._utils import dynamic, static
from linalg.matmul.gpu.sm90.config import MatmulConfig as MatmulConfigSM90
from linalg.matmul.gpu.sm90.testbed_swapAB import (
    test_matmul_sm90_swapAB_comparison,
    test_matmul_sm90_swapAB_comparison_v2,
)
from linalg.utils import elementwise_compute_lambda_type

from utils.index import Index, IndexList

comptime bf16 = DType.bfloat16


def main():
    print("\n" + "=" * 60)
    print("SWAPAB COMPARISON TEST SUITE")
    print("=" * 60 + "\n")

    with DeviceContext() as ctx:
        # =====================================================================
        # Test 1: 16x64x256 (M=16, N=64, K=256)
        # =====================================================================
        print("\n" + "=" * 60)
        print("TEST: 16x64x256 (M=16, N=64, K=256)")
        print("=" * 60 + "\n")

        # Build configs using MatmulConfigSM90's heuristics
        comptime config_1 = MatmulConfigSM90[bf16, bf16, bf16](
            m=16, n=64, k=256, swapAB=False
        )
        comptime config_1_swapAB = MatmulConfigSM90[bf16, bf16, bf16](
            m=16, n=64, k=256, swapAB=True
        )

        test_matmul_sm90_swapAB_comparison[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            config=config_1,
            config_swapAB=config_1_swapAB,
        ](ctx, dynamic(16), static[64](), static[256]())

        # =====================================================================
        # Test 2: 16x80x1024 (M=16, N=80, K=1024)
        # =====================================================================

        print("\n" + "=" * 60)
        print("TEST: 16x80x1024 (M=16, N=80, K=1024)")
        print("=" * 60 + "\n")

        comptime config_2 = MatmulConfigSM90[bf16, bf16, bf16](
            m=16, n=80, k=1024, swapAB=False
        )
        comptime config_2_swapAB = MatmulConfigSM90[bf16, bf16, bf16](
            m=16, n=80, k=1024, swapAB=True
        )

        test_matmul_sm90_swapAB_comparison[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            config=config_2,
            config_swapAB=config_2_swapAB,
        ](ctx, dynamic(16), static[80](), static[1024]())

        # =====================================================================
        # Test 3: 32x128x1024 (M=32, N=128, K=1024)
        # =====================================================================
        print("\n" + "=" * 60)
        print("TEST: 32x128x1024 (M=32, N=128, K=1024)")
        print("=" * 60 + "\n")

        comptime config_3 = MatmulConfigSM90[bf16, bf16, bf16](
            m=32, n=128, k=1024, swapAB=False
        )
        comptime config_3_swapAB = MatmulConfigSM90[bf16, bf16, bf16](
            m=32, n=128, k=1024, swapAB=True
        )

        test_matmul_sm90_swapAB_comparison[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            config=config_3,
            config_swapAB=config_3_swapAB,
        ](ctx, dynamic(32), static[128](), static[1024]())

        # =====================================================================
        # Test 4: 32x128x1024 (M=32, N=128, K=1024)
        # =====================================================================

        print("\n" + "=" * 60)
        print("TEST: 476x1024x128 (M=476, N=1024, K=128)")
        print("=" * 60 + "\n")

        # tests 2sm
        # num consecutive stmatrix > 1
        # tests iteratively storing tiles when BMxBN is larger
        # then shared memory size (num_tiles == 2)
        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=128,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=2,
            BM_SWAPAB=128,
            BN_SWAPAB=256,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=256,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            use_vendor_reference=True,
        ](ctx, dynamic(476), static[1024](), static[128]())

        print("\n" + "=" * 60)
        print("TEST: 48x1536x4096 (M=48, N=1536, K=4096)")
        print("=" * 60 + "\n")

        # tests 2sm
        # num consecutive stmatrix == 1
        # tests iteratively storing tiles when BMxBN is larger (num_tiles == 3)
        # where shared memory tile is not equal
        # then shared memory size
        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=64,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=1,
            BM_SWAPAB=128,
            BN_SWAPAB=48,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=48,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            use_vendor_reference=True,
        ](ctx, dynamic(48), static[4096](), static[1536]())

        print("\n" + "=" * 60)
        print("TEST: 48x1536x4096 (M=48, N=1536, K=4096)")
        print("=" * 60 + "\n")

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=64,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=1,
            BM_SWAPAB=128,
            BN_SWAPAB=40,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=40,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            use_vendor_reference=True,
        ](ctx, dynamic(48), static[4096](), static[1536]())

        print("\n" + "=" * 60)
        print("TEST: 100x1536x4096 (M=100, N=1536, K=4096)")
        print("=" * 60 + "\n")

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=64,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=1,
            BM_SWAPAB=128,
            BN_SWAPAB=96,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=96,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            use_vendor_reference=True,
        ](ctx, dynamic(100), static[1536](), static[4096]())

        print("\n" + "=" * 60)
        print("TEST: 249x4096x1536 (M=249, N=4096, K=1536)")
        print("=" * 60 + "\n")

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=128,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=2,
            BM_SWAPAB=128,
            BN_SWAPAB=248,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=248,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            use_vendor_reference=True,
        ](ctx, dynamic(249), static[4096](), static[1536]())

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=128,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=2,
            BM_SWAPAB=64,
            BN_SWAPAB=8,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=8,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=1,
            use_vendor_reference=True,
        ](ctx, dynamic(249), static[4096](), static[1536]())

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=128,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=2,
            BM_SWAPAB=64,
            BN_SWAPAB=8,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=8,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=1,
            use_vendor_reference=True,
        ](ctx, dynamic(15), static[17](), static[1536]())

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=128,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=2,
            BM_SWAPAB=128,
            BN_SWAPAB=256,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=256,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            use_vendor_reference=True,
        ](ctx, dynamic(800), static[999](), static[1536]())

        # =================================================================
        # EPILOGUE TESTS
        # =================================================================
        print("\n" + "=" * 60)
        print("SWAPAB EPILOGUE TESTS")
        print("=" * 60 + "\n")

        # Test with default_epilogue (triggers _write_with_transform path)
        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=128,
            BN=256,
            BK=64,
            MMA_M=64,
            MMA_N=256,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=2,
            BM_SWAPAB=128,
            BN_SWAPAB=256,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=256,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            default_epilogue=True,
            use_vendor_reference=True,
        ](ctx, dynamic(500), static[2232](), static[64]())

        # Test with coordinate-based lambda to verify indexing pattern
        # Each position gets val + row + col * 0.5, creating unique values
        # Using larger offsets that are significant in bf16 precision
        # If indexing is wrong, the reference and swapAB results will mismatch
        @parameter
        @always_inline
        fn coord_lambda[
            _dtype: DType,
            width: Int,
            *,
            alignment: Int = align_of[SIMD[_dtype, width]](),
        ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
            _dtype, width
        ]:
            var row = Scalar[_dtype](idx[0])
            var base_col = idx[1]
            var result = val

            # Iterate through each element in SIMD vector with correct per-element column
            comptime for i in range(width):
                var col = Scalar[_dtype](base_col + i)
                result[i] = val[i] + row + col * Scalar[_dtype](0.5)
            return result

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=64,
            BN=128,
            BK=64,
            MMA_M=64,
            MMA_N=128,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=1,
            BM_SWAPAB=128,
            BN_SWAPAB=256,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=256,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            elementwise_compute_lambda_fn = Optional[
                elementwise_compute_lambda_type
            ](coord_lambda),
            use_vendor_reference=True,
        ](ctx, dynamic(1000), static[1024](), static[256]())

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=64,
            BN=128,
            BK=64,
            MMA_M=64,
            MMA_N=128,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=1,
            BM_SWAPAB=128,
            BN_SWAPAB=256,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=256,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=2,
            num_consumer_swapAB=2,
            elementwise_compute_lambda_fn = Optional[
                elementwise_compute_lambda_type
            ](coord_lambda),
            use_vendor_reference=True,
        ](ctx, dynamic(900), static[1532](), static[256]())

        test_matmul_sm90_swapAB_comparison_v2[
            a_type=bf16,
            b_type=bf16,
            c_type=bf16,
            BM=64,
            BN=128,
            BK=64,
            MMA_M=64,
            MMA_N=128,
            MMA_K=16,
            num_pipeline_stages=2,
            num_consumer=1,
            BM_SWAPAB=64,
            BN_SWAPAB=72,
            BK_SWAPAB=64,
            MMA_M_SWAPAB=64,
            MMA_N_SWAPAB=72,
            MMA_K_SWAPAB=16,
            num_pipeline_stages_swapAB=10,
            num_consumer_swapAB=1,
            k_group_size_swapAB=2,
            use_vendor_reference=True,
        ](ctx, dynamic(130), static[1536](), static[4096]())

    print("\n" + "=" * 60)
    print("ALL SWAPAB TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")
