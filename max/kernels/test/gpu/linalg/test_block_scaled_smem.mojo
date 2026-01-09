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

"""Unit tests for block-scaled SMEM and tile pipeline structures."""

from sys import size_of
from testing import assert_equal, assert_true

from utils.index import Index

from linalg.fp4_utils import MXFP8_SF_DTYPE
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100_structured.block_scaled_smem import (
    get_sfa_num_cols,
    get_sfb_num_cols,
)


# Create a valid SM100 config for testing
# SM100 uses much larger MMA shapes than Hopper (256x256 or 128x128)
fn _make_test_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
]() -> BlockScaledMatmulConfig[
    a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
]:
    """Create a valid SM100 config for testing."""
    # SM100 with cta_group=2 uses MMA shapes like (256, 256, MMA_K)
    # where MMA_K = 128 // size_of[a_type] for FP8
    comptime MMA_K = 128 // size_of[a_type]()
    return BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ](
        cta_group=2,
        mma_shape=Index(256, 256, MMA_K),
        cluster_shape=Index(2, 1, 1),
    )


# =============================================================================
# Test scaling factor TMEM column calculations
# =============================================================================


fn test_scaling_factor_tmem_cols() raises:
    """Test TMEM column calculations for scaling factors."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # SFA_NUM_COLS = BM // 32
    comptime sfa_cols = get_sfa_num_cols[config]()
    comptime expected_sfa_cols = config.block_tile_shape[0] // 32
    assert_equal(sfa_cols, expected_sfa_cols)

    # SFB_NUM_COLS = MMA_N // 32
    comptime sfb_cols = get_sfb_num_cols[config]()
    comptime expected_sfb_cols = config.mma_shape[1] // 32
    assert_equal(sfb_cols, expected_sfb_cols)

    print("✓ test_scaling_factor_tmem_cols passed")


fn test_block_scaled_config_constraints() raises:
    """Test that BlockScaledMatmulConfig enforces proper constraints."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Verify CTA group is 1 or 2
    assert_true(
        config.cta_group == 1 or config.cta_group == 2,
        "cta_group must be 1 or 2",
    )

    # Verify MMA shape constraints - MMA_M should be 256 for cta_group=2
    comptime mma_m = config.mma_shape[0]
    assert_true(
        mma_m == 128 or mma_m == 256,
        "MMA_M must be 128 or 256 for SM100",
    )

    # Verify block tile shape is derived correctly
    # BM = MMA_M // cta_group
    comptime expected_bm = mma_m // config.cta_group
    assert_equal(config.block_tile_shape[0], expected_bm)

    # Verify pipeline stages is multiple of k_group_size
    assert_true(
        config.num_pipeline_stages % config.k_group_size == 0,
        "num_pipeline_stages must be multiple of k_group_size",
    )

    print("✓ test_block_scaled_config_constraints passed")


fn test_config_derived_constants() raises:
    """Test that config-derived constants are computed correctly."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # For SM100 with MMA shape (256, 256, K) and cta_group=2:
    # BM = 256/2 = 128
    # BN = 256/2 = 128
    # BK = 128 // size_of[FP8] = 128
    assert_equal(config.block_tile_shape[0], 128)
    assert_equal(config.block_tile_shape[1], 128)
    assert_equal(config.block_tile_shape[2], 128)

    # MMA_N = 256 (full MMA_N for MMA_M=256 or cta_group=1)
    assert_equal(config.mma_shape[1], 256)

    # Output tile shape should be valid
    assert_true(config.output_tile_shape[0] > 0, "OutputM should be positive")
    assert_true(config.output_tile_shape[1] > 0, "OutputN should be positive")

    print("✓ test_config_derived_constants passed")


fn test_scaling_factor_calculations() raises:
    """Test scaling factor size and offset calculations."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Scaling factor TMEM columns should be BM/32 and MMA_N/32
    comptime sfa_cols = get_sfa_num_cols[config]()
    comptime sfb_cols = get_sfb_num_cols[config]()

    # For BM=128: sfa_cols = 128/32 = 4
    assert_equal(sfa_cols, 4)

    # For MMA_N=256: sfb_cols = 256/32 = 8
    assert_equal(sfb_cols, 8)

    print("✓ test_scaling_factor_calculations passed")


fn test_pipeline_stage_counts() raises:
    """Test that pipeline stage counts are valid."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Verify pipeline stages are positive
    assert_true(
        config.num_pipeline_stages > 0, "num_pipeline_stages should be positive"
    )
    assert_true(
        config.num_accum_pipeline_stages > 0,
        "num_accum_pipeline_stages should be positive",
    )
    assert_true(
        config.num_output_stages > 0, "num_output_stages should be positive"
    )

    # Verify k_group_size divides num_pipeline_stages
    comptime num_group_stages = (
        Int(config.num_pipeline_stages) // Int(config.k_group_size)
    )
    assert_true(num_group_stages > 0, "num_group_stages should be positive")

    print("✓ test_pipeline_stage_counts passed")


# =============================================================================
# Main entry point
# =============================================================================


fn main() raises:
    """Run all block-scaled SMEM unit tests."""
    print("Running block-scaled SMEM unit tests...")
    print()

    test_scaling_factor_tmem_cols()
    test_block_scaled_config_constraints()
    test_config_derived_constants()
    test_scaling_factor_calculations()
    test_pipeline_stage_counts()

    print()
    print("All tests passed! ✓")
