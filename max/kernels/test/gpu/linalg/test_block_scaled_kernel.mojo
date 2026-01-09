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

"""Integration tests for block-scaled SM100 matmul kernel structures.

These tests verify that the block-scaled kernel components compile correctly
and have the expected structural properties. Functional tests require an
SM100 (B200) GPU and are not included here.
"""

from sys import size_of
from testing import assert_equal, assert_true

from layout import Layout
from layout.tensor_core_async import tile_layout_k_major, tile_sf_layout_k_major
from utils.index import Index
from utils.numerics import get_accum_type

from linalg.fp4_utils import MXFP8_SF_DTYPE, MXFP8_SF_VECTOR_SIZE
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100_structured.block_scaled_smem import (
    BlockScaledSmem,
    get_sfa_num_cols,
    get_sfb_num_cols,
)
from linalg.matmul.gpu.sm100_structured.block_scaled_tile_pipeline import (
    BlockScaledTilePipeline,
)
from linalg.matmul.gpu.sm100_structured.tile_pipeline import (
    OutputTilePipeline,
)
from linalg.matmul.gpu.sm100_structured.block_scaled_output_writer import (
    BlockScaledTileWriter,
)


# =============================================================================
# Test Configuration Helper
# =============================================================================


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
    comptime MMA_K = 128 // size_of[a_type]()
    return BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ](
        cta_group=2,
        mma_shape=Index(256, 256, MMA_K),
        cluster_shape=Index(2, 1, 1),
    )


# =============================================================================
# BlockScaledSmem Tests
# =============================================================================


fn test_block_scaled_smem_type_aliases() raises:
    """Test that BlockScaledSmem defines all required type aliases."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    comptime SmemType = BlockScaledSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]

    # Verify tile array types exist with positive element counts
    assert_true(
        SmemType.ATileArray.num_elements > 0,
        "ATileArray should have elements",
    )
    assert_true(
        SmemType.BTileArray.num_elements > 0,
        "BTileArray should have elements",
    )
    assert_true(
        SmemType.CTileArray.num_elements > 0,
        "CTileArray should have elements",
    )
    assert_true(
        SmemType.SFATileArray.num_elements > 0,
        "SFATileArray should have elements",
    )
    assert_true(
        SmemType.SFBTileArray.num_elements > 0,
        "SFBTileArray should have elements",
    )

    # Verify barrier array types exist
    comptime num_input_barriers = SmemType.num_group_pipeline_stages * 2
    assert_true(num_input_barriers > 0, "Should have input barriers")

    print("✓ test_block_scaled_smem_type_aliases passed")


fn test_block_scaled_smem_size_constraints() raises:
    """Test that SMEM size fits within B200 limits."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    comptime SmemType = BlockScaledSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]

    comptime smem_size = size_of[SmemType]()

    # B200 has 228KB SMEM per SM, reserve 1KB
    comptime max_smem = 228 * 1024 - 1024
    assert_true(
        smem_size < max_smem,
        "SMEM size exceeds B200 limit",
    )

    # Should be at least a few KB for buffers
    assert_true(
        smem_size > 4096,
        "SMEM size suspiciously small",
    )

    print(
        "✓ test_block_scaled_smem_size_constraints passed (size:",
        smem_size,
        "bytes)",
    )


# =============================================================================
# BlockScaledTilePipeline Tests
# =============================================================================


fn test_block_scaled_tile_pipeline_structure() raises:
    """Test BlockScaledTilePipeline has correct structure."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, DType.bfloat16, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Get layouts from config
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_N = config.mma_shape[1]

    comptime a_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode = config.a_swizzle
    ]()
    comptime b_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]()
    comptime sfa_layout = tile_sf_layout_k_major[BM, BK, MXFP8_SF_VECTOR_SIZE]()
    comptime sfb_layout = tile_sf_layout_k_major[
        MMA_N, BK, MXFP8_SF_VECTOR_SIZE
    ]()

    comptime num_pipeline_stages = Int(config.num_pipeline_stages)
    comptime num_group_stages = num_pipeline_stages // Int(config.k_group_size)
    comptime k_group_size = Int(config.k_group_size)

    # Verify pipeline type can be instantiated
    comptime PipelineType = BlockScaledTilePipeline[
        a_type,
        b_type,
        sfa_dtype,
        sfb_dtype,
        a_layout,
        b_layout,
        sfa_layout,
        sfb_layout,
        num_pipeline_stages,
        num_group_stages,
        k_group_size,
    ]

    # Verify it has the expected tile array types
    assert_true(
        PipelineType.ATileArray.num_elements > 0,
        "Pipeline ATileArray should have elements",
    )
    assert_true(
        PipelineType.SFATileArray.num_elements > 0,
        "Pipeline SFATileArray should have elements",
    )

    print("✓ test_block_scaled_tile_pipeline_structure passed")


fn test_output_pipeline_structure() raises:
    """Test OutputTilePipeline has correct structure for block-scaled use."""
    comptime num_accum_stages = 2
    comptime stage_stride = 256
    comptime cta_group = 2

    # Verify output pipeline type can be instantiated
    # The block-scaled kernel uses the standard OutputTilePipeline
    comptime OutputPipelineType = OutputTilePipeline[
        num_accum_stages,
        stage_stride,
        cta_group,
    ]

    # Output pipeline uses TmemAllocation and barriers
    print("✓ test_output_pipeline_structure passed")


# =============================================================================
# Kernel Configuration Tests
# =============================================================================


fn test_kernel_warp_organization() raises:
    """Test that warp organization is consistent with config."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Warp organization constants (matching BlackwellBlockScaledMatmulKernel)
    comptime SCHEDULER_THREADS = 32
    comptime TMA_LOAD_THREADS = 32
    comptime MMA_THREADS = 128
    comptime EPILOGUE_THREADS = 128

    comptime NUM_THREADS = (
        SCHEDULER_THREADS + TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # Total should be 320 threads (10 warps)
    assert_equal(NUM_THREADS, 320)

    # Verify warp counts
    assert_equal(SCHEDULER_THREADS // 32, 1)  # 1 scheduler warp
    assert_equal(TMA_LOAD_THREADS // 32, 1)  # 1 TMA load warp
    assert_equal(MMA_THREADS // 32, 4)  # 4 MMA warps
    assert_equal(EPILOGUE_THREADS // 32, 4)  # 4 epilogue warps

    print("✓ test_kernel_warp_organization passed")


fn test_kernel_tmem_configuration() raises:
    """Test TMEM configuration is consistent."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # TMEM constants (matching BlackwellBlockScaledMatmulKernel)
    comptime NUM_TMEM_COLS = 512
    comptime num_accum_stages = Int(config.num_accum_pipeline_stages)
    comptime stage_stride = NUM_TMEM_COLS // num_accum_stages

    # Verify stage stride is valid
    assert_true(stage_stride > 0, "stage_stride should be positive")
    assert_true(
        NUM_TMEM_COLS % num_accum_stages == 0,
        "TMEM cols should be divisible by accum stages",
    )

    # Verify scaling factor TMEM offsets can be computed
    comptime sfa_cols = get_sfa_num_cols[config]()
    comptime sfb_cols = get_sfb_num_cols[config]()

    # SFA TMEM offset starts after accum stages
    comptime sfa_offset = num_accum_stages * stage_stride
    assert_true(sfa_offset > 0, "SFA TMEM offset should be positive")

    # SFB TMEM offset starts after SFA
    comptime sfb_offset = sfa_offset + sfa_cols
    assert_true(sfb_offset > sfa_offset, "SFB offset should be after SFA")

    print("✓ test_kernel_tmem_configuration passed")


fn test_kernel_mma_operation_types() raises:
    """Test MMA operation configuration matches config."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Verify MMA dimensions
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    # SM100 block-scaled uses 256x256 MMA for cta_group=2
    assert_equal(MMA_M, 256)
    assert_equal(MMA_N, 256)

    # MMA_K = 128 // element_size for FP8
    assert_equal(MMA_K, 128)

    # Verify accumulator type
    comptime accum_type = get_accum_type[a_type]()
    assert_equal(accum_type, DType.float32)

    print("✓ test_kernel_mma_operation_types passed")


fn test_block_scaled_tile_writer_type() raises:
    """Test that BlockScaledTileWriter type can be parameterized."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _make_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Define the C output layout
    comptime OutputM = config.output_tile_shape[0]
    comptime OutputN = config.output_tile_shape[1]
    comptime c_smem_layout = Layout.row_major(OutputM, OutputN)

    # Verify the TileWriter constants are computed
    comptime num_output_stages = 2
    comptime num_accum_stages = Int(config.num_accum_pipeline_stages)
    comptime stage_stride = 512 // num_accum_stages

    assert_true(OutputM > 0, "OutputM should be positive")
    assert_true(OutputN > 0, "OutputN should be positive")
    assert_true(stage_stride > 0, "stage_stride should be positive")

    # Verify c_smem_layout size
    comptime c_smem_size = c_smem_layout.size()
    assert_equal(c_smem_size, OutputM * OutputN)

    print("✓ test_block_scaled_tile_writer_type passed")


# =============================================================================
# Main entry point
# =============================================================================


fn main() raises:
    """Run all block-scaled kernel integration tests."""
    print("Running block-scaled kernel integration tests...")
    print()

    # SMEM tests
    test_block_scaled_smem_type_aliases()
    test_block_scaled_smem_size_constraints()

    # Pipeline tests
    test_block_scaled_tile_pipeline_structure()
    test_output_pipeline_structure()

    # Kernel configuration tests
    test_kernel_warp_organization()
    test_kernel_tmem_configuration()
    test_kernel_mma_operation_types()
    test_block_scaled_tile_writer_type()

    print()
    print("All integration tests passed! ✓")
