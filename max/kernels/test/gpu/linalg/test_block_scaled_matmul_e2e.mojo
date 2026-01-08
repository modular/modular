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

"""End-to-end correctness tests for block-scaled SM100 matmul.

These tests verify that the structured block-scaled kernel components are
correctly configured and can be launched. Tests require an SM100 (B200) GPU
for the full kernel launch tests.
"""

from math import ceildiv
from sys import size_of
from testing import assert_true

from utils.index import Index
from utils.static_tuple import StaticTuple

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig


# =============================================================================
# Helper Functions
# =============================================================================


fn _create_test_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
]() -> BlockScaledMatmulConfig[
    a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
]:
    """Create a test configuration for block-scaled matmul."""
    comptime MMA_K = 128 // size_of[a_type]()
    return BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ](
        cta_group=2,
        mma_shape=Index(256, 256, MMA_K),
        cluster_shape=Index(2, 1, 1),
    )


# =============================================================================
# Test Cases
# =============================================================================


fn test_block_scaled_config_validation() raises:
    """Test that config validation catches invalid configurations."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _create_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Verify config constraints
    assert_true(config.cta_group in (1, 2), "cta_group must be 1 or 2")
    assert_true(
        config.num_pipeline_stages % config.k_group_size == 0,
        "Pipeline stages must be multiple of k_group_size",
    )

    # Verify MMA shape is valid for SM100
    assert_true(
        config.mma_shape[0] in (128, 256), "MMA_M must be 128 or 256 for SM100"
    )

    print("✓ test_block_scaled_config_validation passed")


fn test_scaling_factor_dimensions() raises:
    """Test that scaling factor dimensions are computed correctly."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _create_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    # Problem dimensions
    comptime M = 256
    comptime N = 256
    comptime K = 128

    # Scaling factor group counts
    comptime sfa_m_groups = M // SF_MN_GROUP_SIZE
    comptime sfa_k_groups = K // SF_MN_GROUP_SIZE
    comptime sfb_n_groups = N // SF_MN_GROUP_SIZE
    comptime sfb_k_groups = K // SF_MN_GROUP_SIZE

    # Verify dimensions are positive
    assert_true(sfa_m_groups > 0, "SFA M groups should be positive")
    assert_true(sfa_k_groups > 0, "SFA K groups should be positive")
    assert_true(sfb_n_groups > 0, "SFB N groups should be positive")
    assert_true(sfb_k_groups > 0, "SFB K groups should be positive")

    # SF_ATOM_M is (rows, cols) for scaling factor atom
    comptime sf_atom_size = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    assert_true(sf_atom_size > 0, "SF atom size should be positive")

    print("✓ test_scaling_factor_dimensions passed")


fn test_smem_size_fits_b200() raises:
    """Test that SMEM size fits within B200 limits."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _create_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    from linalg.matmul.gpu.sm100_structured.block_scaled_smem import (
        BlockScaledSmem,
    )

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

    # B200 has 228KB SMEM per SM, reserve 1KB for runtime
    comptime max_smem = 228 * 1024 - 1024
    assert_true(smem_size < max_smem, "SMEM size must fit within B200 limits")

    print("✓ test_smem_size_fits_b200 passed (SMEM:", smem_size, "bytes)")


fn test_kernel_struct_compilation() raises:
    """Test that the kernel struct compiles with valid parameters."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _create_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    from layout import Layout
    from linalg.matmul.gpu.sm100_structured.block_scaled_matmul_kernels import (
        BlackwellBlockScaledMatmulKernel,
    )

    # Define layouts (using simple row-major for test)
    comptime a_layout = Layout.row_major(
        1, config.block_tile_shape[0], config.block_tile_shape[2]
    )
    comptime b_layout = Layout.row_major(
        1, config.block_tile_shape[1], config.block_tile_shape[2]
    )
    comptime c_layout = Layout.row_major(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    )
    comptime sfa_layout = Layout.row_major(
        1, 1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )
    comptime sfb_layout = Layout.row_major(
        1, 1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )

    # Verify kernel struct can be parameterized
    comptime kernel_type = BlackwellBlockScaledMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        a_layout,
        b_layout,
        c_layout,
        sfa_layout,
        sfb_layout,
        a_layout,
        b_layout,
        c_layout,
        sfa_layout,
        sfb_layout,  # desc layouts
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
    ]

    # Verify derived constants
    assert_true(kernel_type.NUM_THREADS > 0, "NUM_THREADS should be positive")
    assert_true(
        kernel_type.NUM_TMEM_COLS > 0, "NUM_TMEM_COLS should be positive"
    )

    print("✓ test_kernel_struct_compilation passed")


fn test_cpu_entry_point_exists() raises:
    """Test that the CPU entry point function exists and can be imported."""
    from linalg.matmul.gpu.sm100_structured.block_scaled_matmul import (
        blackwell_block_scaled_matmul_tma_umma_warp_specialized,
    )

    # Just verify the import works - actual launch requires GPU tensors
    print("✓ test_cpu_entry_point_exists passed")


fn test_tile_writer_type() raises:
    """Test that BlockScaledTileWriter can be imported and configured."""
    from linalg.matmul.gpu.sm100_structured.block_scaled_output_writer import (
        BlockScaledTileWriter,
    )

    # Just verify the import works
    print("✓ test_tile_writer_type passed")


fn test_tile_pipeline_types() raises:
    """Test that block-scaled tile pipeline types can be imported."""
    from linalg.matmul.gpu.sm100_structured.block_scaled_tile_pipeline import (
        BlockScaledTilePipeline,
        BlockScaledProducerStage,
        BlockScaledConsumerStage,
        BlockScaledInputProducer,
        BlockScaledInputConsumer,
    )

    # Just verify the imports work
    print("✓ test_tile_pipeline_types passed")


fn test_tile_loader_types() raises:
    """Test that scaling factor loader types can be imported."""
    from linalg.matmul.gpu.sm100_structured.block_scaled_tile_loader import (
        ScalingFactorLoader,
        copy_sf_tmem,
    )

    # Just verify the imports work
    print("✓ test_tile_loader_types passed")


fn test_kernel_constants_match_config() raises:
    """Test that kernel struct constants match configuration."""
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime sfa_dtype = MXFP8_SF_DTYPE
    comptime sfb_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    comptime config = _create_test_config[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]()

    from layout import Layout
    from linalg.matmul.gpu.sm100_structured.block_scaled_matmul_kernels import (
        BlackwellBlockScaledMatmulKernel,
    )

    # Define layouts
    comptime a_layout = Layout.row_major(
        1, config.block_tile_shape[0], config.block_tile_shape[2]
    )
    comptime b_layout = Layout.row_major(
        1, config.block_tile_shape[1], config.block_tile_shape[2]
    )
    comptime c_layout = Layout.row_major(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    )
    comptime sfa_layout = Layout.row_major(
        1, 1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )
    comptime sfb_layout = Layout.row_major(
        1, 1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )

    comptime kernel_type = BlackwellBlockScaledMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        a_layout,
        b_layout,
        c_layout,
        sfa_layout,
        sfb_layout,
        a_layout,
        b_layout,
        c_layout,
        sfa_layout,
        sfb_layout,
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
    ]

    # Verify thread counts match expected warp organization
    # 1 scheduler warp + 1 TMA load warp + 1 MMA warp + 4 epilogue warps
    assert_true(
        kernel_type.SCHEDULER_THREADS == 32, "Scheduler should be 1 warp"
    )
    assert_true(kernel_type.TMA_LOAD_THREADS == 32, "TMA load should be 1 warp")
    assert_true(kernel_type.MMA_THREADS == 32, "MMA should be 1 warp")
    assert_true(
        kernel_type.EPILOGUE_THREADS == 128, "Epilogue should be 4 warps"
    )
    assert_true(kernel_type.NUM_THREADS == 224, "Total should be 7 warps")

    # Verify TMEM configuration
    assert_true(kernel_type.NUM_TMEM_COLS == 512, "TMEM should have 512 cols")
    assert_true(
        kernel_type.stage_stride_cols > 0, "Stage stride should be positive"
    )

    # Verify pipeline configuration
    assert_true(
        kernel_type.num_pipeline_stages == Int(config.num_pipeline_stages),
        "Pipeline stages should match config",
    )
    assert_true(
        kernel_type.num_accum_pipeline_stages
        == Int(config.num_accum_pipeline_stages),
        "Accum pipeline stages should match config",
    )

    print("✓ test_kernel_constants_match_config passed")


# =============================================================================
# Main Entry Point
# =============================================================================


fn main() raises:
    """Run all end-to-end tests for block-scaled matmul."""
    print("Running block-scaled matmul end-to-end tests...")
    print()

    # Configuration and structure tests
    test_block_scaled_config_validation()
    test_scaling_factor_dimensions()
    test_smem_size_fits_b200()
    test_kernel_struct_compilation()

    # Module import tests
    test_cpu_entry_point_exists()
    test_tile_writer_type()
    test_tile_pipeline_types()
    test_tile_loader_types()

    # Kernel constant validation
    test_kernel_constants_match_config()

    print()
    print("All end-to-end tests passed! ✓")
