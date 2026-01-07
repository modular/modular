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
"""Compare PTX output between sm100 and sm100_structured kernels.

This test generates PTX for both kernel implementations and compares them
to identify differences that could explain performance variations.

Usage:
    mojo max/kernels/test/gpu/linalg/test_matmul_sm100_ptx_comparison.mojo
"""

from math import align_up
from sys import argv, size_of

from bit import next_power_of_two, prev_power_of_two
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.compile import _compile_code, get_gpu_target
from gpu.host.info import B200
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout.layout import Layout
from layout.tma_async import _tma_desc_tile_layout
from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.matmul.gpu.sm100.tile_scheduler import RasterOrder

# Import the original sm100 kernel function
from linalg.matmul.gpu.sm100.matmul import (
    blackwell_tma_umma_warp_specialized_kernel,
)

# Import the structured kernel struct
from linalg.matmul.gpu.sm100_structured.matmul_kernels import (
    BlackwellMatmulSM100Kernel,
)

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


fn generate_ptx_sm100[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]() raises -> String:
    """Generate PTX for original sm100 kernel."""

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime a_swizzle = config.a_swizzle
    comptime b_swizzle = config.b_swizzle
    comptime cluster_shape = config.cluster_shape

    comptime a_tma_shape = Index(BM // cluster_shape[1], BK)
    comptime a_tma_layout = Layout.row_major(a_tma_shape[0], a_tma_shape[1])
    comptime a_tma_desc_layout = _tma_desc_tile_layout[
        a_type, 2, a_tma_shape, a_swizzle
    ]()

    comptime b_tma_shape = Index(
        BN // (cluster_shape[0] // config.cta_group), BK
    )
    comptime b_tma_layout = Layout.row_major(b_tma_shape[0], b_tma_shape[1])
    comptime b_tma_desc_layout = _tma_desc_tile_layout[
        b_type, 2, b_tma_shape, b_swizzle
    ]()

    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if MMA_M == 256 else c_tma_tile_shape_mma128

    comptime c_tma_shape = c_tma_tile_shape if not config.AB_swapped else Index(
        c_tma_tile_shape[0], c_tma_tile_shape[1] // 8
    )
    comptime c_tma_layout = Layout.row_major(c_tma_shape[0], c_tma_shape[1])
    comptime c_tma_desc_layout = _tma_desc_tile_layout[
        c_type, 2, c_tma_shape, config.c_swizzle
    ]()

    comptime kernel = blackwell_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_layout,
        b_tma_layout,
        c_tma_layout,
        a_tma_desc_layout,
        b_tma_desc_layout,
        c_tma_desc_layout,
        transpose_b=transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        elementwise_compute_lambda_fn=None,
    ]

    return _compile_code[kernel, target = get_gpu_target["sm_100a"]()]().asm


fn generate_ptx_sm100_structured[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]() raises -> String:
    """Generate PTX for sm100_structured kernel."""

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime a_swizzle = config.a_swizzle
    comptime b_swizzle = config.b_swizzle
    comptime cluster_shape = config.cluster_shape

    comptime a_tma_shape = Index(BM // cluster_shape[1], BK)
    comptime a_tma_layout = Layout.row_major(a_tma_shape[0], a_tma_shape[1])
    comptime a_tma_desc_layout = _tma_desc_tile_layout[
        a_type, 2, a_tma_shape, a_swizzle
    ]()

    comptime b_tma_shape = Index(
        BN // (cluster_shape[0] // config.cta_group), BK
    )
    comptime b_tma_layout = Layout.row_major(b_tma_shape[0], b_tma_shape[1])
    comptime b_tma_desc_layout = _tma_desc_tile_layout[
        b_type, 2, b_tma_shape, b_swizzle
    ]()

    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if MMA_M == 256 else c_tma_tile_shape_mma128

    comptime c_tma_shape = c_tma_tile_shape if not config.AB_swapped else Index(
        c_tma_tile_shape[0], c_tma_tile_shape[1] // 8
    )
    comptime c_tma_layout = Layout.row_major(c_tma_shape[0], c_tma_shape[1])
    comptime c_tma_desc_layout = _tma_desc_tile_layout[
        c_type, 2, c_tma_shape, config.c_swizzle
    ]()

    # Use the structured kernel struct's run method
    comptime kernel = BlackwellMatmulSM100Kernel[
        a_type,
        b_type,
        c_type,
        a_tma_layout,
        b_tma_layout,
        c_tma_layout,
        a_tma_desc_layout,
        b_tma_desc_layout,
        c_tma_desc_layout,
        transpose_b=transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        elementwise_compute_lambda_fn=None,
    ].run

    return _compile_code[kernel, target = get_gpu_target["sm_100a"]()]().asm


fn count_instructions(ptx: String, instr: String) -> Int:
    """Count occurrences of an instruction in PTX."""
    var count = 0
    var pos = 0
    while True:
        var idx = ptx.find(instr, pos)
        if idx < 0:
            break
        count += 1
        pos = idx + len(instr)
    return count


fn analyze_ptx(name: String, ptx: String):
    """Print summary statistics of PTX code."""
    print("=== " + name + " ===")
    print("Total lines:", ptx.count("\n"))

    # Count key instructions
    print("Instructions:")
    print("  mov.b32:", count_instructions(ptx, "mov.b32"))
    print("  mov.b64:", count_instructions(ptx, "mov.b64"))
    print("  add.s32:", count_instructions(ptx, "add.s32"))
    print("  add.s64:", count_instructions(ptx, "add.s64"))
    print("  mul:", count_instructions(ptx, "mul."))
    print("  ld.shared:", count_instructions(ptx, "ld.shared"))
    print("  st.shared:", count_instructions(ptx, "st.shared"))
    print("  ld.global:", count_instructions(ptx, "ld.global"))
    print("  st.global:", count_instructions(ptx, "st.global"))
    print("  bar.sync:", count_instructions(ptx, "bar.sync"))
    print("  mbarrier:", count_instructions(ptx, "mbarrier"))
    print("  tcgen05:", count_instructions(ptx, "tcgen05"))
    print("  call:", count_instructions(ptx, "call"))
    print("")


def main():
    comptime BK = 64
    comptime MMA_K = 16
    comptime a_type = DType.bfloat16
    comptime b_type = DType.bfloat16
    comptime c_type = DType.bfloat16
    comptime transpose_b = True

    comptime M = 4096
    comptime N = 4096
    comptime K = 4096
    comptime c_layout = Layout.row_major(M, N)
    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(N, K)

    comptime block_tile_shape = Index(128, 128, BK)
    comptime mma_shape = Index(
        block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
    )
    comptime cluster_shape = Index(2, 1, 1)
    comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        cta_group=2,
        block_swizzle_size=0,
        raster_order=RasterOrder.AlongM,
    )

    print("Generating PTX for sm100...")
    var ptx_sm100 = generate_ptx_sm100[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b=transpose_b,
        config=config,
    ]()

    print("Generating PTX for sm100_structured...")
    var ptx_structured = generate_ptx_sm100_structured[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b=transpose_b,
        config=config,
    ]()

    print("\n" + "=" * 60)
    print("PTX COMPARISON RESULTS")
    print("=" * 60 + "\n")

    analyze_ptx("sm100 (original)", ptx_sm100)
    analyze_ptx("sm100_structured", ptx_structured)

    print("=== SIZE COMPARISON ===")
    print("sm100 PTX size:", len(ptx_sm100), "bytes")
    print("sm100_structured PTX size:", len(ptx_structured), "bytes")
    var diff = len(ptx_structured) - len(ptx_sm100)
    var pct = (diff * 100) // len(ptx_sm100)
    print("Difference:", diff, "bytes (", pct, "%)")
    print("")

    # Write PTX files for detailed diff
    print("Writing PTX files for manual comparison...")
    with open("/tmp/sm100_ptx.txt", "w") as f:
        f.write(ptx_sm100)
    with open("/tmp/sm100_structured_ptx.txt", "w") as f:
        f.write(ptx_structured)
    print("Files written to:")
    print("  /tmp/sm100_ptx.txt")
    print("  /tmp/sm100_structured_ptx.txt")
    print("")
    print(
        "Run 'diff /tmp/sm100_ptx.txt /tmp/sm100_structured_ptx.txt' to see"
        " differences"
    )
