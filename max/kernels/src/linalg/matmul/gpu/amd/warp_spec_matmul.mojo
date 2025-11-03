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
from linalg.matmul.gpu.amd.structured import (
    AmdTileOperator,
    AmdWarpBlockScatterGather,
    ThreadRole,
    RingBuffer,
    SMemBuffer,
    MMAConfig,
)

from gpu import (
    WARP_SIZE,
    block_idx,
    thread_idx,
    barrier,
    warp_id as get_warp_id,
)

from layout import Layout, LayoutTensor
from layout.layout_tensor import (
    ThreadScope,
    copy_local_to_dram,
)
from utils import IndexList
from gpu import MAX_THREADS_PER_BLOCK_METADATA
from utils import StaticTuple
from layout.layout import blocked_product
from layout.swizzle import Swizzle


@parameter
fn smem_tile_layout[
    k_tile_size: Int, block_rows: Int, block_cols: Int
]() -> Layout:
    # Shared memory layout
    #
    # - base_layout: Layout.row_major(block_rows, k_tile_size) -> block_rows x k_tile_size tiles
    # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
    # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
    #
    # Resulting shape: block_rowsx(k_tile_size x num_repeats) = block_rows x block_cols tensor
    # Where block_cols = k_tile_size x num_repeats, k_tile_size = MMA_K x k_group_size
    #
    # This creates num_repeats blocks of block_rows x k_tile_size arranged horizontally:
    # Within each k_tile_size-column block, elements are consecutive (stride 1)
    # Between blocks: stride = block_rows x k_tile_size
    #
    # ASCII diagram for block_rows=64, k_tile_size=32, block_cols=64 (showing first 2 of 2 blocks):
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │         Block 0 (64x32)             │         Block 1 (64x32)           │
    # ├─────────────────────────────────────┼───────────────────────────────────┤
    # │   0    1    2  ...   30   31        │ 2048 2049 2050 ... 2078 2079      │
    # │  32   33   34  ...   62   63        │ 2080 2081 2082 ... 2110 2111      │
    # │  64   65   66  ...   94   95        │ 2112 2113 2114 ... 2142 2143      │
    # │  96   97   98  ...  126  127        │ 2144 2145 2146 ... 2174 2175      │
    # │ ...                                 │  ...                              │
    # │2016 2017 2018  ... 2046 2047        │ 4064 4065 4066 ... 4094 4095      │
    # └─────────────────────────────────────────────────────────────────────────┘
    # stride between blocks = block_rows x k_tile_size = 64 x 32 = 2048

    constrained[
        block_cols % k_tile_size == 0,
        "block_cols must be a multiple of k_tile_size",
    ]()

    alias base_layout = Layout.row_major(block_rows, k_tile_size)
    alias num_repeats = block_cols // k_tile_size
    alias tiler_layout = Layout.row_major(1, num_repeats)
    return blocked_product(base_layout, tiler_layout, coalesce_output=True)


@parameter
fn get_producer_warp_thread_layout[
    k_tile_size: Int, simd_width: Int, block_rows: Int, block_cols: Int
]() -> Layout:
    # TODO: Document the logic behind this layout
    # Define a layout that corresponds to the below pattern:
    #
    # | T00 T01 T02 T03 | T16 T17 T18 T19 |
    # | T04 T05 T06 T07 | T20 T21 T22 T23 |
    # | T08 T09 T10 T11 | T24 T25 T26 T27 |
    # | T12 T13 T14 T15 | T28 T29 T30 T31 |
    # | T32 T33 T34 T35 | T48 T49 T50 T51 |
    # | T36 T37 T38 T39 | T52 T53 T54 T55 |
    # | T40 T41 T42 T43 | T56 T57 T58 T59 |
    # | T44 T45 T46 T47 | T60 T61 T62 T63 |

    alias inner_block_size = 16  # total number of threads in the inner block

    # a row of inner blocks will load one k_tile, so here we calculate
    # threads per row
    alias inner_block_cols = k_tile_size // simd_width
    alias inner_block_rows = inner_block_size // inner_block_cols

    alias base_layout = Layout.row_major(inner_block_rows, inner_block_cols)

    alias num_repeats_col = block_cols // k_tile_size

    constrained[
        num_repeats_col < (WARP_SIZE // inner_block_size),
        "not enough threads per warp to cover block k dimension",
    ]()
    alias outer_block_size = num_repeats_col * inner_block_size
    alias num_repeats_row = WARP_SIZE // outer_block_size

    constrained[
        block_rows % (inner_block_rows * num_repeats_row) == 0,
        "shared block size is not evenly distributable among threads",
    ]()

    alias tiler_layout = Layout.row_major(
        num_repeats_row,
        num_repeats_col,
    )
    return blocked_product(base_layout, tiler_layout)


# NOTE: This is a hardcoded pipeline but in reality this should be struct
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        (a_producer_warps + b_producer_warps + consumer_warps) * WARP_SIZE
    )
)
fn warp_specialized_matmul[
    in_type: DType,
    out_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    a_producer_warps: Int = 1,
    b_producer_warps: Int = 1,
    consumer_warps: Int = 1,
    pipeline_stages: Int = 1,
](
    a: LayoutTensor[
        in_type, a_layout, MutableAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    b: LayoutTensor[
        in_type, b_layout, MutableAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    c: LayoutTensor[
        out_type,
        c_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.GLOBAL,
    ],
):
    alias K = a.shape[1]()

    # NOTE: hardcoded MMA for now, but in theory this pipeline will work with
    # any MMA
    alias MMA_M = 16
    alias MMA_N = 16
    alias MMA_K = 16

    alias m_warps_per_block = BM // WM
    alias n_warps_per_block = BN // WN

    constrained[m_warps_per_block % a_producer_warps == 0]()
    constrained[n_warps_per_block % b_producer_warps == 0]()
    constrained[m_warps_per_block * n_warps_per_block % consumer_warps == 0]()
    constrained[
        consumer_warps >= a_producer_warps
        and consumer_warps >= b_producer_warps
    ]()
    constrained[
        consumer_warps.is_power_of_two(), "consumer_warps must be a power of 2"
    ]()

    var role: ThreadRole
    var warp_id = get_warp_id()
    alias producer_thread_count = (
        a_producer_warps + b_producer_warps
    ) * WARP_SIZE

    if thread_idx.x < UInt(producer_thread_count):
        role = ThreadRole.PRODUCER
    else:
        role = ThreadRole.CONSUMER

    alias MMAConfigType = MMAConfig[
        in_type,
        out_type,
        IndexList[3](MMA_M, MMA_N, MMA_K),
        True,
    ]

    alias swizzle = None

    constrained[
        MMAConfigType.adjusted_mma_k_shape_a()
        == MMAConfigType.adjusted_mma_k_shape_b(),
        "MMA_K shapes must be equal",
    ]()

    alias smem_layout_a = smem_tile_layout[
        MMAConfigType.adjusted_mma_k_shape_a(), BM, BK
    ]()
    alias smem_layout_b = smem_tile_layout[
        MMAConfigType.adjusted_mma_k_shape_b(), BN, BK
    ]()

    alias SMemBufferTypeA = SMemBuffer[
        in_type, smem_layout_a, pipeline_stages, BM, BK, WM, WK
    ]
    alias SMemBufferTypeB = SMemBuffer[
        in_type, smem_layout_b, pipeline_stages, BN, BK, WN, WK
    ]

    var smem_buffer_a = SMemBufferTypeA()
    var smem_buffer_b = SMemBufferTypeB()

    # NOTE: Every thread needs access to the ring buffers
    var ring_buffer_a = RingBuffer[
        in_type,
        smem_layout_a,
        pipeline_stages,
        BM,
        BK,
        WM,
        WK,
        consumer_warps,
    ](smem_buffer_a)

    var ring_buffer_b = RingBuffer[
        in_type,
        smem_layout_b,
        pipeline_stages,
        BN,
        BK,
        WN,
        WK,
        consumer_warps,
    ](smem_buffer_b)

    alias consumer_thread_layout_a = get_producer_warp_thread_layout[
        MMAConfigType.adjusted_mma_k_shape_a(), MMAConfigType.simd_width, BM, BK
    ]()

    alias consumer_thread_layout_b = get_producer_warp_thread_layout[
        MMAConfigType.adjusted_mma_k_shape_b(), MMAConfigType.simd_width, BN, BK
    ]()

    barrier()  # NOTE: probably not necessary but I saw it in the HF code around the same point

    alias tile_count = K // BK
    alias warps_processed_per_producer_a = Int(
        m_warps_per_block // a_producer_warps
    )
    alias warps_processed_per_producer_b = Int(
        n_warps_per_block // b_producer_warps
    )

    # NOTE: the 2 producer blocks are almost identical, you can proabbly make this a
    # a function
    if role is ThreadRole.PRODUCER:
        if warp_id < UInt(a_producer_warps):
            # NOTE: If there is a way to hide the phases that would be great, maybe the ringbuffer
            # handles the thread specific phase, or its encapsulated in the ThreadRole Struct.

            var phases = InlineArray[
                Int, pipeline_stages * warps_processed_per_producer_a
            ](
                fill=0,
            )

            var scatter_gather_a = AmdWarpBlockScatterGather[
                in_type,
                consumer_thread_layout_a,
                smem_buffer_a.WarpTileType.layout,
                MMAConfigType.simd_width,  # NOTE: hardcoded simd width for now, but in theory this should be derived
                True,
                WM,
                WK,
                swizzle=swizzle,
            ]()

            @parameter
            for tile_num in range(tile_count):
                alias stage = tile_num % pipeline_stages
                var a_tile = a.tile[BM, BK](Int(block_idx.x), tile_num)

                # NOTE: producers and consumers can process more than one tile this loop
                # makes sure this is possible
                @parameter
                for local_tile_count in range(warps_processed_per_producer_a):
                    var warp_tile_idx = (
                        Int(warp_id) + local_tile_count * a_producer_warps
                    )

                    # NOTE phase_idx needs to be known at compile time otherwise
                    # the compiler will not place phases
                    alias phase_idx = warps_processed_per_producer_a * stage + local_tile_count

                    ref phase = phases[phase_idx]

                    # Load A tile from global to shared memory
                    scatter_gather_a.load_compute_tile(
                        ring_buffer_a,
                        phase,
                        a_tile,
                        stage,
                        warp_tile_idx,
                    )

        else:
            var relative_warp_id = warp_id - UInt(a_producer_warps)

            var phases = InlineArray[
                Int, pipeline_stages * warps_processed_per_producer_b
            ](
                fill=0,
            )

            var scatter_gather_b = AmdWarpBlockScatterGather[
                in_type,
                consumer_thread_layout_b,
                smem_buffer_b.WarpTileType.layout,
                MMAConfigType.simd_width,
                False,
                WN,
                WK,
                swizzle=swizzle,
            ]()

            @parameter
            for tile_num in range(tile_count):
                var b_tile = b.tile[BN, BK](Int(block_idx.y), tile_num)
                alias stage = tile_num % pipeline_stages

                @parameter
                for local_tile_count in range(warps_processed_per_producer_b):
                    var warp_tile_idx = (
                        Int(relative_warp_id)
                        + local_tile_count * b_producer_warps
                    )
                    alias phase_idx = warps_processed_per_producer_b * stage + local_tile_count

                    ref phase = phases[phase_idx]

                    # Load B tile from global to shared memory
                    scatter_gather_b.load_compute_tile(
                        ring_buffer_b,
                        phase,
                        b_tile,
                        stage,
                        warp_tile_idx,
                    )

    else:
        alias total_consumer_operations = m_warps_per_block * n_warps_per_block

        var phases_a = InlineArray[
            Int, pipeline_stages * (total_consumer_operations // consumer_warps)
        ](
            fill=1,
        )

        var phases_b = InlineArray[
            Int, pipeline_stages * (total_consumer_operations // consumer_warps)
        ](
            fill=1,
        )

        var tile_operator = AmdTileOperator[
            MMAConfigType,
            smem_buffer_a.WarpTileType.layout,
            smem_buffer_b.WarpTileType.layout,
            tile_being_processed_per_warp = total_consumer_operations
            // consumer_warps,
            swizzle=swizzle,
        ]()

        var relative_warp_id = warp_id - UInt(
            a_producer_warps + b_producer_warps
        )

        alias warps_computed_per_consumer = total_consumer_operations // consumer_warps

        @parameter
        for i in range(tile_count):
            alias stage = i % pipeline_stages

            @parameter
            for local_tile_count in range(warps_computed_per_consumer):
                var warp_tile_idx = (
                    Int(relative_warp_id) + local_tile_count * consumer_warps
                )
                var m_warp_idx = warp_tile_idx // n_warps_per_block
                var n_warp_idx = warp_tile_idx % n_warps_per_block

                alias phase_idx = warps_computed_per_consumer * stage + local_tile_count

                ref phase_a = phases_a[phase_idx]
                ref phase_b = phases_b[phase_idx]

                # Perform matrix multiplication
                tile_operator.mma(
                    ring_buffer_a,
                    ring_buffer_b,
                    phase_a,
                    phase_b,
                    stage,
                    m_warp_idx,
                    n_warp_idx,
                    local_tile_count,
                    i,
                )

        var local_tile_count = 0
        for warp_tile_idx in range(
            relative_warp_id, total_consumer_operations, consumer_warps
        ):
            var m_warp_idx = warp_tile_idx // n_warps_per_block
            var n_warp_idx = warp_tile_idx % n_warps_per_block

            var c_reg_tile = tile_operator.get_c_reg_tile_slice(
                local_tile_count
            )
            store_c[
                c.dtype,
                c.layout,
                c_reg_tile.layout,
                BM,
                BN,
                WM,
                WN,
                c.shape[1](),
            ](c, c_reg_tile, m_warp_idx, n_warp_idx)

            local_tile_count += 1


fn store_c[
    c_type: DType,
    c_layout: Layout,
    c_reg_layout: Layout,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    static_N: Int,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_reg_tile: LayoutTensor[c_type, c_reg_layout, MutableAnyOrigin, *_, **_],
    warp_m: Int,
    warp_n: Int,
):
    var c_block_tile = c.tile[BM, BN](Int(block_idx.x), Int(block_idx.y))
    var c_warp_tile = c_block_tile.tile[WM, WN](Int(warp_m), Int(warp_n))

    # NOTE: these numbers are hardcoded based on register fragments shapes
    # these should be derived

    alias output_thread_layout = Layout.col_major(16, 4)

    copy_local_to_dram[output_thread_layout, thread_scope = ThreadScope.WARP](
        c_warp_tile.vectorize[1, 4](), c_reg_tile.vectorize[1, 4](), c
    )
