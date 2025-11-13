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
"""
AMD Warp-Specialized Matrix Multiplication

Architecture Overview:
- Producer warps: Load tiles from global to shared memory
  - A producers: Load M×K tiles from matrix A
  - B producers: Load N×K tiles from matrix B
- Consumer warps: Perform matrix multiplication using shared memory tiles
- Ring buffer: Coordinates producer-consumer synchronization with barriers

Data Flow:
1. Producers load tiles into shared memory stages
2. Barriers ensure data is ready before consumers access it
3. Consumers compute partial results and accumulate
4. Final results written back to global memory

Memory Layout:
- Shared memory is divided into pipeline stages for overlapping
- Each stage contains block tiles that are further divided into warp tiles
- Swizzling may be applied to avoid bank conflicts
"""
from gpu import (
    WARP_SIZE,
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
    warp_id as get_warp_id,
)
from gpu.intrinsics import inlined_assembly
from layout import Layout, LayoutTensor
from layout.layout import blocked_product
from layout.layout_tensor import (
    ThreadScope,
    copy_local_to_dram,
    copy_local_to_shared,
)
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg
from linalg.structuring import ScatterGatherAmd
from utils import IndexList, StaticTuple

from .ring_buffer import RingBuffer
from .structured import (
    AmdTileOperator,
    SMemBuffer,
    ThreadRole,
)

# Type aliases for cleaner code
alias GlobalTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype, layout, MutAnyOrigin, address_space = AddressSpace.GLOBAL
]


@parameter
fn validate_config[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    m_warps: Int,
    n_warps: Int,
    producer_a: Int,
    producer_b: Int,
    consumer: Int,
]():
    """Validates the configuration parameters for the matrix multiplication kernel.
    """
    constrained[
        BM % WM == 0 and BN % WN == 0,
        "Block dims must be divisible by warp dims",
    ]()
    constrained[
        m_warps % producer_a == 0, "M warps must be divisible by A producers"
    ]()
    constrained[
        n_warps % producer_b == 0, "N warps must be divisible by B producers"
    ]()
    constrained[
        m_warps * n_warps % consumer == 0,
        "Total warps must be divisible by consumers",
    ]()
    constrained[
        consumer >= producer_a and consumer >= producer_b,
        "Need enough consumers",
    ]()
    constrained[
        consumer.is_power_of_two(), "Consumer warps must be power of 2"
    ]()


@always_inline
fn determine_thread_role[
    producer_a_warps: Int,
    producer_b_warps: Int,
]() -> Tuple[ThreadRole, Int]:
    """Returns (role, consumer_warp_id within role group)."""
    var warp_id = get_warp_id()
    alias producer_thread_count = (
        producer_a_warps + producer_b_warps
    ) * WARP_SIZE

    if thread_idx.x < UInt(producer_thread_count):
        if warp_id < UInt(producer_a_warps):
            return (ThreadRole.PRODUCER, 0)  # A producer
        else:
            return (ThreadRole.PRODUCER, 1)  # B producer
    else:
        return (ThreadRole.CONSUMER, 2)


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


@always_inline
fn lgkm_wait():
    inlined_assembly[
        "s_waitcnt lgkmcnt(0)",
        NoneType,
        constraints="",
        has_side_effect=True,
    ]()


@always_inline
fn run_producer[
    dtype: DType,
    layout: Layout,
    warp_tile_layout: Layout,
    block_rows: Int,  # BM for A, BN for B
    block_cols: Int,  # BK
    warp_rows: Int,  # WM for A, WN for B
    warp_cols: Int,  # WK
    producer_warps: Int,
    pipeline_stages: Int,
    k_tile_size: Int,
    simd_width: Int,
    warps_processed_per_producer: Int,
    tile_count: Int,
    swizzle: OptionalReg[Swizzle],
](
    matrix: GlobalTensor[dtype, layout],
    mut ring_buffer: RingBuffer,
    warp_id: UInt,
    block_idx_dim: Int,
):
    """Generic producer function for loading matrix tiles from global to shared memory.
    """

    alias thread_layout = get_producer_warp_thread_layout[
        k_tile_size, simd_width, block_rows, block_cols
    ]()

    alias total_participating_threads = thread_layout.size()
    alias elements_loaded_per_thread = warp_tile_layout.size() // total_participating_threads
    alias simd_loads_per_thread = elements_loaded_per_thread // simd_width

    var fragment = LayoutTensor[
        dtype,
        Layout.row_major(simd_loads_per_thread, simd_width),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    # Use producer view as context manager
    with ring_buffer.producer[
        warps_processed_per_producer, producer_warps
    ]() as producer_view:
        var scatter_gather = ScatterGatherAmd[thread_layout](matrix)

        @parameter
        for tile_num in range(tile_count):
            alias stage = tile_num % pipeline_stages
            var gmem_tile = matrix.tile[block_rows, block_cols](
                block_idx_dim, tile_num
            )

            @parameter
            for local_tile_count in range(warps_processed_per_producer):
                var warp_tile_idx = (
                    Int(warp_id) + local_tile_count * producer_warps
                )

                # Load gmem tile to register fragments
                var gmem_warp_tile = gmem_tile.tile[warp_rows, warp_cols](
                    warp_tile_idx, 0
                )
                var reg_tile_frag = fragment.vectorize[1, simd_width]()
                scatter_gather.copy(
                    reg_tile_frag,
                    gmem_warp_tile.vectorize[1, simd_width](),
                )

                # Acquire SMEM tile using producer view context manager
                with producer_view.acquire_tile(
                    stage, warp_tile_idx
                ) as smem_warp_tile:
                    # Store register fragment to SMEM tile
                    copy_local_to_shared[
                        thread_layout=thread_layout,
                        swizzle=swizzle,
                        thread_scope = ThreadScope.WARP,
                        row_major=True,
                    ](
                        smem_warp_tile.vectorize[1, simd_width](),
                        reg_tile_frag,
                    )
                    # Wait for data to land
                    lgkm_wait()
                    # Tile is automatically released when exiting the context


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
        in_type, a_layout, MutAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    b: LayoutTensor[
        in_type, b_layout, MutAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    c: LayoutTensor[
        out_type,
        c_layout,
        MutAnyOrigin,
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

    # Validate configuration
    validate_config[
        BM,
        BN,
        BK,
        WM,
        WN,
        WK,
        m_warps_per_block,
        n_warps_per_block,
        a_producer_warps,
        b_producer_warps,
        consumer_warps,
    ]()

    var role_info = determine_thread_role[a_producer_warps, b_producer_warps]()
    var role = role_info[0]
    var role_group = role_info[1]
    var warp_id = get_warp_id()

    alias swizzle = Swizzle(3, 0, 1)

    # Compute k_group_size like MMAConfig does
    alias simd_width = simd_width_of[in_type]()
    alias registers_per_thread_a = num_matrix_reg[MMA_M, MMA_K]()
    alias registers_per_thread_b = num_matrix_reg[MMA_N, MMA_K]()
    alias k_group_size_a = simd_width // registers_per_thread_a
    alias k_group_size_b = simd_width // registers_per_thread_b

    alias adjusted_mma_k_shape_a = MMA_K * k_group_size_a
    alias adjusted_mma_k_shape_b = MMA_K * k_group_size_b

    constrained[
        adjusted_mma_k_shape_a == adjusted_mma_k_shape_b,
        "MMA_K shapes must be equal",
    ]()

    alias smem_layout_a = smem_tile_layout[adjusted_mma_k_shape_a, BM, BK]()
    alias smem_layout_b = smem_tile_layout[adjusted_mma_k_shape_b, BN, BK]()

    var smem_buffer_a = SMemBuffer[
        in_type, smem_layout_a, pipeline_stages, BM, BK, WM, WK
    ]()
    var smem_buffer_b = SMemBuffer[
        in_type, smem_layout_b, pipeline_stages, BN, BK, WN, WK
    ]()

    var ring_buffer_a = RingBuffer[
        consumer_warps,
        n_warps_per_block,  # reads_per_warp_block
    ](smem_buffer_a)

    var ring_buffer_b = RingBuffer[
        consumer_warps,
        m_warps_per_block,  # reads_per_warp_block
    ](smem_buffer_b)

    alias consumer_thread_layout_a = get_producer_warp_thread_layout[
        adjusted_mma_k_shape_a, simd_width, BM, BK
    ]()

    alias consumer_thread_layout_b = get_producer_warp_thread_layout[
        adjusted_mma_k_shape_b, simd_width, BN, BK
    ]()

    barrier()  # NOTE: probably not necessary but I saw it in the HF code around the same point

    alias tile_count = K // BK
    alias warps_processed_per_producer_a = Int(
        m_warps_per_block // a_producer_warps
    )
    alias warps_processed_per_producer_b = Int(
        n_warps_per_block // b_producer_warps
    )

    # Producer logic - simplified using generic function
    if role is ThreadRole.PRODUCER:
        if role_group == 0:  # A producer
            run_producer[
                in_type,
                a_layout,
                Layout.row_major(WM, WK),  # warp_tile_layout for A
                BM,
                BK,
                WM,
                WK,
                a_producer_warps,
                pipeline_stages,
                adjusted_mma_k_shape_a,
                simd_width,
                warps_processed_per_producer_a,
                tile_count,
                swizzle,
            ](
                a,
                ring_buffer_a,
                warp_id,
                Int(block_idx.x),
            )
        else:  # B producer
            var producer_warp_id = warp_id - UInt(a_producer_warps)
            run_producer[
                in_type,
                b_layout,
                Layout.row_major(WN, WK),  # warp_tile_layout for B
                BN,
                BK,
                WN,
                WK,
                b_producer_warps,
                pipeline_stages,
                adjusted_mma_k_shape_b,
                simd_width,
                warps_processed_per_producer_b,
                tile_count,
                swizzle,
            ](
                b,
                ring_buffer_b,
                producer_warp_id,
                Int(block_idx.y),
            )

    else:  # Consumer
        # NOTE: these numbers are hardcoded based on register fragments shapes
        alias output_thread_layout = Layout.col_major(16, 4)

        var c_block_tile = c.tile[BM, BN](Int(block_idx.x), Int(block_idx.y))
        var c_scatter_gather = ScatterGatherAmd[
            output_thread_layout, thread_scope = ThreadScope.WARP
        ](c)

        alias total_consumer_operations = m_warps_per_block * n_warps_per_block
        alias warps_computed_per_consumer = total_consumer_operations // consumer_warps

        var consumer_warp_id = (
            Int(warp_id) - a_producer_warps - b_producer_warps
        )

        # Create a single tile operator that we'll reuse for each tile
        var tile_operator = AmdTileOperator[
            in_type,
            out_type,
            smem_buffer_a.WarpTileType.layout,
            smem_buffer_b.WarpTileType.layout,
            IndexList[3](MMA_M, MMA_N, MMA_K),
            swizzle=swizzle,
        ]()

        @parameter
        fn compute_indices(local_tile_count: Int) -> Tuple[Int, Int]:
            """Computes warp tile index, m_warp_idx, and n_warp_idx."""
            var warp_tile_idx = (
                consumer_warp_id + local_tile_count * consumer_warps
            )
            var m_warp_idx, n_warp_idx = divmod(
                warp_tile_idx, n_warps_per_block
            )
            return (m_warp_idx, n_warp_idx)

        # Use consumer views as context managers
        with ring_buffer_a.consumer[
            warps_computed_per_consumer, consumer_warps
        ](consumer_warp_id) as consumer_view_a, ring_buffer_b.consumer[
            warps_computed_per_consumer, consumer_warps
        ](
            consumer_warp_id
        ) as consumer_view_b:
            # Process each tile completely before moving to the next
            @parameter
            for local_tile_count in range(warps_computed_per_consumer):
                var m_warp_idx, n_warp_idx = compute_indices(local_tile_count)

                # Reset accumulator for this new M,N position
                tile_operator.reset_accumulator()

                # Accumulate across all K tiles for this M,N position
                @parameter
                for i in range(tile_count):
                    alias stage = i % pipeline_stages

                    # Get tiles using consumer view context managers
                    with consumer_view_a.acquire_tile(
                        stage, local_tile_count, m_warp_idx
                    ) as smem_tile_a, consumer_view_b.acquire_tile(
                        stage, local_tile_count, n_warp_idx
                    ) as smem_tile_b:
                        alias num_k_tiles = tile_operator.total_k_tiles

                        # Load all K tiles
                        @parameter
                        for k_idx in range(num_k_tiles):
                            tile_operator.load_tile_fragment[k_idx](
                                smem_tile_a, smem_tile_b
                            )

                        # Perform MMA computation
                        @parameter
                        for k_idx in range(num_k_tiles):
                            tile_operator.mma_compute[k_idx]()

                        # Tiles are automatically released when exiting the context

                # Write this tile's result to global memory immediately
                var warp_tile_idx = (
                    consumer_warp_id + local_tile_count * consumer_warps
                )
                var m_warp_idx_out, n_warp_idx_out = divmod(
                    warp_tile_idx, n_warps_per_block
                )

                # Store results to global memory using the public out_reg_tile
                # This matches MmaOpAMD's interface
                var c_warp_tile = c_block_tile.tile[WM, WN](
                    Int(m_warp_idx_out), Int(n_warp_idx_out)
                )

                c_scatter_gather.copy(
                    c_warp_tile.vectorize[1, 4](),
                    tile_operator.out_reg_tile.vectorize[1, 4](),
                )


@always_inline
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
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin, *_, **_],
    c_reg_tile: LayoutTensor[c_type, c_reg_layout, MutAnyOrigin, *_, **_],
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


@always_inline
fn warp_specialized_matmul[
    M: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    a_producer_warps: Int,
    b_producer_warps: Int,
    consumer_warps: Int,
    pipeline_stages: Int = 1,
](
    a_device_tensor: LayoutTensor[
        DType.bfloat16,
        Layout.row_major(M, K),
    ],
    b_device_tensor: LayoutTensor[DType.bfloat16, Layout.row_major(N, K)],
    c_device_tensor: LayoutTensor[DType.float32, Layout.row_major(M, N)],
    ctx: DeviceContext,
) raises:
    alias kernel = warp_specialized_matmul[
        a_device_tensor.dtype,
        c_device_tensor.dtype,
        a_device_tensor.layout,
        b_device_tensor.layout,
        c_device_tensor.layout,
        BM,
        BN,
        BK,
        WM,
        WN,
        WK,
        a_producer_warps=a_producer_warps,
        b_producer_warps=b_producer_warps,
        consumer_warps=consumer_warps,
        pipeline_stages=pipeline_stages,
    ]

    var global_c_device_tensor = c_device_tensor.address_space_cast[
        AddressSpace.GLOBAL
    ]()
    var global_a_device_tensor = a_device_tensor.address_space_cast[
        AddressSpace.GLOBAL
    ]()
    var global_b_device_tensor = b_device_tensor.address_space_cast[
        AddressSpace.GLOBAL
    ]()

    ctx.enqueue_function_checked[kernel, kernel](
        global_a_device_tensor,
        global_b_device_tensor,
        global_c_device_tensor,
        grid_dim=(M // BM, N // BN),
        block_dim=(
            WARP_SIZE * (a_producer_warps + b_producer_warps + consumer_warps)
        ),
    )
