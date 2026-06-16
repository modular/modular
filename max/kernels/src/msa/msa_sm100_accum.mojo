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
"""MSA-private copy of the SM100 tcgen05 accumulator / operand machinery.

Duplicate of the dense SM100 MHA accumulator types from
`nn.attention.gpu.nvidia.sm100.mha_1q`, lifted here so the block-major forward
(`msa_2q.mojo`) can carry the `num_m_mmas == 2` ping-pong without dragging the
dense MHA / MLA decode kernels through their full test matrix.
"""

from std.sys import size_of

import std.gpu.primitives.warp as warp
from std.gpu import lane_id
from std.gpu.compute.mma import MMAOperandDescriptor
from std.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
)
from std.gpu.sync import named_barrier
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_st,
    tcgen05_store_wait,
)
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import AddressSpace
from std.memory import bitcast

from layout import IntTuple, Layout, LayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
    tile_to_descriptor,
)
from layout.tma_async import PipelineState, SharedMemBarrier

from std.utils.index import Index


struct RegisterAccumulatorDescription:
    var num_mmas: Int
    var frag_size: Int

    @always_inline
    def __init__(out self, num_mmas: Int, frag_size: Int):
        self.num_mmas = num_mmas
        self.frag_size = frag_size


# consumer_group_size equals
# sm90: 128 (warp group size)
# sm100: num_softmax_threads
struct RegisterAccumulatorLayout[
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    consumer_group_size: Int,
    *,
    frag_simdwidth: Int = 2,
](TrivialRegisterPassable):
    comptime frag_size: Int = Self.MMA_M * Self.MMA_N // Self.consumer_group_size
    comptime num_row_blocks_per_mma = 2
    comptime element_layout: Layout = Layout.row_major(1, Self.frag_simdwidth)
    comptime rows_of_frags_layout: Layout = Layout.row_major(
        Self.num_m_mmas * Self.num_n_mmas, Self.frag_size
    )
    comptime vec_output_layout: Layout = Layout(
        IntTuple(
            IntTuple(Self.num_row_blocks_per_mma, Self.num_m_mmas),
            IntTuple(
                Self.frag_size
                // (Self.num_row_blocks_per_mma * Self.frag_simdwidth),
                Self.num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(Self.frag_simdwidth, Self.frag_size),
            IntTuple(
                Self.num_row_blocks_per_mma * Self.frag_simdwidth,
                Self.num_m_mmas * Self.frag_size,
            ),
        ),
    )

    @staticmethod
    @always_inline
    def description() -> RegisterAccumulatorDescription:
        comptime assert Self.vec_output_layout.size() > 0, "layout: " + String(
            Self.vec_output_layout
        )

        return RegisterAccumulatorDescription(
            Self.num_m_mmas * Self.num_n_mmas, Self.frag_size
        )


struct MMAOperandOffsetFn[
    dtype: DType,
    BMN: Int,
    BK: Int,
    swizzle: TensorMapSwizzle,
    is_k_major: Bool,
    WMMA_MN: Int,
    WMMA_K: Int,
](TrivialRegisterPassable):
    # Use typed layouts as source of truth; bridge to legacy Layout for
    # LayoutTensor and MMA descriptor pipeline.
    comptime layout = tile_layout_k_major_typed[
        Self.dtype, Self.BMN, Self.BK, Self.swizzle
    ].to_layout() if Self.is_k_major else tile_layout_mn_major_typed[
        Self.dtype, Self.BMN, Self.BK, Self.swizzle
    ].to_layout()
    comptime layout_size: Int = Self.layout.size()

    comptime canonical_K = Self.swizzle.bytes() // size_of[
        Self.dtype
    ]() if Self.swizzle != TensorMapSwizzle.SWIZZLE_NONE else Self.BK
    comptime canonical_layout_flat = tile_layout_k_major_typed[
        Self.dtype, Self.BMN, Self.canonical_K, Self.swizzle
    ].to_layout() if Self.is_k_major else Self.layout
    comptime canonical_layout = tile_to_descriptor[
        Self.dtype, Self.canonical_layout_flat, Self.is_k_major
    ]()
    comptime canonical_layout_size = Self.canonical_layout.size()

    @always_inline
    def __init__(out self):
        pass


trait DescriptorPair(TrivialRegisterPassable):
    comptime a_t: MMAOperandDescriptor
    comptime b_t: MMAOperandDescriptor

    @always_inline
    def get_a(self) -> Self.a_t:
        ...

    @always_inline
    def get_b(self) -> Self.b_t:
        ...


trait WriteableMMAOperandDescriptor(TrivialRegisterPassable):
    @always_inline
    def copy_from[
        src_type: DType, src_layout: Layout, src_element_layout: Layout, //
    ](
        self,
        src: LayoutTensor[
            src_type,
            src_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
            element_layout=src_element_layout,
        ],
    ):
        ...


trait DescriptorPairTS(TrivialRegisterPassable):
    comptime a_t: WriteableMMAOperandDescriptor
    comptime b_t: MMAOperandDescriptor

    @always_inline
    def get_a(self) -> Self.a_t:
        ...

    @always_inline
    def get_b(self) -> Self.b_t:
        ...


def local_tensor_type[
    dtype: DType, layout: Layout, element_layout: Layout
](
    out dummy_arg: LayoutTensor[
        dtype,
        layout,
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
        element_layout=element_layout,
    ]
):
    dummy_arg = {None}


trait AccumulatorTile(TrivialRegisterPassable):
    comptime dtype: DType
    comptime element_layout: Layout
    comptime vec_output_layout: Layout
    comptime rows_of_frags_layout: Layout

    @staticmethod
    @always_inline
    def _empty_tensor() -> (
        type_of(
            local_tensor_type[
                Self.dtype, Self.vec_output_layout, Self.element_layout
            ]()
        )
    ):
        ...

    @staticmethod
    @always_inline
    def rows_of_frags(
        src: type_of(Self._empty_tensor()),
        out res: LayoutTensor[
            Self.dtype,
            Self.rows_of_frags_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ],
    ):
        ...

    @staticmethod
    @always_inline
    def allocate_register_tile(
        out res: type_of(Self._empty_tensor()),
    ):
        ...

    @always_inline
    def copy_from(
        self,
        src: type_of(Self._empty_tensor()),
    ):
        ...

    @always_inline
    def copy_to(
        self,
        dst: type_of(Self._empty_tensor()),
    ):
        ...


struct UMMADescriptorSS[operand_type: DType](
    DescriptorPair, TrivialRegisterPassable
):
    comptime operand_t = Self.operand_type
    comptime a_t = MMASmemDescriptor
    comptime b_t = MMASmemDescriptor

    var a: Self.a_t
    var b: Self.b_t

    @always_inline
    def __init__(out self, a: Self.a_t, b: Self.b_t):
        self.a = a
        self.b = b

    @always_inline
    def get_a(self) -> Self.a_t:
        return self.a

    @always_inline
    def get_b(self) -> Self.b_t:
        return self.b


@always_inline
def _tmem_offset(dtype_size: Int, *, MMA_N: Int, m_mma: Int, n_mma: Int) -> Int:
    row = 16 * m_mma
    col = (MMA_N * n_mma * dtype_size) // 4
    return (row << 16) + col


@always_inline
def _tmem_offset[dtype: DType, *, MMA_N: Int, m_mma: Int, n_mma: Int]() -> Int:
    comptime linear = _tmem_offset(
        size_of[dtype](), MMA_N=MMA_N, m_mma=m_mma, n_mma=n_mma
    )
    return linear


struct TMemAccumulator[
    dtype_: DType,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_softmax_threads: Int,
](AccumulatorTile, TrivialRegisterPassable):
    comptime dtype: DType = Self.dtype_
    comptime layout_t = RegisterAccumulatorLayout[
        Self.MMA_M,
        Self.MMA_N,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]
    comptime vec_output_layout = Self.layout_t.vec_output_layout
    comptime element_layout = Self.layout_t.element_layout
    comptime rows_of_frags_layout = Self.layout_t.rows_of_frags_layout
    comptime frag_size = Self.layout_t.frag_size

    var tmem_addr: UInt32

    @always_inline
    def __init__(out self, tmem_addr: UInt32):
        Self.check_constraints()
        self.tmem_addr = tmem_addr

    @staticmethod
    @always_inline
    def _empty_tensor() -> (
        type_of(
            local_tensor_type[
                Self.dtype, Self.vec_output_layout, Self.layout_t.element_layout
            ]()
        )
    ):
        Self.check_constraints()
        return local_tensor_type[
            Self.dtype, Self.vec_output_layout, Self.layout_t.element_layout
        ]()

    @always_inline
    def __getitem__(self, i: UInt32) -> Self:
        return {self.tmem_addr + i * UInt32(Self.MMA_N)}

    @always_inline
    @staticmethod
    def check_constraints():
        comptime assert Self.vec_output_layout[0].size() > 0, (
            "layout: "
            + String(Self.vec_output_layout)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
        )
        comptime assert (
            Self.vec_output_layout[1].size() > 0
        ), "layout: " + String(Self.vec_output_layout)
        comptime assert Self.MMA_M > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )
        comptime assert Self.MMA_N > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )
        comptime assert Self.num_m_mmas > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nnum_m_mmas = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )
        comptime assert Self.num_n_mmas > 0, (
            "MMA_M = "
            + String(Self.MMA_M)
            + "\nMMA_N = "
            + String(Self.MMA_N)
            + "\nm_mma = "
            + String(Self.num_m_mmas)
            + "\nnum_n_mmas = "
            + String(Self.num_n_mmas)
            + "\n"
        )

    @always_inline
    def offset[m_mma: Int, n_mma: Int](self) -> UInt32:
        Self.check_constraints()

        comptime if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                Self.dtype, MMA_N=Self.MMA_N, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + UInt32(linear)

    @staticmethod
    @always_inline
    def rows_of_frags(
        src: type_of(Self._empty_tensor()),
        out res: LayoutTensor[
            Self.dtype,
            Self.rows_of_frags_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ],
    ):
        Self.check_constraints()
        res = {src.ptr}

    @staticmethod
    @always_inline
    def allocate_register_tile(
        out res: type_of(Self._empty_tensor()),
    ):
        res = type_of(res).stack_allocation()

    @always_inline
    def copy_from(
        self,
        src: type_of(Self._empty_tensor()),
    ):
        frags = Self.rows_of_frags(src).vectorize[1, Self.frag_size]()
        comptime dtype_size = size_of[Self.dtype]()
        comptime assert dtype_size == 4
        comptime frag_size_b32 = Self.frag_size * dtype_size // 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus fills 8 columns
        # and writes 4 values per thread.
        comptime repeat = frag_size_b32 // 4

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma
                comptime tmem_offset = _tmem_offset(
                    dtype_size,
                    MMA_N=Self.MMA_N,
                    m_mma=m_mma,
                    n_mma=n_mma,
                )
                tmem = self.tmem_addr + UInt32(tmem_offset)
                frag = bitcast[DType.uint32, frag_size_b32](frags[mma_id, 0])
                # 16 x 256b results in repeated 8x4 matrix of <1,2> vector pattern
                var frag_st = InlineArray[Scalar[DType.uint32], frag_size_b32](
                    uninitialized=True
                )

                comptime for _i in range(frag_size_b32):
                    frag_st[_i] = frag[_i]
                tcgen05_st[
                    datapaths=16,  # first dimension of the shape
                    bits=256,  # second dimension of the shape
                    repeat=repeat,
                    pack=False,
                ](tmem, frag_st)
        tcgen05_store_wait()
        tcgen05_fence_before()
        named_barrier[Int32(Self.num_softmax_threads)]()

    @always_inline
    def copy_to(
        self,
        dst: type_of(Self._empty_tensor()),
    ):
        frags = Self.rows_of_frags(dst).vectorize[1, Self.frag_size]()
        comptime dtype_size = size_of[Self.dtype]()
        comptime assert dtype_size == 4
        comptime frag_size_b32 = (Self.frag_size * dtype_size) // 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        comptime repeat = frag_size_b32 // 4
        comptime assert (
            Self.num_m_mmas * Self.num_n_mmas == type_of(frags).layout.size()
        )

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma
                comptime tmem_offset = _tmem_offset(
                    dtype_size,
                    MMA_N=Self.MMA_N,
                    m_mma=m_mma,
                    n_mma=n_mma,
                )
                tmem = self.tmem_addr + UInt32(tmem_offset)
                comptime if repeat > 16:
                    # Split into two halves to reduce register pressure.
                    comptime half_repeat = repeat // 2
                    comptime half_frag = frag_size_b32 // 2
                    comptime half_col_offset = half_repeat * 8 * dtype_size // 4
                    var _ld_lo = tcgen05_ld[
                        datapaths=16,
                        bits=256,
                        repeat=half_repeat,
                        dtype=DType.uint32,
                        pack=False,
                        width=half_frag,
                    ](tmem)
                    var _ld_hi = tcgen05_ld[
                        datapaths=16,
                        bits=256,
                        repeat=half_repeat,
                        dtype=DType.uint32,
                        pack=False,
                        width=half_frag,
                    ](tmem + UInt32(half_col_offset))
                    var _ld_simd = SIMD[DType.uint32, frag_size_b32]()

                    comptime for _i in range(half_frag):
                        _ld_simd[_i] = _ld_lo[_i]
                        _ld_simd[_i + half_frag] = _ld_hi[_i]
                    frags[mma_id, 0] = bitcast[
                        Self.dtype, frags.element_layout.size()
                    ](_ld_simd)
                else:
                    var _ld_result = tcgen05_ld[
                        datapaths=16,
                        bits=256,
                        repeat=repeat,
                        dtype=DType.uint32,
                        pack=False,
                        width=frag_size_b32,
                    ](tmem)
                    var _ld_simd = SIMD[DType.uint32, frag_size_b32]()

                    comptime for _i in range(frag_size_b32):
                        _ld_simd[_i] = _ld_result[_i]
                    frags[mma_id, 0] = bitcast[
                        Self.dtype, frags.element_layout.size()
                    ](_ld_simd)

        tcgen05_load_wait()


struct TMemOperand[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    num_softmax_threads: Int,
](TrivialRegisterPassable, WriteableMMAOperandDescriptor):
    var tmem_addr: UInt32

    comptime reg_layout = RegisterAccumulatorLayout[
        Self.MMA_M,
        Self.MMA_N,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]
    comptime frag_size = Self.reg_layout.frag_size
    comptime vec_output_layout = Self.reg_layout.vec_output_layout
    comptime reg_tile_t = type_of(
        local_tensor_type[
            Self.dtype, Self.vec_output_layout, Self.reg_layout.element_layout
        ]()
    )

    @always_inline
    def __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    def offset[m_mma: Int, k_mma: Int](self) -> UInt32:
        comptime assert Self.MMA_M > 0, "MMA_M = " + String(Self.MMA_M) + "\n"
        comptime assert Self.MMA_K > 0, "MMA_K = " + String(Self.MMA_K) + "\n"

        comptime if m_mma == 0 and k_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                Self.dtype, MMA_N=Self.MMA_K, m_mma=m_mma, n_mma=k_mma
            ]()
            return self.tmem_addr + UInt32(linear)

    @always_inline
    def copy_from[
        src_type: DType,
        src_layout: Layout,
        src_element_layout: Layout,
        //,
    ](
        self,
        src: LayoutTensor[
            src_type,
            src_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
            element_layout=src_element_layout,
        ],
    ):
        # src has row of frags layout
        comptime num_frags = src_layout[0].size()
        comptime assert num_frags == Self.num_m_mmas * Self.num_n_mmas
        comptime assert Self.num_n_mmas == 1
        comptime assert Self.frag_size == src_layout[1].size(), (
            "Self.frag_size = "
            + String(Self.frag_size)
            + "\nsrc_layout = "
            + String(src_layout)
        )
        comptime assert src_element_layout.size() == 1
        comptime src_size = size_of[src_type]()
        comptime dst_size = size_of[Self.dtype]()
        comptime frag_size_b32 = (Self.frag_size * dst_size) // 4
        # 16 x 256b results in repeated 8x4<1xN> pattern, where
        comptime N = 32 // (4 * src_size)
        comptime bytes = 4 * dst_size * N
        # For fp8, the tcgen05.mma.kind::f8f6f4 reader expects K laid
        # out in 8-col groups (MMA_K=32 fp8 = 8 32-bit cols = 256 bits),
        # so use bits=256 with repeat=4 for frag_size_b32=16. bf16 keeps
        # the natural bits=128 (2 cols/repeat, 16 repeats).
        comptime bits = 256 if Self.dtype.is_float8() else 8 * bytes
        # e.g., N = 2 for fp32
        #
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        # width == (repeat * bits * datapaths) // (32 * 32)
        comptime repeat = 64 * frag_size_b32 // bits
        # We need to reshape into a row of frags
        comptime assert (
            Self.num_m_mmas * Self.num_n_mmas * Self.frag_size
            == src_layout.size() * src_element_layout.size()
        )
        frags = LayoutTensor[
            src_type,
            Layout(
                IntTuple(Self.num_m_mmas * Self.num_n_mmas),
                IntTuple(Self.frag_size),
            ),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
            element_layout=Layout.row_major(Self.frag_size),
        ](src.ptr)
        # frags = src.vectorize[1, Self.frag_size]()
        # assume src loaded with 256 bits
        comptime assert src_size >= dst_size
        # MSA ping-pong: each 128-thread WG owns a full BM=128 tile = 2 m-blocks
        # (num_m_mmas==2).  Loop them; self.offset routes block b to TMEM row 16*b.
        comptime for m_mma in range(Self.num_m_mmas):
            tmem = self.offset[m_mma, 0]()
            frag = bitcast[DType.uint32, frag_size_b32](
                frags[m_mma].cast[Self.dtype]()
            )
            # 16 x 256b results in repeated 8x4<1x64b> pattern
            # 256b means 256 // 4 = 64b per thread
            var frag_st2 = InlineArray[Scalar[DType.uint32], frag_size_b32](
                uninitialized=True
            )

            comptime if Self.dtype.is_float8():
                # The SS-D fragment per thread (output of Q@K^T MMA) puts
                # each 4-lane group at a 2x2 (M, N) block. For fp8 with
                # 1 u32 = 4 fp8, the TS MMA (kind::f8f6f4) reader expects
                # 4 K-consecutive fp8 in ONE M-row per u32. The two layouts
                # disagree at thread-granularity: the data we need also
                # lives in OTHER threads' registers. Redistribute via
                # warp shuffles.
                # 1. Each thread iterates s_dst over its 16 destination u32 slots.
                # 2. For each slot, identifies which (M, K) quartet it owns in TS A layout via mma_n_tile, k_lo_half, m_local_src.
                # 3. Uses warp.shuffle_idx to pull u32 values from two peer lanes in the same lane_row (the SS-D layout puts the K-cells we want on different lane_cols of the same row).
                # 4. Picks the m_local_src-th u16 half of each of the two received u32s and concatenates them as the destination u32.

                # Warp grid is 8 lane_rows x 4 lane_cols (32 lanes total).
                comptime lane_cols_per_row = 4
                # SS-D M-pair: each thread owns 2 M positions
                # {lane_row, lane_row + 8} in its own registers.
                comptime m_per_pair = 2
                # TS A packs a K-quartet (4 fp8) per u32. We assemble it
                # from a low half (K, K+1) and a high half (K+2, K+3),
                # each sourced from a distinct peer lane.
                comptime k_halves_per_n_tile = 2
                comptime lane_cols_per_k_half = (
                    lane_cols_per_row // k_halves_per_n_tile  # = 2
                )
                # 4 destination u32 slots per N-tile = the (M, K-half)
                # outer product across the M-pair and the two K-halves.
                comptime dst_slots_per_n_tile = (
                    m_per_pair * k_halves_per_n_tile
                )
                # Source SS-D layout also packs 4 u32 slots per N-tile:
                # the 4 lane_cols of one lane_row each contribute one slot.
                comptime src_slots_per_n_tile = lane_cols_per_row

                var lane_row_ui = UInt32(lane_id()) // lane_cols_per_row
                var lane_col_ui = UInt32(lane_id()) % lane_cols_per_row

                comptime for s_dst in range(frag_size_b32):
                    comptime mma_n_tile = s_dst // dst_slots_per_n_tile
                    comptime k_lo_half = s_dst % k_halves_per_n_tile
                    comptime m_local_src = (
                        s_dst // k_halves_per_n_tile
                    ) % m_per_pair

                    var l_src = lane_row_ui
                    var src_lane_a = l_src * lane_cols_per_row + UInt32(
                        k_lo_half * lane_cols_per_k_half
                    )
                    var src_lane_b = src_lane_a + 1
                    var received_a: Scalar[DType.uint32] = 0
                    var received_b: Scalar[DType.uint32] = 0
                    # Each lane_col publishes a different slot of a_frag;
                    # only the iteration matching this thread's lane_col
                    # contributes to its output u32.
                    comptime for c_val in range(src_slots_per_n_tile):
                        comptime publisher_slot = (
                            mma_n_tile * src_slots_per_n_tile + c_val
                        )
                        var val: Scalar[DType.uint32] = frag[publisher_slot]
                        var ra = warp.shuffle_idx(val, src_lane_a)
                        var rb = warp.shuffle_idx(val, src_lane_b)
                        if lane_col_ui == UInt32(c_val):
                            received_a = ra
                            received_b = rb

                    comptime which_half = Int(m_local_src)
                    var ab_halves = bitcast[DType.uint16, 4](
                        SIMD[DType.uint32, 2](received_a, received_b)
                    )
                    # ab_halves = [a_lo, a_hi, b_lo, b_hi]
                    var packed = SIMD[DType.uint16, 2](
                        ab_halves[which_half],
                        ab_halves[which_half + 2],
                    )
                    frag_st2[s_dst] = bitcast[DType.uint32, 1](packed)
            else:
                comptime for _i in range(frag_size_b32):
                    frag_st2[_i] = frag[_i]

            tcgen05_st[
                datapaths=16,  # first dimension of the shape
                bits=bits,  # second dimension of the shape
                repeat=repeat,
                pack=False,
            ](tmem, frag_st2)
        tcgen05_store_wait()
        named_barrier[Int32(Self.num_softmax_threads)]()

    @always_inline
    def copy_to[
        dst_type: DType,
        dst_layout: Layout,
        dst_element_layout: Layout,
        //,
    ](
        self,
        dst: LayoutTensor[
            dst_type,
            dst_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
            element_layout=dst_element_layout,
        ],
    ):
        # src has row of frags layout
        comptime num_frags = dst_layout[0].size()
        comptime assert num_frags == Self.num_m_mmas * Self.num_n_mmas
        comptime assert Self.frag_size == dst_layout[1].size()
        comptime assert dst_element_layout.size() == 1
        comptime assert size_of[dst_type]() == 4
        # 16 x 256b results in repeated 8x4<1x2> pattern
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        comptime src_size = size_of[Self.dtype]()
        comptime dst_size = size_of[dst_type]()
        comptime frag_size_b32 = (Self.frag_size * src_size) // 4
        # 16 x 256b results in repeated 8x4<1xN> pattern, where
        comptime N = 32 // (4 * dst_size)
        comptime bytes = 4 * src_size * N
        comptime bits = 8 * bytes
        # e.g., N = 2 for fp32
        #
        # each repetition thus loads 8 columns
        # and loads 4 values per thread.
        # width == (repeat * bits * datapaths) // (32 * 32)
        comptime repeat = 64 * frag_size_b32 // bits
        #
        frags = dst.vectorize[1, Self.frag_size]()
        # assume src loaded with 256 bits
        comptime assert src_size <= dst_size
        comptime assert Self.num_n_mmas == 1

        comptime for m_mma in range(Self.num_m_mmas):
            tmem = self.offset[m_mma, 0]()
            # 16 x 256b results in repeated 8x4<1x2> pattern
            var _ld_result2 = tcgen05_ld[
                datapaths=16,  # first dimension of the shape
                bits=bits,  # second dimension of the shape
                repeat=repeat,
                dtype=DType.uint32,
                pack=False,
                width=frag_size_b32,
            ](tmem)
            var _ld_simd2 = SIMD[DType.uint32, frag_size_b32]()

            comptime for _i in range(frag_size_b32):
                _ld_simd2[_i] = _ld_result2[_i]
            frags[m_mma, 0] = rebind[
                SIMD[dst_type, type_of(frags).element_size]
            ](bitcast[Self.dtype, Self.frag_size](_ld_simd2).cast[dst_type]())
        tcgen05_load_wait()


struct UMMADescriptorTS[
    operand_type: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    *,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    consumer_group_size: Int,
](DescriptorPairTS, TrivialRegisterPassable):
    comptime operand_t = Self.operand_type
    comptime a_t = TMemOperand[
        Self.operand_type,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.MMA_M,
        Self.MMA_N,
        Self.MMA_K,
        Self.consumer_group_size,
    ]
    comptime b_t = MMASmemDescriptor

    var a: Self.a_t
    var b: Self.b_t

    @always_inline
    def __init__(out self, a: Self.a_t, b: Self.b_t):
        self.a = a
        self.b = b

    @always_inline
    def get_a(self) -> Self.a_t:
        return self.a

    @always_inline
    def get_b(self) -> Self.b_t:
        return self.b


struct MSASM100TensorAccumulatorSS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    compute_BK: Int,
    num_softmax_threads: Int,
    swizzle_a: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    *,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    pipeline_stages: Int = 1,
](TrivialRegisterPassable):
    comptime operand_t: DType = Self.operand_type
    comptime accum_t: DType = Self.accum_type

    comptime MMA_K = 16 if Self.operand_t.is_half_float() else 32
    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if Self.operand_t.is_float8() else UMMAKind.KIND_F16
    )

    comptime num_m_mmas = Self.BM // Self.MMA_M
    comptime num_n_mmas = Self.BN // Self.MMA_N
    comptime num_k_mmas = Self.compute_BK // Self.MMA_K

    comptime num_m_blocks_per_warp = 2 * Self.BM // Self.num_softmax_threads

    comptime smem_ptr_t = UnsafePointer[
        Scalar[Self.operand_t],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    comptime a_offset = MMAOperandOffsetFn[
        Self.operand_t,
        Self.BM,
        Self.BK,
        Self.swizzle_a,
        True,
        Self.MMA_M,
        Self.MMA_K,
    ]()
    comptime b_offset = MMAOperandOffsetFn[
        Self.operand_t,
        Self.BN,
        Self.BK,
        Self.swizzle_b,
        Self.transpose_b,
        Self.MMA_N,
        Self.MMA_K,
    ]()

    comptime idesc = UMMAInsDescriptor[Self.mma_kind].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=Self.transpose_b,
    ]()

    comptime ab_t: DescriptorPair = UMMADescriptorSS[Self.operand_t]
    comptime a_t: MMAOperandDescriptor = Self.ab_t.a_t
    comptime b_t: MMAOperandDescriptor = Self.ab_t.b_t
    comptime c_t: AccumulatorTile = TMemAccumulator[
        Self.accum_t,
        Self.BM // Self.num_m_blocks_per_warp,
        Self.MMA_N,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]

    var mbar: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
    ]
    var pipeline: PipelineState[Self.pipeline_stages]

    @always_inline
    @staticmethod
    def check_constraints():
        comptime assert (Self.BM % Self.MMA_M) == 0, (
            "BM, MMA_M = " + String(Self.BM) + ", " + String(Self.MMA_M)
        )
        comptime assert ((Self.BN % Self.MMA_N) == 0) and (
            Self.num_n_mmas > 0
        ), ("BN, MMA_N = " + String(Self.BN) + ", " + String(Self.MMA_N))
        comptime assert ((Self.compute_BK % Self.MMA_K) == 0) and (
            Self.num_k_mmas > 0
        ), (
            "compute_BK, MMA_K = "
            + String(Self.compute_BK)
            + ", "
            + String(Self.MMA_K)
        )

    @always_inline
    def __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
        ],
    ):
        Self.check_constraints()
        self.mbar = smem
        self.pipeline = {}

    @always_inline
    def init(self):
        comptime for i in range(Self.pipeline_stages):
            self.mbar[i].init()
            self.mbar[i + Self.pipeline_stages].init(
                Int32(Self.num_softmax_threads)
            )

    @staticmethod
    @always_inline
    def mma_descriptors[
        dtype_a: DType, dtype_b: DType
    ](
        p_a: UnsafePointer[
            Scalar[dtype_a], MutAnyOrigin, address_space=AddressSpace.SHARED
        ],
        p_b: UnsafePointer[
            Scalar[dtype_b], MutAnyOrigin, address_space=AddressSpace.SHARED
        ],
    ) -> Self.ab_t:
        Self.check_constraints()
        comptime a_canonical_layout = Self.a_offset.canonical_layout
        comptime a_type = Self.operand_t
        comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[
            a_type
        ]()
        comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[
            a_type
        ]()
        adesc_base = MMASmemDescriptor.create[aSBO, aLBO, Self.swizzle_a](p_a)

        comptime b_canonical_layout = Self.b_offset.canonical_layout
        comptime b_type = Self.operand_t
        comptime b_stride01 = b_canonical_layout[0].stride[1].value()
        comptime b_stride11 = b_canonical_layout[1].stride[1].value()
        comptime bSBO = (
            b_stride01 if Self.transpose_b else b_stride11
        ) * size_of[b_type]()
        comptime bLBO = (
            b_stride11 if Self.transpose_b else b_stride01
        ) * size_of[b_type]()
        bdesc_base = MMASmemDescriptor.create[bSBO, bLBO, Self.swizzle_b](p_b)

        return Self.ab_t(adesc_base, bdesc_base)

    @always_inline
    def mma(
        mut self,
        a: Self.a_t,
        b: Self.b_t,
        c_base: Self.c_t,
        scale_c: UInt32,
    ):
        c = c_base[self.pipeline.index()]

        comptime for n_mma in range(Self.num_n_mmas):
            comptime for m_mma in range(Self.num_m_mmas):
                c_tmem = c.offset[m_mma, n_mma]()
                comptime for k_mma in range(Self.num_k_mmas):
                    comptime a_offset = Self.a_offset.layout(
                        IntTuple(Self.MMA_M * m_mma, Self.MMA_K * k_mma)
                    )
                    comptime a_offset_bytes = a_offset * size_of[
                        Self.operand_t
                    ]()
                    a_desc = a + a_offset_bytes

                    comptime b_offset = Self.b_offset.layout(
                        IntTuple(Self.MMA_N * n_mma, Self.MMA_K * k_mma)
                    ) * size_of[Self.operand_t]()
                    b_desc = b + b_offset

                    comptime if k_mma == 0:
                        mma[Self.cta_group](
                            a_desc,
                            b_desc,
                            c_tmem,
                            Self.idesc,
                            scale_c,
                        )
                    else:
                        mma[Self.cta_group, c_scale=1](
                            a_desc, b_desc, c_tmem, Self.idesc
                        )

        mma_arrive(self.mbar + self.pipeline.index())
        self.pipeline.step()

    # the mma thread
    # loop:
    #   wait_for_tmem() # self.mbar[Stages + index()].wait(phase())
    #   mma()           # self.mbar[index()].arrive(), step()
    #
    # the softmax thread
    #
    # tmem_arrive_init() # for i in range(Stages): self.mbar[Stages + i].arrive()
    #
    # loop:
    #   wait_for_mma()   # self.mbar[index()].wait(phase())
    #   use accumulator
    #   tmem_arrive()    # self.mbar[Stages + index()].arrive(), step()
    @always_inline
    def wait_for_tmem(self):
        """
        Wait for the accumulator tmem to finish being read.
        """
        self.mbar[UInt32(Self.pipeline_stages) + self.pipeline.index()].wait(
            self.pipeline.phase()
        )

    @always_inline
    def wait_for_mma(self, c_base: Self.c_t) -> Self.c_t:
        """
        Wait for the accumulator tmem to finish being read.
        """
        var idx: UInt32 = self.pipeline.index()
        self.mbar[idx].wait(self.pipeline.phase())
        return c_base[idx]

    @always_inline
    def tmem_arrive_init(self):
        comptime for i in range(Self.pipeline_stages):
            _ = self.mbar[Self.pipeline_stages + i].arrive()

    @always_inline
    def tmem_arrive(mut self):
        """
        Indicate that the accumulator is ready to be updated.
        """
        _ = self.mbar[
            UInt32(Self.pipeline_stages) + self.pipeline.index()
        ].arrive()
        self.pipeline.step()


struct MSASM100TensorAccumulatorTS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    num_softmax_threads: Int,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    comptime operand_t: DType = Self.operand_type
    comptime accum_t: DType = Self.accum_type

    comptime MMA_K = 16 if Self.operand_t.is_half_float() else 32
    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if Self.operand_t.is_float8() else UMMAKind.KIND_F16
    )
    comptime smem_ptr_t = UnsafePointer[
        Scalar[Self.operand_t],
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    comptime num_m_mmas = Self.BM // Self.MMA_M
    comptime num_n_mmas = Self.BN // Self.MMA_N
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime c_frag_size = Self.MMA_M * Self.MMA_N // Self.num_softmax_threads
    comptime a_frag_size = Self.MMA_M * Self.MMA_K // Self.num_softmax_threads
    comptime num_m_blocks_per_warp = 2 * Self.BM // Self.num_softmax_threads
    comptime ab_t: DescriptorPairTS = UMMADescriptorTS[
        Self.operand_t,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        MMA_M=Self.BM // Self.num_m_blocks_per_warp,
        MMA_N=Self.BK,
        MMA_K=Self.MMA_K,
        consumer_group_size=Self.num_softmax_threads,
    ]
    comptime a_t: WriteableMMAOperandDescriptor = Self.ab_t.a_t
    comptime b_t: MMAOperandDescriptor = Self.ab_t.b_t

    comptime b_offset = MMAOperandOffsetFn[
        Self.operand_t,
        Self.BN,
        Self.BK,
        Self.swizzle_b,
        Self.transpose_b,
        Self.MMA_N,
        Self.MMA_K,
    ]()
    comptime c_t: AccumulatorTile = TMemAccumulator[
        Self.accum_t,
        Self.BM // Self.num_m_blocks_per_warp,
        Self.MMA_N,
        Self.num_m_blocks_per_warp,
        Self.num_n_mmas,
        Self.num_softmax_threads,
    ]

    comptime idesc = UMMAInsDescriptor[Self.mma_kind].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=Self.transpose_b,
    ]()

    var mbar: UnsafePointer[
        SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
    ]
    var phase: UInt32

    @staticmethod
    @always_inline
    def check_constraints():
        comptime assert (Self.BM % Self.MMA_M) == 0, (
            "BM, MMA_M = " + String(Self.BM) + ", " + String(Self.MMA_M)
        )
        comptime assert ((Self.BN % Self.MMA_N) == 0) and (
            Self.num_n_mmas > 0
        ), ("BN, MMA_N = " + String(Self.BN) + ", " + String(Self.MMA_N))
        comptime assert ((Self.BK % Self.MMA_K) == 0) and (
            Self.num_k_mmas > 0
        ), ("BK, MMA_K = " + String(Self.BK) + ", " + String(Self.MMA_K))

    @always_inline
    def __init__(
        out self,
        smem: UnsafePointer[
            SharedMemBarrier, MutAnyOrigin, address_space=AddressSpace.SHARED
        ],
    ):
        Self.check_constraints()
        self.mbar = smem
        self.phase = 0

    @always_inline
    def init(self):
        self.mbar[0].init()
        self.mbar[1].init(Int32(Self.num_softmax_threads))

    @staticmethod
    @always_inline
    def a_mma_descriptor(a_tmem: UInt32) -> Self.ab_t.a_t:
        Self.check_constraints()
        return Self.ab_t.a_t(a_tmem)

    @staticmethod
    @always_inline
    def b_mma_descriptor[
        dtype_b: DType
    ](
        p_b: UnsafePointer[
            Scalar[dtype_b], MutAnyOrigin, address_space=AddressSpace.SHARED
        ],
    ) -> Self.ab_t.b_t:
        Self.check_constraints()
        comptime b_canonical_layout = Self.b_offset.canonical_layout
        comptime b_type = Self.operand_t
        comptime b_stride01 = b_canonical_layout[0].stride[1].value()
        comptime b_stride11 = b_canonical_layout[1].stride[1].value()
        comptime bSBO = (
            b_stride01 if Self.transpose_b else b_stride11
        ) * size_of[b_type]()
        comptime bLBO = (
            b_stride11 if Self.transpose_b else b_stride01
        ) * size_of[b_type]()

        return MMASmemDescriptor.create[bSBO, bLBO, Self.swizzle_b](p_b)

    @always_inline
    def mma(
        self,
        a: Self.a_t,
        b: Self.b_t,
        c: Self.c_t,
        c_scale: UInt32,
    ):
        comptime for k_mma in range(Self.num_k_mmas):
            comptime for m_mma in range(Self.num_m_mmas):
                a_tmem = a.offset[m_mma=m_mma, k_mma=k_mma]()

                comptime for n_mma in range(Self.num_n_mmas):
                    c_tmem = c.offset[m_mma=m_mma, n_mma=n_mma]()
                    comptime b_offset = Self.b_offset.layout(
                        IntTuple(Self.MMA_N * n_mma, Self.MMA_K * k_mma)
                    ) * size_of[Self.operand_t]()
                    b_desc = b + b_offset

                    comptime if k_mma == 0:
                        mma[Self.cta_group](
                            a_tmem,
                            b_desc,
                            c_tmem,
                            Self.idesc,
                            c_scale,
                        )
                    else:
                        mma[Self.cta_group, c_scale=1](
                            a_tmem, b_desc, c_tmem, Self.idesc
                        )
        mma_arrive(self.mbar)

    # the mma thread
    # loop:
    #   wait_for_tmem()   # self.mbar[1].wait(self.phase), self.phase ^= 1
    #   mma()             # self.mbar[0].arrive()
    #
    # the softmax thread
    # tmem_arrive()       # self.mbar[1].arrive()
    #
    # loop:
    #   wait_for_mma()    # self.mbar[0].wait(self.phase), self.phase ^= 1
    #   scale output, write P
    #   tmem_arrive()     # self.mbar[1].arrive()
    @always_inline
    def wait(mut self, idx: UInt32):
        # update the phase before waiting
        var old_phase: UInt32 = self.phase
        self.phase = old_phase ^ 1
        self.mbar[idx].wait(old_phase)

    @always_inline
    def wait_for_mma(mut self):
        """
        Wait for the mma to be complete.
        """
        self.wait(0)

    @always_inline
    def wait_for_tmem(mut self):
        """
        Wait for the `output` and `A` tmem to be ready.
        """
        self.wait(1)

    @always_inline
    def tmem_arrive(self):
        """
        Indicate that the accumulator and the tensor memory arguments
        are ready for the MMA to begin.
        """
        _ = self.mbar[1].arrive()
