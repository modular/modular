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
from math import align_up, ceildiv
from memory import UnsafePointer
from sys import (
    alignof,
    env_get_int,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_nvidia_gpu,
    simdwidthof,
    sizeof,
)
from sys.info import _accelerator_arch

from buffer import NDBuffer
from gpu import WARP_SIZE, lane_id
from gpu.memory import AddressSpace
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor, LayoutTensorIter
from layout.swizzle import make_ldmatrix_swizzle
from nn.mha_mask import (
    CausalMask,
    ChunkedCausalMask,
    ChunkedMask,
    MaskName,
    MaterializedMask,
    MHAMask,
    NullMask,
    SlidingWindowCausalMask,
)
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod, ScoreModTrait

from utils.index import Index, IndexList
from utils.numerics import min_or_neg_inf

# ===-----------------------------------------------------------------------===#
# Multi-Head Attention
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct FlashAttentionAlgorithm(
    Copyable, Defaultable, Movable, Stringable, Writable
):
    var _value: Int32

    alias NAIVE = Self(0)
    alias FLASH_ATTENTION_1 = Self(1)
    alias FLASH_ATTENTION_2 = Self(2)
    alias FLASH_ATTENTION_3 = Self(3)

    fn __init__(out self):
        self._value = 3

    @implicit
    fn __init__(out self, value: Int):
        self._value = Int32(value)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @always_inline
    fn __str__(self) -> String:
        return String.write(self)

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self._value == 0:
            writer.write("naive-attention")
        elif self._value == 1:
            writer.write("flash-attention-1")
        elif self._value == 2:
            writer.write("flash-attention-2")
        elif self._value == 3:
            writer.write("flash-attention-3")
        else:
            writer.write("invalid algorithm")


alias is_sm90 = ":90" in _accelerator_arch()
alias is_sm100 = ":100" in _accelerator_arch()
alias is_sm90or100 = is_sm90 or is_sm100


@fieldwise_init
@register_passable("trivial")
struct MHAConfig(Copyable, Movable, Writable):
    var type: DType

    # Q, K, V, output should have the same type.
    var num_heads: UInt
    var depth: UInt
    var num_queries_per_block: UInt
    var num_keys_per_block: UInt
    var BK: UInt  # tile size in depth dimension
    var WM: UInt
    var WN: UInt
    var num_pipeline_stages: UInt
    var k_group_size: UInt
    var algorithm: FlashAttentionAlgorithm

    fn block_m(self) -> UInt:
        return self.num_queries_per_block

    fn block_n(self) -> UInt:
        return self.num_keys_per_block

    fn block_k(self) -> UInt:
        return self.BK

    fn warp_m(self) -> UInt:
        return self.WM

    fn warp_n(self) -> UInt:
        return self.WN

    fn num_warps_m(self) -> UInt:
        return self.block_m() // self.warp_m()

    fn num_warps_n(self) -> UInt:
        return self.block_n() // self.warp_n()

    fn num_consumer_threads(self) -> UInt:
        return self.num_warps_m() * self.num_warps_n() * WARP_SIZE

    fn num_producer_threads[
        producer_consumer_kernel: Bool = False
    ](self) -> UInt:
        return 128 if (producer_consumer_kernel and self.algorithm == 3) else 0

    fn num_threads[producer_consumer_kernel: Bool = False](self) -> UInt:
        return (
            self.num_consumer_threads()
            + self.num_producer_threads[producer_consumer_kernel]()
        )

    fn q_smem_size(self, fa3: Bool = False) -> UInt:
        q_smem = self.block_m() * self.depth
        return UInt(2) * q_smem if fa3 else q_smem

    fn kv_smem_size(self, fa3: Bool = False) -> UInt:
        kv_smem = self.block_n() * self.depth
        return kv_smem * self.num_pipeline_stages if fa3 else kv_smem

    fn k_smem_size(self, sm_90: Bool = False) -> UInt:
        k_smem = self.block_n() * self.depth
        return k_smem * self.num_pipeline_stages if sm_90 else k_smem

    fn v_smem_size(self, sm_90: Bool = False) -> UInt:
        BN = self.block_n()
        kv_smem = BN * BN
        return kv_smem * self.num_pipeline_stages if sm_90 else kv_smem

    fn p_smem_size(self) -> UInt:
        return self.block_m() * self.block_n()

    fn warp_scratch_smem_size(self) -> UInt:
        n_warps_n = self.num_warps_n()
        return 2 * n_warps_n * self.block_m() if n_warps_n > 1 else 0

    fn shared_mem_bytes[
        shared_kv: Bool = False, sm_90: Bool = False
    ](self) -> UInt:
        if not has_nvidia_gpu_accelerator():
            return 0

        sm_90_fa3 = sm_90 and (self.algorithm == 3)

        @parameter
        if shared_kv:
            num_smem_elements = (
                self.q_smem_size()
                + self.kv_smem_size(sm_90_fa3)
                + self.warp_scratch_smem_size()
            )
        else:
            num_smem_elements = (
                self.q_smem_size(sm_90_fa3)
                + self.k_smem_size(sm_90_fa3)
                + self.v_smem_size(sm_90_fa3)
                + self.warp_scratch_smem_size()
            )

        if self.num_warps_n() > 1 or has_amd_gpu_accelerator():
            num_smem_elements += self.p_smem_size()

        num_smem_bytes = self.type.sizeof() * num_smem_elements
        if sm_90_fa3:
            alias persistent = env_get_int["USE_EXPERIMENTAL_KERNELS", 0]()
            num_smem_bytes += (4 * self.num_pipeline_stages + 4) * sizeof[
                DType.int64
            ]() + (2 * sizeof[DType.uint32]() if persistent != 0 else 0)
        return num_smem_bytes

    fn __init__(
        out self,
        type: DType,
        num_heads: UInt,
        depth: UInt,
        num_queries_per_block: OptionalReg[UInt] = None,
        num_keys_per_block: OptionalReg[UInt] = None,
        BK: OptionalReg[UInt] = None,
        WM: OptionalReg[UInt] = None,
        WN: OptionalReg[UInt] = None,
        num_pipeline_stages: UInt = 2 if is_sm90or100 else 4,
        k_group_size: UInt = 1,
        algorithm: FlashAttentionAlgorithm = FlashAttentionAlgorithm(),
    ):
        self.type = type
        self.num_heads = num_heads
        self.depth = depth
        self.num_pipeline_stages = num_pipeline_stages
        self.k_group_size = k_group_size
        # Not all of these have to be `OptionalReg`, only
        # those that depend on `depth`.
        # Currently, all are `OptionalReg` for consistency.
        # BN
        self.num_keys_per_block = num_keys_per_block.or_else(depth)

        if is_sm90or100 and type.is_half_float():
            # BM
            self.num_queries_per_block = num_queries_per_block.or_else(
                128 if algorithm == 3 else 64
            )
            self.BK = BK.or_else(64)
        else:
            # BM
            self.num_queries_per_block = num_queries_per_block.or_else(
                32 if type
                is DType.float32 else (128 if has_amd_gpu_accelerator() else 64)
            )
            var bk_arch_factor = 2 if num_pipeline_stages <= 2 else 1
            var bk_type_factor = 1 if type is DType.float32 else 2
            self.BK = BK.or_else(
                16 * bk_arch_factor * bk_type_factor
            ) if has_nvidia_gpu_accelerator() else 32
        self.WM = WM.or_else(
            32 if type
            is DType.float32 else (32 if has_amd_gpu_accelerator() else 16)
        )
        self.WN = WN.or_else(32 if type is DType.float32 else depth)
        self.algorithm = algorithm

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("ampere_")
        writer.write(self.type, "_")
        # Use BNxBM to match MatmulConfig, which matches cublas
        writer.write(self.block_n(), "x", self.block_m(), "_")
        writer.write(self.block_k(), "x")
        writer.write(self.num_pipeline_stages)


@always_inline
fn _kernel_mask[
    type: DType, width: Int
](
    coord: IndexList[2, **_], bound: IndexList[2, **_], vec: SIMD[type, width]
) -> SIMD[type, width]:
    var masked_vec = SIMD[type, width]()

    # TODO: use `select` to see if it generates the same code.
    @parameter
    for i in range(width):
        masked_vec[i] = (
            vec[i] if coord[0] < bound[0]
            and coord[1] + UInt32(i) < bound[1] else min_or_neg_inf[type]()
        )

    return masked_vec


@always_inline
fn _copy_frag_to_smem_nvidia[
    BM: UInt,
    BN: UInt,
    BK: UInt,
    WM: UInt,
    WN: UInt,
    MMA_M: UInt,
    MMA_N: UInt,
    frag_simd_width: UInt,
    *,
    type0: DType,
    layout0: Layout,
    type1: DType,
    layout1: Layout,
](
    p_smem_iter: LayoutTensorIter[
        type0, layout0, address_space = AddressSpace.SHARED, **_
    ],
    p_reg_tile: LayoutTensor[
        type1, layout1, address_space = AddressSpace.LOCAL
    ],
    warp_x: UInt32,
    warp_y: UInt32,
):
    """Copy p fragments to shared memory.

    Logically P has shape BM x BN. It's sharded across threads in 16x8 mma layout.
    The BM x BN matrix is divided to BM x BK tiles, each tile is CONTIGUOUS for
    the 2nd mma. This function maps each fragment to the right BM x BK tile and
    swizzle to avoid bank conflict.
    """

    alias simd_width = simdwidthof[p_smem_tile.dtype]()
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    # This tile is used for offset computation because 1st mma output is organized
    # for BM x BN output tile. The layout for 2nd mma is in p_smem_iter.
    var p_smem_tile = LayoutTensor[
        p_smem_iter.type,
        Layout.row_major(BM, BN),
        address_space = AddressSpace.SHARED,
    ](p_smem_iter.ptr)
    var p_smem_warp_tile = p_smem_tile.tile[WM, WN](Int(warp_y), Int(warp_x))
    var p_reg_vecs = p_reg_tile.vectorize[1, frag_simd_width]()

    alias swizzle_fn = make_ldmatrix_swizzle[p_smem_tile.dtype, BK]()

    @parameter
    for n_mma in range(Int(num_n_mmas)):

        @parameter
        for m_mma in range(Int(num_m_mmas)):
            var p_smem_mma_tile = p_smem_warp_tile.tile[MMA_M, MMA_N](
                m_mma, n_mma
            ).vectorize[1, frag_simd_width]()
            var p_smem_frag = p_smem_mma_tile.distribute[
                Layout.row_major(8, 4)
            ](lane_id())
            var frag_offset = p_smem_frag.distance(p_smem_tile)

            @parameter
            for i in range(p_reg_vecs.shape[1]()):
                alias offset_in_frag = __type_of(p_smem_frag).layout(i)

                # Translate offset in BM x BN matrix to the right BM x BK tile.
                var offset_BMxBN = frag_offset + offset_in_frag
                var offset_BMxBK = (offset_BMxBN // BN) * BK + offset_BMxBN % BK
                # Convert offset to vectorized domain, since BM x BK will be loaded
                # by vectors in 2nd mma, and swizzle
                var swizzle_offset = swizzle_fn(offset_BMxBK // simd_width)
                # Convert offset back to where the frag will be stored.
                offset_BMxBK = (
                    swizzle_offset * simd_width + offset_BMxBK % simd_width
                )
                # E.g. fp32x2 -> bf16x2 for bf16 mma.
                var vec = p_reg_vecs[n_mma * num_m_mmas + m_mma, i].cast[
                    p_smem_tile.dtype
                ]()
                # Grep the right BMxBK tile and store the casted vec.
                var tile_BMxBK = p_smem_iter.next_unsafe(
                    Int((offset_BMxBN % BN) // BK)
                )[]
                alias align = alignof[SIMD[p_smem_iter.type, frag_simd_width]]()
                tile_BMxBK.ptr.store[alignment=align](offset_BMxBK, vec)


@always_inline
fn _copy_frag_to_smem_amd[
    BM: UInt,
    BN: UInt,
    BK: UInt,
    WM: UInt,
    WN: UInt,
    MMA_M: UInt,
    MMA_N: UInt,
    frag_simd_width: UInt,
    *,
    type0: DType,
    layout0: Layout,
    type1: DType,
    layout1: Layout,
](
    p_smem_iter: LayoutTensorIter[
        type0, layout0, address_space = AddressSpace.SHARED, **_
    ],
    p_reg_tile: LayoutTensor[
        type1, layout1, address_space = AddressSpace.LOCAL
    ],
    warp_x: UInt32,
    warp_y: UInt32,
):
    """Copy p fragments to shared memory.
    Logically P has shape BM x BN. It's sharded across threads in 16x16 mma layout.
    The BM x BN matrix is divided to BM x BK tiles, each tile is CONTIGUOUS for
    the 2nd mma. This function maps each fragment to the right BM x BK tile.
    """
    alias simd_width = 1
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    # This tile is used for offset computation because 1st mma output is organized
    # for BM x BN output tile. The layout for 2nd mma is in p_smem_iter.
    var p_smem_tile = LayoutTensor[
        p_smem_iter.type,
        Layout.row_major(BM, BN),
        address_space = AddressSpace.SHARED,
    ](p_smem_iter.ptr)

    var p_smem_warp_tile = p_smem_tile.tile[WM, WN](Int(warp_y), Int(warp_x))
    var p_reg_vecs = p_reg_tile.vectorize[1, frag_simd_width]()

    @parameter
    for n_mma in range(Int(num_n_mmas)):

        @parameter
        for m_mma in range(Int(num_m_mmas)):
            var p_smem_mma_tile = p_smem_warp_tile.tile[MMA_M, MMA_N](
                m_mma, n_mma
            ).vectorize[frag_simd_width, 1]()
            var p_smem_frag = p_smem_mma_tile.distribute[
                Layout.row_major(4, 16)
            ](lane_id())
            var frag_offset = p_smem_frag.distance(p_smem_tile)

            @parameter
            for i in range(Int(frag_simd_width)):
                alias offset_in_frag = BN * i
                # Translate offset in BM x BN matrix to the right BM x BK tile.
                var offset_BMxBN = frag_offset + offset_in_frag
                var offset_BMxBK = (offset_BMxBN // BN) * BK + offset_BMxBN % BK

                var vec = p_reg_vecs[n_mma * num_m_mmas + m_mma, 0][i].cast[
                    p_smem_tile.dtype
                ]()
                # Grep the right BMxBK tile and store the casted vec.
                var tile_BMxBK = p_smem_iter.next_unsafe(
                    Int((offset_BMxBN % BN) // BK)
                )[]
                tile_BMxBK.ptr.store(offset_BMxBK, vec)


@always_inline
fn _copy_frag_to_smem[
    BM: UInt,
    BN: UInt,
    BK: UInt,
    WM: UInt,
    WN: UInt,
    MMA_M: UInt,
    MMA_N: UInt,
    frag_simd_width: UInt,
    *,
    type0: DType,
    layout0: Layout,
    type1: DType,
    layout1: Layout,
](
    p_smem_iter: LayoutTensorIter[
        type0, layout0, address_space = AddressSpace.SHARED, **_
    ],
    p_reg_tile: LayoutTensor[
        type1, layout1, address_space = AddressSpace.LOCAL
    ],
    warp_x: UInt32,
    warp_y: UInt32,
):
    @parameter
    if is_nvidia_gpu():
        _copy_frag_to_smem_nvidia[
            BM, BN, BK, WM, WN, MMA_M, MMA_N, frag_simd_width
        ](p_smem_iter, p_reg_tile, warp_x, warp_y)
    else:
        _copy_frag_to_smem_amd[
            BM, BN, BK, WM, WN, MMA_M, MMA_N, frag_simd_width
        ](p_smem_iter, p_reg_tile, warp_x, warp_y)


@always_inline
fn get_start_and_end_for_partitions[
    tile_size: Int
](num_keys: Int, num_partitions: Int, partition_idx: Int) -> Tuple[Int, Int]:
    """Calculate start and end indices for a partition.

    Args:
        num_keys: Total number of keys (sequence length).
        num_partitions: Number of partitions to split keys into.
        partition_idx: Index of current partition (0 to num_partitions-1).

    Returns:
        Tuple of (start_idx, end_idx) for the partition, aligned to tile_size.
    """
    var num_keys_per_partition = ceildiv(num_keys, num_partitions)

    # Align start to tile_size
    var start = align_up(num_keys_per_partition * partition_idx, tile_size)
    # If start is already beyond num_keys, return empty range
    if start >= num_keys:
        return (num_keys, num_keys)
    var next_start = align_up(
        num_keys_per_partition * (partition_idx + 1), tile_size
    )
    var end = min(num_keys, next_start)
    return (start, end)

    # ^ may lead to non-uniform distribution of keys across partitions because of alignment requirement,
    # we may want to use the following instead for non-paged kvcache but then we will have to know which cache is being used.
    # Keep this here for now, can remove it later if we are only using paged kvcache.
    # var start = num_keys_per_partition * partition_idx
    # var end = min(num_keys, start + num_keys_per_partition)
    # return (start, end)


alias callback_fn_type = fn[mask_t: MHAMask, score_mod_t: ScoreModTrait] (
    mask: mask_t, score_mod: score_mod_t
) raises capturing -> None


@always_inline
fn dispatch_mask_and_score_mod[
    mask_type: String,
    score_mod_type: String,
    callback_fn: callback_fn_type,
    local_window_size: Int = -1,
    num_heads: Int = -1,
]() raises -> None:
    @always_inline
    @parameter
    fn outer_wrapper[mask_t: MHAMask](mask: mask_t) raises:
        @always_inline
        @parameter
        fn wrapper[score_mod_t: ScoreModTrait](score_mod: score_mod_t) raises:
            return callback_fn(mask, score_mod)

        return _dispatch_score_mod[score_mod_type, wrapper, num_heads]()

    # TODO: attach string constants to mask types themselves.
    @parameter
    if MaskName.CAUSAL == mask_type:
        return outer_wrapper(CausalMask())
    elif MaskName.CHUNKED == mask_type:
        constrained[
            local_window_size > 0,
            "You must specify local_window_size for ChunkedMask",
        ]()
        return outer_wrapper(ChunkedMask[local_window_size]())
    elif MaskName.NULL == mask_type:
        return outer_wrapper(NullMask())
    elif MaskName.SLIDING_WINDOW_CAUSAL == mask_type:
        constrained[
            local_window_size > 0,
            "You must specify local_window_size for SlidingWindowCausalMask",
        ]()
        return outer_wrapper(SlidingWindowCausalMask[local_window_size]())
    elif MaskName.CHUNKED_CAUSAL == mask_type:
        constrained[
            local_window_size > 0,
            "You must specify local_window_size for ChunkedCausalMask",
        ]()
        return outer_wrapper(ChunkedCausalMask[local_window_size]())
    else:
        constrained[False, "Unsupported mask type: " + mask_type]()


@always_inline
fn dispatch_materialized_mask_and_score_mod[
    score_mod_type: String, callback_fn: callback_fn_type, num_heads: Int = -1
](
    mask_nd: NDBuffer,
    start_pos_nd: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ] = None,
) raises -> None:
    var mask = MaterializedMask(mask_nd, start_pos_nd)

    @always_inline
    @__copy_capture(mask)
    @parameter
    fn wrapper[score_mod_t: ScoreModTrait](score_mod: score_mod_t) raises:
        return callback_fn(mask, score_mod)

    return _dispatch_score_mod[score_mod_type, wrapper, num_heads]()


@always_inline
fn _dispatch_score_mod[
    score_mod_type: String,
    callback_fn: fn[score_mod_t: ScoreModTrait] (
        score_mod: score_mod_t
    ) raises capturing -> None,
    num_heads: Int = -1,
]() raises -> None:
    @always_inline
    @parameter
    fn wrapper[score_mod_t: ScoreModTrait](score_mod: score_mod_t) raises:
        return callback_fn(score_mod)

    @parameter
    if score_mod_type == AlibiScoreMod.name_str:
        constrained[
            num_heads > 0, "You must specify num_heads for AlibiScoreMod"
        ]()
        return wrapper(AlibiScoreMod[num_heads]())
    elif score_mod_type == IdentityScoreMod.name_str:
        return wrapper(IdentityScoreMod())
    else:
        constrained[False, "Unsupported score mod type: " + score_mod_type]()


# The motivation here is to be able to pass `StaticInt[1]()`
# to indicate `decoding=True`, and have this not generate any code
# when passing as a function argument.
# That is, we want different specializations of a function to have
# different numbers of arguments post-compilation.
trait OptionallyStaticInt(Intable):
    alias static_value: OptionalReg[Int]

    fn as_uint32(self) -> UInt32:
        ...


# These are used to avoid generating code for passing unused values to kernels.
# That is, if we have a static int, no argument should be passed.
@register_passable("trivial")
struct StaticInt[value: Int](Defaultable, OptionallyStaticInt):
    alias static_value: OptionalReg[Int] = OptionalReg[Int](value)

    @always_inline("nodebug")
    fn __init__(out self):
        pass

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        return Self.value

    @always_inline("nodebug")
    fn as_uint32(self) -> UInt32:
        return UInt32(Self.value)


@register_passable("trivial")
struct DynamicInt(OptionallyStaticInt):
    var value: UInt32
    alias static_value: OptionalReg[Int] = None

    @always_inline("nodebug")
    fn __init__(out self, value: Int):
        self.value = UInt32(value)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        return Int(self.value)

    @always_inline("nodebug")
    fn as_uint32(self) -> UInt32:
        return self.value


@always_inline
fn _is_decoding[int_t: OptionallyStaticInt]() -> Bool:
    return int_t.static_value.or_else(0) == 1


@register_passable("trivial")
trait MHAPartitionScheme:
    alias do_partition: Bool
    alias accum_dtype: DType

    @always_inline
    fn num_partitions(self) -> UInt32:
        ...

    @always_inline
    fn get_exp_sum_qk_max_pointer(
        self,
    ) -> UnsafePointer[Scalar[Self.accum_dtype]]:
        ...


@register_passable("trivial")
struct NoPartition[dtype: DType](
    Copyable, Defaultable, MHAPartitionScheme, Movable
):
    alias do_partition: Bool = False
    alias accum_dtype: DType = dtype

    @always_inline
    fn __init__(out self):
        pass

    @always_inline
    fn num_partitions(self) -> UInt32:
        return 1

    @always_inline
    fn get_exp_sum_qk_max_pointer(
        self,
    ) -> UnsafePointer[Scalar[Self.accum_dtype]]:
        return UnsafePointer[Scalar[Self.accum_dtype]]()


@register_passable("trivial")
struct SplitKPartition[dtype: DType](Copyable, MHAPartitionScheme, Movable):
    alias do_partition: Bool = True
    alias accum_dtype: DType = Self.dtype
    var ptr: UnsafePointer[Scalar[Self.accum_dtype]]
    var num_partitions_value: UInt32

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.accum_dtype]],
        num_partitions_value: UInt32,
    ):
        debug_assert(ptr != UnsafePointer[Scalar[Self.accum_dtype]]())
        self.ptr = ptr
        self.num_partitions_value = num_partitions_value

    @always_inline
    fn num_partitions(self) -> UInt32:
        return self.num_partitions_value

    @always_inline
    fn get_exp_sum_qk_max_pointer(
        self,
    ) -> UnsafePointer[Scalar[Self.accum_dtype]]:
        return self.ptr
