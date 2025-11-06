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

from hashlib.hasher import Hasher

from bit import next_power_of_two
from collections.set import Set
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from itertools.itertools import product
from layout.tensor_core import get_mma_shape
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.math import align_down
from ..tile_scheduler import RasterOrder


@fieldwise_init
@register_passable("trivial")
struct MatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](Copyable, EqualityComparable, Hashable, Movable, Stringable, Writable):
    """Static configuration of GPU matmul."""

    # Mandatory parameters
    var cta_group: Int
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var AB_swapped: Bool
    var block_swizzle_size: Int
    var raster_order: RasterOrder

    alias accum_type = get_accum_type[a_type]()  # TODO: factor b_type

    # Has default values or derivible from mandatory parameters
    var block_tile_shape: IndexList[3]
    var num_split_k: Int
    var num_pipeline_stages: UInt
    var num_clc_pipeline_stages: UInt
    var num_accum_pipeline_stages: UInt
    var num_output_stages: UInt
    var output_tile_shape: IndexList[2]
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle
    var c_swizzle: TensorMapSwizzle

    var k_group_size: UInt

    fn __init__(
        out self,
        *,
        cta_group: Int = 2,
        mma_shape: IndexList[3] = get_mma_shape[a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(2, 1, 1),
        AB_swapped: Bool = False,
        num_split_k: Int = 1,
        block_swizzle_size: Int = 0,
        raster_order: RasterOrder = RasterOrder.AlongM,
        k_group_size: UInt = 1,
        num_pipeline_stages: Optional[UInt] = None,
        num_accum_pipeline_stages: UInt = 2,
        num_clc_pipeline_stages: UInt = 2,
    ):
        constrained[a_type == b_type]()

        self.cta_group = cta_group
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.AB_swapped = AB_swapped
        self.block_swizzle_size = block_swizzle_size
        self.raster_order = raster_order
        self.k_group_size = k_group_size
        self.block_tile_shape = Index(
            self.mma_shape[0] // self.cta_group,
            self.mma_shape[1] // self.cta_group,
            128 // a_type.size_of(),
        )

        # If MMA_M is 256, each of the pair ctas has the entire MMA_N
        # If MMA_M is 128, each of the pair ctas has 1/2 of MMA_N
        # If cta_group=1, the cta has the entire MMA_N
        var c_tile_n = (
            self.mma_shape[1] if (
                self.mma_shape[0] == 256 or self.cta_group == 1
            ) else self.mma_shape[1]
            // 2
        )
        var output_tile_n = 8
        if c_tile_n % 32 == 0:
            output_tile_n = 32
        elif c_tile_n % 16 == 0:
            output_tile_n = 16

        # MMA_M=128/256 cta_group=2 all use 128 rows in output tile.
        var output_tile_m = 128 if self.cta_group == 2 else self.mma_shape[0]
        self.output_tile_shape = Index(
            output_tile_n, output_tile_m
        ) if self.AB_swapped else Index(output_tile_m, output_tile_n)

        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_output_stages = 2
        self.num_split_k = num_split_k

        self.a_swizzle = TensorMapSwizzle.SWIZZLE_128B
        self.b_swizzle = TensorMapSwizzle.SWIZZLE_128B
        self.c_swizzle = TensorMapSwizzle.SWIZZLE_NONE
        if self.AB_swapped:
            self.c_swizzle = TensorMapSwizzle.SWIZZLE_128B
        elif output_tile_n == 32:
            self.c_swizzle = TensorMapSwizzle.SWIZZLE_64B
        elif output_tile_n == 16:
            self.c_swizzle = TensorMapSwizzle.SWIZZLE_32B

        self.num_pipeline_stages = 4  # Need for compilation
        self._maximize_pipeline_stages_by_default()

        if num_pipeline_stages:
            self.num_pipeline_stages = num_pipeline_stages.value()

        # SM100 kernel only supports k grouping when num_pipeline_stages is a multiple of k_group_size.
        self.num_pipeline_stages = align_down(
            self.num_pipeline_stages, self.k_group_size
        )

    fn _maximize_pipeline_stages_by_default(mut self):
        alias b200_smem = B200.shared_memory_per_multiprocessor - 1024

        var c_smem_bytes = (
            self.output_tile_shape[0]
            * self.output_tile_shape[1]
            * Int(self.num_output_stages)
            * size_of[c_type]()
        )
        # Add tmem addr (4) and tmem dealloc mbar(8)
        var output_smem_bytes = c_smem_bytes + 12

        # response 128B, clc mbar 16B, clc-load pipeline mbar 16B
        var clc_smem_bytes = 160 * Int(self.num_clc_pipeline_stages)

        # Usage by mma-output-pipeline
        var mma_output_smem_bytes = self.num_accum_pipeline_stages * 16

        var a_smem_bytes_per_stage = (
            self.block_tile_shape[0]
            * self.block_tile_shape[2]
            * size_of[a_type]()
        )
        var b_smem_bytes_per_stage = (
            self.block_tile_shape[1]
            * self.block_tile_shape[2]
            * size_of[b_type]()
        )
        # Include 16 for consumer and producer mbar per stage
        var AB_smem_per_stage = (
            a_smem_bytes_per_stage + b_smem_bytes_per_stage + 16
        )

        # Subtract 511B for mbar usage etc
        self.num_pipeline_stages = UInt(
            (
                b200_smem
                - output_smem_bytes
                - clc_smem_bytes
                - Int(mma_output_smem_bytes)
            )
            // AB_smem_per_stage
        )

    fn __eq__(
        self, other: MatmulConfig[a_type, b_type, c_type, transpose_b]
    ) -> Bool:
        return (
            self.cta_group == other.cta_group
            and self.mma_shape == other.mma_shape
            and self.cluster_shape == other.cluster_shape
            and self.AB_swapped == other.AB_swapped
            and self.num_pipeline_stages == other.num_pipeline_stages
            and self.num_clc_pipeline_stages == other.num_clc_pipeline_stages
            and self.num_accum_pipeline_stages
            == other.num_accum_pipeline_stages
            and self.num_output_stages == other.num_output_stages
            and self.output_tile_shape == other.output_tile_shape
            and self.a_swizzle == other.a_swizzle
            and self.b_swizzle == other.b_swizzle
            and self.c_swizzle == other.c_swizzle
            and self.block_swizzle_size == other.block_swizzle_size
            and self.raster_order == other.raster_order
            and self.k_group_size == other.k_group_size
            and self.num_split_k == other.num_split_k
        )

    fn swap_AB_type(self) -> MatmulConfig[b_type, a_type, c_type, transpose_b]:
        return MatmulConfig[b_type, a_type, c_type, transpose_b](
            cta_group=self.cta_group,
            mma_shape=self.mma_shape,
            cluster_shape=self.cluster_shape,
            AB_swapped=self.AB_swapped,
            num_pipeline_stages=self.num_pipeline_stages,
            num_accum_pipeline_stages=self.num_accum_pipeline_stages,
            num_clc_pipeline_stages=self.num_clc_pipeline_stages,
            block_swizzle_size=self.block_swizzle_size,
            raster_order=self.raster_order,
            k_group_size=self.k_group_size,
            num_split_k=self.num_split_k,
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("kernel_")
        writer.write(a_type, "_")
        writer.write(c_type, "_")
        writer.write("cta", self.cta_group, "_")
        writer.write(
            "mma",
            self.mma_shape[0],
            "x",
            self.mma_shape[1],
            "x",
            self.mma_shape[2],
            "_",
        )
        writer.write(
            "cluster",
            self.cluster_shape[0],
            "x",
            self.cluster_shape[1],
            "x",
            self.cluster_shape[2],
            "_",
        )
        writer.write("stages", self.num_pipeline_stages, "_")
        writer.write("k_group", self.k_group_size, "_")
        writer.write("clc", self.num_clc_pipeline_stages, "_")
        writer.write("accum", self.num_accum_pipeline_stages, "_")
        writer.write("out", self.num_output_stages, "_")
        writer.write(
            self.output_tile_shape[0], "x", self.output_tile_shape[1], "_"
        )
        writer.write("swap" if self.AB_swapped else "noswap", "_")
        writer.write("K_")
        writer.write("K_" if transpose_b else "MN_")
        writer.write("asz", self.a_swizzle.bytes(), "_")
        writer.write("bsz", self.b_swizzle.bytes(), "_")
        writer.write("csz", self.c_swizzle.bytes(), "_")
        writer.write("bz", self.block_swizzle_size, "_", self.raster_order)
        writer.write("splitk", self.num_split_k, "_")

    fn __repr__(self) -> String:
        return String.write(self)

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher.update(a_type)
        hasher.update(b_type)
        hasher.update(c_type)
        hasher.update(transpose_b)
        hasher.update(self.cta_group)
        hasher.update(self.mma_shape)
        hasher.update(self.block_tile_shape)
        hasher.update(self.cluster_shape)
        hasher.update(self.AB_swapped)
        hasher.update(self.num_pipeline_stages)
        hasher.update(self.num_clc_pipeline_stages)
        hasher.update(self.num_accum_pipeline_stages)
        hasher.update(self.num_output_stages)
        hasher.update(self.output_tile_shape)
        hasher.update(Int(self.a_swizzle))
        hasher.update(Int(self.b_swizzle))
        hasher.update(Int(self.c_swizzle))
        hasher.update(self.raster_order)
        hasher.update(self.k_group_size)
        hasher.update(self.num_split_k)


fn choose_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](M: Int, N: Int, K: Int) -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    constrained[a_type == b_type, "a_type and b_type must be the same"]()

    alias num_SMs = B200.sm_count
    # Nvidia mma instruction process 32B in K.
    alias Kbytes_per_mma = 32
    # We use 128B swizzle, tile size in K is 128B over element size.
    alias BK = 128 // a_type.size_of()

    alias M_pivote = 32

    var cta_group = 1 if M < M_pivote else 2
    var swapAB = True if M < M_pivote else False
    var k_group_size = 1  # maybe increased for small M later

    var mma_mn = Tuple[Int, Int](256, 256)
    var min_num_waves = Int.MAX

    # Travers possible combinations of BM x MMA_N to choose the one minimizes the
    # workload per SM. The computation per SM is the flops (ignoring 2x in 2MNK)
    # timed by max number of ctas per SM i.e. number of waves.
    # We first minimize the number of waves, then use the flops to break tie.

    # For small M, swap A and B so that the small M maps to mma_n since it supports
    # a larger range than mma_m.
    if M < M_pivote:
        for bm, mma_n in product([64, 128], range(8, align_up(M, 8) + 1, 8)):
            num_ctas = ceildiv(M, mma_n) * ceildiv(N, bm)
            num_waves = ceildiv(num_ctas, num_SMs)
            if num_waves < min_num_waves or (
                num_waves == min_num_waves
                and bm * mma_n < mma_mn[0] * mma_mn[1]
            ):
                min_num_waves = num_waves
                mma_mn[0] = bm
                mma_mn[1] = mma_n

    # For large M, use 2xSM mma
    else:

        @parameter
        @always_inline
        fn select_mma_mn(M: Int, N: Int, _swapAB: Bool = False):
            N_alignby16 = align_up(N, 16)
            max_mma_n = min(N_alignby16, 256)
            # In pratice 64x16 mma creates too many ctas and increase L2
            # load volume, ends up hurting performance.
            min_mma_n = min(N_alignby16, 32)
            for bm in [64, 128]:
                for mma_n in range(max_mma_n, min_mma_n - 1, -16):
                    var mma_m = bm * cta_group
                    var num_clusters = ceildiv(M, mma_m) * ceildiv(N, mma_n)
                    var num_waves = ceildiv(num_clusters, num_SMs // cta_group)
                    if num_waves > min_num_waves:
                        break
                    elif num_waves < min_num_waves or (
                        num_waves == min_num_waves
                        and mma_m * mma_n < mma_mn[0] * mma_mn[1]
                    ):
                        min_num_waves = num_waves
                        mma_mn[0] = mma_m
                        mma_mn[1] = mma_n
                        swapAB = _swapAB

        # Swap AB may work better for M = 192 and not-multiple-of-128 values.
        # Capture and update min_num_waves, mma_mn
        select_mma_mn(M, N)
        select_mma_mn(N, M, True)

    # For small mmas, we group multiple tiles per tma-mma synchronization.
    var output_block_size = (mma_mn[0] // cta_group) * mma_mn[1]
    if output_block_size <= 64 * 96 and ceildiv(K, BK) % 2 == 0:
        k_group_size = 2
    # For very small mmas we can group more aggressively.
    if output_block_size <= 64 * 16 and ceildiv(K, BK) % 4 == 0:
        k_group_size = 4

    var min_load_volume = Int.MAX
    var optimal_block_swizzle_size = 0

    # Tile waves when there are >= 4 waves. In theory it should be >=2, but let's
    # be conservative.
    if min_num_waves >= 4:
        # Represent the load volume by
        #    BM * num_ctas_per_wave_m + MMA_N * num_ctas_per_wave_N
        # Use MMA_N because cta_group = 2, 2 ctas cover entire MMA_N. cta_group = 1
        # has BN = MMA_N.
        # Traverse the tile sizes to find min load volume per wave.
        # TODO: consider the L2 resue across waves.
        var BM = mma_mn[0] // cta_group
        for tile_size in List[Int](1, 2, 4, 8):
            var num_ctas_m = ceildiv(M, BM)
            # When tile_size is small, it's possible that a wave has more ctas
            # then num_ctas_m * tile_size and num_ctas_per_wave_m > num_ctas_m.
            # The ctas mapping will "wrap around" and include following tile_sizes.
            var num_ctas_per_wave_m = ceildiv(num_SMs, tile_size)
            var num_ctas_per_wave_n = tile_size * ceildiv(
                num_ctas_per_wave_m, num_ctas_m
            )
            num_ctas_per_wave_m = min(num_ctas_per_wave_m, num_ctas_m)
            var load_volume_per_wave = (
                num_ctas_per_wave_m * BM + num_ctas_per_wave_n * mma_mn[1]
            )
            if load_volume_per_wave < min_load_volume:
                min_load_volume = load_volume_per_wave
                optimal_block_swizzle_size = tile_size

    # TODO: evaluate the comment's perf impact
    # var num_clc_pipeline_stages: UInt = UInt(min(min_num_waves-1, 2))
    var num_clc_pipeline_stages = 0 if min_num_waves == 1 else 2

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        mma_shape=IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // a_type.size_of()
        ),
        cta_group=cta_group,
        cluster_shape=Index(cta_group, 1, 1),
        AB_swapped=swapAB,
        block_swizzle_size=optimal_block_swizzle_size,
        num_accum_pipeline_stages=UInt(min(2, min_num_waves)),
        num_clc_pipeline_stages=UInt(num_clc_pipeline_stages),
        k_group_size=UInt(k_group_size),
    )


fn build_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
]() -> Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    alias config_t = MatmulConfig[a_type, b_type, c_type, transpose_b]

    var set = Set[config_t]()

    for m in range(8, 128, 8):  # [8, 128]
        config = choose_config[a_type, b_type, c_type, transpose_b](m, N, K)
        if config not in set:
            set.add(config)

    for m in range(128, 8193, 64):  # [128, 8192]
        config = choose_config[a_type, b_type, c_type, transpose_b](m, N, K)
        if config not in set:
            set.add(config)

    return set^
