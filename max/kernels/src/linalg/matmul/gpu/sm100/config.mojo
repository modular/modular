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

from collections.set import Set
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200
from layout.tensor_core import get_mma_shape
from memory.unsafe_pointer import UnsafePointer
from utils.index import Index, IndexList
from utils.numerics import get_accum_type

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
    var num_pipeline_stages: UInt
    var num_clc_pipeline_stages: UInt
    var num_accum_pipeline_stages: UInt
    var num_output_stages: UInt
    var output_tile_shape: IndexList[2]
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle
    var c_swizzle: TensorMapSwizzle

    fn __init__(
        out self,
        *,
        cta_group: Int = 2,
        mma_shape: IndexList[3] = get_mma_shape[a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(2, 1, 1),
        AB_swapped: Bool = False,
        block_swizzle_size: Int = 0,
        raster_order: RasterOrder = RasterOrder.AlongM,
        num_pipeline_stages: Optional[UInt] = None,
    ):
        constrained[a_type == b_type]()

        self.cta_group = cta_group
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.AB_swapped = AB_swapped
        self.block_swizzle_size = block_swizzle_size
        self.raster_order = raster_order

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
        var output_tile_n = 32 if c_tile_n % 32 == 0 else 16

        self.output_tile_shape = Index(
            output_tile_n, self.block_tile_shape[0]
        ) if self.AB_swapped else Index(self.block_tile_shape[0], output_tile_n)

        self.num_clc_pipeline_stages = 2
        self.num_accum_pipeline_stages = 2
        self.num_output_stages = 2

        var c_smem_bytes = (
            self.output_tile_shape[0]
            * self.output_tile_shape[1]
            * self.num_output_stages
            * size_of[c_type]()
        )

        alias b200_smem = B200.shared_memory_per_multiprocessor - 1024
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
        var AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage
        # Substract 512B for mbar usage etc
        self.num_pipeline_stages = UInt(
            (b200_smem - c_smem_bytes - 512) // AB_smem_per_stage
        )

        if num_pipeline_stages:
            self.num_pipeline_stages = num_pipeline_stages.value()

        self.a_swizzle = TensorMapSwizzle.SWIZZLE_128B
        self.b_swizzle = TensorMapSwizzle.SWIZZLE_128B
        self.c_swizzle = TensorMapSwizzle.SWIZZLE_32B if self.AB_swapped else (
            TensorMapSwizzle.SWIZZLE_64B if output_tile_n
            == 32 else TensorMapSwizzle.SWIZZLE_32B
        )

    fn copy_field(mut self, other: MatmulConfig):
        self.cta_group = other.cta_group
        self.mma_shape = other.mma_shape
        self.cluster_shape = other.cluster_shape
        self.AB_swapped = other.AB_swapped
        self.num_pipeline_stages = other.num_pipeline_stages
        self.num_clc_pipeline_stages = other.num_clc_pipeline_stages
        self.num_accum_pipeline_stages = other.num_accum_pipeline_stages
        self.num_output_stages = other.num_output_stages
        self.output_tile_shape = other.output_tile_shape
        self.a_swizzle = other.a_swizzle
        self.b_swizzle = other.b_swizzle
        self.c_swizzle = other.c_swizzle

    fn swapAB(self) -> MatmulConfig[b_type, a_type, c_type, transpose_b]:
        var new_config = UnsafePointer(to=self).bitcast[
            MatmulConfig[b_type, a_type, c_type, transpose_b]
        ]()[0]
        # new_config.copy_field(self)
        new_config.AB_swapped = not self.AB_swapped
        # Swap A and B swizzles
        new_config.a_swizzle = self.b_swizzle
        new_config.b_swizzle = self.a_swizzle
        return new_config

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
            and self.raster_order == other.raster_order
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
        writer.write("clc", self.num_clc_pipeline_stages, "_")
        writer.write("accum", self.num_accum_pipeline_stages, "_")
        writer.write("out", self.num_output_stages, "_")
        writer.write("swap" if self.AB_swapped else "noswap", "_")
        writer.write("K", "_")
        writer.write("K" if transpose_b else "MN")
        writer.write("bz", self.block_swizzle_size, "_", self.raster_order)

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


fn choose_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](M: Int, N: Int, K: Int) -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    constrained[a_type == b_type, "a_type and b_type must be the same"]()

    # Hardcode to 2 now since main only dispatches 2xSM MMA
    alias cta_group = 2
    alias num_SMs = B200.sm_count
    # Nvidia mma instruction process 32B in K.
    alias Kbytes_per_mma = 32

    var mma_mn = Tuple[Int, Int](64, 128)
    var min_num_waves = Int.MAX

    # Travers possible combinations of BM x MMA_N to choose the one minizes the
    # workload per SM. The computation per SM is the flops (ignoring 2x in 2MNK)
    # timed by max number of ctas per SM i.e. number of waves.
    for bm in [64, 128]:
        for mma_n in range(16, min(257, N), 16):
            num_ctas = ceildiv(M, bm) * ceildiv(N, mma_n)
            num_waves = ceildiv(num_ctas, num_SMs)
            if num_waves < min_num_waves or (
                num_waves == min_num_waves
                and bm * cta_group * mma_n < mma_mn[0] * mma_mn[1]
            ):
                min_num_waves = num_waves
                mma_mn[0] = bm * cta_group
                mma_mn[1] = mma_n

    var min_load_volume = Int.MAX
    var optimal_block_swizzle_size = 0

    # Tile waves when there are >= 4 waves. In theory it should be >=2, but let's
    # be conservative.
    if min_num_waves >= 4:
        # Represent the load volume by
        #    BM * num_ctas_per_wave_m + MMA_N * num_ctas_per_wave_N
        # Use MMA_N because cta_group = 2, 2 ctas cover entire MMA_N. cta_group = 1
        # has BN = MMA_N.
        # Traverse the tile sizes to find min load volume.
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
            var load_volume = (
                num_ctas_per_wave_m * BM + num_ctas_per_wave_n * mma_mn[1]
            )
            if load_volume < min_load_volume:
                min_load_volume = load_volume
                optimal_block_swizzle_size = tile_size

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        mma_shape=IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // a_type.size_of()
        ),
        block_swizzle_size=optimal_block_swizzle_size,
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

    for m in range(8, 129, 8):  # [8, 128]
        config = choose_config[a_type, b_type, c_type, transpose_b](m, N, K)
        if config not in set:
            set.add(config)

    for m in range(128, 8193, 64):  # [128, 8192]
        config = choose_config[a_type, b_type, c_type, transpose_b](m, N, K)
        if config not in set:
            set.add(config)

    return set^
