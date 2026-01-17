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
from gpu.primitives.grid_controls import PDLLevel
from gpu.host.info import H100
from utils.index import Index, IndexList
from ....utils_gpu import MatmulConfig as BaseMatmulConfig
from collections import OptionalReg


@register_passable("trivial")
struct MatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](Copyable, Equatable, Hashable, Stringable, Writable):
    """Static configuration of SM90 GPU matmul."""

    # Mandatory parameters
    var block_tile_shape: IndexList[3]
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var num_pipeline_stages: UInt
    var num_k_partitions: UInt
    var num_consumer: UInt
    var partitioned_multicast: Bool
    var _pdl_level: PDLLevel
    var k_group_size: UInt

    fn __init__(
        out self,
        m: Int,
        n: Int,
        k: Int,
        num_k_partitions: UInt = 1,
        partitioned_multicast: Bool = False,
        pdl_level: PDLLevel = PDLLevel.OFF,
        k_groups: OptionalReg[UInt] = None,
        consumer_groups: OptionalReg[Int] = None,
        swapAB: Bool = False,
    ):
        """Initialize MatmulConfig by computing optimal values from M, N, K.

        Args:
            m: The M dimension of the matmul.
            n: The N dimension of the matmul.
            k: The K dimension of the matmul.
            num_k_partitions: Number of K partitions.
            partitioned_multicast: Whether to use partitioned multicast.
            pdl_level: PDL level for grid controls.
            k_groups: How many pipeline (loads and stores) are grouped together.
            consumer_groups: The number of consumer groups.
            swapAB: Whether to swap A and B.
        """
        constrained[
            Self.a_type == Self.b_type, "a_type and b_type must be the same"
        ]()

        var M = n if swapAB else m
        var N = m if swapAB else n
        var K = k

        # Heuristic: Use 1 consumer group for small M, 2 otherwise
        # TODO: Once SwapAB is added, this should probably always be 2

        var num_consumer_groups: Int
        if consumer_groups:
            num_consumer_groups = consumer_groups.value()
        else:
            num_consumer_groups = 1 if M <= 64 else 2

        comptime num_SMs = H100.sm_count
        # Nvidia mma instruction process 32B in K.
        comptime Kbytes_per_mma = 32  # 16 * 2
        # We use 128B swizzle, tile size in K is 128B over element size.
        comptime BK = 128 // size_of[Self.a_type]()

        var mma_mn = Tuple[Int, Int](256, 256)
        var min_num_waves = Int.MAX

        comptime bm = 64
        mma_mn[0] = bm

        # Tries it maximize active SM's and minimize waves
        for mma_n in range(8, 256 + 1, 8):
            var num_ctas = ceildiv(M, bm * num_consumer_groups) * ceildiv(
                N, mma_n
            )
            var num_waves = ceildiv(num_ctas, num_SMs)

            if (
                num_waves < min_num_waves
                or (
                    (num_waves == min_num_waves)
                    and (bm * mma_n < mma_mn[0] * mma_mn[1])
                )
            ) and (
                N % mma_n == 0
            ):  # NOTE: Matmul only works if this condition is true
                min_num_waves = num_waves
                mma_mn[1] = mma_n

        self.block_tile_shape = Index(
            mma_mn[0] * num_consumer_groups, mma_mn[1], BK
        )
        self.mma_shape = IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // size_of[Self.a_type]()
        )
        self.cluster_shape = Index(1, 1, 1)
        self.num_k_partitions = num_k_partitions
        self.num_consumer = UInt(num_consumer_groups)
        self.partitioned_multicast = partitioned_multicast
        self._pdl_level = pdl_level

        # Compute max pipeline stages.
        self.num_pipeline_stages = 4  # Default for compilation
        self.k_group_size = 1

        if k_groups:
            self.k_group_size = k_groups.value()
        else:
            var output_block_size = mma_mn[0] * mma_mn[1]

            if output_block_size <= 64 * 64 and ceildiv(K, BK) % 2 == 0:
                self.k_group_size = 2

            # For very small mmas we can group more aggressively.
            if output_block_size <= 64 * 48 and ceildiv(K, BK) % 4 == 0:
                self.k_group_size = 4

        self._maximize_pipeline_stages_by_default()

        self.num_pipeline_stages = align_down(
            self.num_pipeline_stages, self.k_group_size
        )

    fn _maximize_pipeline_stages_by_default(mut self):
        var BM = Int(self.block_tile_shape[0])
        var BN = Int(self.block_tile_shape[1])
        var BK = Int(self.block_tile_shape[2])

        var MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
        var tma_mbar_bytes_per_stage = MBAR_BYTES
        var mma_mbar_bytes_per_stage = MBAR_BYTES

        comptime h100_smem = Int(H100.shared_memory_per_multiprocessor - 1024)
        # Assume largest c smem tile is BM * 128
        var c_smem_bytes = BM * 128 * size_of[Self.c_type]()

        var a_smem_bytes_per_stage = BM * BK * size_of[Self.a_type]()
        var b_smem_bytes_per_stage = BN * BK * size_of[Self.b_type]()
        # A and B per pipeline stage
        var AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage
        var producer_consumer_smem_per_stage = (
            AB_smem_per_stage
            + tma_mbar_bytes_per_stage
            + mma_mbar_bytes_per_stage
        )

        var smem_leftover = h100_smem - c_smem_bytes
        self.num_pipeline_stages = UInt(
            smem_leftover // producer_consumer_smem_per_stage
        )

    fn pdl_level(self) -> PDLLevel:
        return self._pdl_level

    fn to_base_config(
        self,
    ) -> BaseMatmulConfig[
        Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
    ]:
        """Convert to base MatmulConfig from utils_gpu."""
        return BaseMatmulConfig[
            Self.a_type, Self.b_type, Self.c_type, Self.transpose_b
        ](
            block_tile_shape=self.block_tile_shape,
            mma_shape=self.mma_shape,
            cluster_shape=self.cluster_shape,
            num_pipeline_stages=self.num_pipeline_stages,
            num_k_partitions=self.num_k_partitions,
            num_consumer=self.num_consumer,
            partitioned_multicast=self.partitioned_multicast,
            pdl_level=self._pdl_level,
            k_group_size=self.k_group_size,
        )

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.block_tile_shape == other.block_tile_shape
            and self.mma_shape == other.mma_shape
            and self.cluster_shape == other.cluster_shape
            and self.num_pipeline_stages == other.num_pipeline_stages
            and self.num_k_partitions == other.num_k_partitions
            and self.num_consumer == other.num_consumer
            and self.partitioned_multicast == other.partitioned_multicast
            and self.k_group_size == other.k_group_size
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("kernel_")
        writer.write(Self.a_type, "_")
        writer.write(Self.c_type, "_")
        writer.write(
            "block",
            self.block_tile_shape[0],
            "x",
            self.block_tile_shape[1],
            "x",
            self.block_tile_shape[2],
            "_",
        )
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
        writer.write("consumer", self.num_consumer, "_")
        writer.write(
            "multicast" if self.partitioned_multicast else "nomulticast"
        )
        writer.write("_K" if Self.transpose_b else "_MN")

    fn __repr__(self) -> String:
        return String.write(self)

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher.update(Self.a_type)
        hasher.update(Self.b_type)
        hasher.update(Self.c_type)
        hasher.update(Self.transpose_b)
        hasher.update(self.block_tile_shape)
        hasher.update(self.mma_shape)
        hasher.update(self.cluster_shape)
        hasher.update(self.num_pipeline_stages)
        hasher.update(self.num_k_partitions)
        hasher.update(self.num_consumer)
        hasher.update(self.partitioned_multicast)


fn build_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
    num_k_partitions: UInt = 1,
    partitioned_multicast: Bool = False,
    pdl_level: PDLLevel = PDLLevel.OFF,
    k_groups: OptionalReg[UInt] = None,
    consumer_groups: OptionalReg[Int] = None,
    swapAB: Bool = False,
]() -> Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    var set = Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]()

    for m in range(8, 128, 8):  # [8, 128]
        config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            m,
            N,
            K,
            num_k_partitions=num_k_partitions,
            partitioned_multicast=partitioned_multicast,
            pdl_level=pdl_level,
            k_groups=k_groups,
            consumer_groups=consumer_groups,
            swapAB=swapAB,
        )
        if config not in set:
            set.add(config)

    for m in range(128, 8193, 64):  # [128, 8192]
        config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            m,
            N,
            K,
            num_k_partitions=num_k_partitions,
            partitioned_multicast=partitioned_multicast,
            pdl_level=pdl_level,
            k_groups=k_groups,
            consumer_groups=consumer_groups,
            swapAB=swapAB,
        )
        if config not in set:
            set.add(config)

    return set^
