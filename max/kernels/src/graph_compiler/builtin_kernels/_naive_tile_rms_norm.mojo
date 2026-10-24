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
"""Transitional tile-codegen scaffolding: the naive `mo.reduce.rms_norm` tile
driver, exercising the STORE variant of the tile-based epilogue.

Companion to `_naive_tile_matmul.mojo`, which exercises the COMPUTE variant
(`ComputeOutputFusionTile` / `TileOperation`, non-terminal). The graph compiler
registers one primary kernel per op, so a kernel author picks store or compute
fusion for that op; this file picks store (`OutputFusionTile` / `TileConsumer`,
terminal) for `mo.reduce.rms_norm`.

This is transitional scaffolding for bringing up the tile-based codegen path,
not the intended end state. It (and the `tile_based_fusion` plumbing it rides
on) is expected to be removed once every kernel for the target device has been
ported to `TileTensor` and the tile path is productionized (see GEX-3919).
"""

import extensibility

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from std.math import sqrt
from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout.tile_tensor import stack_allocation
from std.utils import IndexList
from linalg.utils import TileConsumer
from extensibility import (
    InputTensor,
    _FusedOutputTileTensor,
    get_kernel_tile_shape,
)

from ._tile_fusion import _tile_store_consumer


@fieldwise_init
struct _NaiveRMSNormTileAdapter[
    dtype: DType,
    InLayout: TensorLayout,
    GammaLayout: TensorLayout,
    TileConsumerType: TileConsumer,
    //,
    tile_shape: IndexList[2],
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Naive per-tile RMSNorm driver for the tile programming model (GPU).

    SM100 (B200) / correctness-only. One thread-block computes one
    `tile_shape` output tile, `thread_layout == tile_shape` so each of the
    `TM * TN` threads owns a `1x1` fragment and computes exactly one output
    element. The row-wise sum-of-squares reduction is over the FULL row
    (`n_dim`), independent of the output tile width `TN`: each thread
    redundantly walks its whole row to get `inv_rms`, rather than sharing the
    reduction across threads/blocks in that row.
    TODO(GEX-3919): O(n_dim) redundant work per element is a correctness-only
    shortcut; a real kernel would compute the row reduction once (e.g. one
    block-row pass with a shared-memory or warp reduction) and broadcast it.
    The bound `TileConsumer` owns the store of each per-thread fragment.

    Transitional scaffolding for the tile-codegen bring-up; expected to be
    removed once the tile path is productionized (see GEX-3919).
    """

    comptime thread_layout = row_major(
        Idx[Self.tile_shape[0]], Idx[Self.tile_shape[1]]
    )

    var input: TileTensor[Self.dtype, Self.InLayout, MutUntrackedOrigin]
    var gamma: TileTensor[Self.dtype, Self.GammaLayout, MutUntrackedOrigin]
    var n_dim: Int
    var epsilon: Float32
    var weight_offset: Scalar[Self.dtype]
    var tile_consumer: Self.TileConsumerType

    @inline(.always)
    def __call__(self):
        comptime TM = Self.tile_shape[0]
        comptime TN = Self.tile_shape[1]

        var tile_row = block_idx.y
        var tile_col = block_idx.x
        var tid = thread_idx.x
        var i, j = divmod(tid, TN)
        var row = tile_row * TM + i
        var col = tile_col * TN + j

        var sum_sq = Float32(0)
        for k in range(self.n_dim):
            var v = self.input.load((row, k)).cast[DType.float32]()
            sum_sq += v * v
        var inv_rms = 1.0 / sqrt(sum_sq / Float32(self.n_dim) + self.epsilon)

        var x_val = self.input.load((row, col)).cast[DType.float32]()
        var g_val = self.gamma.load((col,)).cast[.float32]()
        var w_off = self.weight_offset.cast[.float32]()
        var y = (x_val * inv_rms) * (g_val + w_off)

        # Driver-owned per-thread output fragment; see `_NaiveMatmulTileAdapter`
        # for the LOCAL->GENERIC staging dance (TODO(GEX-3912)).
        var dst_local = stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](row_major[1, 1]())
        var dst = TileTensor(
            dst_local._storage.unsafe_address_space_cast[
                .GENERIC
            ]().unsafe_origin_cast[MutAnyOrigin](),
            row_major[1, 1](),
        )
        dst.store((0, 0), y.cast[Self.dtype]())

        var consumer = self.tile_consumer
        consumer(
            Coord(tile_row * TM, tile_col * TN),
            dst.address_space_cast[.LOCAL]().address_space_cast[
                Self.TileConsumerType.src_address_space
            ](),
            Self.thread_layout,
        )


# Transitional tile-codegen scaffolding (remove per GEX-3941): a SEPARATE,
# device-specific `mo.reduce.rms_norm` registration for sm_100a (B200). It
# declares the tile fused-output IOSpec (`_FusedOutputTileTensor`); the
# generic SIMD `ReduceRMSNorm` in `reductions.mojo` stays untouched and keeps
# serving non-tile devices. Registering the tile kernel as its own struct
# (rather than a second `execute` overload) avoids the importer's single-
# `execute`-slot collapse. It must NOT become the canonical B200 rms_norm: the
# graph-compiler selection filter in `MojoRegistry::getResolvedKernel`
# discounts this tile candidate unless the target device has
# `tile_based_fusion` set, so a default B200 (flag off) resolves to the
# generic SIMD `mo.reduce.rms_norm` and only the tile-codegen path (flag on)
# selects this kernel. Expected to be removed once the tile path is
# productionized (see GEX-3919).
@extensibility.register(
    "mo.reduce.rms_norm", type="gpu", api="cuda", arch="sm_100a"
)
struct RMSNormTileSM100:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
        multiply_before_cast: Bool = True,
    ](
        output: _FusedOutputTileTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        # The tile programming model (`tile_based_fusion=True`) binds `output`
        # to `_FusedOutputTile`, and the selection filter routes the op to
        # this sm_100a registration. Naive, correctness-only GPU rms_norm on
        # MAX's own Mojo kernel (no vendor BLAS); see
        # `_NaiveRMSNormTileAdapter`. Tested at (M, N) = (256, 256) f32 on
        # B200.
        comptime assert is_gpu[target](), "tile-codegen rms_norm is GPU-only"
        comptime assert (
            rank == 2
        ), "tile-codegen rms_norm only supports rank-2 tensors"
        comptime assert (
            dtype == .float32
        ), "tile-codegen rms_norm is correctness-only and f32-only"

        # TODO(GEX-3905): a real rms_norm must own its own tile shape rather
        # than inheriting this shared elementwise `get_kernel_tile_shape`
        # default, and that shape must be threaded into the fusion emitter so
        # the epilogue conforms to it -- instead of the driver and emitter
        # each independently reading `get_kernel_tile_shape`.
        comptime tile_shape = get_kernel_tile_shape[output.dtype, target]()
        comptime TM = tile_shape[0]
        comptime TN = tile_shape[1]

        var m_dim = Int(output.dim_size[0]())
        var n_dim = Int(output.dim_size[1]())
        debug_assert(
            m_dim % TM == 0 and n_dim % TN == 0,
            "tile-codegen rms_norm requires tile-divisible output shapes",
        )

        var adapter = _NaiveRMSNormTileAdapter[tile_shape=tile_shape](
            input.to_tile_tensor(),
            gamma.to_tile_tensor(),
            n_dim,
            epsilon,
            weight_offset.cast[dtype](),
            _tile_store_consumer(output),
        )
        ctx.enqueue_function(
            adapter,
            grid_dim=(n_dim // TN, m_dim // TM),
            block_dim=(TM * TN),
        )
