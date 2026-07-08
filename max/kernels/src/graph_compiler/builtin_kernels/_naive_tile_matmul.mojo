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
"""Transitional tile-codegen scaffolding: the naive `mo.matmul` tile driver.

This file is deliberately separate from `linalg.mojo` so the tile-programming-
model driver (and the GPU/tile-only imports it needs) stays isolated from the
main linalg kernels: `linalg.mojo` just imports `_NaiveMatmulTileAdapter`.

This is transitional scaffolding for bringing up the tile-based codegen path,
not the intended end state. It (and the `tile_based_fusion` plumbing it rides
on) is expected to be removed once every kernel for the target device has been
ported to `TileTensor` and the tile path is productionized (see GEX-3919).
"""

import extensibility as compiler

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_gpu
from std.gpu.memory import AddressSpace
from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout.tile_tensor import stack_allocation
from layout.tile_io import GenericToLocalTileCopier, LocalToGenericTileCopier
from std.utils import IndexList
from extensibility import (
    ComputeOutputFusion,
    ComputeOutputFusionTile,
    InputFusion,
    InputTensor,
    IOSpec,
    ManagedTensorSlice,
    OutputFusion,
    StaticTensorSpec,
    _FusedComputeOutputTileTensor,
    get_kernel_tile_shape,
)


@fieldwise_init
struct _NaiveMatmulTileAdapter[
    dtype: DType,
    InFusion: InputFusion,
    OutFusion: OutputFusion,
    ComputeFusion: ComputeOutputFusion,
    ComputeFusionTile: ComputeOutputFusionTile,
    io_spec: IOSpec[True, _],
    static_spec: StaticTensorSpec[
        dtype, 2, _, InFusion, OutFusion, ComputeFusion, ComputeFusionTile
    ],
    a_dtype: DType,
    ALayout: TensorLayout,
    b_dtype: DType,
    BLayout: TensorLayout,
    //,
    has_epilogue_fusion: Bool,
    tile_shape: IndexList[2],
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Naive per-tile matmul driver for the tile programming model (GPU).

    SM100 (B200) / correctness-only. Structurally the tile twin of
    `_ElementwiseFusionTileAdapter` in `builtin_primitives/primitives.mojo`:
    one thread-block computes one `tile_shape` output tile, `thread_layout ==
    tile_shape` so each of the `TM * TN` threads owns a `1x1` fragment and
    computes exactly one output element `C[row, col] = sum_k A[row, k] *
    B[k, col]`. The per-thread fragment is routed through the output
    compute-fusion lambda (matmul+add, case 5) and stored via the same store
    copier the elementwise driver uses. No tensor cores / SMEM tiling: this is
    the validation kernel for the tile-codegen path, not a perf kernel.

    Transitional scaffolding for the tile-codegen bring-up; expected to be
    removed once the tile path is productionized (see GEX-3919).
    """

    comptime thread_layout = row_major(
        Idx[Self.tile_shape[0]], Idx[Self.tile_shape[1]]
    )
    comptime frag_layout = type_of(row_major[1, 1]())

    var c: ManagedTensorSlice[
        io_spec=Self.io_spec, static_spec=Self.static_spec
    ]
    var a: TileTensor[Self.a_dtype, Self.ALayout, MutUntrackedOrigin]
    var b: TileTensor[Self.b_dtype, Self.BLayout, MutUntrackedOrigin]
    var k_dim: Int

    @always_inline
    def __call__(self) capturing:
        comptime TM = Self.tile_shape[0]
        comptime TN = Self.tile_shape[1]

        # One block per output tile; `thread_idx.x` selects this thread's
        # element within the tile (row-major over `thread_layout`).
        var tile_row = Int(block_idx.y)
        var tile_col = Int(block_idx.x)
        var tid = Int(thread_idx.x)
        var i = tid // TN
        var j = tid % TN
        var row = tile_row * TM + i
        var col = tile_col * TN + j

        # Naive dot product over K for this output element.
        var acc = Scalar[Self.dtype](0)
        for k in range(self.k_dim):
            acc += (
                self.a.load[width=1](Coord(row, k)).cast[Self.dtype]()
                * self.b.load[width=1](Coord(k, col)).cast[Self.dtype]()
            )

        # Driver-owned per-thread output fragment, allocated in LOCAL (so the
        # store copier's src space lines up) and viewed GENERIC / `MutAnyOrigin`
        # to match the compute-fusion lambda's `val` type. Mirrors the `dst`
        # pattern in `_ElementwiseFusionTileAdapter`.
        #
        # TODO(GEX-3912): this LOCAL->GENERIC staging + the `address_space_cast`
        # back to LOCAL at the store below is a known limitation of the current
        # tile compute interface, which pins its tile argument to GENERIC and
        # returns a fresh tile. Generalizing that interface (an address-space-
        # general tile argument so the fragment can stay LOCAL throughout, plus
        # an `inout` `dst` so the passed-in tile is the output buffer instead of
        # a layout carrier) would remove this dance entirely.
        var dst_local = stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](row_major[1, 1]())
        var dst = TileTensor(
            dst_local.ptr.address_space_cast[
                AddressSpace.GENERIC
            ]().unsafe_origin_cast[MutAnyOrigin](),
            row_major[1, 1](),
        )
        dst.store[width=1](Coord(0, 0), acc)

        # Apply the fused epilogue (matmul+add, case 5) via the output
        # compute-fusion lambda; the supplied load copier lets the fusion pull
        # its own inputs (e.g. the broadcast bias tile) into local. Plain
        # matmul (case 4) skips it and stores the raw tile.
        var load_copier = GenericToLocalTileCopier[Self.thread_layout]()
        var tile_coords = IndexList[2](tile_row, tile_col)
        var res = dst
        comptime if Self.has_epilogue_fusion:
            res = self.c._fused_compute_output_tile_lambda(
                tile_coords, load_copier, dst
            )

        # Store the result fragment into the output tile at `tile_coords`. The
        # `address_space_cast` back to LOCAL here is the store-side half of the
        # staging dance noted above (see TODO(GEX-3912)).
        var tc = Coord(tile_row, tile_col)
        var out_tile = self.c.to_tile_tensor().tile[TM, TN](tc)
        var res_local = res.address_space_cast[AddressSpace.LOCAL]()
        LocalToGenericTileCopier[Self.thread_layout]().copy(out_tile, res_local)


# Transitional tile-codegen scaffolding (remove per GEX-3941): a SEPARATE,
# device-specific `mo.matmul` registration for sm_100a (B200). It declares the
# tile fused-output IOSpec (`_FusedComputeOutputTileTensor`); the generic
# `Matmul` in `linalg.mojo` stays SIMD-only. Registering the tile matmul as its
# own kernel (rather than a second `execute` overload on the generic struct)
# avoids the importer's single-`execute`-slot collapse. It must NOT become the
# canonical B200 matmul: the graph-compiler selection filter in
# `MojoRegistry::getResolvedKernel` discounts this tile candidate unless the
# target device has `tile_based_fusion` set, so a default B200 (flag off)
# resolves to the generic SIMD `mo.matmul` and only the tile-codegen path
# (flag on) selects this kernel. Expected to be removed once the real tile
# matmul lands and the tile-codegen transition completes (see GEX-3919).
@compiler.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a")
struct MatmulTileSM100:
    @staticmethod
    def execute[
        transpose_b: Bool,
        packed_b: Bool,
        has_epilogue_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: _FusedComputeOutputTileTensor[rank=2, ...],
        a: InputTensor[rank=2, ...],
        b: InputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # The tile programming model (`tile_based_fusion=True`) binds `c` to
        # `_FusedComputeOutputTile`, and the selection filter routes the op to
        # this sm_100a registration. Naive, correctness-only GPU matmul on MAX's
        # own Mojo kernel (no vendor BLAS); see `_NaiveMatmulTileAdapter`. Tested
        # at (M, N, K) = (256, 256, 256) f32 on B200.
        comptime assert is_gpu[target](), "tile-codegen matmul is GPU-only"
        comptime assert (
            not transpose_b
        ), "tile-codegen matmul only supports transpose_b=False"
        comptime assert (
            not packed_b
        ), "tile-codegen matmul does not support packed_b"

        # TODO(GEX-3905): a real matmul must own its own tile shape (tensor-core
        # tiling) rather than inheriting this shared elementwise
        # `get_kernel_tile_shape` default, and that shape must be threaded into
        # the fusion emitter so the epilogue conforms to it — instead of the
        # driver and emitter each independently reading `get_kernel_tile_shape`.
        comptime tile_shape = get_kernel_tile_shape[c.dtype, target]()
        comptime TM = tile_shape[0]
        comptime TN = tile_shape[1]

        var m_dim = Int(c.dim_size[0]())
        var n_dim = Int(c.dim_size[1]())
        var k_dim = Int(a.dim_size[1]())
        debug_assert(
            m_dim % TM == 0 and n_dim % TN == 0,
            "tile-codegen matmul requires tile-divisible output shapes",
        )

        var adapter = _NaiveMatmulTileAdapter[
            has_epilogue_fusion=has_epilogue_fusion, tile_shape=tile_shape
        ](c, a.to_tile_tensor(), b.to_tile_tensor(), k_dim)
        ctx.enqueue_function(
            adapter,
            grid_dim=(n_dim // TN, m_dim // TM),
            block_dim=(TM * TN),
        )
