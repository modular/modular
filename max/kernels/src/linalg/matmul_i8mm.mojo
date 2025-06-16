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

from math import align_up
from sys import prefetch
from sys.info import alignof
from sys.intrinsics import PrefetchOptions

from buffer.buffer import NDBuffer, partial_simd_load, partial_simd_store
from memory import UnsafePointer

from utils.index import Index, IndexList

from .accumulate import _Accumulator
from .matmul import InnerMatmulKernel
from .neon_intrinsics import _neon_matmul
from .utils import GemmShape, get_matmul_prefetch_b_distance_k


struct LoadStore_i8mm[
    type: DType,
    simd_size: Int,
    single_row: Bool,
    tile_rows: Int,
    tile_columns: Int,
]:
    alias num_simd_cols = tile_columns // simd_size
    var output_tile: _Accumulator[
        type, tile_rows, Self.num_simd_cols, simd_size
    ]
    var skip_boundary_check: Bool

    @always_inline
    @implicit
    fn __init__(out self, skip_boundary_check: Bool):
        self.output_tile = _Accumulator[
            type, tile_rows, Self.num_simd_cols, simd_size
        ]()
        self.skip_boundary_check = skip_boundary_check

    @always_inline
    fn _initialize_c_tile(mut self):
        self.output_tile.init(0)

    @always_inline
    fn _load_c_tile(
        mut self,
        c_ptr: UnsafePointer[Scalar[type]],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        @parameter
        for idx0 in range(tile_rows):

            @parameter
            for idx1 in range(tile_columns // simd_size):
                var c_data: SIMD[type, simd_size] = 0
                if self.skip_boundary_check or (
                    idx1 * 2 + 2 <= c_bound[1] - tile_n_idx
                ):
                    var t0 = c_ptr_loc.load[width=2](
                        c_stride * (2 * idx0) + 2 * idx1
                    )
                    var t1 = c_ptr_loc.load[width=2](
                        c_stride * (2 * idx0 + 1) + 2 * idx1
                    ) if not single_row else SIMD[type, 2](0)
                    c_data = rebind[SIMD[type, simd_size]](t0.join(t1))
                elif idx1 * 2 <= c_bound[1]:
                    var t0 = partial_simd_load[2](
                        c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    )
                    var t1 = partial_simd_load[2](
                        c_ptr_loc.offset(c_stride * (2 * idx0 + 1) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    ) if not single_row else SIMD[type, 2](0)
                    c_data = rebind[SIMD[type, simd_size]](t0.join(t1))

                self.output_tile[idx0, idx1] = c_data

    @always_inline
    fn _store_c_tile(
        mut self,
        c_ptr: UnsafePointer[Scalar[type]],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
    ):
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        @parameter
        for idx0 in range(tile_rows):

            @parameter
            for idx1 in range(tile_columns // simd_size):
                var c_data = self.output_tile[idx0, idx1]
                if self.skip_boundary_check or (
                    idx1 * 2 + 2 <= c_bound[1] - tile_n_idx
                ):
                    c_ptr_loc.offset(
                        c_stride * (2 * idx0 + 0) + 2 * idx1
                    ).store(
                        c_data.slice[2](),
                    )

                    @parameter
                    if not single_row:
                        c_ptr_loc.offset(
                            c_stride * (2 * idx0 + 1) + 2 * idx1
                        ).store(
                            c_data.slice[2, offset=2](),
                        )
                elif idx1 * 2 <= c_bound[1]:
                    partial_simd_store(
                        c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        c_data.slice[2](),
                    )

                    @parameter
                    if not single_row:
                        partial_simd_store(
                            c_ptr_loc.offset(
                                c_stride * (2 * idx0 + 1) + 2 * idx1
                            ),
                            0,
                            c_bound[1] - tile_n_idx - idx1 * 2,
                            c_data.slice[2, offset=2](),
                        )


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the I8MM microkernel.
@fieldwise_init
struct Inner_matmul_i8mm(InnerMatmulKernel, Movable):
    # Parameters for global reference.

    @always_inline
    fn _accumulate[
        simd_size: Int, kernel_rows: Int, kernel_cols: Int
    ](
        self,
        a: NDBuffer,
        b_packed: NDBuffer[_, 3, _, _],
        mut c_local: _Accumulator[
            _, kernel_rows, kernel_cols // simd_size, simd_size
        ],
        global_offset: GemmShape,
        tile_n_k_idx: IndexList[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            a: TODO.
            b_packed: TODO.
            c_local: Pre-allocated local buffer for c partial sums.
            global_offset: TODO.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """

        var n_outer_idx = tile_n_k_idx[0] // (kernel_cols // 2)
        var kl = tile_n_k_idx[1]
        var b_ptr = b_packed._offset(Index(n_outer_idx, kl // 8, 0))

        # This inner kernels works with non-transposed A.
        var K = a.dim(1)
        var a_ptr = a.data.offset(
            global_offset.M * K + 2 * global_offset.K + 2 * kl
        )

        # Prefetch B matrix.
        alias prefetch_distance = get_matmul_prefetch_b_distance_k()
        constrained[simd_size == 4]()

        @parameter
        if prefetch_distance > 0:
            alias prefetch_offset = prefetch_distance * kernel_cols

            @parameter
            for idx in range(kernel_cols // simd_size):
                prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ](b_ptr.offset(prefetch_offset + idx * simd_size))

        # Loop over local accumulator tiles.
        @parameter
        for idx0 in range(kernel_rows):

            @parameter
            for idx1 in range(kernel_cols // simd_size):
                alias alignment = alignof[SIMD[c_local.type, simd_size]]()
                var a_val = a_ptr.load[width = simd_size * 4](2 * idx0 * K)
                var b_val = b_ptr.offset(16 * idx1).load[
                    width = simd_size * 4, alignment=alignment
                ]()
                var c_val = c_local[idx0, idx1]
                c_val = _neon_matmul(c_val, a_val, b_val)
                c_local[idx0, idx1] = c_val

    @always_inline
    fn __inner_matmul__[
        kernel_rows: Int,
        kernel_cols: Int,
        simd_size: Int,
    ](
        self,
        c: NDBuffer,
        a: NDBuffer,
        b_packed: NDBuffer[_, 3, _, _],
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: IndexList[2],
        skip_boundary_check: Bool,
    ):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (kernel_rows2, TileN, TileK) tile.
        """

        alias kernel_rows2 = kernel_rows // 2 if kernel_rows != 1 else kernel_rows
        alias single_row = (kernel_rows == 1)

        var c_stride = c.dim[1]()

        var c_ptr = c.data.offset(global_offset.M * c_stride + global_offset.N)

        var c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

        var acc = LoadStore_i8mm[
            c.type,
            simd_size,
            single_row,
            kernel_rows2,
            kernel_cols,
        ](skip_boundary_check)

        for idx_n in range(0, tile_n_k[0], kernel_cols // 2):
            if global_offset.K == 0:
                acc._initialize_c_tile()
            else:
                acc._load_c_tile(
                    rebind[UnsafePointer[Scalar[c.type]]](c_ptr),
                    c_stride,
                    idx_n,
                    c_bound,
                )
            var kl = align_up(tile_n_k[1], 8)
            for idx_k in range(0, kl, 8):
                self._accumulate[simd_size, kernel_rows2, kernel_cols](
                    a,
                    b_packed,
                    acc.output_tile,
                    global_offset,
                    Index(idx_n, idx_k),
                )
            acc._store_c_tile(
                rebind[UnsafePointer[Scalar[c.type]]](c_ptr),
                c_stride,
                idx_n,
                c_bound,
            )
