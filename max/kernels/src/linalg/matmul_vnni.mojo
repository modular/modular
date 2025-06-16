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

from math import align_down
from sys import prefetch
from sys.info import alignof, has_avx512f, has_neon_int8_dotprod
from sys.intrinsics import PrefetchOptions

from buffer.buffer import NDBuffer, partial_simd_load
from memory import UnsafePointer
from memory.unsafe import bitcast

from utils.index import Index, IndexList

from .accumulate import _Accumulator
from .matmul import InnerMatmulKernel
from .neon_intrinsics import _neon_dotprod
from .utils import GemmShape, get_matmul_prefetch_b_distance_k
from .vnni_intrinsics import dot_i8_to_i32_saturated_x86, dot_i8_to_i32_x86


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the VNNI microkernel.
@fieldwise_init
struct Inner_matmul_vnni[saturated_vnni: Bool](InnerMatmulKernel, Movable):
    # Parameters for global reference.

    @always_inline
    fn _accumulate[
        is_tail: Bool,
        simd_size: Int,
        kernel_rows: Int,
        kernel_cols: Int,
    ](
        self,
        a: NDBuffer,
        b_packed: NDBuffer[_, 3, _, _],
        mut c_local: _Accumulator[
            _, kernel_rows, kernel_cols // simd_size, simd_size
        ],
        global_offset: GemmShape,
        tile_n_k_idx: IndexList[2],
        tile_n_k: IndexList[2],
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
            tile_n_k: TODO
        """
        alias c_type = c_local.type
        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // kernel_cols
        var kl = tile_n_k_idx[1]

        # Global K index.
        var global_k = global_offset.K + kl
        var b_ptr = b_packed._offset(Index(n_outer_idx, kl // 4, 0)).bitcast[
            Scalar[c_type]
        ]()

        @parameter
        if not is_tail:
            # Prefetch B matrix.
            alias prefetch_distance = get_matmul_prefetch_b_distance_k()

            @parameter
            if prefetch_distance > 0:
                alias prefetch_offset = prefetch_distance * kernel_cols

                @parameter
                for idx in range(kernel_cols // simd_size):
                    prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ](b_ptr.offset(prefetch_offset + idx * simd_size))

        # This inner kernels works with non-transposed A.
        var K = a.dim[1]()

        var a_local = NDBuffer[
            a.type,
            1,
            MutableAnyOrigin,
            4 * kernel_rows,
            address_space = a.address_space,
        ].stack_allocation()
        var a_base_ptr = a.data.offset(global_offset.M * K + global_k)
        var a_ptr = a_local.data if (
            is_tail
            and not has_avx512f()
            # This origin cast is not ideal since we give up
            # exclusivity checking, but it is safe in the sense that
            # `a` will be guaranteed to remain alive because
            # it is an argument to the function.
        ) else a_base_ptr.origin_cast[True, MutableAnyOrigin]()
        var a_ptr_stride = 4 if (is_tail and not has_avx512f()) else K

        var tail_length = tile_n_k[1] - kl

        # pack A if (tile_n_k_idx[1] - kl) is 1, 2, or 3
        @parameter
        if is_tail and not has_avx512f():
            for idx0 in range(kernel_rows):
                for idx_k in range(tail_length):
                    a_local[4 * idx0 + idx_k] = a_base_ptr[idx0 * K + idx_k]

        # Loop over local accumulator tiles.
        @parameter
        for idx0 in range(kernel_rows):

            @parameter
            for idx1 in range(kernel_cols // simd_size):
                # width K bytes or K/4 ints, a_ptr is pointer to ints
                var a_val = (
                    bitcast[c_type, 1](
                        partial_simd_load[4](
                            a_ptr.offset(idx0 * a_ptr_stride), 0, tail_length, 0
                        )
                    ) if (is_tail and has_avx512f()) else a_ptr.offset(
                        idx0 * a_ptr_stride
                    )
                    .bitcast[Scalar[c_type]]()
                    .load()
                )

                alias alignment = alignof[SIMD[c_type, simd_size]]()
                # var c_idx = Index(idx0, idx1 * simd_size)
                var c_val = c_local[idx0, idx1]
                var b_val = b_ptr.offset(idx1 * simd_size).load[
                    width=simd_size, alignment=alignment
                ]()

                @parameter
                if has_neon_int8_dotprod():
                    var a_val2 = SIMD[c_type, simd_size](a_val)
                    c_val = _neon_dotprod(
                        c_val,
                        bitcast[a.type, simd_size * 4](a_val2),
                        bitcast[b_packed.type, simd_size * 4](b_val),
                    )
                elif saturated_vnni:
                    c_val = dot_i8_to_i32_saturated_x86[simd_size](
                        c_val, a_val, b_val
                    )
                else:
                    c_val = dot_i8_to_i32_x86[simd_size](c_val, a_val, b_val)
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
        (kernel_rows, TileN, TileK) tile.
        """
        debug_assert(
            tile_n_k[1] % 0 == 0, "K dimension must be a multiple of 4"
        )

        var c_stride = c.dim[1]()

        var c_ptr = c.data.offset(global_offset.M * c_stride + global_offset.N)
        var c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

        var acc = _Accumulator[
            c.type, kernel_rows, kernel_cols // simd_size, simd_size
        ]()

        for idx_n in range(0, tile_n_k[0], kernel_cols):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if global_offset.K == 0:
                acc.init(0)
            else:
                acc.load(
                    rebind[UnsafePointer[Scalar[c.type]]](c_ptr),
                    c_stride,
                    idx_n,
                    c_bound,
                    skip_boundary_check,
                )

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            var kl = align_down(tile_n_k[1], 4)
            for idx_k in range(0, kl, 4):
                # accumulate data for this (n, k) index
                self._accumulate[False, simd_size, kernel_rows, kernel_cols](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, idx_k),
                    tile_n_k,
                )
            if kl != tile_n_k[1]:
                self._accumulate[True, simd_size, kernel_rows, kernel_cols](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, kl),
                    tile_n_k,
                )
            acc.store(
                rebind[UnsafePointer[Scalar[c.type]]](c_ptr),
                c_stride,
                idx_n,
                c_bound,
                skip_boundary_check,
            )
