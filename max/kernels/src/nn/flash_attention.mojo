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

from collections import OptionalReg
from math import align_down, align_up, ceildiv, exp

from os import abort
from sys import align_of, simd_width_of
from sys.info import CompilationTarget

from algorithm import sync_parallelize, tile, vectorize
from algorithm.reduction import (
    _simd_max,
    _simd_max_elementwise,
    _simd_sum,
    _simd_sum_elementwise,
    map_reduce,
)
from kv_cache.types import KVCacheT
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from layout.int_tuple import to_index_list
from linalg.accumulate import _Accumulator
from linalg.matmul.cpu.apple_accelerate import (
    _cblas_f32,
    use_apple_accelerate_lib,
)
from linalg.transpose import transpose_inplace
from linalg.utils import partition_work
from memory import memset_zero, stack_allocation
from nn.mha_mask import MHAMask
from runtime.asyncrt import parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils import Index, IndexList


struct _MatmulConfig:
    var col_sizes: VariadicList[Int]
    var row_sizes: VariadicList[Int]
    var gemv_sizes: VariadicList[Int]
    var pack_sizes: VariadicList[Int]

    fn __init__(
        out self,
        *,
        col_sizes: VariadicList[Int],
        row_sizes: VariadicList[Int],
        gemv_sizes: VariadicList[Int],
        pack_sizes: VariadicList[Int],
    ):
        self.col_sizes = col_sizes
        self.row_sizes = row_sizes
        self.gemv_sizes = gemv_sizes
        self.pack_sizes = pack_sizes

    @staticmethod
    fn _get_config() -> _MatmulConfig:
        comptime if CompilationTarget.has_neon():
            return _MatmulConfig(
                col_sizes=VariadicList[Int](4, 3, 2, 1),
                row_sizes=VariadicList[Int](6, 4, 1),
                gemv_sizes=VariadicList[Int](32, 4, 1),
                pack_sizes=VariadicList[Int](32, 8, 4, 1),
            )
        elif CompilationTarget.has_avx512f():
            return _MatmulConfig(
                col_sizes=VariadicList[Int](4, 3, 2, 1),
                row_sizes=VariadicList[Int](6, 4, 1),
                gemv_sizes=VariadicList[Int](64, 16, 4, 1),
                pack_sizes=VariadicList[Int](64, 16, 8, 4, 1),
            )
        else:
            return _MatmulConfig(
                col_sizes=VariadicList[Int](3, 2, 1),
                row_sizes=VariadicList[Int](4, 1),
                gemv_sizes=VariadicList[Int](64, 16, 4, 1),
                pack_sizes=VariadicList[Int](64, 16, 8, 4, 1),
            )


struct _Matmul[dtype: DType, simd_width: Int]:
    comptime _matmul_config = _MatmulConfig._get_config()

    comptime _input_fn_type = fn[simd_width: Int](
        x: Int, y: Int
    ) capturing -> SIMD[Self.dtype, simd_width]

    @staticmethod
    @always_inline
    fn _inner_loop_a_lane[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype]],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[Self.dtype]],
        b_stride: Int,
        mut c_tile: _Accumulator[Self.dtype, tile_m, tile_n, Self.simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @parameter
        @always_inline
        fn loop_body[lane_count: Int](k: Int):
            var a_tile = InlineArray[SIMD[Self.dtype, lane_count], tile_m](
                fill=0
            )

            comptime for m in range(tile_m):
                a_tile[m] = ak_ptr.load[width=lane_count](m * a_stride)

            ak_ptr += lane_count

            comptime for k in range(lane_count):
                comptime for n in range(tile_n):
                    var b_data = bk_ptr.load[width = Self.simd_width](
                        n * Self.simd_width
                    )

                    comptime for m in range(tile_m):
                        c_tile.fma(m, n, a_tile[m][k], b_data)

                bk_ptr += b_stride

        tile[loop_body, VariadicList[Int](Self.simd_width, 1)](0, K)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = ak_ptr
        _ = bk_ptr

    @staticmethod
    @always_inline
    fn _inner_loop_a_broadcast[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype]],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[Self.dtype]],
        b_stride: Int,
        mut c_tile: _Accumulator[Self.dtype, tile_m, tile_n, Self.simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @parameter
        @always_inline
        fn loop_body[unroll_factor: Int](k: Int):
            var b_tile = InlineArray[SIMD[Self.dtype, Self.simd_width], tile_n](
                fill=0
            )

            comptime for k in range(unroll_factor):
                comptime for n in range(tile_n):
                    b_tile[n] = bk_ptr.load[width = Self.simd_width](
                        n * Self.simd_width
                    )

                comptime for m in range(tile_m):
                    var a_data = ak_ptr.load(m * a_stride)

                    comptime for n in range(tile_n):
                        c_tile.fma(m, n, a_data, b_tile[n])

                ak_ptr += 1
                bk_ptr += b_stride

        tile[loop_body, VariadicList[Int](2, 1)](0, K)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = ak_ptr
        _ = bk_ptr

    @no_inline
    @staticmethod
    fn _matmul_packed(
        M: Int,
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype]],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[Self.dtype]],
        c_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        c_stride: Int,
        accumulate: Bool = False,
    ):
        var am_ptr = a_ptr
        var cm_ptr = c_ptr

        @parameter
        fn process_rows[tile_m: Int](m: Int):
            var bn_ptr = b_ptr
            var cn_ptr = cm_ptr

            @parameter
            fn process_cols[tile_n: Int](n_unscaled: Int):
                var c_tile = _Accumulator[
                    Self.dtype, tile_m, tile_n, Self.simd_width
                ]()

                if accumulate:
                    c_tile.load(cn_ptr, c_stride)
                else:
                    c_tile.init(0.0)

                comptime if CompilationTarget.has_neon():
                    Self._inner_loop_a_lane(
                        K, am_ptr, a_stride, bn_ptr, N, c_tile
                    )
                else:
                    Self._inner_loop_a_broadcast(
                        K, am_ptr, a_stride, bn_ptr, N, c_tile
                    )

                c_tile.store(cn_ptr, c_stride)

                bn_ptr += tile_n * Self.simd_width
                cn_ptr += tile_n * Self.simd_width

            tile[process_cols, Self._matmul_config.col_sizes](
                0, ceildiv(N, Self.simd_width)
            )

            am_ptr += tile_m * a_stride
            cm_ptr += tile_m * c_stride

            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = bn_ptr
            _ = cn_ptr

        tile[process_rows, Self._matmul_config.row_sizes](0, M)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = am_ptr
        _ = cm_ptr

    @no_inline
    @staticmethod
    fn _pack_buffer_transposed[
        input_b_fn: Self._input_fn_type, static_k: Int
    ](
        packed_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        N: Int,
        dynamic_k: Int,
    ):
        var K = static_k if static_k != UNKNOWN_VALUE else dynamic_k

        var aligned_n = align_up(N, Self.simd_width)

        # Use a conservative SIMD width for transposing. Using a wider native
        # SIMD width has not been observed to improve performance and causes
        # code size to unnecessarily increase.
        comptime transpose_width = 4
        comptime tile_sizes = VariadicList[Int](transpose_width, 1)

        comptime layout = Layout.row_major(transpose_width, transpose_width)
        var transpose_stack = InlineArray[Scalar[Self.dtype], layout.size()](
            uninitialized=True
        )
        var transpose_buffer = LayoutTensor[
            Self.dtype,
            layout,
        ](transpose_stack)

        @parameter
        @always_inline
        fn process_tile[tile_n: Int, tile_k: Int](n: Int, k: Int):
            comptime if transpose_width == tile_n == tile_k:
                # Use an optimized path to transpose a square tile of the
                # input tensor.
                comptime for i in range(transpose_width):
                    var val = input_b_fn[simd_width=transpose_width](n + i, k)
                    transpose_buffer.store(Index(i, 0), val)

                transpose_inplace[4, 4](transpose_buffer)

                comptime for i in range(transpose_width):
                    var val = transpose_buffer.load[width=transpose_width](
                        Index(i, 0)
                    )
                    packed_ptr.store((k + i) * aligned_n + n, val)

            else:
                # Fallback to strided loads and stores of the tensors.
                #
                # Note that in the common case, `K` is statically known and is
                # a multiple of `transpose_width`, so the case to optimize for
                # `tile_n=1` and `tile_k=transpose_width`.
                comptime for nn in range(tile_n):
                    var val = input_b_fn[simd_width=tile_k](n + nn, k)

                    comptime for kk in range(tile_k):
                        packed_ptr.store(
                            (k + kk) * aligned_n + (n + nn), val[kk]
                        )

        tile[process_tile, tile_sizes, tile_sizes](0, 0, N, K)

        if aligned_n != N:
            for k in range(K):
                memset_zero(packed_ptr + k * aligned_n + N, aligned_n - N)

    @no_inline
    @staticmethod
    fn _pack_buffer[
        input_b_fn: Self._input_fn_type
    ](
        packed_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        N: Int,
        K: Int,
    ):
        var output_ptr = packed_ptr
        var aligned_n = align_up(N, Self.simd_width)

        for k in range(K):

            @parameter
            @always_inline
            fn packed_copy[_simd_width: Int](idx: Int):
                var val = input_b_fn[_simd_width](idx, k)
                output_ptr.store(idx, val)

            tile[packed_copy, Self._matmul_config.pack_sizes](0, N)

            if aligned_n != N:
                memset_zero(output_ptr + N, aligned_n - N)

            output_ptr += aligned_n

    @no_inline
    @staticmethod
    fn _gemv_transposed[
        input_b_fn: Self._input_fn_type, static_k: Int
    ](
        N: Int,
        dynamic_k: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype]],
        c_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        var K = static_k if static_k != UNKNOWN_VALUE else dynamic_k
        var cn_ptr = c_ptr

        @parameter
        @always_inline
        fn process_cols[tile_n: Int](n: Int):
            @parameter
            @always_inline
            fn do_reduce[
                _simd_width: Int
            ](
                start: Int,
                end: Int,
                mut accum: InlineArray[SIMD[Self.dtype, _simd_width], tile_n],
            ):
                for k in range(start, end, _simd_width):
                    var a_data = a_ptr.load[width=_simd_width](k)

                    comptime for nn in range(tile_n):
                        var b_data = input_b_fn[_simd_width](n + nn, k)
                        accum[nn] = b_data.fma(a_data, accum[nn])

            @parameter
            @always_inline
            fn do_reduce_accum[
                target_width: Int, _simd_width: Int
            ](
                accum: InlineArray[SIMD[Self.dtype, _simd_width], tile_n]
            ) -> InlineArray[SIMD[Self.dtype, target_width], tile_n]:
                var accum_reduce = InlineArray[
                    SIMD[Self.dtype, target_width], tile_n
                ](fill=0)

                comptime for nn in range(tile_n):
                    accum_reduce[nn] = accum[nn].reduce_add[target_width]()
                return accum_reduce^

            comptime unroll_factor = 2
            comptime unroll_simd_width = Self.simd_width * unroll_factor

            var unroll_loop_end = align_down(K, unroll_simd_width)
            var unroll_accum = InlineArray[
                SIMD[Self.dtype, unroll_simd_width], tile_n
            ](fill=0)
            do_reduce(0, unroll_loop_end, unroll_accum)

            var simd_loop_end = align_down(K, Self.simd_width)
            var simd_accum = do_reduce_accum[Self.simd_width](unroll_accum)
            do_reduce(unroll_loop_end, simd_loop_end, simd_accum)

            var scalar_accum = do_reduce_accum[1](simd_accum)
            do_reduce(simd_loop_end, K, scalar_accum)

            comptime for nn in range(tile_n):
                cn_ptr.store(nn, scalar_accum[nn])

            cn_ptr += tile_n

        tile[process_cols, VariadicList[Int](4, 1)](0, N)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = K
        _ = cn_ptr

    @no_inline
    @staticmethod
    fn _gemv[
        input_b_fn: Self._input_fn_type
    ](
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype]],
        c_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        accumulate: Bool = False,
    ):
        var cn_ptr = c_ptr

        @parameter
        @always_inline
        fn process_cols[_simd_width: Int](n: Int):
            var accum = SIMD[Self.dtype, _simd_width]()

            for k in range(K):
                var b_data = input_b_fn[_simd_width](n, k)
                accum = b_data.fma(a_ptr[k], accum)

            if accumulate:
                accum += cn_ptr.load[width=_simd_width]()

            cn_ptr.store(accum)
            cn_ptr += _simd_width

        tile[process_cols, Self._matmul_config.gemv_sizes](0, N)
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = cn_ptr

    @no_inline
    @staticmethod
    fn _matmul[
        input_b_fn: Self._input_fn_type,
        *,
        transpose_b: Bool = False,
        static_k: Int = UNKNOWN_VALUE,
    ](
        M: Int,
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[Self.dtype]],
        a_stride: Int,
        packed_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        c_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        c_stride: Int,
        accumulate: Bool = False,
    ) raises:
        if M == 1:
            comptime if transpose_b:
                # Transpose is implemented for the K tensor and accumulation
                # is used with the V tensor, so simplify the implementation by
                # falling back to the general path.
                if not accumulate:
                    return Self._gemv_transposed[input_b_fn, static_k](
                        N, K, a_ptr, c_ptr
                    )
            else:
                return Self._gemv[input_b_fn](
                    N, K, a_ptr, c_ptr, accumulate=accumulate
                )

        comptime if transpose_b:
            Self._pack_buffer_transposed[input_b_fn, static_k](packed_ptr, N, K)
        else:
            Self._pack_buffer[input_b_fn](packed_ptr, N, K)

        comptime if use_apple_accelerate_lib[
            Self.dtype, Self.dtype, Self.dtype
        ]():
            return _cblas_f32(
                Int32(M),
                Int32(N),
                Int32(K),
                Int32(a_stride),
                Int32(align_up(N, Self.simd_width)),
                Int32(c_stride),
                Float32(1.0),
                Float32(1.0) if accumulate else Float32(0.0),
                rebind[UnsafePointer[Float32, MutAnyOrigin]](c_ptr),
                rebind[UnsafePointer[Float32, ImmutAnyOrigin]](a_ptr),
                rebind[UnsafePointer[Float32, MutAnyOrigin]](packed_ptr),
            )

        Self._matmul_packed(
            M,
            align_up(N, Self.simd_width),
            K,
            a_ptr,
            a_stride,
            packed_ptr,
            c_ptr,
            c_stride,
            accumulate=accumulate,
        )


struct _FlashAttentionConfig[
    dtype: DType,
    rank: Int,
    simd_width: Int,
    output_static_shape: IndexList[rank],
](Defaultable):
    var block_m: Int
    var qk_block_n: Int
    var o_block_n: Int

    fn __init__(out self):
        self.qk_block_n = 128
        self.o_block_n = 128

        # Set a target size for the output block array.
        comptime output_target_size = 8192

        comptime depth_static_dim = Self.output_static_shape[Self.rank - 1]

        comptime if depth_static_dim != UNKNOWN_VALUE:
            # Extract the static depth dimension with a guard against zero.
            var depth_dim = max(depth_static_dim, 1)

            # Compute the number of columns for the output block array. If the
            # count is too large, then use the default size.
            self.o_block_n = align_up(
                depth_dim if depth_dim <= 256 else self.o_block_n,
                Self.simd_width,
            )

        # Compute the number of rows per iteration, but constrain this number
        # as other buffers are allocated to this size too.
        self.block_m = align_down(output_target_size // self.o_block_n, 4)
        self.block_m = min(max(self.block_m, 1), 64)


struct _FlashAttention[
    dtype: DType,
    rank: Int,
    //,
    input_q_ptr_fn: fn(IndexList[rank]) capturing -> UnsafePointer[
        Scalar[dtype], ImmutAnyOrigin
    ],
    input_k_fn: fn[simd_width: Int, rank: Int](
        idx: IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int](
        idx: IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    mask_fn: fn[simd_width: Int, mask_rank: Int](
        idx: IndexList[mask_rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[dtype, simd_width],
    mask_rank: Int,
    output_ptr_fn: fn(IndexList[rank]) capturing -> UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ],
    q_length_fn: fn(batch: Int) capturing -> Int,
    kv_length_fn: fn(batch: Int) capturing -> Int,
    kv_cache_length_fn: fn(batch: Int) capturing -> Int,
    padded_output_shape: IndexList[rank],
    *,
    simd_width: Int = simd_width_of[dtype](),
]:
    comptime _matmul = _Matmul[Self.dtype, Self.simd_width]
    comptime _config = _FlashAttentionConfig[
        Self.dtype, Self.rank, Self.simd_width, Self.padded_output_shape
    ]()
    comptime _depth_static_dim = Self.padded_output_shape[Self.rank - 1]

    @staticmethod
    fn _online_softmax[
        _mask_fn: fn[simd_width: Int](
            m: Int, n: Int, score_vec: SIMD[Self.dtype, simd_width]
        ) capturing -> SIMD[Self.dtype, simd_width],
    ](
        qk_block_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        o_block_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        max_vals: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        sum_vals: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        count_m: Int,
        count_n: Int,
        kv_seq_cnt: Int,
        scale: Float32,
        sink_weight: Optional[Scalar[Self.dtype]] = None,
    ):
        comptime assert (
            Self.dtype.is_floating_point()
        ), "dtype must be floating point"
        var qk_row_ptr = qk_block_ptr
        var o_row_ptr = o_block_ptr

        var sink_logit: Scalar[Self.dtype] = 0
        var do_sink = sink_weight is not None
        if do_sink:
            sink_logit = sink_weight.value()

        comptime layout_1d = Layout.row_major(UNKNOWN_VALUE)
        for m in range(count_m):
            var qk_row = LayoutTensor[Self.dtype, layout_1d](
                qk_row_ptr,
                RuntimeLayout[layout_1d].row_major(IndexList[1](kv_seq_cnt)),
            )

            @parameter
            @always_inline
            fn pass1_input_gen_fn[
                _dtype: DType, _simd_width: Int
            ](idx: Int) -> SIMD[_dtype, _simd_width]:
                var val = qk_row_ptr.load[width=_simd_width](idx)
                return _mask_fn(m, idx, val * scale.cast[Self.dtype]()).cast[
                    _dtype
                ]()

            @always_inline
            @parameter
            fn output_fn[
                _dtype: DType, width: Int, rank: Int
            ](idx: Int, val: SIMD[_dtype, width]):
                qk_row.store(
                    IndexList[1](idx), rebind[SIMD[Self.dtype, width]](val)
                )

            # Update the row with the scale and mask. Find the maximum value
            # of the row to bias the exponential function below for numeric
            # stability.
            var max_val = map_reduce[
                Self.simd_width,
                Self.dtype,
                Self.dtype,
                origin_of()._mlir_origin,
                pass1_input_gen_fn,
                origin_of()._mlir_origin,
                _simd_max_elementwise,
                _simd_max,
                output_fn,
            ](qk_row.size(), max_vals[m])

            if do_sink:
                max_val = max(max_val, sink_logit)

            @parameter
            @always_inline
            fn pass2_input_gen_fn[
                _dtype: DType, _simd_width: Int
            ](idx: Int) -> SIMD[_dtype, _simd_width]:
                var val = qk_row_ptr.load[width=_simd_width](idx)
                return rebind[SIMD[_dtype, _simd_width]](exp(val - max_val))

            # Update the row with the exponential of each value and accumulate
            # the result.
            var accum_val = map_reduce[
                Self.simd_width,
                Self.dtype,
                Self.dtype,
                origin_of()._mlir_origin,
                pass2_input_gen_fn,
                origin_of()._mlir_origin,
                _simd_sum_elementwise,
                _simd_sum,
                output_fn,
            ](qk_row.size(), 0)

            if do_sink:
                accum_val += exp(sink_logit - max_val)

            var fixup_val = exp(max_vals[m] - max_val)

            # Update the running maximum and sum for the row.
            max_vals[m] = max_val
            sum_vals[m] = sum_vals[m] * fixup_val + accum_val

            @always_inline
            fn do_correction[_simd_width: Int](idx: Int) unified {mut}:
                var val = o_row_ptr.load[width=_simd_width](idx)
                o_row_ptr.store(idx, val * fixup_val)

            vectorize[Self.simd_width, unroll_factor=2](count_n, do_correction)

            qk_row_ptr += Self._config.qk_block_n
            o_row_ptr += Self._config.o_block_n

    @staticmethod
    fn run(
        num_batches: Int,
        num_heads: Int,
        depth_dim: Int,
        num_kv_heads: Int,
        # Max sequence length of query states.
        max_seq_len: Int,
        scale: Float32,
        sink_weights: OptionalReg[
            LayoutTensor[
                Self.dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ]
        ] = None,
    ):
        var kv_group_count = num_heads // num_kv_heads

        # Compute the maximum size in elements for the common packed buffer.
        var packed_qk_size = Self._config.qk_block_n * depth_dim
        var packed_o_size = Self._config.o_block_n * Self._config.qk_block_n
        var packed_size = max(packed_qk_size, packed_o_size)

        var num_blocks_m = ceildiv(max_seq_len, Self._config.block_m)
        var num_blocks_n = ceildiv(depth_dim, Self._config.o_block_n)
        var work_count = num_batches * num_heads * num_blocks_m * num_blocks_n

        var num_threads = min(work_count, parallelism_level())

        @__copy_capture(
            num_threads,
            work_count,
            num_blocks_n,
            num_blocks_m,
            packed_size,
            kv_group_count,
            depth_dim,
            max_seq_len,
            num_heads,
            sink_weights,
        )
        @parameter
        fn task_func(task_id: Int):
            var qk_block_ptr = stack_allocation[
                Self._config.block_m * Self._config.qk_block_n,
                Self.dtype,
                alignment = align_of[SIMD[Self.dtype, Self.simd_width]](),
            ]()
            var o_block_ptr = stack_allocation[
                Self._config.block_m * Self._config.o_block_n,
                Self.dtype,
                alignment = align_of[SIMD[Self.dtype, Self.simd_width]](),
            ]()
            comptime layout = Layout.row_major(Self._config.block_m)
            var max_vals_stack = InlineArray[
                Scalar[Self.dtype], Self._config.block_m
            ](uninitialized=True)
            var max_vals = LayoutTensor[Self.dtype, layout](max_vals_stack)
            var sum_vals_stack = InlineArray[
                Scalar[Self.dtype], Self._config.block_m
            ](uninitialized=True)
            var sum_vals = LayoutTensor[Self.dtype, layout](sum_vals_stack)

            var packed_ptr = UnsafePointer[
                Scalar[Self.dtype], MutExternalOrigin
            ]()
            if max_seq_len != 1:
                packed_ptr = alloc[Scalar[Self.dtype]](
                    packed_size,
                    alignment=align_of[SIMD[Self.dtype, Self.simd_width]](),
                )

            var q_seq_stride = num_heads * depth_dim

            var block_range = partition_work(
                task_id, num_threads, work_count, 1
            )

            for i in range(block_range[0], block_range[0] + block_range[1]):
                var n = (i % num_blocks_n) * Self._config.o_block_n
                var j = i // num_blocks_n
                var m = (j % num_blocks_m) * Self._config.block_m
                var batch_head = j // num_blocks_m
                var head = batch_head % num_heads
                var batch = batch_head // num_heads
                var kv_head = head // kv_group_count
                var kv_cache_len = Self.kv_cache_length_fn(batch)
                var seq_len = Self.q_length_fn(batch)
                var kv_seq_len = kv_cache_len + Self.kv_length_fn(batch)

                # Exit early if there's no more work to do for this batch.
                if m >= seq_len:
                    continue

                @parameter
                @__copy_capture(batch, batch_head, kv_head, head)
                @always_inline
                fn get_nd_index[
                    is_kv: Bool = False
                ](x: Int, y: Int) -> IndexList[Self.rank]:
                    comptime if Self.rank == 4:
                        return IndexList[Self.rank](
                            batch, x, kv_head if is_kv else head, y
                        )
                    else:
                        return IndexList[Self.rank](batch, x, y)

                @parameter
                @__copy_capture(batch, head)
                @always_inline
                fn get_mask_nd_index(
                    x: Int, y: Int
                ) -> IndexList[Self.mask_rank]:
                    comptime if Self.mask_rank == 4:
                        return IndexList[Self.mask_rank](batch, head, x, y)
                    elif Self.mask_rank == 3:
                        return IndexList[Self.mask_rank](batch, x, y)
                    elif Self.mask_rank == 2:
                        return IndexList[Self.mask_rank](x, y)
                    else:
                        return IndexList[Self.mask_rank]()

                var count_m = min(Self._config.block_m, seq_len - m)
                var count_n = min(Self._config.o_block_n, depth_dim - n)

                var o_ptr = Self.output_ptr_fn(get_nd_index(m, n))
                var q_ptr = Self.input_q_ptr_fn(get_nd_index(m, 0))

                _ = max_vals.fill(Scalar[Self.dtype].MIN)
                _ = sum_vals.fill(0)

                for kv_seq_idx in range(0, kv_seq_len, Self._config.qk_block_n):
                    var kv_seq_cnt = min(
                        kv_seq_len - kv_seq_idx, Self._config.qk_block_n
                    )

                    @parameter
                    @always_inline
                    fn input_k_2d_fn[
                        _simd_width: Int
                    ](_n: Int, _k: Int) -> SIMD[Self.dtype, _simd_width]:
                        return Self.input_k_fn[_simd_width, Self.rank](
                            get_nd_index[is_kv=True](_n + kv_seq_idx, _k)
                        )

                    try:
                        Self._matmul._matmul[
                            input_k_2d_fn,
                            transpose_b=True,
                            static_k = Self._depth_static_dim,
                        ](
                            count_m,
                            kv_seq_cnt,
                            depth_dim,
                            q_ptr,
                            q_seq_stride,
                            packed_ptr,
                            qk_block_ptr,
                            Self._config.qk_block_n,
                        )
                    except e:
                        # This won't trigger in practice, but we want to keep
                        # this function non-raising.
                        abort(String(e))

                    @parameter
                    @always_inline
                    fn mask_2d_fn[
                        _simd_width: Int
                    ](
                        _m: Int,
                        _n: Int,
                        score_vec: SIMD[Self.dtype, _simd_width],
                    ) -> SIMD[Self.dtype, _simd_width]:
                        return Self.mask_fn[_simd_width, Self.mask_rank](
                            get_mask_nd_index(_m + m, _n + kv_seq_idx),
                            score_vec,
                            kv_cache_len,
                        )

                    var sink_weight: Optional[Scalar[Self.dtype]] = None
                    if sink_weights:
                        sink_weight = sink_weights.value()[head][0]

                    Self._online_softmax[mask_2d_fn](
                        qk_block_ptr,
                        o_block_ptr,
                        max_vals.ptr.as_unsafe_pointer(),
                        sum_vals.ptr.as_unsafe_pointer(),
                        count_m,
                        count_n,
                        kv_seq_cnt,
                        scale,
                        sink_weight,
                    )

                    @parameter
                    @always_inline
                    fn input_v_2d_fn[
                        _simd_width: Int
                    ](_n: Int, _k: Int) -> SIMD[Self.dtype, _simd_width]:
                        return Self.input_v_fn[_simd_width, Self.rank](
                            get_nd_index[is_kv=True](_k + kv_seq_idx, n + _n)
                        )

                    try:
                        Self._matmul._matmul[input_v_2d_fn](
                            count_m,
                            count_n,
                            kv_seq_cnt,
                            qk_block_ptr,
                            Self._config.qk_block_n,
                            packed_ptr,
                            o_block_ptr,
                            Self._config.o_block_n,
                            accumulate=(kv_seq_idx > 0),
                        )
                    except e:
                        abort(String(e))
                    _ = kv_seq_idx

                _ = m
                _ = n
                var oz_ptr = o_block_ptr

                for m in range(count_m):
                    var reciprocal = 1 / sum_vals[m][0]

                    @always_inline
                    fn do_final[_simd_width: Int](idx: Int) unified {mut}:
                        var v = oz_ptr.load[width=_simd_width](idx)
                        o_ptr.store(idx, v * reciprocal)

                    vectorize[Self.simd_width, unroll_factor=4](
                        count_n, do_final
                    )

                    o_ptr += q_seq_stride
                    oz_ptr += Self._config.o_block_n

            if packed_ptr:
                packed_ptr.free()

        sync_parallelize[task_func](num_threads)


@always_inline
fn _flash_attention[
    dtype: DType,
    rank: Int,
    mask_rank: Int,
    //,
    input_k_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_mask_fn: fn[simd_width: Int, mask_rank: Int](
        IndexList[mask_rank]
    ) capturing -> SIMD[dtype, simd_width],
](
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    mask_shape: IndexList[mask_rank],
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    var num_batches = output.dim[0]()
    var max_seq_len = output.dim[1]()
    var num_heads = output.dim[rank - 2]() if rank == 4 else 1
    var depth_dim = output.dim[rank - 1]()
    var kv_cache_len = v_shape[1] - max_seq_len
    var num_kv_heads = k_shape[rank - 2] if rank == 4 else 1

    @always_inline
    @parameter
    fn input_q_ptr_fn(
        coords: IndexList[rank],
    ) -> UnsafePointer[Scalar[dtype], ImmutAnyOrigin]:
        var idx = q._offset(coords)
        return (q.ptr + idx).as_unsafe_pointer()

    @always_inline
    @parameter
    fn output_ptr_fn(
        coords: IndexList[rank],
    ) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        var idx = output._offset(coords)
        return (output.ptr + idx).as_unsafe_pointer()

    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int, rank: Int
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        return score_vec + input_mask_fn[simd_width, rank](idx)

    @always_inline
    @__copy_capture(kv_cache_len)
    @parameter
    fn kv_cache_length_fn(batch: Int) -> Int:
        return kv_cache_len

    @always_inline
    @__copy_capture(max_seq_len)
    @parameter
    fn q_length_fn(batch: Int) -> Int:
        return max_seq_len

    _FlashAttention[
        input_q_ptr_fn,
        input_k_fn,
        input_v_fn,
        mask_fn,
        mask_rank,
        output_ptr_fn,
        q_length_fn,
        # Use the `q_length_fn` also for the KV length for now.
        # Note that this is only correct for self attention and is broken for
        # cross attention, which has different KV lengths.
        q_length_fn,
        kv_cache_length_fn,
        rebind[IndexList[rank]](
            to_index_list[output.rank](output.layout.shape)
        ),
    ].run(
        num_batches,
        num_heads,
        depth_dim,
        num_kv_heads,
        max_seq_len,
        scale,
        sink_weights,
    )


fn flash_attention[
    dtype: DType,
    rank: Int,
    mask_rank: Int,
    //,
    input_k_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_mask_fn: fn[simd_width: Int, mask_rank: Int](
        IndexList[mask_rank]
    ) capturing -> SIMD[dtype, simd_width],
](
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    mask_shape: IndexList[mask_rank],
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    _flash_attention[input_k_fn, input_v_fn, input_mask_fn](
        q,
        k_shape,
        v_shape,
        mask_shape,
        output,
        scale,
        sink_weights,
    )


fn flash_attention_split_kv[
    dtype: DType,
    rank: Int,
    mask_rank: Int,
    //,
    input_k_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_k_cache_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_v_cache_fn: fn[simd_width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, simd_width],
    input_mask_fn: fn[simd_width: Int, mask_rank: Int](
        IndexList[mask_rank]
    ) capturing -> SIMD[dtype, simd_width],
](
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    # {k,v}_cache_shape are rank + 1 because reshape in MO IR prevents fusion.
    k_cache_shape: IndexList[rank + 1],
    v_cache_shape: IndexList[rank + 1],
    mask_shape: IndexList[mask_rank],
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
) raises:
    """Variant of flash attention that takes the previous KV cache
    `input_{k,v}_cache_fn` and the current KV tensors `input_k_fn` and
    `input_v_fn` as separate arguments.

    This works around the fact that fusion can't currently look through concat.
    So this kernel does an in-place concat fusion by changing the input lambdas
    `input_{k,v}_cache_fn_wrapper` to take previous sequence KV elements from
    the KV cache, and current KV elements from tensors `k` and `v`.
    """
    # This expects the following layouts:
    # q: BSHD
    # k (input_k_fn): BSHD
    # v (input_v_fn): BSHD
    # k_cache (input_k_cache_fn): 1BHS'D
    # v_cache (input_v_cache_fn): 1BHS'D
    comptime assert rank == 4

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg("k", k_shape),
                    trace_arg("v", v_shape),
                    trace_arg("k_cache", k_cache_shape),
                    trace_arg("v_cache", v_cache_shape),
                    trace_arg("output", output.runtime_layout.shape.value),
                ]
            )
        )

    with Trace[TraceLevel.OP, target = StaticString("cpu")](
        "flash_attention_split_kv",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        comptime kv_rank = rank + 1

        var kv_cache_len = v_cache_shape[3]

        @always_inline
        @parameter
        fn kv_index[rank: Int](idx: IndexList[rank]) -> IndexList[kv_rank]:
            # Index into the previous kv_cache by unsqueezing dim 0.
            return IndexList[kv_rank](0, idx[0], idx[2], idx[1], idx[3])

        @always_inline
        @__copy_capture(kv_cache_len)
        @parameter
        fn load_from_split_cache[
            curr_fn: fn[simd_width: Int, rank: Int](
                IndexList[rank]
            ) capturing -> SIMD[dtype, simd_width],
            cache_fn: fn[simd_width: Int, rank: Int](
                IndexList[rank]
            ) capturing -> SIMD[dtype, simd_width],
            rank: Int,
            simd_width: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
            # Load directly from either `curr_fn` or `cache_fn` depending on the
            # sequence index.
            # Boundary condition handling is done by the caller since
            # the last dim `depth_dim` is contiguous.
            var seq_idx = idx[1]

            if seq_idx >= kv_cache_len:
                return curr_fn[simd_width, rank](
                    IndexList[rank](
                        idx[0], seq_idx - kv_cache_len, idx[2], idx[3]
                    )
                )

            return cache_fn[simd_width, kv_rank](kv_index(idx))

        @always_inline
        @parameter
        fn input_k_cache_fn_wrapper[
            simd_width: Int,
            rank: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
            return load_from_split_cache[
                input_k_fn, input_k_cache_fn, rank, simd_width
            ](idx)

        @always_inline
        @parameter
        fn input_v_cache_fn_wrapper[
            simd_width: Int,
            rank: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
            return load_from_split_cache[
                input_v_fn, input_v_cache_fn, rank, simd_width
            ](idx)

        var combined_k_shape = IndexList[rank](
            k_shape[0], k_shape[1] + k_cache_shape[3], k_shape[2], k_shape[3]
        )
        var combined_v_shape = IndexList[rank](
            v_shape[0], v_shape[1] + v_cache_shape[3], v_shape[2], v_shape[3]
        )
        _flash_attention[
            input_k_cache_fn_wrapper,
            input_v_cache_fn_wrapper,
            input_mask_fn,
        ](
            q,
            combined_k_shape,
            combined_v_shape,
            mask_shape,
            output,
            scale,
        )


@always_inline
fn _flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    //,
    mask_fn: fn[simd_width: Int, mask_rank: Int](
        idx: IndexList[mask_rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[dtype, simd_width],
    mask_rank: Int,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    comptime kv_params = cache_t.kv_params

    var max_seq_len = q.dim[1]()
    var num_batches = q.dim[0]()
    comptime num_heads = Int(q.layout.shape[2])
    comptime head_size = cache_t.kv_params.head_size
    comptime output_shape = IndexList[4](
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, Int(head_size)
    )

    @always_inline
    @parameter
    fn input_q_ptr_fn(
        coords: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], ImmutAnyOrigin]:
        var idx = q._offset(coords)
        return (q.ptr + idx).as_unsafe_pointer()

    @always_inline
    @parameter
    fn output_ptr_fn(
        coords: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        var idx = output._offset(coords)
        return (output.ptr + idx).as_unsafe_pointer()

    @always_inline
    @__copy_capture(max_seq_len)
    @parameter
    fn q_length_fn(batch: Int) -> Int:
        return max_seq_len

    return _flash_attention_kv_cache[
        input_q_ptr_fn,
        output_ptr_fn,
        q_length_fn,
        # NOTE: kv_length_fn = q_length_fn is only correct for self attention.
        kv_length_fn=q_length_fn,
        mask_fn=mask_fn,
        mask_rank=mask_rank,
        output_shape=output_shape,
    ](k, v, num_batches, num_heads, max_seq_len, scale, sink_weights)


@always_inline
fn _flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    //,
    input_q_ptr_fn: fn(IndexList[4]) capturing -> UnsafePointer[
        Scalar[dtype], ImmutAnyOrigin
    ],
    output_ptr_fn: fn(IndexList[4]) capturing -> UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ],
    q_length_fn: fn(batch: Int) capturing -> Int,
    kv_length_fn: fn(batch: Int) capturing -> Int,
    mask_fn: fn[simd_width: Int, mask_rank: Int](
        idx: IndexList[mask_rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[dtype, simd_width],
    mask_rank: Int,
    output_shape: IndexList[4],
](
    k: cache_t,
    v: cache_t,
    num_batches: Int,
    num_heads: Int,
    max_seq_len: Int,
    scale: Float32,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    comptime num_kv_heads = cache_t.kv_params.num_heads
    comptime depth_dim = cache_t.kv_params.head_size
    comptime cache_type = cache_t.dtype

    comptime assert cache_type == dtype, (
        "Expected cache dtype ("
        + String(cache_type)
        + ") to match input dtype ("
        + String(dtype)
        + ")"
    )

    @parameter
    fn input_k_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        # Unwrap BSHD->BHSD indices.
        return rebind[SIMD[dtype, width]](
            k.load[width=width](idx[0], idx[2], idx[1], idx[3])
        )

    @parameter
    fn input_v_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        # Unwrap BSHD->BHSD indices.
        return rebind[SIMD[dtype, width]](
            v.load[width=width](idx[0], idx[2], idx[1], idx[3])
        )

    @always_inline
    @parameter
    fn kv_cache_length_fn(batch: Int) -> Int:
        return k.cache_length(batch)

    _FlashAttention[
        input_q_ptr_fn,
        input_k_fn,
        input_v_fn,
        mask_fn,
        mask_rank,
        output_ptr_fn,
        q_length_fn,
        kv_length_fn,
        kv_cache_length_fn,
        output_shape,
    ].run(
        num_batches,
        num_heads,
        Int(depth_dim),
        Int(num_kv_heads),
        max_seq_len,
        scale,
        sink_weights,
    )


fn flash_attention_kv_cache[
    dtype: DType, cache_t: KVCacheT, //
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    mask: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int, rank: Int
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        return score_vec + mask.load[width=simd_width](idx)

    _flash_attention_kv_cache[mask_fn, mask.rank](
        q, k, v, scale, output, sink_weights
    )


fn flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    //,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    mask: mask_t,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
):
    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int,
        rank: Int,
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        # Shift the mask index from local->global space.
        return mask.mask(
            Index(idx[0], idx[1], idx[2] + kv_cache_len, idx[3]), score_vec
        )

    _flash_attention_kv_cache[mask_fn, 4](q, k, v, scale, output, sink_weights)


fn flash_attention_kv_cache[
    dtype: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    //,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q_input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    mask: mask_t,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
):
    """Entrypoint for ragged tensors."""

    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int,
        rank: Int,
    ](
        idx: IndexList[rank],
        score_vec: SIMD[dtype, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[dtype, simd_width]:
        # Shift the mask index from local->global space.
        return mask.mask(
            Index(idx[0], idx[1], idx[2] + kv_cache_len, idx[3]), score_vec
        )

    @always_inline
    @parameter
    fn q_length_fn(batch: Int) -> Int:
        return Int(q_input_row_offsets[batch + 1] - q_input_row_offsets[batch])

    @always_inline
    @parameter
    fn kv_length_fn(batch: Int) -> Int:
        return Int(
            kv_input_row_offsets[batch + 1] - kv_input_row_offsets[batch]
        )

    @always_inline
    @parameter
    fn input_q_ptr_fn(
        idx: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], ImmutAnyOrigin]:
        var bs = idx[0]
        var tok_idx = idx[1]
        var q_start = Int(q_input_row_offsets[bs]) + tok_idx
        var flat_idx = IndexList[3](q_start, idx[2], idx[3])
        var out_idx = q._offset(flat_idx)
        return (q.ptr + out_idx).as_unsafe_pointer()

    @always_inline
    @parameter
    fn output_ptr_fn(
        idx: IndexList[4],
    ) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        var bs = idx[0]
        var tok_idx = idx[1]
        var q_start = Int(q_input_row_offsets[bs]) + tok_idx
        var flat_idx = IndexList[3](q_start, idx[2], idx[3])
        var out_idx = output._offset(flat_idx)
        return (output.ptr + out_idx).as_unsafe_pointer()

    comptime mask_rank = 4
    var num_batches = q_input_row_offsets.dim[0]() - 1
    var max_seq_len = k.max_prompt_length()
    comptime num_heads = Int(q.layout.shape[q.rank - 2])
    comptime head_size = cache_t.kv_params.head_size
    comptime output_shape = IndexList[4](
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, Int(head_size)
    )

    _flash_attention_kv_cache[
        input_q_ptr_fn,
        output_ptr_fn,
        q_length_fn,
        kv_length_fn,
        mask_fn,
        mask_rank,
        output_shape,
    ](k, v, num_batches, num_heads, Int(max_seq_len), scale, sink_weights)
