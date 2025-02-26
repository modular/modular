# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray
from math import align_down, align_up, ceildiv, exp
from sys import alignof, has_avx512f, has_neon, simdwidthof

from algorithm import sync_parallelize, tile, vectorize
from algorithm.reduction import (
    _simd_max,
    _simd_max_elementwise,
    _simd_sum,
    _simd_sum_elementwise,
    map_reduce,
)
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from kv_cache.types import KVCacheT
from linalg.accumulate import _Accumulator
from linalg.apple_accelerate import _cblas_f32, use_apple_accelerate_lib
from linalg.transpose import transpose_inplace
from linalg.utils import partition_work
from memory import UnsafePointer, memset_zero, stack_allocation
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
        mut self,
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
        @parameter
        if has_neon():
            return _MatmulConfig(
                col_sizes=VariadicList[Int](4, 3, 2, 1),
                row_sizes=VariadicList[Int](6, 4, 1),
                gemv_sizes=VariadicList[Int](32, 4, 1),
                pack_sizes=VariadicList[Int](32, 8, 4, 1),
            )
        elif has_avx512f():
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


struct _Matmul[
    type: DType,
    simd_width: Int,
]:
    alias _matmul_config = _MatmulConfig._get_config()

    alias _input_fn_type = fn[simd_width: Int] (
        x: Int, y: Int
    ) capturing -> SIMD[type, simd_width]

    @staticmethod
    @always_inline
    fn _inner_loop_a_lane[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: UnsafePointer[Scalar[type]],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[type]],
        b_stride: Int,
        mut c_tile: _Accumulator[type, tile_m, tile_n, simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @parameter
        @always_inline
        fn loop_body[lane_count: Int](k: Int):
            var a_tile = InlineArray[SIMD[type, lane_count], tile_m](0)

            @parameter
            for m in range(tile_m):
                a_tile[m] = ak_ptr.load[width=lane_count](m * a_stride)

            ak_ptr += lane_count

            @parameter
            for k in range(lane_count):

                @parameter
                for n in range(tile_n):
                    var b_data = bk_ptr.load[width=simd_width](n * simd_width)

                    @parameter
                    for m in range(tile_m):
                        c_tile.fma(m, n, a_tile[m][k], b_data)

                bk_ptr += b_stride

        tile[loop_body, VariadicList[Int](simd_width, 1)](0, K)

    @staticmethod
    @always_inline
    fn _inner_loop_a_broadcast[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: UnsafePointer[Scalar[type]],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[type]],
        b_stride: Int,
        mut c_tile: _Accumulator[type, tile_m, tile_n, simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @parameter
        @always_inline
        fn loop_body[unroll_factor: Int](k: Int):
            var b_tile = InlineArray[SIMD[type, simd_width], tile_n](0)

            @parameter
            for k in range(unroll_factor):

                @parameter
                for n in range(tile_n):
                    b_tile[n] = bk_ptr.load[width=simd_width](n * simd_width)

                @parameter
                for m in range(tile_m):
                    var a_data = ak_ptr.load(m * a_stride)

                    @parameter
                    for n in range(tile_n):
                        c_tile.fma(m, n, a_data, b_tile[n])

                ak_ptr += 1
                bk_ptr += b_stride

        tile[loop_body, VariadicList[Int](2, 1)](0, K)

    @no_inline
    @staticmethod
    fn _matmul_packed(
        M: Int,
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[type]],
        a_stride: Int,
        b_ptr: UnsafePointer[Scalar[type]],
        c_ptr: UnsafePointer[Scalar[type]],
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
                var c_tile = _Accumulator[type, tile_m, tile_n, simd_width]()

                if accumulate:
                    c_tile.load(cn_ptr, c_stride)
                else:
                    c_tile.init(0.0)

                @parameter
                if has_neon():
                    Self._inner_loop_a_lane(
                        K, am_ptr, a_stride, bn_ptr, N, c_tile
                    )
                else:
                    Self._inner_loop_a_broadcast(
                        K, am_ptr, a_stride, bn_ptr, N, c_tile
                    )

                c_tile.store(cn_ptr, c_stride)

                bn_ptr += tile_n * simd_width
                cn_ptr += tile_n * simd_width

            tile[process_cols, Self._matmul_config.col_sizes](
                0, ceildiv(N, simd_width)
            )

            am_ptr += tile_m * a_stride
            cm_ptr += tile_m * c_stride

        tile[process_rows, Self._matmul_config.row_sizes](0, M)

    @no_inline
    @staticmethod
    fn _pack_buffer_transposed[
        input_b_fn: Self._input_fn_type, static_k: Dim
    ](packed_ptr: UnsafePointer[Scalar[type]], N: Int, dynamic_k: Int):
        var K = Int(static_k) if static_k else dynamic_k

        var aligned_n = align_up(N, simd_width)

        # Use a conservative SIMD width for transposing. Using a wider native
        # SIMD width has not been observed to improve performance and causes
        # code size to unnecessarily increase.
        alias transpose_width = 4
        alias tile_sizes = VariadicList[Int](transpose_width, 1)

        var transpose_buffer = NDBuffer[
            type, 2, DimList(transpose_width, transpose_width)
        ].stack_allocation()

        @parameter
        @always_inline
        fn process_tile[tile_n: Int, tile_k: Int](n: Int, k: Int):
            @parameter
            if transpose_width == tile_n == tile_k:
                # Use an optimized path to transpose a square tile of the
                # input tensor.
                @parameter
                for i in range(transpose_width):
                    var val = input_b_fn[simd_width=transpose_width](n + i, k)
                    transpose_buffer.store(Index(i, 0), val)

                transpose_inplace[4, 4](transpose_buffer)

                @parameter
                for i in range(transpose_width):
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
                @parameter
                for nn in range(tile_n):
                    var val = input_b_fn[simd_width=tile_k](n + nn, k)

                    @parameter
                    for kk in range(tile_k):
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
    ](packed_ptr: UnsafePointer[Scalar[type]], N: Int, K: Int):
        var output_ptr = packed_ptr
        var aligned_n = align_up(N, simd_width)

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
        input_b_fn: Self._input_fn_type, static_k: Dim
    ](
        N: Int,
        dynamic_k: Int,
        a_ptr: UnsafePointer[Scalar[type]],
        c_ptr: UnsafePointer[Scalar[type]],
    ):
        var K = Int(static_k) if static_k else dynamic_k

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
                mut accum: InlineArray[SIMD[type, _simd_width], tile_n],
            ):
                for k in range(start, end, _simd_width):
                    var a_data = a_ptr.load[width=_simd_width](k)

                    @parameter
                    for nn in range(tile_n):
                        var b_data = input_b_fn[_simd_width](n + nn, k)
                        accum[nn] = b_data.fma(a_data, accum[nn])

            @parameter
            @always_inline
            fn do_reduce_accum[
                target_width: Int, _simd_width: Int
            ](
                accum: InlineArray[SIMD[type, _simd_width], tile_n]
            ) -> InlineArray[SIMD[type, target_width], tile_n]:
                var accum_reduce = InlineArray[
                    SIMD[type, target_width], tile_n
                ](0)

                @parameter
                for nn in range(tile_n):
                    accum_reduce[nn] = accum[nn].reduce_add[target_width]()
                return accum_reduce

            alias unroll_factor = 2
            alias unroll_simd_width = simd_width * unroll_factor

            var unroll_loop_end = align_down(K, unroll_simd_width)
            var unroll_accum = InlineArray[
                SIMD[type, unroll_simd_width], tile_n
            ](0)
            do_reduce(0, unroll_loop_end, unroll_accum)

            var simd_loop_end = align_down(K, simd_width)
            var simd_accum = do_reduce_accum[simd_width](unroll_accum)
            do_reduce(unroll_loop_end, simd_loop_end, simd_accum)

            var scalar_accum = do_reduce_accum[1](simd_accum)
            do_reduce(simd_loop_end, K, scalar_accum)

            @parameter
            for nn in range(tile_n):
                cn_ptr.store(nn, scalar_accum[nn])

            cn_ptr += tile_n

        tile[process_cols, VariadicList[Int](4, 1)](0, N)

    @no_inline
    @staticmethod
    fn _gemv[
        input_b_fn: Self._input_fn_type
    ](
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[type]],
        c_ptr: UnsafePointer[Scalar[type]],
        accumulate: Bool = False,
    ):
        var cn_ptr = c_ptr

        @parameter
        @always_inline
        fn process_cols[_simd_width: Int](n: Int):
            var accum = SIMD[type, _simd_width]()

            for k in range(K):
                var b_data = input_b_fn[_simd_width](n, k)
                accum = b_data.fma(a_ptr[k], accum)

            if accumulate:
                accum += cn_ptr.load[width=_simd_width]()

            cn_ptr.store(accum)
            cn_ptr += _simd_width

        tile[process_cols, Self._matmul_config.gemv_sizes](0, N)

    @no_inline
    @staticmethod
    fn _matmul[
        input_b_fn: Self._input_fn_type,
        *,
        transpose_b: Bool = False,
        static_k: Dim = Dim(),
    ](
        M: Int,
        N: Int,
        K: Int,
        a_ptr: UnsafePointer[Scalar[type]],
        a_stride: Int,
        packed_ptr: UnsafePointer[Scalar[type]],
        c_ptr: UnsafePointer[Scalar[type]],
        c_stride: Int,
        accumulate: Bool = False,
    ):
        if M == 1:

            @parameter
            if transpose_b:
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

        @parameter
        if transpose_b:
            Self._pack_buffer_transposed[input_b_fn, static_k](packed_ptr, N, K)
        else:
            Self._pack_buffer[input_b_fn](packed_ptr, N, K)

        @parameter
        if use_apple_accelerate_lib[type, type, type]():
            return _cblas_f32(
                M,
                N,
                K,
                a_stride,
                align_up(N, simd_width),
                c_stride,
                Float32(1.0),
                Float32(1.0) if accumulate else Float32(0.0),
                rebind[UnsafePointer[Float32]](c_ptr),
                rebind[UnsafePointer[Float32]](a_ptr),
                rebind[UnsafePointer[Float32]](packed_ptr),
            )

        Self._matmul_packed(
            M,
            align_up(N, simd_width),
            K,
            a_ptr,
            a_stride,
            packed_ptr,
            c_ptr,
            c_stride,
            accumulate=accumulate,
        )


struct _FlashAttentionConfig[
    type: DType,
    rank: Int,
    simd_width: Int,
    output_static_shape: DimList,
]:
    var block_m: Int
    var qk_block_n: Int
    var o_block_n: Int

    fn __init__(out self):
        self.qk_block_n = 128
        self.o_block_n = 128

        # Set a target size for the output block array.
        alias output_target_size = 8192

        alias depth_static_dim = output_static_shape.at[rank - 1]()

        @parameter
        if depth_static_dim:
            # Extract the static depth dimension with a guard against zero.
            var depth_dim = max(Int(depth_static_dim), 1)

            # Compute the number of columns for the output block array. If the
            # count is too large, then use the default size.
            self.o_block_n = align_up(
                depth_dim if depth_dim <= 256 else self.o_block_n, simd_width
            )

        # Compute the number of rows per iteration, but constrain this number
        # as other buffers are allocated to this size too.
        self.block_m = align_down(output_target_size // self.o_block_n, 4)
        self.block_m = min(max(self.block_m, 1), 64)


struct _FlashAttention[
    type: DType,
    rank: Int, //,
    input_q_ptr_fn: fn (IndexList[rank]) capturing -> UnsafePointer[
        Scalar[type]
    ],
    input_k_fn: fn[simd_width: Int, rank: Int] (
        idx: IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int] (
        idx: IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    mask_fn: fn[simd_width: Int, mask_rank: Int] (
        idx: IndexList[mask_rank],
        score_vec: SIMD[type, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[type, simd_width],
    mask_rank: Int,
    output_ptr_fn: fn (IndexList[rank]) capturing -> UnsafePointer[
        Scalar[type]
    ],
    q_length_fn: fn (batch: Int) capturing -> Int,
    kv_length_fn: fn (batch: Int) capturing -> Int,
    kv_cache_length_fn: fn (batch: Int) capturing -> Int,
    padded_output_shape: DimList,
    *,
    simd_width: Int = simdwidthof[type](),
]:
    alias _matmul = _Matmul[type, simd_width]
    alias _config = _FlashAttentionConfig[
        type, rank, simd_width, padded_output_shape
    ]()
    alias _depth_static_dim = padded_output_shape.at[rank - 1]()

    @staticmethod
    fn _online_softmax[
        mask_fn: fn[simd_width: Int] (
            m: Int, n: Int, score_vec: SIMD[type, simd_width]
        ) capturing -> SIMD[type, simd_width],
    ](
        qk_block_ptr: UnsafePointer[Scalar[type]],
        o_block_ptr: UnsafePointer[Scalar[type]],
        max_vals: UnsafePointer[Scalar[type]],
        sum_vals: UnsafePointer[Scalar[type]],
        count_m: Int,
        count_n: Int,
        kv_seq_cnt: Int,
        scale: Float32,
    ):
        var qk_row_ptr = qk_block_ptr
        var o_row_ptr = o_block_ptr

        for m in range(count_m):
            var qk_row = NDBuffer[type, 1](qk_row_ptr, kv_seq_cnt)

            @parameter
            @always_inline
            fn pass1_input_gen_fn[
                _type: DType, _simd_width: Int
            ](idx: Int) -> SIMD[_type, _simd_width]:
                var val = qk_row_ptr.load[width=_simd_width](idx)
                return mask_fn(m, idx, val * scale.cast[type]()).cast[_type]()

            # Update the row with the scale and mask. Find the maximum value
            # of the row to bias the exponential function below for numeric
            # stability.
            var max_val = map_reduce[
                simd_width,
                Dim(),
                type,
                type,
                __origin_of(),
                pass1_input_gen_fn,
                __origin_of(),
                _simd_max_elementwise,
                _simd_max,
            ](qk_row, max_vals[m])

            @parameter
            @always_inline
            fn pass2_input_gen_fn[
                _type: DType, _simd_width: Int
            ](idx: Int) -> SIMD[_type, _simd_width]:
                var val = qk_row_ptr.load[width=_simd_width](idx)
                return rebind[SIMD[_type, _simd_width]](exp(val - max_val))

            # Update the row with the exponential of each value and accumulate
            # the result.
            var accum_val = map_reduce[
                simd_width,
                Dim(),
                type,
                type,
                __origin_of(),
                pass2_input_gen_fn,
                __origin_of(),
                _simd_sum_elementwise,
                _simd_sum,
            ](qk_row, 0)

            var fixup_val = exp(max_vals[m] - max_val)

            # Update the running maximum and sum for the row.
            max_vals[m] = max_val
            sum_vals[m] = sum_vals[m] * fixup_val + accum_val

            @parameter
            @always_inline
            fn do_correction[_simd_width: Int](idx: Int):
                var val = o_row_ptr.load[width=_simd_width](idx)
                o_row_ptr.store(idx, val * fixup_val)

            vectorize[do_correction, simd_width, unroll_factor=2](count_n)

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
        )
        @parameter
        fn task_func(task_id: Int):
            var qk_block_ptr = stack_allocation[
                Self._config.block_m * Self._config.qk_block_n,
                type,
                alignment = alignof[SIMD[type, simd_width]](),
            ]()
            var o_block_ptr = stack_allocation[
                Self._config.block_m * Self._config.o_block_n,
                type,
                alignment = alignof[SIMD[type, simd_width]](),
            ]()
            var max_vals = NDBuffer[
                type, 1, Dim(Self._config.block_m)
            ]().stack_allocation()
            var sum_vals = NDBuffer[
                type, 1, Dim(Self._config.block_m)
            ]().stack_allocation()

            var packed_ptr = UnsafePointer[
                Scalar[type],
                alignment = alignof[SIMD[type, simd_width]](),
            ]()
            if max_seq_len != 1:
                packed_ptr = packed_ptr.alloc(packed_size)

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
                var kv_cache_len = kv_cache_length_fn(batch)
                var seq_len = q_length_fn(batch)
                var kv_seq_len = kv_cache_len + kv_length_fn(batch)

                # Exit early if there's no more work to do for this batch.
                if m >= seq_len:
                    continue

                @parameter
                @__copy_capture(batch, batch_head, kv_head, head)
                @always_inline
                fn get_nd_index[
                    is_kv: Bool = False
                ](x: Int, y: Int) -> IndexList[rank]:
                    @parameter
                    if rank == 4:
                        return IndexList[rank](
                            batch, x, kv_head if is_kv else head, y
                        )
                    else:
                        return IndexList[rank](batch, x, y)

                @parameter
                @__copy_capture(batch, head)
                @always_inline
                fn get_mask_nd_index(x: Int, y: Int) -> IndexList[mask_rank]:
                    @parameter
                    if mask_rank == 4:
                        return IndexList[mask_rank](batch, head, x, y)
                    elif mask_rank == 3:
                        return IndexList[mask_rank](batch, x, y)
                    elif mask_rank == 2:
                        return IndexList[mask_rank](x, y)
                    else:
                        return IndexList[mask_rank]()
                    constrained[False, "unsupported mask rank"]()

                var count_m = min(Self._config.block_m, seq_len - m)
                var count_n = min(Self._config.o_block_n, depth_dim - n)

                var o_ptr = output_ptr_fn(get_nd_index(m, n))
                var q_ptr = input_q_ptr_fn(get_nd_index(m, 0))

                max_vals.fill(Scalar[type].MIN)
                sum_vals.fill(0)

                for kv_seq_idx in range(0, kv_seq_len, Self._config.qk_block_n):
                    var kv_seq_cnt = min(
                        kv_seq_len - kv_seq_idx, Self._config.qk_block_n
                    )

                    @parameter
                    @always_inline
                    fn input_k_2d_fn[
                        _simd_width: Int
                    ](_n: Int, _k: Int) -> SIMD[type, _simd_width]:
                        return input_k_fn[_simd_width, rank](
                            get_nd_index[is_kv=True](_n + kv_seq_idx, _k)
                        )

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

                    @parameter
                    @always_inline
                    fn mask_2d_fn[
                        _simd_width: Int
                    ](
                        _m: Int, _n: Int, score_vec: SIMD[type, _simd_width]
                    ) -> SIMD[type, _simd_width]:
                        return mask_fn[_simd_width, mask_rank](
                            get_mask_nd_index(_m + m, _n + kv_seq_idx),
                            score_vec,
                            kv_cache_len,
                        )

                    Self._online_softmax[mask_2d_fn](
                        qk_block_ptr,
                        o_block_ptr,
                        max_vals.data,
                        sum_vals.data,
                        count_m,
                        count_n,
                        kv_seq_cnt,
                        scale,
                    )

                    @parameter
                    @always_inline
                    fn input_v_2d_fn[
                        _simd_width: Int
                    ](_n: Int, _k: Int) -> SIMD[type, _simd_width]:
                        return input_v_fn[_simd_width, rank](
                            get_nd_index[is_kv=True](_k + kv_seq_idx, n + _n)
                        )

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
                    _ = kv_seq_idx

                _ = m
                _ = n
                var oz_ptr = o_block_ptr

                for m in range(count_m):
                    var reciprocal = 1 / sum_vals[m]

                    @parameter
                    @always_inline
                    fn do_final[_simd_width: Int](idx: Int):
                        var v = oz_ptr.load[width=_simd_width](idx)
                        o_ptr.store(idx, v * reciprocal)

                    vectorize[do_final, simd_width, unroll_factor=4](count_n)

                    o_ptr += q_seq_stride
                    oz_ptr += Self._config.o_block_n

            if packed_ptr:
                packed_ptr.free()

        sync_parallelize[task_func](num_threads)


@always_inline
fn _flash_attention[
    type: DType,
    rank: Int,
    mask_rank: Int, //,
    input_k_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_mask_fn: fn[simd_width: Int, mask_rank: Int] (
        IndexList[mask_rank]
    ) capturing -> SIMD[type, simd_width],
](
    q: NDBuffer[type, rank, *_],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    mask_shape: IndexList[mask_rank],
    output: NDBuffer[type, rank, *_],
    scale: Float32,
):
    var num_batches = output.dim[0]()
    var max_seq_len = output.dim[1]()
    var num_heads = output.dim[rank - 2]() if rank == 4 else 1
    var depth_dim = output.dim[rank - 1]()
    var kv_cache_len = v_shape[1] - max_seq_len
    var num_kv_heads = k_shape[rank - 2] if rank == 4 else 1

    @always_inline
    @parameter
    fn input_q_ptr_fn(idx: IndexList[rank]) -> UnsafePointer[Scalar[type]]:
        return q._offset(idx)

    @always_inline
    @parameter
    fn output_ptr_fn(idx: IndexList[rank]) -> UnsafePointer[Scalar[type]]:
        return output._offset(idx)

    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int, rank: Int
    ](
        idx: IndexList[rank],
        score_vec: SIMD[type, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[type, simd_width]:
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
        output.shape,
    ].run(num_batches, num_heads, depth_dim, num_kv_heads, max_seq_len, scale)


fn flash_attention[
    type: DType,
    rank: Int,
    mask_rank: Int, //,
    input_k_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_mask_fn: fn[simd_width: Int, mask_rank: Int] (
        IndexList[mask_rank]
    ) capturing -> SIMD[type, simd_width],
](
    q: NDBuffer[type, rank, *_],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    mask_shape: IndexList[mask_rank],
    output: NDBuffer[type, rank, *_],
    scale: Float32,
):
    _flash_attention[input_k_fn, input_v_fn, input_mask_fn](
        q,
        k_shape,
        v_shape,
        mask_shape,
        output,
        scale,
    )


fn flash_attention_split_kv[
    type: DType,
    rank: Int,
    mask_rank: Int, //,
    input_k_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_k_cache_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_v_cache_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_mask_fn: fn[simd_width: Int, mask_rank: Int] (
        IndexList[mask_rank]
    ) capturing -> SIMD[type, simd_width],
](
    q: NDBuffer[type, rank, *_],
    k_shape: IndexList[rank],
    v_shape: IndexList[rank],
    # {k,v}_cache_shape are rank + 1 because reshape in MO IR prevents fusion.
    k_cache_shape: IndexList[rank + 1],
    v_cache_shape: IndexList[rank + 1],
    mask_shape: IndexList[mask_rank],
    output: NDBuffer[type, rank, *_],
    scale: Float32,
):
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
    constrained[rank == 4]()

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("k", k_shape),
            trace_arg("v", v_shape),
            trace_arg("k_cache", k_cache_shape),
            trace_arg("v_cache", v_cache_shape),
            trace_arg("output", output),
        )

    with Trace[TraceLevel.OP, target="cpu"](
        "flash_attention_split_kv",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        alias kv_rank = rank + 1

        var num_kv_heads = v_cache_shape[2]
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
            curr_fn: fn[simd_width: Int, rank: Int] (
                IndexList[rank]
            ) capturing -> SIMD[type, simd_width],
            cache_fn: fn[simd_width: Int, rank: Int] (
                IndexList[rank]
            ) capturing -> SIMD[type, simd_width],
            rank: Int,
            simd_width: Int,
        ](idx: IndexList[rank]) -> SIMD[type, simd_width]:
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
        ](idx: IndexList[rank]) -> SIMD[type, simd_width]:
            return load_from_split_cache[
                input_k_fn, input_k_cache_fn, rank, simd_width
            ](idx)

        @always_inline
        @parameter
        fn input_v_cache_fn_wrapper[
            simd_width: Int,
            rank: Int,
        ](idx: IndexList[rank]) -> SIMD[type, simd_width]:
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
            input_k_cache_fn_wrapper, input_v_cache_fn_wrapper, input_mask_fn
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
    type: DType,
    cache_t: KVCacheT, //,
    mask_fn: fn[simd_width: Int, mask_rank: Int] (
        idx: IndexList[mask_rank],
        score_vec: SIMD[type, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[type, simd_width],
    mask_rank: Int,
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    scale: Float32,
    output: NDBuffer[type, 4, *_],
):
    alias kv_params = cache_t.kv_params

    var max_seq_len = q.dim[1]()
    var num_batches = q.dim[0]()
    alias num_heads = q.shape.get[2]()
    alias head_size = cache_t.kv_params.head_size
    alias output_shape = DimList(Dim(), Dim(), num_heads, head_size)

    @always_inline
    @parameter
    fn input_q_ptr_fn(idx: IndexList[4]) -> UnsafePointer[Scalar[type]]:
        return q._offset(idx)

    @always_inline
    @parameter
    fn output_ptr_fn(idx: IndexList[4]) -> UnsafePointer[Scalar[type]]:
        return output._offset(idx)

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
    ](k, v, num_batches, num_heads, max_seq_len, scale)


@always_inline
fn _flash_attention_kv_cache[
    type: DType,
    cache_t: KVCacheT, //,
    input_q_ptr_fn: fn (IndexList[4]) capturing -> UnsafePointer[Scalar[type]],
    output_ptr_fn: fn (IndexList[4]) capturing -> UnsafePointer[Scalar[type]],
    q_length_fn: fn (batch: Int) capturing -> Int,
    kv_length_fn: fn (batch: Int) capturing -> Int,
    mask_fn: fn[simd_width: Int, mask_rank: Int] (
        idx: IndexList[mask_rank],
        score_vec: SIMD[type, simd_width],
        kv_cache_length: Int,
    ) capturing -> SIMD[type, simd_width],
    mask_rank: Int,
    output_shape: DimList,
](
    k: cache_t,
    v: cache_t,
    num_batches: Int,
    num_heads: Int,
    max_seq_len: Int,
    scale: Float32,
):
    alias num_kv_heads = cache_t.kv_params.num_heads
    alias depth_dim = cache_t.kv_params.head_size
    alias cache_type = cache_t.type

    constrained[
        cache_type == type,
        "Expected cache type ("
        + String(cache_type)
        + ") to match input type ("
        + String(type)
        + ")",
    ]()

    @parameter
    fn input_k_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[type, width]:
        # Unwrap BSHD->BHSD indices.
        return rebind[SIMD[type, width]](
            k.load[width=width](idx[0], idx[2], idx[1], idx[3])
        )

    @parameter
    fn input_v_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[type, width]:
        # Unwrap BSHD->BHSD indices.
        return rebind[SIMD[type, width]](
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
    ].run(num_batches, num_heads, depth_dim, num_kv_heads, max_seq_len, scale)


fn flash_attention_kv_cache[
    type: DType, cache_t: KVCacheT, //
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    mask: NDBuffer[type, *_],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
):
    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int, rank: Int
    ](
        idx: IndexList[rank],
        score_vec: SIMD[type, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[type, simd_width]:
        return score_vec + mask.load[width=simd_width](idx)

    _flash_attention_kv_cache[mask_fn, mask.rank](q, k, v, scale, output)


fn flash_attention_kv_cache[
    type: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask, //,
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    mask: mask_t,
    scale: Float32,
    output: NDBuffer[type, 4, *_],
):
    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int,
        rank: Int,
    ](
        idx: IndexList[rank],
        score_vec: SIMD[type, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[type, simd_width]:
        # Shift the mask index from local->global space.
        return mask.mask(
            Index(idx[0], idx[1], idx[2] + kv_cache_len, idx[3]), score_vec
        )

    _flash_attention_kv_cache[mask_fn, 4](q, k, v, scale, output)


fn flash_attention_kv_cache[
    type: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask, //,
](
    q: NDBuffer[type, 3, *_],
    q_input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    k: cache_t,
    v: cache_t,
    mask: mask_t,
    scale: Float32,
    output: NDBuffer[type, 3, *_],
):
    """Entrypoint for ragged tensors."""

    @always_inline
    @parameter
    fn mask_fn[
        simd_width: Int,
        rank: Int,
    ](
        idx: IndexList[rank],
        score_vec: SIMD[type, simd_width],
        kv_cache_len: Int,
    ) -> SIMD[type, simd_width]:
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
    fn input_q_ptr_fn(idx: IndexList[4]) -> UnsafePointer[Scalar[type]]:
        var bs = idx[0]
        var tok_idx = idx[1]
        var q_start = Int(q_input_row_offsets[bs]) + tok_idx
        var flat_idx = IndexList[3](q_start, idx[2], idx[3])
        return q._offset(flat_idx)

    @always_inline
    @parameter
    fn output_ptr_fn(idx: IndexList[4]) -> UnsafePointer[Scalar[type]]:
        var bs = idx[0]
        var tok_idx = idx[1]
        var q_start = Int(q_input_row_offsets[bs]) + tok_idx
        var flat_idx = IndexList[3](q_start, idx[2], idx[3])
        return output._offset(flat_idx)

    alias mask_rank = 4
    var num_batches = q_input_row_offsets.dim[0]() - 1
    var max_seq_len = k.get_max_seq_length()
    alias num_heads = q.shape.get[q.rank - 2]()
    alias head_size = cache_t.kv_params.head_size
    alias output_shape = DimList(Dim(), Dim(), num_heads, head_size)

    _flash_attention_kv_cache[
        input_q_ptr_fn,
        output_ptr_fn,
        q_length_fn,
        kv_length_fn,
        mask_fn,
        mask_rank,
        output_shape,
    ](k, v, num_batches, num_heads, Int(max_seq_len), scale)
