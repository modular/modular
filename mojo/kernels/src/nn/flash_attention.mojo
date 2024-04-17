# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, align_up, div_ceil, exp, max, min
from sys.info import has_avx512f, has_neon

from algorithm import sync_parallelize, tile, vectorize
from algorithm.reduction import (
    _simd_max,
    _simd_max_elementwise,
    _simd_sum,
    _simd_sum_elementwise,
    map_reduce,
)
from buffer import Buffer, NDBuffer
from buffer.list import Dim, DimList
from LinAlg.MatmulUtils import partition_work
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer
from runtime.llcl import Runtime

from utils.index import Index
from nn.accumulate import _Accumulator


struct _MatmulConfig:
    var col_sizes: VariadicList[Int]
    var row_sizes: VariadicList[Int]
    var gemv_sizes: VariadicList[Int]
    var pack_sizes: VariadicList[Int]

    fn __init__(
        inout self,
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

    @always_inline
    @staticmethod
    fn _inner_loop_a_lane[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: DTypePointer[type],
        a_stride: Int,
        b_ptr: DTypePointer[type],
        b_stride: Int,
        inout c_tile: _Accumulator[type, tile_m, tile_n, simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @parameter
        @always_inline
        fn loop_body[lane_count: Int](k: Int):
            var a_tile = StaticTuple[SIMD[type, lane_count], tile_m]()

            @unroll
            for m in range(tile_m):
                a_tile[m] = ak_ptr.load[width=lane_count](m * a_stride)

            ak_ptr += lane_count

            @unroll
            for k in range(lane_count):

                @unroll
                for n in range(tile_n):
                    var b_data = bk_ptr.load[width=simd_width](n * simd_width)

                    @unroll
                    for m in range(tile_m):
                        c_tile.fma(m, n, a_tile[m][k], b_data)

                bk_ptr += b_stride

        tile[loop_body, VariadicList[Int](simd_width, 1)](0, K)

    @always_inline
    @staticmethod
    fn _inner_loop_a_broadcast[
        tile_m: Int, tile_n: Int
    ](
        K: Int,
        a_ptr: DTypePointer[type],
        a_stride: Int,
        b_ptr: DTypePointer[type],
        b_stride: Int,
        inout c_tile: _Accumulator[type, tile_m, tile_n, simd_width],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @parameter
        @always_inline
        fn loop_body[unroll_factor: Int](k: Int):
            var b_tile = StaticTuple[SIMD[type, simd_width], tile_n]()

            @unroll
            for k in range(unroll_factor):

                @unroll
                for n in range(tile_n):
                    b_tile[n] = bk_ptr.load[width=simd_width](n * simd_width)

                @unroll
                for m in range(tile_m):
                    var a_data = ak_ptr.load(m * a_stride)

                    @unroll
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
        a_ptr: DTypePointer[type],
        a_stride: Int,
        b_ptr: DTypePointer[type],
        c_ptr: DTypePointer[type],
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
                0, div_ceil(N, simd_width)
            )

            am_ptr += tile_m * a_stride
            cm_ptr += tile_m * c_stride

        tile[process_rows, Self._matmul_config.row_sizes](0, M)

    @no_inline
    @staticmethod
    fn _pack_buffer[
        input_b_fn: Self._input_fn_type,
    ](packed_ptr: DTypePointer[type], N: Int, K: Int):
        var output_ptr = packed_ptr
        var aligned_n = align_up(N, simd_width)

        for k in range(K):

            @always_inline
            @parameter
            fn packed_copy[_simd_width: Int](idx: Int):
                var val = input_b_fn[_simd_width](idx, k)
                output_ptr.store(idx, val)

            tile[packed_copy, Self._matmul_config.pack_sizes](0, N)

            if aligned_n != N:
                memset_zero(output_ptr + N, aligned_n - N)

            output_ptr += aligned_n

    @no_inline
    @staticmethod
    fn _gemv[
        input_b_fn: Self._input_fn_type,
    ](
        N: Int,
        K: Int,
        a_ptr: DTypePointer[type],
        c_ptr: DTypePointer[type],
        accumulate: Bool = False,
    ):
        var cn_ptr = c_ptr

        @parameter
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
    ](
        M: Int,
        N: Int,
        K: Int,
        a_ptr: DTypePointer[type],
        a_stride: Int,
        packed_ptr: DTypePointer[type],
        c_ptr: DTypePointer[type],
        c_stride: Int,
        accumulate: Bool = False,
    ):
        if M == 1:
            Self._gemv[input_b_fn](
                N,
                K,
                a_ptr,
                c_ptr,
                accumulate=accumulate,
            )

        else:
            Self._pack_buffer[input_b_fn](packed_ptr, N, K)

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

    fn __init__(inout self):
        self.qk_block_n = 128
        self.o_block_n = 128

        # Set a target size for the output block array.
        alias output_target_size = 8192

        alias depth_static_dim = output_static_shape.at[rank - 1]()

        @parameter
        if depth_static_dim:
            # Extract the static depth dimension with a guard against zero.
            var depth_dim = max(int(depth_static_dim), 1)

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
    rank: Int,
    simd_width: Int,
    input_k_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_mask_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    output_static_shape: DimList,
]:
    alias _matmul = _Matmul[type, simd_width]
    alias _config = _FlashAttentionConfig[
        type, rank, simd_width, output_static_shape
    ]()

    @staticmethod
    fn _online_softmax[
        input_mask_fn: fn[simd_width: Int] (m: Int, n: Int) capturing -> SIMD[
            type, simd_width
        ],
    ](
        qk_block_ptr: DTypePointer[type],
        o_block_ptr: DTypePointer[type],
        max_vals: DTypePointer[type],
        sum_vals: DTypePointer[type],
        count_m: Int,
        count_n: Int,
        kv_seq_cnt: Int,
        scale: Float32,
    ):
        var qk_row_ptr = qk_block_ptr
        var o_row_ptr = o_block_ptr

        for m in range(count_m):
            var qk_row = Buffer[type](qk_row_ptr, kv_seq_cnt)

            @always_inline
            @parameter
            fn pass1_input_gen_fn[
                _type: DType, _simd_width: Int
            ](idx: Int) -> SIMD[_type, _simd_width]:
                var val = qk_row_ptr.load[width=_simd_width](idx)
                var mask = input_mask_fn[_simd_width](m, idx)
                return rebind[SIMD[_type, _simd_width]](
                    val * scale.cast[type]() + mask
                )

            # Update the row with the scale and mask. Find the maximum value
            # of the row to bias the exponential function below for numeric
            # stability.
            var max_val = map_reduce[
                simd_width,
                Dim(),
                type,
                type,
                pass1_input_gen_fn,
                _simd_max_elementwise,
                _simd_max,
            ](qk_row, max_vals[m])

            @always_inline
            @parameter
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
                pass2_input_gen_fn,
                _simd_sum_elementwise,
                _simd_sum,
            ](qk_row, 0)

            var fixup_val = exp(max_vals[m] - max_val)

            # Update the running maximum and sum for the row.
            max_vals[m] = max_val
            sum_vals[m] = sum_vals[m] * fixup_val + accum_val

            @always_inline
            @parameter
            fn do_correction[_simd_width: Int](idx: Int):
                var val = o_row_ptr.load[width=_simd_width](idx)
                o_row_ptr.store(idx, val * fixup_val)

            vectorize[do_correction, simd_width, unroll_factor=2](count_n)

            qk_row_ptr += Self._config.qk_block_n
            o_row_ptr += Self._config.o_block_n

    @staticmethod
    fn run(
        q: NDBuffer[type, rank],
        k_shape: StaticIntTuple[rank],
        v_shape: StaticIntTuple[rank],
        output: NDBuffer[type, rank, output_static_shape],
        scale: Float32,
    ):
        var num_batches = output.dim[0]()
        var num_heads = output.dim[1]() if rank == 4 else 1
        var seq_len = output.dim[rank - 2]()
        var depth_dim = output.dim[rank - 1]()
        var kv_seq_len = v_shape[rank - 2]

        # Compute the maximum size in elements for the common packed buffer.
        var packed_qk_size = Self._config.qk_block_n * depth_dim
        var packed_o_size = Self._config.o_block_n * Self._config.qk_block_n
        var packed_size = max(packed_qk_size, packed_o_size)

        var num_blocks_m = div_ceil(seq_len, Self._config.block_m)
        var num_blocks_n = div_ceil(depth_dim, Self._config.o_block_n)
        var work_count = num_batches * num_heads * num_blocks_m * num_blocks_n

        var num_threads = min(work_count, Runtime().parallelism_level())

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
            var max_vals = Buffer[
                type, Dim(Self._config.block_m)
            ]().stack_allocation()
            var sum_vals = Buffer[
                type, Dim(Self._config.block_m)
            ]().stack_allocation()

            var packed_ptr = DTypePointer[
                type
            ]() if seq_len == 1 else DTypePointer[type].alloc(
                packed_size,
                alignment=alignof[SIMD[type, simd_width]](),
            )

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

                @always_inline
                @parameter
                fn get_nd_index(x: Int, y: Int) -> StaticIntTuple[rank]:
                    var idx: StaticIntTuple[rank]

                    @parameter
                    if rank == 4:
                        idx = rebind[StaticIntTuple[rank]](
                            Index(batch, head, x, y)
                        )
                    else:
                        idx = rebind[StaticIntTuple[rank]](
                            Index(batch_head, x, y)
                        )
                    return idx

                var count_m = min(Self._config.block_m, seq_len - m)
                var count_n = min(Self._config.o_block_n, depth_dim - n)

                var o_ptr = output._offset(get_nd_index(m, n))
                var q_ptr = q._offset(get_nd_index(m, 0))

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
                            get_nd_index(_k, _n + kv_seq_idx)
                        )

                    Self._matmul._matmul[input_k_2d_fn](
                        count_m,
                        kv_seq_cnt,
                        depth_dim,
                        q_ptr,
                        depth_dim,
                        packed_ptr,
                        qk_block_ptr,
                        Self._config.qk_block_n,
                    )

                    @parameter
                    @always_inline
                    fn input_mask_2d_fn[
                        _simd_width: Int
                    ](_m: Int, _n: Int) -> SIMD[type, _simd_width]:
                        return input_mask_fn[_simd_width, rank](
                            get_nd_index(_m + m, _n + kv_seq_idx)
                        )

                    Self._online_softmax[input_mask_2d_fn](
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
                            get_nd_index(_k + kv_seq_idx, n + _n)
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

                var oz_ptr = o_block_ptr

                for m in range(count_m):
                    var reciprocal = 1 / sum_vals[m]

                    @always_inline
                    @parameter
                    fn do_final[_simd_width: Int](idx: Int):
                        var v = oz_ptr.load[width=_simd_width](idx)
                        o_ptr.store(idx, v * reciprocal)

                    vectorize[do_final, simd_width, unroll_factor=4](count_n)

                    o_ptr += depth_dim
                    oz_ptr += Self._config.o_block_n

            if packed_ptr:
                packed_ptr.free()

        sync_parallelize[task_func](num_threads)


fn flash_attention[
    type: DType,
    rank: Int,
    input_k_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_v_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_mask_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    output_static_shape: DimList = DimList.create_unknown[rank](),
](
    q: NDBuffer[type, rank],
    k_shape: StaticIntTuple[rank],
    v_shape: StaticIntTuple[rank],
    output: NDBuffer[type, rank, output_static_shape],
    scale: Float32,
):
    _FlashAttention[
        type,
        rank,
        simdwidthof[type](),
        input_k_fn,
        input_v_fn,
        input_mask_fn,
        output_static_shape,
    ].run(q, k_shape, v_shape, output, scale)
