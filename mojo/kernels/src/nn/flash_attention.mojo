# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_up, div_ceil, exp, max, min
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
from buffer.list import Dim
from LinAlg.MatmulUtils import partition_work
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer
from runtime.llcl import Runtime

from utils.index import Index


struct _MatmulAccumulators[
    type: DType, simd_width: Int, tile_m: Int, tile_n: Int
]:
    var _storage: StaticTuple[SIMD[type, simd_width], tile_m * tile_n]

    fn __init__(inout self):
        self._storage = StaticTuple[SIMD[type, simd_width], tile_m * tile_n](0)

    @staticmethod
    fn _storage_index(m: Int, n: Int) -> Int:
        return m * tile_n + n

    fn __getitem__(self, m: Int, n: Int) -> SIMD[type, simd_width]:
        return self._storage[self._storage_index(m, n)]

    fn __setitem__(inout self, m: Int, n: Int, value: SIMD[type, simd_width]):
        self._storage[self._storage_index(m, n)] = value

    fn fma(
        inout self,
        m: Int,
        n: Int,
        a: SIMD[type, simd_width],
        b: SIMD[type, simd_width],
    ):
        self[m, n] = b.fma(a, self[m, n])

    fn _transfer[
        func: fn (m: Int, n: Int, ptr: DTypePointer[type]) capturing -> None
    ](inout self, base_ptr: DTypePointer[type], stride: Int):
        var row_ptr = base_ptr

        @unroll
        for m in range(tile_m):

            @unroll
            for n in range(tile_n):
                func(m, n, row_ptr.offset(n * simd_width))
            row_ptr = row_ptr.offset(stride)

    fn load(inout self, base_ptr: DTypePointer[type], stride: Int):
        @parameter
        @always_inline
        fn do_transfer(m: Int, n: Int, ptr: DTypePointer[type]):
            self[m, n] = ptr.load[width=simd_width]()

        self._transfer[do_transfer](base_ptr, stride)

    fn store(inout self, base_ptr: DTypePointer[type], stride: Int):
        @parameter
        @always_inline
        fn do_transfer(m: Int, n: Int, ptr: DTypePointer[type]):
            ptr.store(self[m, n])

        self._transfer[do_transfer](base_ptr, stride)


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
    fn _get_matmul_config() -> _MatmulConfig:
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
    alias _matmul_config = _MatmulConfig._get_matmul_config()

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
        inout c_tile: _MatmulAccumulators[type, simd_width, tile_m, tile_n],
    ):
        var ak_ptr = a_ptr
        var bk_ptr = b_ptr

        @parameter
        @always_inline
        fn loop_body[lane_count: Int](k: Int):
            var a_tile = StaticTuple[SIMD[type, lane_count], tile_m]()

            @unroll
            for m in range(tile_m):
                a_tile[m] = ak_ptr.offset(m * a_stride).load[width=lane_count]()

            ak_ptr += lane_count

            @unroll
            for k in range(lane_count):

                @unroll
                for n in range(tile_n):
                    var b_data = bk_ptr.offset(n * simd_width).load[
                        width=simd_width
                    ]()

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
        inout c_tile: _MatmulAccumulators[type, simd_width, tile_m, tile_n],
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
                    b_tile[n] = bk_ptr.offset(n * simd_width).load[
                        width=simd_width
                    ]()

                @unroll
                for m in range(tile_m):
                    var a_data = ak_ptr.offset(m * a_stride).load()

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
                var c_tile = _MatmulAccumulators[
                    type, simd_width, tile_m, tile_n
                ]()

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

                bn_ptr = bn_ptr.offset(tile_n * simd_width)
                cn_ptr = cn_ptr.offset(tile_n * simd_width)

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
                output_ptr.offset(idx).store[width=_simd_width](val)

            tile[packed_copy, Self._matmul_config.pack_sizes](0, N)

            if aligned_n != N:
                memset_zero(output_ptr.offset(N), aligned_n - N)

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
    block_m: Int,
    qk_block_n: Int,
    o_block_n: Int,
]:
    alias _matmul = _Matmul[type, simd_width]

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
                o_row_ptr.store[width=_simd_width](idx, val * fixup_val)

            vectorize[do_correction, simd_width, unroll_factor=2](count_n)

            qk_row_ptr += qk_block_n
            o_row_ptr += o_block_n

    @staticmethod
    fn run(
        q: NDBuffer[type, rank],
        k_shape: StaticIntTuple[rank],
        v_shape: StaticIntTuple[rank],
        output: NDBuffer[type, rank],
        scale: Float32,
    ):
        var num_batches = output.dim[0]()
        var num_heads = output.dim[1]() if rank == 4 else 1
        var seq_len = output.dim[rank - 2]()
        var depth_dim = output.dim[rank - 1]()
        var kv_seq_len = v_shape[rank - 2]

        # Compute the maximum size in elements for the common packed buffer.
        var packed_qk_size = qk_block_n * depth_dim
        var packed_o_size = o_block_n * qk_block_n
        var packed_size = max(packed_qk_size, packed_o_size)

        var num_blocks_m = div_ceil(seq_len, block_m)
        var num_blocks_n = div_ceil(depth_dim, o_block_n)
        var work_count = num_batches * num_heads * num_blocks_m * num_blocks_n

        var num_threads = min(work_count, Runtime().parallelism_level())

        @parameter
        fn task_func(task_id: Int):
            var qk_block_ptr = stack_allocation[
                block_m * qk_block_n,
                type,
                alignment = alignof[SIMD[type, simd_width]](),
            ]()
            var o_block_ptr = stack_allocation[
                block_m * o_block_n,
                type,
                alignment = alignof[SIMD[type, simd_width]](),
            ]()
            var max_vals = Buffer[type, Dim(block_m)]().stack_allocation()
            var sum_vals = Buffer[type, Dim(block_m)]().stack_allocation()

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
                var n = (i % num_blocks_n) * o_block_n
                var j = i // num_blocks_n
                var m = (j % num_blocks_m) * block_m
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

                var count_m = min(block_m, seq_len - m)
                var count_n = min(o_block_n, depth_dim - n)

                var o_ptr = output._offset(get_nd_index(m, n))
                var q_ptr = q._offset(get_nd_index(m, 0))

                max_vals.fill(Scalar[type].MIN)
                sum_vals.fill(0)

                for kv_seq_idx in range(0, kv_seq_len, qk_block_n):
                    var kv_seq_cnt = min(kv_seq_len - kv_seq_idx, qk_block_n)

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
                        qk_block_n,
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
                        qk_block_n,
                        packed_ptr,
                        o_block_ptr,
                        o_block_n,
                        accumulate=(kv_seq_idx > 0),
                    )

                var oz_ptr = o_block_ptr

                for m in range(count_m):
                    var recip = 1 / sum_vals[m]

                    @always_inline
                    @parameter
                    fn do_final[_simd_width: Int](idx: Int):
                        var v = oz_ptr.load[width=_simd_width](idx)
                        var e = v * recip
                        o_ptr.store[width=_simd_width](idx, e)

                    vectorize[do_final, simd_width, unroll_factor=4](count_n)

                    o_ptr += depth_dim
                    oz_ptr += o_block_n

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
](
    q: NDBuffer[type, rank],
    k_shape: StaticIntTuple[rank],
    v_shape: StaticIntTuple[rank],
    output: NDBuffer[type, rank],
    scale: Float32,
):
    alias simd_width = simdwidthof[type]()

    _FlashAttention[
        type,
        rank,
        simd_width,
        input_k_fn,
        input_v_fn,
        input_mask_fn,
        block_m=64,
        qk_block_n=128,
        o_block_n=128,
    ].run(q, k_shape, v_shape, output, scale)
