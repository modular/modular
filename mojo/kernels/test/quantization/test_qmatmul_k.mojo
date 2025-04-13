# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from collections import InlineArray
from math import ceildiv, isclose
from random import rand, random_float64
from sys import sizeof

from algorithm import sync_parallelize
from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from quantization.qmatmul import matmul_qint4, matmul_qint4_pack_b
from quantization.qmatmul_k import (
    _block_Q4_K,
    _block_Q6_K,
    _block_QK_K,
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
    matmul_Q6_K,
    matmul_Q6_K_pack_b,
)

from utils.index import Index


fn fill_random[type: DType](mut array: InlineArray[Scalar[type]]):
    rand(array.unsafe_ptr(), len(array))


fn random_float16(min: Float64 = 0, max: Float64 = 1) -> Float16:
    # Avoid pulling in a __truncdfhf2 dependency for a float64->float16
    # conversion by casting through float32 first.
    return (
        random_float64(min=min, max=max)
        .cast[DType.float32]()
        .cast[DType.float16]()
    )


fn quantize_a_Q8[
    group_size: Int
](a: UnsafePointer[Float32, **_], a_quant: UnsafePointer[Int8, **_]) -> Float32:
    var fp_data = a.load[width=group_size]()
    var max_value = abs(fp_data).reduce_max()
    var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0
    var scale = (max_value / 127.0).cast[DType.float32]()
    var quant_data = round(fp_data * multiplier).cast[DType.int8]()

    a_quant.store(quant_data)
    return scale


fn dot_product_QK_K[
    b_scales_type: DType,
    *,
    group_size: Int,
    b_zero_point: Int32 = 0,
](
    a_quant_data: UnsafePointer[Int8, **_],
    b_quant_data: UnsafePointer[UInt8, **_],
    b_scales: UnsafePointer[Scalar[b_scales_type], **_],
) -> Int32:
    var sum: Int32 = 0
    for i in range(_block_QK_K.quantized_k):
        sum += (
            a_quant_data[i].cast[DType.int32]()
            * (b_quant_data[i].cast[DType.int32]() - b_zero_point)
            * b_scales[i // group_size].cast[DType.int32]()
        )
    return sum


trait QuantizedGemm:
    @staticmethod
    fn k_group_size() -> Int:
        ...

    @staticmethod
    fn build_b_buffer(
        N: Int, K: Int
    ) -> NDBuffer[DType.uint8, 2, MutableAnyOrigin]:
        ...

    @staticmethod
    def pack_b_buffer(
        b: NDBuffer[mut=True, DType.uint8, 2],
        b_packed: NDBuffer[mut=True, DType.uint8, 2],
    ):
        ...

    @staticmethod
    fn kernel(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        c: NDBuffer[DType.float32, 2],
    ):
        ...

    @staticmethod
    fn dot_product(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        ...


struct _block_Q4_0:
    alias group_size = 32

    var base_scale: Float16
    var q_bits: InlineArray[UInt8, Self.group_size // 2]


struct qgemm_Q4_0(QuantizedGemm):
    @staticmethod
    fn k_group_size() -> Int:
        return _block_Q4_0.group_size

    @staticmethod
    fn build_b_buffer(
        N: Int, K: Int
    ) -> NDBuffer[DType.uint8, 2, MutableAnyOrigin]:
        var k_groups = ceildiv(K, Self.k_group_size())
        var b_ptr = UnsafePointer[UInt8].alloc(
            N * k_groups * sizeof[_block_Q4_0]()
        )
        var block_ptr = b_ptr.bitcast[_block_Q4_0]()

        for _n in range(N):
            for _k in range(k_groups):
                block_ptr[].base_scale = random_float16(max=0.001)
                fill_random(block_ptr[].q_bits)
                block_ptr += 1

        return NDBuffer[DType.uint8, 2](
            b_ptr, Index(N, k_groups * sizeof[_block_Q4_0]())
        )

    @staticmethod
    def pack_b_buffer(
        b: NDBuffer[mut=True, DType.uint8, 2],
        b_packed: NDBuffer[mut=True, DType.uint8, 2],
    ):
        matmul_qint4_pack_b[_block_Q4_0.group_size](b, b_packed)

    @staticmethod
    fn kernel(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        c: NDBuffer[DType.float32, 2],
    ):
        matmul_qint4[_block_Q4_0.group_size](a, b, c)

    @staticmethod
    fn dot_product(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        var block_ptr = b._offset(Index(n, 0)).bitcast[_block_Q4_0]() + (
            k // Self.k_group_size()
        )

        var a_quant_data = InlineArray[Int8, _block_Q4_0.group_size](
            uninitialized=True
        )

        var a_scale = quantize_a_Q8[_block_Q4_0.group_size](
            a._offset(Index(m, k)), a_quant_data.unsafe_ptr()
        )

        var b_quant_data = InlineArray[UInt8, _block_Q4_0.group_size](
            uninitialized=True
        )

        # Decode the bits of the weight data.
        var q_packed_bits = block_ptr[].q_bits.unsafe_ptr().load[
            width = _block_Q4_0.group_size // 2
        ]()

        for j in range(2):
            var idx = j * _block_Q4_0.group_size // 2
            var q_bits = ((q_packed_bits >> (j * 4)) & 15)
            b_quant_data.unsafe_ptr().store(idx, q_bits)

        var sum: Int32 = 0

        alias b_zero_point = 8

        for i in range(_block_Q4_0.group_size):
            sum += a_quant_data[i].cast[DType.int32]() * (
                (b_quant_data[i].cast[DType.int32]() - b_zero_point)
            )

        var sumf = sum.cast[DType.float32]() * block_ptr[].base_scale.cast[
            DType.float32
        ]()

        return sumf * a_scale


struct qgemm_Q4_K(QuantizedGemm):
    @staticmethod
    fn k_group_size() -> Int:
        return _block_QK_K.quantized_k

    @staticmethod
    fn build_b_buffer(
        N: Int, K: Int
    ) -> NDBuffer[DType.uint8, 2, MutableAnyOrigin]:
        var k_groups = ceildiv(K, Self.k_group_size())
        var b_ptr = UnsafePointer[UInt8].alloc(
            N * k_groups * sizeof[_block_Q4_K]()
        )
        var block_ptr = b_ptr.bitcast[_block_Q4_K]()

        for _n in range(N):
            for _k in range(k_groups):
                block_ptr[].base_scale = random_float16(max=0.001)
                block_ptr[].base_min = random_float16(min=-0.01, max=0.01)
                fill_random(block_ptr[].q_scales_and_mins)
                fill_random(block_ptr[].q_bits)
                block_ptr += 1

        return NDBuffer[DType.uint8, 2](
            b_ptr, Index(N, k_groups * sizeof[_block_Q4_K]())
        )

    @staticmethod
    def pack_b_buffer(
        b: NDBuffer[mut=True, DType.uint8, 2],
        b_packed: NDBuffer[mut=True, DType.uint8, 2],
    ):
        matmul_Q4_K_pack_b(b, b_packed)

    @staticmethod
    fn kernel(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        c: NDBuffer[DType.float32, 2],
    ):
        matmul_Q4_K(a, b, c)

    @staticmethod
    fn dot_product(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        var block_ptr = b._offset(Index(n, 0)).bitcast[_block_Q4_K]() + (
            k // Self.k_group_size()
        )

        var a_quant_data = InlineArray[Int8, _block_QK_K.quantized_k](
            uninitialized=True
        )

        var a_scale = quantize_a_Q8[_block_QK_K.quantized_k](
            a._offset(Index(m, k)), a_quant_data.unsafe_ptr()
        )

        var a_block_sums = InlineArray[Int32, _block_Q4_K.group_count](
            uninitialized=True
        )
        for i in range(_block_Q4_K.group_count):
            a_block_sums[i] = (
                a_quant_data.unsafe_ptr()
                .load[width = _block_Q4_K.group_size](
                    i * _block_Q4_K.group_size
                )
                .cast[DType.int32]()
                .reduce_add()
            )

        var b_scales = InlineArray[UInt8, _block_Q4_K.group_count](
            uninitialized=True
        )
        var b_mins = InlineArray[UInt8, _block_Q4_K.group_count](
            uninitialized=True
        )

        for i in range(_block_Q4_K.group_count):
            if i < 4:
                b_scales[i] = block_ptr[].q_scales_and_mins[i] & 63
                b_mins[i] = block_ptr[].q_scales_and_mins[i + 4] & 63
            else:
                b_scales[i] = (block_ptr[].q_scales_and_mins[i + 4] & 15) | (
                    (block_ptr[].q_scales_and_mins[i - 4] >> 6) << 4
                )
                b_mins[i] = (block_ptr[].q_scales_and_mins[i + 4] >> 4) | (
                    (block_ptr[].q_scales_and_mins[i - 0] >> 6) << 4
                )

        var b_quant_data = InlineArray[UInt8, _block_QK_K.quantized_k](
            uninitialized=True
        )

        # Decode the bits of the weight data.
        for i in range(0, _block_QK_K.quantized_k // 2, 32):
            var q_bits_ptr = block_ptr[].q_bits.unsafe_ptr()
            var q_packed_bits = q_bits_ptr.load[width=32](i)

            for j in range(2):
                var idx = i * 2 + j * 32
                var q_bits = ((q_packed_bits >> (j * 4)) & 15)
                b_quant_data.unsafe_ptr().store(idx, q_bits)

        var sum2: Int32 = 0

        for i in range(_block_Q4_K.group_count):
            sum2 += a_block_sums[i] * b_mins[i].cast[DType.int32]()

        var sum = dot_product_QK_K[group_size = _block_Q4_K.group_size](
            a_quant_data.unsafe_ptr(),
            b_quant_data.unsafe_ptr(),
            b_scales.unsafe_ptr(),
        )

        var sumf = sum.cast[DType.float32]() * block_ptr[].base_scale.cast[
            DType.float32
        ]()
        sumf = (
            sumf
            - sum2.cast[DType.float32]()
            * block_ptr[].base_min.cast[DType.float32]()
        )

        return sumf * a_scale


struct qgemm_Q6_K(QuantizedGemm):
    @staticmethod
    fn k_group_size() -> Int:
        return _block_QK_K.quantized_k

    @staticmethod
    fn build_b_buffer(
        N: Int, K: Int
    ) -> NDBuffer[DType.uint8, 2, MutableAnyOrigin]:
        var k_groups = ceildiv(K, Self.k_group_size())
        var b_ptr = UnsafePointer[UInt8].alloc(
            N * k_groups * sizeof[_block_Q6_K]()
        )
        var block_ptr = b_ptr.bitcast[_block_Q6_K]()

        for _n in range(N):
            for _k in range(k_groups):
                fill_random(block_ptr[].q_bits_lo)
                fill_random(block_ptr[].q_bits_hi)
                fill_random(block_ptr[].q_scales)
                block_ptr[].base_scale = random_float16(max=0.001)
                block_ptr += 1

        return NDBuffer[DType.uint8, 2](
            b_ptr, Index(N, k_groups * sizeof[_block_Q6_K]())
        )

    @staticmethod
    def pack_b_buffer(
        b: NDBuffer[mut=True, DType.uint8, 2],
        b_packed: NDBuffer[mut=True, DType.uint8, 2],
    ):
        matmul_Q6_K_pack_b(b, b_packed)

    @staticmethod
    fn kernel(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        c: NDBuffer[DType.float32, 2],
    ):
        matmul_Q6_K(a, b, c)

    @staticmethod
    fn dot_product(
        a: NDBuffer[DType.float32, 2],
        b: NDBuffer[DType.uint8, 2],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        var block_ptr = b._offset(Index(n, 0)).bitcast[_block_Q6_K]() + (
            k // Self.k_group_size()
        )

        var a_quant_data = InlineArray[Int8, _block_QK_K.quantized_k](
            uninitialized=True
        )

        var a_scale = quantize_a_Q8[_block_QK_K.quantized_k](
            a._offset(Index(m, k)), a_quant_data.unsafe_ptr()
        )

        var b_quant_data = InlineArray[UInt8, _block_QK_K.quantized_k](
            uninitialized=True
        )

        # Decode the bottom bits of the weight data.
        for i in range(0, _block_QK_K.quantized_k // 2, 64):
            var q_bits_lo_ptr = block_ptr[].q_bits_lo.unsafe_ptr()
            var q_packed_bits = q_bits_lo_ptr.load[width=64](i)

            for j in range(2):
                var idx = i * 2 + j * 64
                var q_bits = ((q_packed_bits >> (j * 4)) & 15)
                b_quant_data.unsafe_ptr().store(idx, q_bits)

        # Decode the top bits of the weight data.
        for i in range(0, _block_QK_K.quantized_k // 4, 32):
            var q_bits_hi_ptr = block_ptr[].q_bits_hi.unsafe_ptr()
            var q_packed_bits = q_bits_hi_ptr.load[width=32](i)

            for j in range(4):
                var idx = i * 4 + j * 32
                var q_bits_lo = b_quant_data.unsafe_ptr().load[width=32](idx)
                var q_bits_hi = (((q_packed_bits >> (j * 2)) & 3) << 4)
                b_quant_data.unsafe_ptr().store(idx, q_bits_hi | q_bits_lo)

        var sum = dot_product_QK_K[
            group_size = _block_Q6_K.group_size, b_zero_point=32
        ](
            a_quant_data.unsafe_ptr(),
            b_quant_data.unsafe_ptr(),
            block_ptr[].q_scales.unsafe_ptr(),
        )

        var sumf = sum.cast[DType.float32]() * block_ptr[].base_scale.cast[
            DType.float32
        ]()

        return sumf * a_scale


fn reference_gemm[
    qgemm: QuantizedGemm
](
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[mut=True, DType.float32, 2],
):
    var M = a.dim[0]()
    var N = b.dim[0]()
    var K = a.dim[1]()

    alias grain_size = 128

    var total_work = M * N
    var num_workers = ceildiv(total_work, grain_size)

    @__copy_capture(total_work, N, K)
    @parameter
    fn task_func(task_id: Int):
        var task_start = task_id * grain_size
        var task_count = min(total_work - task_start, grain_size)

        for i in range(task_start, task_start + task_count):
            var n = i % N
            var m = i // N

            var result: Float32 = 0

            for k in range(0, K, qgemm.k_group_size()):
                result += qgemm.dot_product(a, b, m, n, k)

            c.store(Index(m, n), result)

    sync_parallelize[task_func](num_workers)


struct GemmContext[qgemm: QuantizedGemm]:
    var a: NDBuffer[DType.float32, 2, MutableAnyOrigin]
    var b: NDBuffer[DType.uint8, 2, MutableAnyOrigin]
    var b_packed: NDBuffer[DType.uint8, 2, MutableAnyOrigin]
    var c: NDBuffer[DType.float32, 2, MutableAnyOrigin]
    var c_golden: NDBuffer[DType.float32, 2, MutableAnyOrigin]

    @staticmethod
    def _build_float_buffer(
        M: Int, N: Int
    ) -> NDBuffer[DType.float32, 2, MutableAnyOrigin]:
        var ptr = UnsafePointer[Float32].alloc(M * N)
        for i in range(M * N):
            ptr[i] = random_float64(min=-1.0, max=+1.0).cast[DType.float32]()
        return NDBuffer[DType.float32, 2](ptr, Index(M, N))

    @staticmethod
    def _build_b_buffer(
        N: Int, K: Int
    ) -> NDBuffer[DType.uint8, 2, MutableAnyOrigin]:
        return qgemm.build_b_buffer(N, K)

    @staticmethod
    def _pack_b_buffer(
        b: NDBuffer[mut=True, DType.uint8, 2]
    ) -> NDBuffer[DType.uint8, 2, MutableAnyOrigin]:
        var b_packed_buffer = UnsafePointer[UInt8].alloc(len(b))
        var b_packed = NDBuffer[DType.uint8, 2](b_packed_buffer, b.get_shape())
        qgemm.pack_b_buffer(b, b_packed)
        return b_packed

    def __init__(out self, M: Int, N: Int, K: Int):
        self.a = Self._build_float_buffer(M, K)
        self.b = Self._build_b_buffer(N, K)
        self.b_packed = Self._pack_b_buffer(self.b)
        self.c = Self._build_float_buffer(M, N)
        self.c_golden = Self._build_float_buffer(M, N)

    def free(mut self):
        self.a.data.free()
        self.b.data.free()
        self.b_packed.data.free()
        self.c.data.free()
        self.c_golden.data.free()


def test_case[qgemm: QuantizedGemm](M: Int, N: Int, K: Int):
    var ctx = GemmContext[qgemm](M, N, K)

    if K % qgemm.k_group_size() != 0:
        raise ("K must be a multiple of qgemm.k_group_size()")

    reference_gemm[qgemm](ctx.a, ctx.b, ctx.c_golden)
    qgemm.kernel(ctx.a, ctx.b_packed, ctx.c)

    var mismatch = False
    for i in range(len(ctx.c)):
        if not isclose(
            ctx.c.data[i], ctx.c_golden.data[i], atol=1e-4, rtol=1e-4
        ):
            print(
                "MISMATCH",
                ctx.c.get_nd_index(i),
                ctx.c.data[i],
                ctx.c_golden.data[i],
            )
            mismatch = True
            break
    if mismatch:
        raise Error("found mismatch")

    ctx.free()


def test_cases[qgemm: QuantizedGemm]():
    for m in range(1, 16):
        test_case[qgemm](m, 128, 256)
        test_case[qgemm](m, 128, 1024)

        @parameter
        if qgemm.k_group_size() == 32:
            test_case[qgemm](m, 256, 32)

    # Typical LLM use case.
    test_case[qgemm](1, 4096, 4096)
    test_case[qgemm](160, 4096, 4096)


def main():
    test_cases[qgemm_Q4_0]()
    test_cases[qgemm_Q4_K]()
    test_cases[qgemm_Q6_K]()
