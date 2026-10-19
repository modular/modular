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

from std.math import ceildiv
from std.sys.info import align_of, size_of

from quantization import (
    Q4sym,
    block_Q4_K,
    block_Q6_K,
    q4_k_dequantize_impl,
    q6_k_dequantize_impl,
    scale_min_k4,
)
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from std.utils import IndexList


def _run_test_quant[group_size: Int, tolerance: Float32]() -> Bool:
    var uniform = SIMD[.float32, group_size]()
    for i in range(group_size):
        uniform[i] = Float32(i)
    uniform -= Float32(group_size // 2)

    var skew_pos = uniform + 30
    var skew_neg = uniform - 30
    var skew_slightly_pos = uniform + 1.842
    var skew_slightly_neg = uniform - 1.842
    var big_range = uniform * 1000
    var unitary = SIMD[.float32, group_size](1.0)

    def run_fake_quant(input_vec: SIMD[.float32, group_size]) -> Bool:
        var packed_result = Q4sym[group_size, .float32](input_vec)
        var decoded_result = packed_result.decode_fully()
        print("input_vec        :", input_vec)
        print("fakeq_result     :", decoded_result)
        print("abs-err          :", input_vec - decoded_result)
        print(
            "max abs-err      :", abs(input_vec - decoded_result).reduce_max()
        )

        var l2_norm_err = (
            (input_vec - decoded_result) * (input_vec - decoded_result)
        ).reduce_add()
        l2_norm_err = l2_norm_err**0.5
        var l2_norm_input = (input_vec * input_vec).reduce_add() ** 0.5
        var rel_l2_norm = l2_norm_err / l2_norm_input
        print("rel-l2-norm      :", rel_l2_norm)

        return rel_l2_norm < tolerance

    var allPass: Bool = True
    allPass = allPass and run_fake_quant(uniform)
    allPass = allPass and run_fake_quant(skew_pos)
    allPass = allPass and run_fake_quant(skew_neg)
    allPass = allPass and run_fake_quant(skew_slightly_pos)
    allPass = allPass and run_fake_quant(skew_slightly_neg)
    allPass = allPass and run_fake_quant(big_range)
    allPass = allPass and run_fake_quant(unitary)
    return allPass


def test_fake_quant_error[l2_tolerance: Float32]() raises:
    # Tests round-trippability of encoding/decoding groups of numbers
    print("------------test_fake_quant_error------------")
    print("********** GROUP SIZE 08 **********")
    var g8_result = _run_test_quant[8, l2_tolerance]()
    print("G08 PASS" if g8_result else "G08 FAIL")
    print()
    assert_true(g8_result)

    print("********** GROUP SIZE 16 **********")
    var g16_result = _run_test_quant[16, l2_tolerance]()
    print("G16 PASS" if g16_result else "G16 FAIL")
    print()
    assert_true(g16_result)

    print("********** GROUP SIZE 32 **********")
    var g32_result = _run_test_quant[32, l2_tolerance]()
    print("------------end test_fake_quant_error------------")
    print("G32 PASS" if g32_result else "G32 FAIL")
    print()
    assert_true(g32_result)


def test_alignment_and_size():
    # Tests the total size and alignment of structs is as expected
    print("-------test_alignment_and_size-------")
    print("StructType, size_of, Alignment")
    print(
        "Q5sym[32, DType.float32]",
        size_of[Q4sym[32, DType.float32]](),
        align_of[Q4sym[32, DType.float32]](),
    )
    print(
        "Q5sym[16, DType.float32]",
        size_of[Q4sym[16, DType.float32]](),
        align_of[Q4sym[16, DType.float32]](),
    )
    print(
        "Q5sym[8, DType.float32]",
        size_of[Q4sym[8, DType.float32]](),
        align_of[Q4sym[8, DType.float32]](),
    )
    # Calculation for group size 8:
    # - 2 bytes for fp16 scale
    # - 8 // 2 = 4 bytes for the low bits
    # 2 + 4 = 6
    # Bits per weight: (8 * 6) / 8 = 6bpw
    comptime assert size_of[Q4sym[8]]() == 6

    # Calculation for group size 16:
    # - 2 bytes for fp16 scale
    # - 16 // 2 = 8 bytes for the low bits
    # 2 + 8 = 10
    # Bits per weight: (8 * 10) / 16 = 5bpw
    comptime assert size_of[Q4sym[16]]() == 10

    # Calculation for group size 32:
    # - 2 bytes for fp16 scale
    # - 32 // 2 = 16 bytes for the low bits
    # 2 + 16 = 18
    # Bits per weight: (8 * 18) / 32 = 4.5bpw
    comptime assert size_of[Q4sym[32]]() == 18
    print("-------end test_alignment_and_size-------")
    print()


def _read_write_to_tensors[
    group_size: Int,
    rtol: Float32,
    atol: Float32,
    num_elements: Int = 64,
    rank: Int = 1,
]() -> Bool:
    # Write a quantized tensor, and then immediately decode, making sure results
    # are close.

    # Allocate and populate tensor to encode
    # Buffer with the original data
    var data_matrix_backing = Array[Float32, num_elements](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var data_matrix_ptr: MutPointer[
        Float32, origin_of(data_matrix_backing)
    ] = data_matrix_backing.unsafe_ptr()
    var data_matrix = TileTensor(
        ptr=data_matrix_ptr, layout=row_major[num_elements]()
    )

    # Tensor to store the packed data
    comptime assert num_elements % group_size == 0
    comptime num_blocks = ceildiv(num_elements, group_size)
    comptime block_size = size_of[Q4sym[group_size]]()
    var packed_blob_backing = Array[UInt8, num_blocks * block_size](fill={})
    var packed_blob = TileTensor(
        packed_blob_backing, row_major[num_blocks * block_size]()
    )

    # Tensor to store the dequantized data
    var out_data_matrix_backing = Array[Float32, num_elements](fill=Float32(0))
    var out_data_matrix = TileTensor(
        out_data_matrix_backing, row_major[num_elements]()
    )

    Q4sym[group_size, DType.float32].quantize_and_write_to_tensor(
        data_matrix.make_dynamic[.int64](),
        packed_blob.make_dynamic[.int64](),
        IndexList[data_matrix.rank](num_elements),
    )

    Q4sym[group_size, DType.float32].dequantize_and_write_to_tensor(
        packed_blob.make_dynamic[.int64](),
        out_data_matrix.make_dynamic[.int64](),
        IndexList[out_data_matrix.rank](num_elements),
    )

    var allClose: Bool = True
    # See if it prints the correct results!
    for i in range(num_elements):
        var localRDiff = abs(
            (data_matrix[i] - out_data_matrix[i]) / (data_matrix[i] + 1e-10)
        )
        var acceptableErr = abs(data_matrix[i] * rtol) + atol
        print(
            "fake-quantized:",
            out_data_matrix[i],
            " vs original:",
            data_matrix[i],
            " -- rel-diff: ",
            localRDiff,
        )
        allClose = allClose and (localRDiff <= acceptableErr)
    return allClose


def test_read_write_to_tensors[rtol: FloatLiteral, atol: FloatLiteral]() raises:
    print("------------test_read_write_to_tensors------------")

    print("********** GROUP SIZE 08 **********")
    var g8_result = _read_write_to_tensors[8, rtol, atol]()
    print("G08 PASS" if g8_result else "G08 FAIL")
    print()
    assert_true(g8_result)

    print("********** GROUP SIZE 16 **********")
    var g16_result = _read_write_to_tensors[16, rtol, atol]()
    print("G16 PASS" if g16_result else "G16 FAIL")
    print()
    assert_true(g16_result)

    print("********** GROUP SIZE 32 **********")
    var g32_result = _read_write_to_tensors[32, rtol, atol]()
    print("G32 PASS" if g32_result else "G32 FAIL")
    print()
    assert_true(g32_result)

    print("------------end test_read_write_to_tensors------------")
    print()


def _fill_pattern(mut bytes: List[UInt8]):
    for i in range(len(bytes)):
        bytes[i] = UInt8((i * 37 + 11) & 0xFF)


def test_q4_k_dequantize() raises:
    comptime num_blocks = 3
    comptime block_nbytes = size_of[block_Q4_K]()
    var encoded = List[UInt8](length=num_blocks * block_nbytes, fill=0)
    _fill_pattern(encoded)
    var blocks = encoded.unsafe_ptr().unsafe_bitcast[block_Q4_K]()
    for b in range(num_blocks):
        blocks[b].base_scale = Float16(0.37 + 0.11 * Float32(b))
        blocks[b].base_min = Float16(0.21 - 0.05 * Float32(b))

    var out = List[Float32](length=num_blocks * 256, fill=-1)
    q4_k_dequantize_impl(
        TileTensor(encoded, row_major(len(encoded))).as_imm(),
        TileTensor(out, row_major(len(out))),
    )

    for b in range(num_blocks):
        var blk = blocks + b
        var d = blk[].base_scale.cast[.float32]()
        var m = blk[].base_min.cast[.float32]()
        for g in range(8):
            var scale, minimum = scale_min_k4(blk, g)
            # Pairs of groups share 32 bytes: low nibbles, then high nibbles.
            var byte_base = (g // 2) * 32
            for e in range(32):
                var byte = blk[].q_bits[byte_base + e]
                var nibble = byte & 0xF if g % 2 == 0 else byte >> 4
                assert_almost_equal(
                    out[b * 256 + g * 32 + e],
                    d * scale * nibble.cast[.float32]() - m * minimum,
                    rtol=1e-5,
                    atol=1e-6,
                )


def test_q6_k_dequantize() raises:
    comptime num_blocks = 3
    comptime block_nbytes = size_of[block_Q6_K]()
    var encoded = List[UInt8](length=num_blocks * block_nbytes, fill=0)
    _fill_pattern(encoded)
    var blocks = encoded.unsafe_ptr().unsafe_bitcast[block_Q6_K]()
    for b in range(num_blocks):
        blocks[b].base_scale = Float16(0.043 + 0.01 * Float32(b))

    var out = List[Float32](length=num_blocks * 256, fill=-1)
    q6_k_dequantize_impl(
        TileTensor(encoded, row_major(len(encoded))).as_imm(),
        TileTensor(out, row_major(len(out))),
        IndexList[2](num_blocks, 256),
    )

    for b in range(num_blocks):
        var blk = blocks + b
        var d = blk[].base_scale.cast[.float32]()
        for half in range(2):
            for l in range(32):
                var ql0 = blk[].q_bits_lo[half * 64 + l]
                var ql1 = blk[].q_bits_lo[half * 64 + l + 32]
                var qh = blk[].q_bits_hi[half * 32 + l]
                var is_ = half * 8 + l // 16
                var q: List[Int] = [
                    Int((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32,
                    Int((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32,
                    Int((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32,
                    Int((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32,
                ]
                for k in range(4):
                    var scale = blk[].q_scales[is_ + 2 * k].cast[.float32]()
                    assert_almost_equal(
                        out[b * 256 + half * 128 + k * 32 + l],
                        d * scale * Float32(q[k]),
                        rtol=1e-5,
                    )


def main() raises:
    comptime l2_tolerance = 0.1

    test_fake_quant_error[l2_tolerance]()

    test_read_write_to_tensors[rtol=0.1, atol=1.0]()

    # Tests via compile-time constraints on size_of(Q4Sym)
    test_alignment_and_size()

    test_q4_k_dequantize()
    test_q6_k_dequantize()
