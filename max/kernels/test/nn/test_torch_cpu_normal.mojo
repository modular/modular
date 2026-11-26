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
"""Pins `torch_cpu_normal_fill` to the bits of PyTorch's CPU `normal_`.

Each golden is a checksum over the bits of
`torch.empty(n, dtype=dtype).normal_(mean, std, generator=g)` with
`g = torch.Generator().manual_seed(seed)` from torch 2.15.0 on an x86 host
whose capability is AVX2 or AVX-512 (both give the same bits). The checksum is
the sum of `bits[i] * (i + 1)` modulo 2^64. The last tests run the
`mo.random.normal` kernel on a CPU context.
"""

from layout import Coord, row_major
from max.gpu.host import DeviceContext
from nn.rand_normal import random_normal, torch_cpu_normal_fill
from std.memory import bitcast
from std.sys.info import CompilationTarget
from std.testing import assert_equal, assert_raises, TestSuite


@fieldwise_init
struct Golden(Copyable):
    var numel: Int
    var seed: UInt64
    var mean: Float32
    var stddev: Float32
    var checksum: UInt64


def _bits[dtype: DType](value: Scalar[dtype]) -> UInt64:
    comptime if dtype == .float32:
        return UInt64(bitcast[.uint32, 1](value)[0])
    elif dtype == .float64:
        return bitcast[.uint64, 1](value)[0]
    else:
        return UInt64(bitcast[.uint16, 1](value)[0])


def _expect(checksum: UInt64, golden: Golden) raises:
    assert_equal(
        checksum,
        golden.checksum,
        String(
            t"numel={golden.numel} seed={golden.seed} mean={golden.mean}"
            t" std={golden.stddev}"
        ),
    )


def _check_float32[dtype: DType](golden: Golden) raises:
    """Compares the checksum of `dtype` values rounded from the float32 stream.

    `float32` takes torch's whole-tensor draw order. The narrow types draw per
    16-block.
    """
    var storage = List[Float32](length=golden.numel, fill=0)
    var values = storage.unsafe_ptr()
    torch_cpu_normal_fill[full_draw=dtype == .float32](
        values, golden.numel, golden.seed, golden.mean, golden.stddev
    )
    var checksum = UInt64(0)
    for i in range(golden.numel):
        checksum += _bits[dtype](values.unsafe_load(i).cast[dtype]()) * UInt64(
            i + 1
        )
    _expect(checksum, golden)


def _check_float64(golden: Golden) raises:
    var storage = List[Float64](length=golden.numel, fill=0)
    var values = storage.unsafe_ptr()
    torch_cpu_normal_fill(
        values, golden.numel, golden.seed, golden.mean, golden.stddev
    )
    var checksum = UInt64(0)
    for i in range(golden.numel):
        checksum += _bits[.float64](values.unsafe_load(i)) * UInt64(i + 1)
    _expect(checksum, golden)


def test_float32() raises:
    var goldens: List[Golden] = [
        Golden(1, 0, 0.0, 1.0, 0x000000003FC53F5C),
        Golden(7, 0, 0.0, 1.0, 0x0000000EECE7EE38),
        Golden(15, 0, 0.0, 1.0, 0x000000429FD37640),
        Golden(16, 0, 0.0, 1.0, 0x0000003AF6C7FE6D),
        Golden(17, 0, 0.0, 1.0, 0x000000404ADEBD18),
        Golden(31, 0, 0.0, 1.0, 0x000000DAC32E24C1),
        Golden(1000, 0, 0.0, 1.0, 0x0003D9F11BDE31EF),
        Golden(524288, 0, 0.0, 1.0, 0xE48C0B20767C86EC),
        Golden(1, 0, 0.5, 2.0, 0x0000000040653F5C),
        Golden(7, 0, 0.5, 2.0, 0x0000000EF9A4DA71),
        Golden(15, 0, 0.5, 2.0, 0x00000042D13AE827),
        Golden(16, 0, 0.5, 2.0, 0x0000003309E66768),
        Golden(17, 0, 0.5, 2.0, 0x0000003B172F4219),
        Golden(31, 0, 0.5, 2.0, 0x000000C782A590D2),
        Golden(1000, 0, 0.5, 2.0, 0x000372DB8E4B59DA),
        Golden(524288, 0, 0.5, 2.0, 0x5EE68E653C514EB7),
        Golden(1, 1, 0.0, 1.0, 0x000000003F294E60),
        Golden(7, 1, 0.0, 1.0, 0x0000000FDF02BD51),
        Golden(15, 1, 0.0, 1.0, 0x00000048F2BAC09E),
        Golden(16, 1, 0.0, 1.0, 0x00000049F681F325),
        Golden(17, 1, 0.0, 1.0, 0x0000005A1C22369E),
        Golden(31, 1, 0.0, 1.0, 0x000000F415C88B3B),
        Golden(1000, 1, 0.0, 1.0, 0x0003BB205D9B7BB5),
        Golden(524288, 1, 0.0, 1.0, 0xE3F7E4F6B6CFE225),
        Golden(1, 1, 0.5, 2.0, 0x000000003FE94E60),
        Golden(7, 1, 0.5, 2.0, 0x0000000CEBD1AFEF),
        Golden(15, 1, 0.5, 2.0, 0x00000039A6D725FB),
        Golden(16, 1, 0.5, 2.0, 0x0000004101A90C68),
        Golden(17, 1, 0.5, 2.0, 0x0000004D697F7DBF),
        Golden(31, 1, 0.5, 2.0, 0x000000BB39AB7311),
        Golden(1000, 1, 0.5, 2.0, 0x000368B559FB7128),
        Golden(524288, 1, 0.5, 2.0, 0x624B1116595680FB),
        Golden(1, 4294967301, 0.0, 1.0, 0x00000000BEF93B54),
        Golden(7, 4294967301, 0.0, 1.0, 0x0000000FEC354C99),
        Golden(15, 4294967301, 0.0, 1.0, 0x0000003C1791E601),
        Golden(16, 4294967301, 0.0, 1.0, 0x000000567A46FC6B),
        Golden(17, 4294967301, 0.0, 1.0, 0x0000003E342D6F88),
        Golden(31, 4294967301, 0.0, 1.0, 0x0000011612E3789A),
        Golden(1000, 4294967301, 0.0, 1.0, 0x0003AC6FFAEFA931),
        Golden(524288, 4294967301, 0.0, 1.0, 0xE3156CC00082C442),
        Golden(1, 4294967301, 0.5, 2.0, 0x00000000BEF276A7),
        Golden(7, 4294967301, 0.5, 2.0, 0x0000000D73687343),
        Golden(15, 4294967301, 0.5, 2.0, 0x00000039CEF2DCD8),
        Golden(16, 4294967301, 0.5, 2.0, 0x0000004E3034F7D1),
        Golden(17, 4294967301, 0.5, 2.0, 0x000000329AA870E2),
        Golden(31, 4294967301, 0.5, 2.0, 0x000000DCBBF461E2),
        Golden(1000, 4294967301, 0.5, 2.0, 0x0003406F0D4039C4),
        Golden(524288, 4294967301, 0.5, 2.0, 0x5D8DFAD36271E681),
    ]
    for golden in goldens:
        _check_float32[.float32](golden)


def test_float64() raises:
    var goldens: List[Golden] = [
        Golden(1, 1, 0.0, 1.0, 0x3FE529CC078113DA),
        Golden(7, 1, 0.0, 1.0, 0xFBE057AA230FF8B3),
        Golden(15, 1, 0.0, 1.0, 0x6E5758125A2EA722),
        Golden(16, 1, 0.0, 1.0, 0xEA8915F042CE4F98),
        Golden(17, 1, 0.0, 1.0, 0x3529761EEB86EBB2),
        Golden(31, 1, 0.0, 1.0, 0x4982FDF16E4DC1C6),
        Golden(1000, 1, 0.0, 1.0, 0x490199A65F0AB544),
        Golden(524288, 1, 0.0, 1.0, 0x9B5289D4B7899CAB),
        Golden(1, 1, 0.5, 2.0, 0x3FFD29CC078113DA),
        Golden(7, 1, 0.5, 2.0, 0xFD7A35FE402CF815),
        Golden(15, 1, 0.5, 2.0, 0xF4DAE4BEF880A584),
        Golden(16, 1, 0.5, 2.0, 0xF4B7B71D4EB64C76),
        Golden(17, 1, 0.5, 2.0, 0x4006AD1A6DBFB8E5),
        Golden(31, 1, 0.5, 2.0, 0x6A7ACDB8185E53C1),
        Golden(1000, 1, 0.5, 2.0, 0x85663BD6EB5715E8),
        Golden(524288, 1, 0.5, 2.0, 0x88AFA0C739B620E3),
    ]
    # Goldens are x86 torch values; aarch64 libm rounds log/cos/sin differently.
    comptime if CompilationTarget.is_x86():
        for golden in goldens:
            _check_float64(golden)


def test_bfloat16() raises:
    var goldens: List[Golden] = [
        Golden(1, 1, 0.0, 1.0, 0x0000000000003F29),
        Golden(7, 1, 0.0, 1.0, 0x00000000000FDF02),
        Golden(15, 1, 0.0, 1.0, 0x000000000048F2B0),
        Golden(16, 1, 0.0, 1.0, 0x000000000049F687),
        Golden(17, 1, 0.0, 1.0, 0x0000000000569D57),
        Golden(31, 1, 0.0, 1.0, 0x00000000011205D3),
        Golden(1000, 1, 0.0, 1.0, 0x00000003B92C0EAE),
        Golden(524288, 1, 0.0, 1.0, 0x000FE3F7E41F0112),
        Golden(1, 1, 0.5, 2.0, 0x0000000000003FE9),
        Golden(7, 1, 0.5, 2.0, 0x00000000000CEBD5),
        Golden(15, 1, 0.5, 2.0, 0x000000000039A6C9),
        Golden(16, 1, 0.5, 2.0, 0x00000000004101B6),
        Golden(17, 1, 0.5, 2.0, 0x000000000048ED99),
        Golden(31, 1, 0.5, 2.0, 0x0000000000EDCE56),
        Golden(1000, 1, 0.5, 2.0, 0x0000000364D5F9F9),
        Golden(524288, 1, 0.5, 2.0, 0x000E624B112FDDDF),
    ]
    for golden in goldens:
        _check_float32[.bfloat16](golden)


def test_float16() raises:
    var goldens: List[Golden] = [
        Golden(1, 1, 0.0, 1.0, 0x000000000000394A),
        Golden(7, 1, 0.0, 1.0, 0x00000000000EF811),
        Golden(15, 1, 0.0, 1.0, 0x00000000004515D0),
        Golden(16, 1, 0.0, 1.0, 0x000000000046340D),
        Golden(17, 1, 0.0, 1.0, 0x0000000000522ACC),
        Golden(31, 1, 0.0, 1.0, 0x0000000001042E41),
        Golden(1000, 1, 0.0, 1.0, 0x0000000385AC2600),
        Golden(524288, 1, 0.0, 1.0, 0x000F07558ADFF6D1),
        Golden(1, 1, 0.5, 2.0, 0x0000000000003F4A),
        Golden(7, 1, 0.5, 2.0, 0x00000000000C5E8A),
        Golden(15, 1, 0.5, 2.0, 0x00000000003736C1),
        Golden(16, 1, 0.5, 2.0, 0x00000000003D8D57),
        Golden(17, 1, 0.5, 2.0, 0x000000000046AD95),
        Golden(31, 1, 0.5, 2.0, 0x0000000000E57414),
        Golden(1000, 1, 0.5, 2.0, 0x000000034DE4FE59),
        Golden(524288, 1, 0.5, 2.0, 0x000DFA3038E11813),
    ]
    for golden in goldens:
        _check_float32[.float16](golden)


def _check_bits(
    expected: List[UInt32],
    numel: Int,
    seed: UInt64,
    mean: Float32,
    std: Float32,
) raises:
    var storage = List[Float32](length=numel, fill=0)
    torch_cpu_normal_fill(storage.unsafe_ptr(), numel, seed, mean, std)
    for i in range(numel):
        assert_equal(_bits[.float32](storage[i]), UInt64(expected[i]))


def test_float32_values() raises:
    # n=7, seed=0, mean=0.0, std=1.0
    _check_bits(
        [
            0x3FC53F5C,
            0xBE963C50,
            0xC00B7149,
            0x3F1184B6,
            0xBF8AD1A1,
            0xBFB3052D,
            0x3ECE837A,
        ],
        7,
        0,
        0.0,
        1.0,
    )
    # n=17, seed=1, mean=0.5, std=2.0
    _check_bits(
        [
            0xC02346BA,
            0xBFE927AA,
            0x3FF31536,
            0x3F65EE07,
            0x400ECF02,
            0x3F7D2282,
            0xBF53691E,
            0x400755E0,
            0x3FB06B95,
            0x3E17AED8,
            0xC07F6E70,
            0xC0192589,
            0x3F1F4FBF,
            0xBF3C4900,
            0xBF8C4E4E,
            0x3E726F88,
            0xBF8C668D,
        ],
        17,
        1,
        0.5,
        2.0,
    )


def _fill[
    width: Int, full_draw: Bool
](numel: Int, seed: UInt64) -> List[Float32]:
    var values = List[Float32](length=numel, fill=0)
    torch_cpu_normal_fill[full_draw=full_draw, vector_width=width](
        values.unsafe_ptr(), numel, seed, 0.5, 2.0
    )
    return values^


def test_vector_width_does_not_change_the_stream() raises:
    # The goldens above run at the host's native width. The stream is defined
    # by the 16-element block and the i / i + 8 pairing, so any width must give
    # the same bits.
    for numel in [16, 17, 31, 1000, 4099]:
        var reference = _fill[8, True](numel, 3)
        var narrow_reference = _fill[8, False](numel, 3)
        comptime for log2_width in range(6):
            comptime width = 1 << log2_width
            var values = _fill[width, True](numel, 3)
            var narrow_values = _fill[width, False](numel, 3)
            for i in range(numel):
                assert_equal(values[i], reference[i], String(t"width {width}"))
                assert_equal(
                    narrow_values[i],
                    narrow_reference[i],
                    String(t"width {width}"),
                )


def test_seed_uses_the_low_32_bits() raises:
    var low = List[Float32](length=64, fill=0)
    var high = List[Float32](length=64, fill=0)
    torch_cpu_normal_fill(low.unsafe_ptr(), 64, 5, 0.0, 1.0)
    torch_cpu_normal_fill(
        high.unsafe_ptr(), 64, (UInt64(1) << 32) + 5, 0.0, 1.0
    )
    for i in range(64):
        assert_equal(low[i], high[i])


def _random_normal_values[
    dtype: DType
](shape: Coord, seed: UInt64, mean: Float32, stddev: Float32) raises -> List[
    Scalar[dtype]
]:
    """Runs the `mo.random.normal` kernel on a CPU context."""
    var values = List[Scalar[dtype]](length=Int(shape.product()), fill=0)
    var out = values.unsafe_ptr()
    var layout = row_major(shape)
    var seeds = List[UInt64]([seed])

    def store[
        width: SIMDLength
    ](idx: Coord, val: SIMD[dtype, width]) {var out, var layout}:
        out.unsafe_store(Int(layout(idx)), val)

    with DeviceContext.CPU(api="cpu") as ctx:
        random_normal(
            shape,
            mean,
            stddev,
            seeds.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            ctx,
            store,
        )
    return values^


def test_random_normal_kernel_matches_torch() raises:
    # torch.empty(17).normal_(0.5, 2.0) with seed 1, as a [1, 17] tensor.
    var values = _random_normal_values[.float32](Coord(1, 17), 1, 0.5, 2.0)
    var expected: List[UInt32] = [
        0xC02346BA,
        0xBFE927AA,
        0x3FF31536,
        0x3F65EE07,
        0x400ECF02,
        0x3F7D2282,
        0xBF53691E,
        0x400755E0,
        0x3FB06B95,
        0x3E17AED8,
        0xC07F6E70,
        0xC0192589,
        0x3F1F4FBF,
        0xBF3C4900,
        0xBF8C4E4E,
        0x3E726F88,
        0xBF8C668D,
    ]
    for i in range(17):
        assert_equal(_bits[.float32](values[i]), UInt64(expected[i]))


def test_random_normal_kernel_shapes_and_dtypes() raises:
    # A rank-3 shape stores through the same flat stream as a flat one.
    var flat = _random_normal_values[.float32](Coord(60), 9, 0.0, 1.0)
    var shaped = _random_normal_values[.float32](Coord(3, 4, 5), 9, 0.0, 1.0)
    for i in range(60):
        assert_equal(flat[i], shaped[i])

    var wide = _random_normal_values[.float64](Coord(3, 7), 9, 0.0, 1.0)
    var wide_flat = List[Float64](length=21, fill=0)
    torch_cpu_normal_fill(wide_flat.unsafe_ptr(), 21, 9, 0.0, 1.0)
    for i in range(21):
        assert_equal(wide[i], wide_flat[i])

    # bfloat16 rounds the float32 stream that draws per 16-block.
    var narrow = _random_normal_values[.bfloat16](Coord(2, 21), 9, 0.0, 1.0)
    var narrow_flat = List[Float32](length=42, fill=0)
    torch_cpu_normal_fill[full_draw=False](
        narrow_flat.unsafe_ptr(), 42, 9, 0.0, 1.0
    )
    for i in range(42):
        assert_equal(narrow[i], narrow_flat[i].cast[.bfloat16]())


def test_random_normal_kernel_rejects_nonpositive_stddev() raises:
    with assert_raises():
        _ = _random_normal_values[.float32](Coord(4), 0, 0.0, 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
