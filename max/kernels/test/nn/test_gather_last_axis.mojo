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

# Checks the CPU `gather` along the input's last axis against a scalar
# reference. That path vectorizes the index load and output store, so the tests
# cover index counts around the SIMD width (remainders), negative and
# out-of-range indices, both index dtypes, uint8 and float32 data, and ranks
# 1 to 4.

from std.sys import simd_width_of
from std.testing import TestSuite, assert_equal, assert_raises

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.gather_scatter import gather


def _data[dtype: DType](i: Int) -> Scalar[dtype]:
    return Scalar[dtype]((i * 13 + 5) % 251)


def _index[indices_type: DType](k: Int, dim: Int) -> Scalar[indices_type]:
    """A deterministic index in `[-dim, dim)`, so half of them are negative."""
    return Scalar[indices_type](((k * 37 + 11) % (2 * dim)) - dim)


def _fill_data[dtype: DType, size: Int](mut array: Array[Scalar[dtype], size]):
    for i in range(size):
        array[i] = _data[dtype](i)


def _fill_indices[
    indices_type: DType, size: Int
](mut array: Array[Scalar[indices_type], size], dim: Int):
    for k in range(size):
        array[k] = _index[indices_type](k, dim)


def _check_last_axis_gather[
    dtype: DType, indices_type: DType, //
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
) raises:
    """Gathers on the last axis and compares against a scalar loop.

    All three tensors are contiguous row-major, so a flat walk over the
    storage is the reference: `output[o, k] = input[o, indices[k]]`.
    """
    comptime axis = input.rank - 1
    gather[axis=axis](
        output.make_dynamic[.int64](),
        input.make_dynamic[.int64](),
        indices.make_dynamic[.int64](),
        context=DeviceContext.cpu(),
    )

    var dim = Int(input.dim[axis]())
    var num_indices = indices.num_elements()
    var outer = input.num_elements() // dim
    for o in range(outer):
        for k in range(num_indices):
            var j = Int(indices.ptr[unsafe_offset=k])
            if j < 0:
                j += dim
            var got = output.ptr[unsafe_offset=o * num_indices + k]
            var expected = input.ptr[unsafe_offset=o * dim + j]
            assert_equal(got, expected)


def _test_ranks[dtype: DType, indices_type: DType, dim: Int, n: Int]() raises:
    """Runs a last-axis gather of `n` indices at every input rank from 1 to 4.
    """
    # Rank 1: input [dim], indices [n], output [n].
    var in1_stack = Array[Scalar[dtype], dim](fill={})
    _fill_data(in1_stack)
    var idx_stack = Array[Scalar[indices_type], n](fill={})
    _fill_indices(idx_stack, dim)
    var out1_stack = Array[Scalar[dtype], n](fill={})
    _check_last_axis_gather(
        TileTensor(out1_stack, row_major[n]()),
        TileTensor(in1_stack, row_major[dim]()),
        TileTensor(idx_stack, row_major[n]()),
    )

    # Rank 2: input [3, dim], indices [n], output [3, n].
    var in2_stack = Array[Scalar[dtype], 3 * dim](fill={})
    _fill_data(in2_stack)
    var out2_stack = Array[Scalar[dtype], 3 * n](fill={})
    _check_last_axis_gather(
        TileTensor(out2_stack, row_major[3, n]()),
        TileTensor(in2_stack, row_major[3, dim]()),
        TileTensor(idx_stack, row_major[n]()),
    )

    # Rank 3: input [2, 3, dim], indices [n], output [2, 3, n].
    var in3_stack = Array[Scalar[dtype], 6 * dim](fill={})
    _fill_data(in3_stack)
    var out3_stack = Array[Scalar[dtype], 6 * n](fill={})
    _check_last_axis_gather(
        TileTensor(out3_stack, row_major[2, 3, n]()),
        TileTensor(in3_stack, row_major[2, 3, dim]()),
        TileTensor(idx_stack, row_major[n]()),
    )

    # Rank 4: input [2, 2, 2, dim], indices [n], output [2, 2, 2, n].
    var in4_stack = Array[Scalar[dtype], 8 * dim](fill={})
    _fill_data(in4_stack)
    var out4_stack = Array[Scalar[dtype], 8 * n](fill={})
    _check_last_axis_gather(
        TileTensor(out4_stack, row_major[2, 2, 2, n]()),
        TileTensor(in4_stack, row_major[2, 2, 2, dim]()),
        TileTensor(idx_stack, row_major[n]()),
    )


def _test_widths[dtype: DType, indices_type: DType, dim: Int]() raises:
    """Index counts below, at, and above the SIMD width, plus whole vectors and
    a remainder.
    """
    comptime width = simd_width_of[dtype]()
    _test_ranks[dtype, indices_type, dim, 1]()
    _test_ranks[dtype, indices_type, dim, width - 1]()
    _test_ranks[dtype, indices_type, dim, width]()
    _test_ranks[dtype, indices_type, dim, width + 1]()
    _test_ranks[dtype, indices_type, dim, 3 * width + 5]()


def test_float32_int32_indices() raises:
    _test_widths[.float32, .int32, 17]()


def test_float32_int64_indices() raises:
    _test_widths[.float32, .int64, 17]()


def test_uint8_int32_indices() raises:
    _test_widths[.uint8, .int32, 17]()


def test_uint8_int64_indices() raises:
    _test_widths[.uint8, .int64, 17]()


def test_uint8_lut() raises:
    """A 256-entry uint8 lookup table applied to an image-like [rows, cols]
    index tensor.
    """
    comptime rows = 33
    comptime cols = 129
    var lut_stack = Array[UInt8, 256](fill={})
    for i in range(256):
        lut_stack[i] = UInt8(255 - i)
    var idx_stack = Array[Int32, rows * cols](fill={})
    for k in range(rows * cols):
        idx_stack[k] = Int32((k * 7) % 256)
    var out_stack = Array[UInt8, rows * cols](fill={})
    var out = TileTensor(out_stack, row_major[rows, cols]())
    gather[axis=0](
        out.make_dynamic[.int64](),
        TileTensor(lut_stack, row_major[256]()).make_dynamic[.int64](),
        TileTensor(idx_stack, row_major[rows, cols]()).make_dynamic[.int64](),
        context=DeviceContext.cpu(),
    )
    for k in range(rows * cols):
        assert_equal(out_stack[k], UInt8(255 - (k * 7) % 256))


def test_multidimensional_indices() raises:
    """Indices of rank 2 against a rank-2 input: output is [3, 2, n]."""
    comptime dim = 17
    comptime n = 2 * simd_width_of[DType.float32]() + 3
    var in_stack = Array[Float32, 3 * dim](fill={})
    _fill_data(in_stack)
    var idx_stack = Array[Int64, 2 * n](fill={})
    _fill_indices(idx_stack, dim)
    var out_stack = Array[Float32, 3 * 2 * n](fill={})
    _check_last_axis_gather(
        TileTensor(out_stack, row_major[3, 2, n]()),
        TileTensor(in_stack, row_major[3, dim]()),
        TileTensor(idx_stack, row_major[2, n]()),
    )


def _expect_out_of_bounds[
    indices_type: DType
](
    bad_lane: Int,
    bad_index: Int,
    later_lane: Int = -1,
    later_index: Int = 0,
) raises:
    """Places an out-of-range index at `bad_lane` and expects the same error
    as the scalar path, which reports the index as given (not normalized). An
    optional second one at `later_lane` must not be the one reported.
    """
    comptime dim = 17
    comptime n = 2 * simd_width_of[DType.float32]() + 3
    var in_stack = Array[Float32, 3 * dim](fill={})
    _fill_data(in_stack)
    var idx_stack = Array[Scalar[indices_type], n](fill={})
    _fill_indices(idx_stack, dim)
    idx_stack[bad_lane] = Scalar[indices_type](bad_index)
    if later_lane >= 0:
        idx_stack[later_lane] = Scalar[indices_type](later_index)
    var out_stack = Array[Float32, 3 * n](fill={})
    with assert_raises(
        contains=String(
            "gather index {} is out of bounds for axis 1 with size {}"
        ).format(bad_index, dim)
    ):
        gather[axis=1](
            TileTensor(out_stack, row_major[3, n]()).make_dynamic[.int64](),
            TileTensor(in_stack, row_major[3, dim]()).make_dynamic[.int64](),
            TileTensor(idx_stack, row_major[n]()).make_dynamic[.int64](),
            context=DeviceContext.cpu(),
        )


def test_out_of_bounds() raises:
    comptime width = simd_width_of[DType.float32]()
    # Too large and too negative, in a vector body lane and in the remainder.
    for bad_index in [17, 100, -18, -100]:
        _expect_out_of_bounds[.int32](1, bad_index)
        _expect_out_of_bounds[.int64](width + 1, bad_index)
        _expect_out_of_bounds[.int32](2 * width + 2, bad_index)


def test_out_of_bounds_reports_first_in_vector() raises:
    """Lanes 0-3 share a vector at every SIMD width of at least 4."""
    _expect_out_of_bounds[.int32](1, 100, later_lane=3, later_index=-100)
    _expect_out_of_bounds[.int64](0, -18, later_lane=2, later_index=17)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
