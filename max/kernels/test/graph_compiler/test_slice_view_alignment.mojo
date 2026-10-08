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
"""Tests the alignment a `mo.slice` view reports for itself.

The reported alignment promises that every address the view can produce is a
multiple of it, and consumers turn it straight into a vector-load width.
`get_view_alignment` is a pure function of shape, strides, starts and steps, so
these cases pin it directly, with no device and no op execution.

Regression test for GEX-4058.
"""

from builtin_kernels.gather_scatter import Slice

from layout import IntTuple, UNKNOWN_VALUE

from std.testing import TestSuite, assert_equal

# A freshly allocated buffer is generously aligned. Every case starts here and
# can only shrink it.
comptime BUFFER_ALIGN = 256

comptime NO_OFFSET_2 = IntTuple(0, 0)
comptime UNIT_STEP_2 = IntTuple(1, 1)
comptime NO_OFFSET_3 = IntTuple(0, 0, 0)
comptime UNIT_STEP_3 = IntTuple(1, 1, 1)
comptime UNKNOWN_SHAPE_3 = IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)


def test_column_slice_bounded_by_row_pitch() raises:
    """`x[:, :8]` of a `[4, 10]` float32 tensor: rows are 40 bytes apart."""
    comptime shape = IntTuple(4, 10)
    comptime strides = IntTuple(10, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, shape, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    assert_equal(got, 8)


def test_aligned_row_pitch_is_a_no_op() raises:
    """`[4, 64]` float32 rows are 256 bytes, so the pitch costs nothing."""
    comptime shape = IntTuple(4, 64)
    comptime strides = IntTuple(64, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, shape, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    # Guards the vectorization the pitch rule exists to enable: clamping every
    # slice to the innermost stride would report 4.
    assert_equal(got, 256)


def test_moe_gate_row_pitch() raises:
    """Selecting 256 routed experts out of 258 logits: 1032-byte rows."""
    comptime shape = IntTuple(8, 258)
    comptime strides = IntTuple(258, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, shape, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    assert_equal(got, 8)


def test_step_scales_the_pitch() raises:
    """A step of 2 doubles what one move along that dimension covers."""
    comptime shape = IntTuple(4, 10)
    comptime strides = IntTuple(10, 1)
    comptime steps = IntTuple(2, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, shape, strides, NO_OFFSET_2, steps
    ](BUFFER_ALIGN)

    assert_equal(got, 16)


def test_innermost_start_offset_shrinks_alignment() raises:
    """`x[:, 2:]` of `[4, 64]` float32: an aligned pitch, an offset base."""
    comptime shape = IntTuple(4, 64)
    comptime strides = IntTuple(64, 1)
    comptime starts = IntTuple(0, 2)
    var got = Slice.get_view_alignment[
        2, DType.float32, shape, strides, starts, UNIT_STEP_2
    ](BUFFER_ALIGN)

    # The pitch is a no-op at 256 bytes, isolating the start-offset path.
    assert_equal(got, 8)


def test_dense_rank3_outer_strides_are_dominated() raises:
    """On a dense source each outer stride is a multiple of the next inner."""
    comptime shape = IntTuple(2, 3, 10)
    comptime strides = IntTuple(30, 10, 1)
    var got = Slice.get_view_alignment[
        3, DType.float32, shape, strides, NO_OFFSET_3, UNIT_STEP_3
    ](BUFFER_ALIGN)

    assert_equal(got, 8)


def test_permuted_rank3_every_dimension_contributes() raises:
    """Strides stop nesting once a view is permuted, so all of them matter."""
    comptime strides = IntTuple(24, 64, 1)
    var got = Slice.get_view_alignment[
        3, DType.float32, UNKNOWN_SHAPE_3, strides, NO_OFFSET_3, UNIT_STEP_3
    ](BUFFER_ALIGN)

    # Folding only `stride[rank - 2]` would report 256, since 64 * 4 is already
    # a multiple of it. Dimension 0 is what brings this down.
    assert_equal(got, 32)


def test_dynamic_stride_bails_to_one() raises:
    """With nothing static inside it, a dynamic stride admits no promise."""
    comptime shape = IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime strides = IntTuple(UNKNOWN_VALUE, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, shape, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    assert_equal(got, 1)


def test_dynamic_batch_stride_keeps_vector_alignment() raises:
    """`[batch, seq_len, 8192]` bfloat16: the batch stride is symbolic."""
    comptime shape = IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, 8192)
    comptime strides = IntTuple(UNKNOWN_VALUE, 8192, 1)
    var got = Slice.get_view_alignment[
        3, DType.bfloat16, shape, strides, NO_OFFSET_3, UNIT_STEP_3
    ](BUFFER_ALIGN)

    # The row pitch below the symbolic dimension is dense and 16 KiB wide, so
    # the batch stride is a multiple of it and costs nothing. Bailing to 1 here
    # is what dropped every activation slice to scalar loads.
    assert_equal(got, 256)


def test_dynamic_batch_stride_with_start_offset() raises:
    """The same view, offset by one along the symbolic dimension."""
    comptime shape = IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, 8192)
    comptime strides = IntTuple(UNKNOWN_VALUE, 8192, 1)
    comptime starts = IntTuple(1, 0, 0)
    var got = Slice.get_view_alignment[
        3, DType.bfloat16, shape, strides, starts, UNIT_STEP_3
    ](BUFFER_ALIGN)

    assert_equal(got, 256)


def test_dynamic_stride_still_bounded_by_row_pitch() raises:
    """The GEX-4058 pitch under `[4, seq_len, 10]`: rows 40 bytes apart."""
    comptime shape = IntTuple(4, UNKNOWN_VALUE, 10)
    comptime strides = IntTuple(UNKNOWN_VALUE, 10, 1)
    var got = Slice.get_view_alignment[
        3, DType.float32, shape, strides, NO_OFFSET_3, UNIT_STEP_3
    ](BUFFER_ALIGN)

    # The dynamic stride inherits the 40-byte bound of the dense row it nests,
    # rather than the buffer's 256.
    assert_equal(got, 8)


def test_dynamic_stride_over_a_strided_suffix_bails() raises:
    """A step already folded into `stride[1]` breaks the dense chain."""
    comptime shape = IntTuple(4, UNKNOWN_VALUE, 10)
    comptime strides = IntTuple(UNKNOWN_VALUE, 160, 1)
    var got = Slice.get_view_alignment[
        3, DType.float32, shape, strides, NO_OFFSET_3, UNIT_STEP_3
    ](BUFFER_ALIGN)

    # 160 is not `shape[2] * stride[2]`, so nothing bounds the outer stride.
    assert_equal(got, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
