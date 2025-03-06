# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.roi_align import roi_align_nhwc
from testing import *

from utils import IndexList


def test_roi_align_avg[scale_type: DType]():
    print("=== test_roi_align_avg")

    alias in_shape = DimList(1, 10, 10, 1)
    alias out_shape = DimList(1, 5, 5, 1)
    alias roi_shape = DimList(1, 5)

    var input_stack = InlineArray[Float32, Int(in_shape.product())](
        unsafe_uninitialized=True
    )
    var input = NDBuffer[DType.float32, 4, in_shape](input_stack.unsafe_ptr())
    var output_stack = InlineArray[Float32, Int(out_shape.product())](
        unsafe_uninitialized=True
    )
    var output = NDBuffer[DType.float32, 4, out_shape](
        output_stack.unsafe_ptr()
    )
    var rois_stack = InlineArray[Float32, Int(roi_shape.product())](
        unsafe_uninitialized=True
    )
    var rois = NDBuffer[DType.float32, 2, roi_shape](rois_stack.unsafe_ptr())

    for i in range(10):
        for j in range(10):
            input[0, i, j, 0] = i * 10 + j

    rois[0, 0] = 0
    rois[0, 1] = 0
    rois[0, 2] = 0
    rois[0, 3] = 4
    rois[0, 4] = 4

    roi_align_nhwc[aligned=False](
        output.make_dims_unknown(),
        input,
        rois,
        out_shape.at[1]().get(),
        out_shape.at[2]().get(),
        Scalar[scale_type](1.0),
        Scalar[scale_type](2.0),
    )

    assert_almost_equal(output[0, 0, 0, 0], 4.4000000953674316)
    assert_almost_equal(output[0, 0, 1, 0], 5.2000002861022949)
    assert_almost_equal(output[0, 0, 2, 0], 6.0000004768371582)
    assert_almost_equal(output[0, 0, 3, 0], 6.8000006675720215)
    assert_almost_equal(output[0, 0, 4, 0], 7.6000008583068848)

    assert_almost_equal(output[0, 1, 0, 0], 12.399999618530273)
    assert_almost_equal(output[0, 1, 1, 0], 13.200000762939453)
    assert_almost_equal(output[0, 1, 2, 0], 14.0)
    assert_almost_equal(output[0, 1, 3, 0], 14.80000114440918)
    assert_almost_equal(output[0, 1, 4, 0], 15.600000381469727)

    assert_almost_equal(output[0, 2, 0, 0], 20.400001525878906)
    assert_almost_equal(output[0, 2, 1, 0], 21.19999885559082)
    assert_almost_equal(output[0, 2, 2, 0], 22.000001907348633)
    assert_almost_equal(output[0, 2, 3, 0], 22.80000114440918)
    assert_almost_equal(output[0, 2, 4, 0], 23.599998474121094)

    assert_almost_equal(output[0, 3, 0, 0], 28.399999618530273)
    assert_almost_equal(output[0, 3, 1, 0], 29.200002670288086)
    assert_almost_equal(output[0, 3, 2, 0], 30.0)
    assert_almost_equal(output[0, 3, 3, 0], 30.799999237060547)
    assert_almost_equal(output[0, 3, 4, 0], 31.600000381469727)

    assert_almost_equal(output[0, 4, 0, 0], 36.400001525878906)
    assert_almost_equal(output[0, 4, 1, 0], 37.200000762939453)
    assert_almost_equal(output[0, 4, 2, 0], 38.0)
    assert_almost_equal(output[0, 4, 3, 0], 38.799999237060547)
    assert_almost_equal(output[0, 4, 4, 0], 39.600006103515625)


def test_roi_align_max():
    print("=== test_roi_align_max")

    alias in_shape = DimList(1, 10, 10, 1)
    alias out_shape = DimList(1, 5, 5, 1)
    alias roi_shape = DimList(1, 5)

    var input_stack = InlineArray[Float32, Int(in_shape.product())](
        unsafe_uninitialized=True
    )
    var input = NDBuffer[DType.float32, 4, in_shape](input_stack.unsafe_ptr())
    var output_stack = InlineArray[Float32, Int(out_shape.product())](
        unsafe_uninitialized=True
    )
    var output = NDBuffer[DType.float32, 4, out_shape](
        output_stack.unsafe_ptr()
    )
    var rois_stack = InlineArray[Float32, Int(roi_shape.product())](
        unsafe_uninitialized=True
    )
    var rois = NDBuffer[DType.float32, 2, roi_shape](rois_stack.unsafe_ptr())

    for i in range(10):
        for j in range(10):
            input[0, i, j, 0] = i * 10 + j

    rois[0, 0] = 0
    rois[0, 1] = 0
    rois[0, 2] = 0
    rois[0, 3] = 4
    rois[0, 4] = 4

    roi_align_nhwc[aligned=False, mode="MAX"](
        output.make_dims_unknown(),
        input,
        rois,
        out_shape.at[1]().get(),
        out_shape.at[2]().get(),
        1.0,
        2.0,
    )

    assert_almost_equal(output[0, 0, 0, 0], 4.8000001907348633)
    assert_almost_equal(output[0, 0, 1, 0], 6.6000003814697266)
    assert_almost_equal(output[0, 0, 2, 0], 5.7600007057189941)
    assert_almost_equal(output[0, 0, 3, 0], 7.8000001907348633)
    assert_almost_equal(output[0, 0, 4, 0], 6.7200021743774414)

    assert_almost_equal(output[0, 1, 0, 0], 8.0)
    assert_almost_equal(output[0, 1, 1, 0], 11.0)
    assert_almost_equal(output[0, 1, 2, 0], 9.6000003814697266)
    assert_almost_equal(output[0, 1, 3, 0], 13.0)
    assert_almost_equal(output[0, 1, 4, 0], 11.200002670288086)

    assert_almost_equal(output[0, 2, 0, 0], 12.80000114440918)
    assert_almost_equal(output[0, 2, 1, 0], 16.80000114440918)
    assert_almost_equal(output[0, 2, 2, 0], 14.080001831054688)
    assert_almost_equal(output[0, 2, 3, 0], 18.400001525878906)
    assert_almost_equal(output[0, 2, 4, 0], 15.360005378723145)

    assert_almost_equal(output[0, 3, 0, 0], 24.0)
    assert_almost_equal(output[0, 3, 1, 0], 31.0)
    assert_almost_equal(output[0, 3, 2, 0], 25.600002288818359)
    assert_almost_equal(output[0, 3, 3, 0], 33.0)
    assert_almost_equal(output[0, 3, 4, 0], 27.200006484985352)

    assert_almost_equal(output[0, 4, 0, 0], 25.600006103515625)
    assert_almost_equal(output[0, 4, 1, 0], 32.800006866455078)
    assert_almost_equal(output[0, 4, 2, 0], 26.880008697509766)
    assert_almost_equal(output[0, 4, 3, 0], 34.400009155273438)
    assert_almost_equal(output[0, 4, 4, 0], 28.160013198852539)


def test_roi_align_KERN_692():
    print("=== test_roi_align_KERN_692")

    alias in_shape = DimList(1, 6, 6, 1)
    alias out_shape = DimList(1, 3, 3, 1)
    alias roi_shape = DimList(1, 5)

    var input_stack = InlineArray[Float32, Int(in_shape.product())](
        unsafe_uninitialized=True
    )
    var input = NDBuffer[DType.float32, 4, in_shape](input_stack.unsafe_ptr())
    var output_stack = InlineArray[Float32, Int(out_shape.product())](
        unsafe_uninitialized=True
    )
    var output = NDBuffer[DType.float32, 4, out_shape](
        output_stack.unsafe_ptr()
    )
    var rois_stack = InlineArray[Float32, Int(roi_shape.product())](
        unsafe_uninitialized=True
    )
    var rois = NDBuffer[DType.float32, 2, roi_shape](rois_stack.unsafe_ptr())

    for i in range(6):
        for j in range(6):
            input[0, i, j, 0] = i * 6 + j + 1

    rois[0, 0] = 0
    rois[0, 1] = -2
    rois[0, 2] = -2
    rois[0, 3] = 22
    rois[0, 4] = 22

    roi_align_nhwc[aligned=False](
        output.make_dims_unknown(),
        input,
        rois,
        out_shape.at[1]().get(),
        out_shape.at[2]().get(),
        0.25,
        2.0,
    )

    assert_almost_equal(output[0, 0, 0, 0], 4.5)
    assert_almost_equal(output[0, 0, 1, 0], 6.5)
    assert_almost_equal(output[0, 0, 2, 0], 8.5)

    assert_almost_equal(output[0, 1, 0, 0], 16.5)
    assert_almost_equal(output[0, 1, 1, 0], 18.5)
    assert_almost_equal(output[0, 1, 2, 0], 20.5)

    assert_almost_equal(output[0, 2, 0, 0], 28.5)
    assert_almost_equal(output[0, 2, 1, 0], 30.5)
    assert_almost_equal(output[0, 2, 2, 0], 32.5)


def main():
    test_roi_align_avg[DType.float32]()
    test_roi_align_avg[DType.float64]()
    test_roi_align_max()
    test_roi_align_KERN_692()
