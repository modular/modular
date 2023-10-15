# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory.buffer import NDBuffer

from nn.roialign import roi_align_nhwc


# CHECK-LABEL: test_roi_align_avg
fn test_roi_align_avg():
    print("=== test_roi_align_avg")

    alias in_shape = DimList(1, 10, 10, 1)
    alias out_shape = DimList(1, 5, 5, 1)
    alias roi_shape = DimList(1, 5)

    var input = NDBuffer[4, in_shape, DType.float32].stack_allocation()
    var output = NDBuffer[4, out_shape, DType.float32].stack_allocation()
    var rois = NDBuffer[2, roi_shape, DType.float32].stack_allocation()

    for i in range(10):
        for j in range(10):
            input[StaticIntTuple[4](0, i, j, 0)] = i * 10 + j

    rois[StaticIntTuple[2](0, 0)] = 0
    rois[StaticIntTuple[2](0, 1)] = 0
    rois[StaticIntTuple[2](0, 2)] = 0
    rois[StaticIntTuple[2](0, 3)] = 4
    rois[StaticIntTuple[2](0, 4)] = 4

    roi_align_nhwc[DType.float32, in_shape, roi_shape, False,](
        output.make_dims_unknown(),
        input,
        rois,
        Int(out_shape.at[1]().get()),
        Int(out_shape.at[2]().get()),
        1.0,
        2.0,
    )
    # CHECK: 4.4000000953674316, 5.2000002861022949, 6.0000004768371582, 6.8000006675720215, 7.6000008583068848,
    # CHECK: 12.399999618530273, 13.200000762939453, 14.0, 14.80000114440918, 15.600000381469727,
    # CHECK: 20.400001525878906, 21.19999885559082, 22.000001907348633, 22.80000114440918, 23.599998474121094,
    # CHECK: 28.399999618530273, 29.200002670288086, 30.0, 30.799999237060547, 31.600000381469727,
    # CHECK: 36.400001525878906, 37.200000762939453, 38.0, 38.799999237060547, 39.600006103515625,
    for i in range(5):
        for j in range(5):
            print_no_newline(output[StaticIntTuple[4](0, i, j, 0)])
            print_no_newline(", ")
        print("")


# CHECK-LABEL: test_roi_align_max
fn test_roi_align_max():
    print("=== test_roi_align_max")

    alias in_shape = DimList(1, 10, 10, 1)
    alias out_shape = DimList(1, 5, 5, 1)
    alias roi_shape = DimList(1, 5)

    var input = NDBuffer[4, in_shape, DType.float32].stack_allocation()
    var output = NDBuffer[4, out_shape, DType.float32].stack_allocation()
    var rois = NDBuffer[2, roi_shape, DType.float32].stack_allocation()

    for i in range(10):
        for j in range(10):
            input[StaticIntTuple[4](0, i, j, 0)] = i * 10 + j

    rois[StaticIntTuple[2](0, 0)] = 0
    rois[StaticIntTuple[2](0, 1)] = 0
    rois[StaticIntTuple[2](0, 2)] = 0
    rois[StaticIntTuple[2](0, 3)] = 4
    rois[StaticIntTuple[2](0, 4)] = 4

    roi_align_nhwc[DType.float32, in_shape, roi_shape, False, "MAX",](
        output.make_dims_unknown(),
        input,
        rois,
        Int(out_shape.at[1]().get()),
        Int(out_shape.at[2]().get()),
        1.0,
        2.0,
    )

    # CHECK: 4.8000001907348633, 6.6000003814697266, 5.7600007057189941, 7.8000001907348633, 6.7200021743774414,
    # CHECK: 8.0, 11.0, 9.6000003814697266, 13.0, 11.200002670288086,
    # CHECK: 12.80000114440918, 16.80000114440918, 14.080001831054688, 18.400001525878906, 15.360005378723145,
    # CHECK: 24.0, 31.0, 25.600002288818359, 33.0, 27.200006484985352,
    # CHECK: 25.600006103515625, 32.800006866455078, 26.880008697509766, 34.400009155273438, 28.160013198852539,
    for i in range(5):
        for j in range(5):
            print_no_newline(output[StaticIntTuple[4](0, i, j, 0)])
            print_no_newline(", ")
        print("")


fn main():
    test_roi_align_avg()
    test_roi_align_max()
