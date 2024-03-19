# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math

from buffer import NDBuffer

from buffer.list import DimList


@register_passable("trivial")
struct Weighted2DPoint[type: DType]:

    """Utility class to wrap 2-d point coordinates and floating point weight for
    biliniear interpolation.
    """

    var y: Int
    var x: Int
    var w: Scalar[type]

    fn __init__(inout self, y: Int, x: Int, weight: Scalar[type]):
        self.y = y
        self.x = x
        self.w = weight


@always_inline
fn _bilinear_interpolate[
    type: DType
](
    ph: Int,
    pw: Int,
    iy: Int,
    ix: Int,
    height: Int,
    width: Int,
    roi_start_h: Float32,
    roi_start_w: Float32,
    bin_size_h: Float32,
    bin_size_w: Float32,
    roi_bin_grid_h: Int,
    roi_bin_grid_w: Int,
) -> Tuple[
    Weighted2DPoint[type],
    Weighted2DPoint[type],
    Weighted2DPoint[type],
    Weighted2DPoint[type],
]:
    # Compute centeral point (y, x) by mapping (py, ph) into a  grid of size
    # [roi_bin_grid_h, roi_bin_grid_w] shifted by (roi_start_h, roi_start_w)
    var y = math.max(
        roi_start_h
        + ph * bin_size_h
        + (iy + 0.5) * bin_size_h / roi_bin_grid_h,
        0,
    )
    var x = math.max(
        roi_start_w
        + pw * bin_size_w
        + (ix + 0.5) * bin_size_w / roi_bin_grid_w,
        0,
    )
    var topLeft = y == 0 and x == 0
    if topLeft or y > height or x > width:
        var zeroPoint = Weighted2DPoint[type](0, 0, 0)
        return (zeroPoint, zeroPoint, zeroPoint, zeroPoint)

    # Compute box coordinates:
    #   (y_low,  x_low)      (y_low,  x_high)
    #
    #                   (x, y)
    #
    #   (y_high, x_low)      (y_high, x_high)
    # and bilinar weights (w1, w2, w3, w4)
    var y_low = math.min(int(y), height - 1)
    var x_low = math.min(int(x), width - 1)
    var y_high = math.min(y_low + 1, height - 1)
    var x_high = math.min(x_low + 1, width - 1)

    var ly = y - y_low
    var lx = x - x_low
    var hy = 1.0 - ly
    var hx = 1.0 - lx

    var w1 = hy * hx
    var w2 = hy * lx
    var w3 = ly * hx
    var w4 = ly * lx

    var p1 = Weighted2DPoint[type](y_low, x_low, w1.cast[type]())
    var p2 = Weighted2DPoint[type](y_low, x_high, w2.cast[type]())
    var p3 = Weighted2DPoint[type](y_high, x_low, w3.cast[type]())
    var p4 = Weighted2DPoint[type](y_high, x_high, w4.cast[type]())

    return (p1, p2, p3, p4)


@always_inline
fn roi_align_nhwc[
    type: DType,
    input_shape: DimList,
    roi_shape: DimList,
    aligned: Bool,
    mode: StringLiteral = "AVG",
](
    output: NDBuffer[type, 4],
    input: NDBuffer[type, 4, input_shape],
    rois: NDBuffer[type, 2, roi_shape],
    output_height: Int,
    output_width: Int,
    spatial_scale: Float32,
    sampling_ratio: Float32,
):
    """
    Compute ROIAlign a batch of rois of shape [M, 5] where the first dim is the
    batch index, followed by region box coordinates (y0, x0) (y1, x1). For
    inputs of NHWC format. The output shape is
    [M, output_height, output_width, C].

    Paramerers:
        type: Type of the input tensor.
        input_shape: Shape of the input tensor.
        roi_shape: Shape of regions of interests (ROI).
        aligned: If not true offset the ROIs by 0.5.
        mode: The pooling mode "AVG" for average and "MAX" for max pooling.
    Args:
        output: Pre-allocated output tensor.
        input: Batched images to the roi_align with NHWC format.
        rois: Batched ROIs box coordinates.
        output_height: Pooled output height.
        output_width: Pooled output width.
        spatial_scale: Scale of ROIs from spatial scale to pooling scale.
        sampling_ratio: Number of sampling points in the interpolation grid
          used to compute the output value of each pooled bin.
    """

    constrained[
        type.is_floating_point(),
        "ROI align input / output must be a floating point",
    ]()

    debug_assert(mode == "AVG" or mode == "MAX", "mode must be AVG or MAX")

    var n_regions = rois.dim(0)
    var height = input.dim(1)
    var width = input.dim(2)
    var channles = input.dim(3)

    var pooled_height = output_height
    var pooled_width = output_width
    var offset = 0.5 if aligned else 0.0

    for ri in range(n_regions):
        # Region coordinates and batch indix
        var roi_batch_idx = int(rois[ri, 0])
        var roi_start_w = rois[ri, 1].cast[
            DType.float32
        ]() * spatial_scale - offset
        var roi_start_h = rois[ri, 2].cast[
            DType.float32
        ]() * spatial_scale - offset
        var roi_end_w = rois[ri, 3].cast[
            DType.float32
        ]() * spatial_scale - offset
        var roi_end_h = rois[ri, 4].cast[
            DType.float32
        ]() * spatial_scale - offset

        # Region size (roi_h, roi_w) with 1x1 lower bound
        var roi_height = roi_end_h - roi_start_h if aligned else math.max(
            roi_end_h - roi_start_h, 1.0
        )
        var roi_width = roi_end_w - roi_start_w if aligned else math.max(
            roi_end_w - roi_start_w, 1.0
        )

        # Bin size for region.
        var bin_size_h = roi_height / pooled_height
        var bin_size_w = roi_width / pooled_width

        # Use pooling window size as either sampling_ratio x sampling_ratio or
        # ⌈bin_size_h x bin_size_w⌉.
        var roi_bin_grid_h = int(
            sampling_ratio if sampling_ratio > 0 else math.ceil(bin_size_h)
        )
        var roi_bin_grid_w = int(
            sampling_ratio if sampling_ratio > 0 else math.ceil(bin_size_w)
        )

        # Number of points in the pooling window.
        var pool_elemn_num = math.max(roi_bin_grid_h * roi_bin_grid_w, 1)

        # Associatve pooling init/update/finalize functions parameterized by
        # mode
        @parameter
        @always_inline
        fn init_fn[type: DType]() -> Scalar[type]:
            if mode == "AVG":
                return 0
            else:
                return math.limit.neginf[type]()

        @parameter
        @always_inline
        fn update_fn[
            type: DType
        ](a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
            if mode == "AVG":
                return a + b
            else:
                return math.max(a, b)

        @parameter
        @always_inline
        fn reduce_fn[
            type: DType
        ](a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
            if mode == "AVG":
                return a / b
            else:
                return a

        for ph in range(pooled_height):
            for pw in range(pooled_width):
                for c in range(channles):
                    var pool_val = init_fn[type]()
                    for iy in range(roi_bin_grid_h):
                        for ix in range(roi_bin_grid_w):
                            # Sample bilinearly mapped points coordinates
                            # and weights.
                            var p = _bilinear_interpolate[type](
                                ph,
                                pw,
                                iy,
                                ix,
                                height,
                                width,
                                roi_start_h,
                                roi_start_w,
                                bin_size_h,
                                bin_size_w,
                                roi_bin_grid_h,
                                roi_bin_grid_w,
                            )
                            var p1 = p.get[0, Weighted2DPoint[type]]()
                            var p2 = p.get[1, Weighted2DPoint[type]]()
                            var p3 = p.get[2, Weighted2DPoint[type]]()
                            var p4 = p.get[3, Weighted2DPoint[type]]()
                            pool_val = update_fn(
                                pool_val,
                                p1.w * input[roi_batch_idx, p1.y, p1.x, c],
                            )
                            pool_val = update_fn(
                                pool_val,
                                p2.w * input[roi_batch_idx, p2.y, p2.x, c],
                            )
                            pool_val = update_fn(
                                pool_val,
                                p3.w * input[roi_batch_idx, p3.y, p3.x, c],
                            )
                            pool_val = update_fn(
                                pool_val,
                                p4.w * input[roi_batch_idx, p4.y, p4.x, c],
                            )
                    output[StaticIntTuple[4](ri, ph, pw, c)] = reduce_fn(
                        pool_val, pool_elemn_num
                    )
