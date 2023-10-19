# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import abs, iota, max, min
from math.limit import min_or_neginf

from algorithm.sort import _quicksort
from memory.buffer import NDBuffer
from tensor import Tensor, TensorShape

from utils.index import Index
from utils.vector import DynamicVector


@value
struct BoundingBox[type: DType]:
    var nw: SIMD[type, 2]
    var se: SIMD[type, 2]

    fn __init__(
        inout self,
        y1: SIMD[type, 1],
        x1: SIMD[type, 1],
        y2: SIMD[type, 1],
        x2: SIMD[type, 1],
    ):
        self.nw = SIMD[type, 2](max(y1, y2), max(x1, x2))
        self.se = SIMD[type, 2](min(y1, y2), min(x1, x2))

    fn iou(self, other: BoundingBox[type]) -> SIMD[type, 1]:
        let intersection_area = self.intersection_area(other)

        let union_area = self.area() + other.area() - intersection_area
        let iou_val = abs(intersection_area) / abs(union_area)
        return iou_val

    fn intersection_area(self, other: BoundingBox[type]) -> SIMD[type, 1]:
        let nw = min(self.nw, other.nw)
        let se = max(self.se, other.se)

        if nw[1] < se[1] or nw[0] < se[0]:
            return 0

        return Self(nw, se).area()

    fn area(self) -> SIMD[type, 1]:
        return (self.se[0] - self.nw[0]) * (self.se[1] - self.nw[1])


@always_inline
fn _get_bounding_box[
    type: DType
](
    batch_size: Int,
    box_idx: Int,
    boxes: NDBuffer[3, DimList.create_unknown[3](), type],
) -> BoundingBox[type]:
    let y1 = boxes[batch_size, box_idx, 0]
    let x1 = boxes[batch_size, box_idx, 1]
    let y2 = boxes[batch_size, box_idx, 2]
    let x2 = boxes[batch_size, box_idx, 3]
    return BoundingBox(y1, x1, y2, x2)


fn non_max_suppression[
    type: DType
](
    boxes: NDBuffer[3, DimList.create_unknown[3](), type],
    scores: NDBuffer[3, DimList.create_unknown[3](), type],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> Tensor[DType.int64]:
    """Value semantic overload. Graph compiler does not support this yet."""
    var output_predictions = DynamicVector[Int64]()

    @parameter
    @always_inline
    fn store_to_outputs(batch_idx: Int64, class_idx: Int64, box_idx: Int64):
        output_predictions.push_back(batch_idx)
        output_predictions.push_back(class_idx)
        output_predictions.push_back(box_idx)

    non_max_suppression[type, store_to_outputs](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )

    # note that the output tensor takes ownership of output_predictions.data
    # output_predictions.data may be larger than the actual tensor size indicated
    # by the shape, but that is OK since the tensor.__del__() frees the pointer
    let output_shape = TensorShape(output_predictions.__len__() // 3, 3)
    return Tensor[DType.int64](
        rebind[DTypePointer[DType.int64]](output_predictions.data), output_shape
    )


fn non_max_suppression[
    type: DType
](
    boxes: NDBuffer[3, DimList.create_unknown[3](), type],
    scores: NDBuffer[3, DimList.create_unknown[3](), type],
    output: NDBuffer[2, DimList.create_unknown[2](), DType.int64],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
):
    """Buffer semantic overload."""
    var pred_count = 0

    @parameter
    @always_inline
    fn store_to_outputs(batch_idx: Int64, class_idx: Int64, box_idx: Int64):
        output[Index(pred_count, 0)] = batch_idx
        output[Index(pred_count, 1)] = class_idx
        output[Index(pred_count, 2)] = box_idx
        pred_count += 1

    non_max_suppression[type, store_to_outputs](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )


fn non_max_suppression_shape_func[
    type: DType
](
    boxes: NDBuffer[3, DimList.create_unknown[3](), type],
    scores: NDBuffer[3, DimList.create_unknown[3](), type],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> StaticIntTuple[2]:
    """Overload to compute the output shape. Can be removed once the graph compiler
    supports value semantic kernels that allocate their own output."""
    var box_pred_count: Int64 = 0

    @parameter
    @always_inline
    fn incr_pred_count(batch_idx: Int64, class_idx: Int64, box_idx: Int64):
        box_pred_count += 1

    non_max_suppression[type, incr_pred_count](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )

    return StaticIntTuple[2](box_pred_count.__int__(), 3)


fn non_max_suppression[
    type: DType,
    func: fn (Int64, Int64, Int64) capturing -> None,
](
    boxes: NDBuffer[3, DimList.create_unknown[3](), type],
    scores: NDBuffer[3, DimList.create_unknown[3](), type],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
):
    """Implements the NonMaxSuppression operator from the ONNX spec https://github.com/onnx/onnx/blob/main/docs/Operators.md#nonmaxsuppression."""

    let batch_size = boxes.dim(0)
    let num_boxes = boxes.dim(1)
    let num_classes = scores.dim(1)

    debug_assert(
        boxes.dim(2) == 4,
        (
            "boxes must be specified with the 2D coords representing the"
            " diagonal corners"
        ),
    )
    debug_assert(
        boxes.dim(0) == scores.dim(0), "dim 0 of boxes and scores must be equal"
    )
    debug_assert(
        boxes.dim(1) == scores.dim(2),
        "boxes and scores must contain the same number of boxes",
    )

    if max_output_boxes_per_class == 0:
        return

    var box_idxs = DynamicVector[Int64](num_boxes)
    box_idxs.resize(num_boxes)

    var per_class_scores = DynamicVector[SIMD[type, 1]](num_boxes)
    per_class_scores.resize(num_boxes)

    for b in range(batch_size):
        for c in range(num_classes):
            # entries of per_class_scores_ptr are set to neginf when they no longer
            # correspond to an eligible box
            # this happens when:
            #   1. score does not meet score threshold
            #   2. iou with an existing prediction is above the IOU threshold
            let per_class_scores_ptr = scores._offset(Index(b, c, 0))

            # filter so that we only consider scores above the threshold
            # reduces the number of box_idxs to sort
            var num_boxes_remaining = 0
            for i in range(num_boxes):
                let score = per_class_scores_ptr.load(i)
                if score > score_threshold.cast[type]():
                    per_class_scores[i] = score
                    num_boxes_remaining += 1
                else:
                    per_class_scores[i] = min_or_neginf[type]()

            iota[DType.int64](box_idxs)

            @parameter
            @always_inline
            fn _greater_than[ty: AnyType](lhs: ty, rhs: ty) -> Bool:
                return (
                    per_class_scores[rebind[Int64](lhs).__int__()]
                    >= per_class_scores[rebind[Int64](rhs).__int__()]
                )

            # sort box_idxs based on corresponding scores
            _quicksort[Int64, _greater_than](box_idxs.data, box_idxs.__len__())

            var pred_idx = 0
            while (
                pred_idx < max_output_boxes_per_class
                and num_boxes_remaining > 0
            ):
                let pred = _get_bounding_box(
                    b, box_idxs[pred_idx].__int__(), boxes
                )
                num_boxes_remaining -= 1
                # each output prediction contains 3 values: [batch_index, class_index, box_index]
                func(b, c, box_idxs[pred_idx])

                # at the beginning of this loop box_idxs are sorted such that scores[box_idxs] looks like this:
                # [1st best score, 2nd best score, ..., num_boxes_remaining'th best score, -inf, ..., -inf]
                let num_boxes_curr_pred = num_boxes_remaining
                # iterate over remaining boxes and set the scores of any whose
                # iou is above the threshold to neginf
                for i in range(
                    pred_idx + 1, pred_idx + 1 + num_boxes_curr_pred
                ):
                    let next_box = _get_bounding_box(
                        b, box_idxs[i].__int__(), boxes
                    )

                    if pred.iou(next_box) > iou_threshold.cast[type]():
                        per_class_scores[box_idxs[i].__int__()] = min_or_neginf[
                            type
                        ]()
                        num_boxes_remaining -= 1
                pred_idx += 1
                # don't need to sort all of box_idxs because:
                #   1. the start of the array contains already outputed predictions whose order cannot change
                #   2. the end of the array contains neginf values
                # note we need to use num_boxes_curr_pred instead of num_boxes_remainig
                # because num_boxes_remaining has been adjusted for the high IOU boxes above
                _quicksort[Int64, _greater_than](
                    box_idxs.data + pred_idx, num_boxes_curr_pred
                )

            @always_inline
            fn sorted() -> Bool:
                for i in range(len(box_idxs) - 1):
                    if (
                        per_class_scores[box_idxs[i].to_int()]
                        < per_class_scores[box_idxs[i + 1].to_int()]
                    ):
                        return False
                return True

            debug_assert(
                sorted(), "NonMaxSuppression boxes not sorted correctly"
            )

    box_idxs._del_old()
    per_class_scores._del_old()
