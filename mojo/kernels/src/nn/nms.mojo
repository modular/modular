# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max, min
from tensor import Tensor, TensorShape
from algorithm.sort import _quicksort
from utils.vector import DynamicVector
from math import iota, abs
from math.limit import min_or_neginf


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


fn non_max_suppression[
    type: DType
](
    boxes: Tensor[type],
    scores: Tensor[type],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> Tensor[DType.int64]:
    """Implements the NonMaxSuppression operator from the ONNX spec https://github.com/onnx/onnx/blob/main/docs/Operators.md#nonmaxsuppression"""

    let batch_size = boxes.dim(0)
    let num_boxes = boxes.dim(1)
    let num_classes = scores.dim(1)

    debug_assert(
        boxes.rank() == 3, "boxes rank must be 3 ([batch_size, num_boxes, 4])"
    )
    debug_assert(
        scores.rank() == 3,
        "scores rank must be 3 ([batch_size, num_classes, num_boxes])",
    )
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
        return Tensor[DType.int64]()

    var box_idxs = DynamicVector[Int64](num_boxes)
    box_idxs.resize(num_boxes)

    var per_class_scores = DynamicVector[SIMD[type, 1]](num_boxes)
    per_class_scores.resize(num_boxes)

    @always_inline
    fn get_bounding_box(batch_size: Int, box_idx: Int) -> BoundingBox[type]:
        let y1 = boxes[batch_size, box_idx, 0]
        let x1 = boxes[batch_size, box_idx, 1]
        let y2 = boxes[batch_size, box_idx, 2]
        let x2 = boxes[batch_size, box_idx, 3]
        return BoundingBox(y1, x1, y2, x2)

    var output_predictions = DynamicVector[Int64]()

    for b in range(batch_size):
        for c in range(num_classes):
            let offset = scores._compute_linear_offset(b, c, 0)
            # entries of per_class_scores_ptr are set to neginf when they no longer
            # correspond to an eligible box
            # this happens when:
            #   1. score does not meet score threshold
            #   2. iou with an existing prediction is above the IOU threshold
            let per_class_scores_ptr = scores.data().offset(offset)

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
                    > per_class_scores[rebind[Int64](rhs).__int__()]
                )

            # sort box_idxs based on corresponding scores
            _quicksort[Int64, _greater_than](box_idxs.data, box_idxs.__len__())

            var pred_idx = 0
            while (
                pred_idx < max_output_boxes_per_class
                and num_boxes_remaining > 0
            ):
                let pred = get_bounding_box(b, box_idxs[pred_idx].__int__())
                num_boxes_remaining -= 1
                # each output prediction contains 3 values: [batch_index, class_index, box_index]
                output_predictions.push_back(b)
                output_predictions.push_back(c)
                output_predictions.push_back(box_idxs[pred_idx])

                # at the beginning of this loop box_idxs are sorted such that scores[box_idxs] looks like this:
                # [1st best score, 2nd best score, ..., num_boxes_remaining'th best score, -inf, ..., -inf]
                let num_boxes_curr_pred = num_boxes_remaining
                # iterate over remaining boxes and set the scores of any whose
                # iou is above the threshold to neginf
                for i in range(
                    pred_idx + 1, pred_idx + 1 + num_boxes_curr_pred
                ):
                    let next_box = get_bounding_box(b, box_idxs[i].__int__())

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

    box_idxs._del_old()
    per_class_scores._del_old()

    let output_shape = TensorShape(output_predictions.__len__() // 3, 3)
    return Tensor[DType.int64](
        rebind[DTypePointer[DType.int64]](output_predictions.data), output_shape
    )
