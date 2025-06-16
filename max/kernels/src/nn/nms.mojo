# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import List
from math import iota

from layout import LayoutTensor, RuntimeTuple
from memory import Span

from utils import IndexList


@fieldwise_init
struct BoundingBox[type: DType](Copyable, Movable):
    var nw: SIMD[type, 2]
    var se: SIMD[type, 2]

    fn __init__(
        out self,
        y1: Scalar[type],
        x1: Scalar[type],
        y2: Scalar[type],
        x2: Scalar[type],
    ):
        self.nw = SIMD[type, 2](max(y1, y2), max(x1, x2))
        self.se = SIMD[type, 2](min(y1, y2), min(x1, x2))

    fn iou(self, other: BoundingBox[type]) -> Scalar[type]:
        var intersection_area = self.intersection_area(other)

        var union_area = self.area() + other.area() - intersection_area
        var iou_val = abs(intersection_area) / abs(union_area)
        return iou_val

    fn intersection_area(self, other: BoundingBox[type]) -> Scalar[type]:
        var nw = min(self.nw, other.nw)
        var se = max(self.se, other.se)

        if nw[1] < se[1] or nw[0] < se[0]:
            return 0

        return Self(nw, se).area()

    fn area(self) -> Scalar[type]:
        return (self.se[0] - self.nw[0]) * (self.se[1] - self.nw[1])


@always_inline
fn _get_bounding_box[
    type: DType
](
    batch_size: Int,
    box_idx: Int,
    boxes: LayoutTensor[type, **_],
) -> BoundingBox[type]:
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    var y1 = boxes[batch_size, box_idx, 0][0]
    var x1 = boxes[batch_size, box_idx, 1][0]
    var y2 = boxes[batch_size, box_idx, 2][0]
    var x2 = boxes[batch_size, box_idx, 3][0]
    return BoundingBox(y1, x1, y2, x2)


fn non_max_suppression[
    type: DType
](
    boxes: LayoutTensor[type, **_],
    scores: LayoutTensor[type, **_],
    output: LayoutTensor[mut=True, DType.int64, **_],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
):
    """Buffer semantic overload."""
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    constrained[scores.rank == 3, "scores must be of rank 3"]()
    constrained[output.rank == 2, "output must be of rank 2"]()

    var pred_count = 0

    @parameter
    @always_inline
    fn store_to_outputs(batch_idx: Int64, class_idx: Int64, box_idx: Int64):
        output[pred_count, 0] = batch_idx
        output[pred_count, 1] = class_idx
        output[pred_count, 2] = box_idx
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
    boxes: LayoutTensor[type, **_],
    scores: LayoutTensor[type, **_],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> IndexList[2]:
    """Overload to compute the output shape. Can be removed once the graph compiler
    supports value semantic kernels that allocate their own output."""
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    constrained[scores.rank == 3, "scores must be of rank 3"]()

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

    return IndexList[2](Int(box_pred_count), 3)


fn non_max_suppression[
    type: DType,
    func: fn (Int64, Int64, Int64) capturing [_] -> None,
](
    boxes: LayoutTensor[type, **_],
    scores: LayoutTensor[type, **_],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
):
    """Implements the NonMaxSuppression operator from the ONNX spec https://github.com/onnx/onnx/blob/main/docs/Operators.md#nonmaxsuppression.
    """
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    constrained[scores.rank == 3, "scores must be of rank 3"]()

    var batch_size = Int(boxes.runtime_layout.shape[0])
    var num_boxes = Int(boxes.runtime_layout.shape[1])
    var num_classes = Int(scores.runtime_layout.shape[1])

    debug_assert(
        Int(boxes.runtime_layout.shape[2]) == 4,
        (
            "boxes must be specified with the 2D coords representing the"
            " diagonal corners"
        ),
    )
    debug_assert(
        Int(boxes.runtime_layout.shape[0])
        == Int(scores.runtime_layout.shape[0]),
        "dim 0 of boxes and scores must be equal",
    )
    debug_assert(
        Int(boxes.runtime_layout.shape[1])
        == Int(scores.runtime_layout.shape[2]),
        "boxes and scores must contain the same number of boxes",
    )

    if max_output_boxes_per_class == 0:
        return

    # Allocate the box indices and scores without initializing their elements.
    var box_idxs = List[Int64](unsafe_uninit_length=num_boxes)
    var per_class_scores = List[Scalar[type]](unsafe_uninit_length=num_boxes)

    for b in range(batch_size):
        for c in range(num_classes):
            # entries of per_class_scores_ptr are set to neginf when they no longer
            # correspond to an eligible box
            # this happens when:
            #   1. score does not meet score threshold
            #   2. iou with an existing prediction is above the IOU threshold
            var offset = scores.runtime_layout(
                RuntimeTuple[scores.layout.shape](b, c, 0)
            )
            var per_class_scores_ptr = scores.ptr.offset(offset)

            # filter so that we only consider scores above the threshold
            # reduces the number of box_idxs to sort
            var num_boxes_remaining = 0
            for i in range(num_boxes):
                var score = per_class_scores_ptr.load(i)
                if score > score_threshold.cast[type]():
                    per_class_scores[i] = score
                    num_boxes_remaining += 1
                else:
                    per_class_scores[i] = Scalar[type].MIN

            iota(box_idxs)

            @parameter
            @always_inline
            fn _greater_than(lhs: Int64, rhs: Int64) -> Bool:
                return per_class_scores[Int(lhs)] > per_class_scores[Int(rhs)]

            # sort box_idxs based on corresponding scores
            sort[_greater_than](box_idxs)

            var pred_idx = 0
            while (
                pred_idx < max_output_boxes_per_class
                and num_boxes_remaining > 0
            ):
                var pred = _get_bounding_box(b, Int(box_idxs[pred_idx]), boxes)
                num_boxes_remaining -= 1
                # each output prediction contains 3 values: [batch_index, class_index, box_index]
                func(b, c, box_idxs[pred_idx])

                # at the beginning of this loop box_idxs are sorted such that scores[box_idxs] looks like this:
                # [1st best score, 2nd best score, ..., num_boxes_remaining'th best score, -inf, ..., -inf]
                var num_boxes_curr_pred = num_boxes_remaining
                # iterate over remaining boxes and set the scores of any whose
                # iou is above the threshold to neginf
                for i in range(
                    pred_idx + 1, pred_idx + 1 + num_boxes_curr_pred
                ):
                    var next_box = _get_bounding_box(b, Int(box_idxs[i]), boxes)

                    if pred.iou(next_box) > iou_threshold.cast[type]():
                        per_class_scores[Int(box_idxs[i])] = Scalar[type].MIN
                        num_boxes_remaining -= 1
                pred_idx += 1
                # don't need to sort all of box_idxs because:
                #   1. the start of the array contains already outputted predictions whose order cannot change
                #   2. the end of the array contains neginf values
                # note we need to use num_boxes_curr_pred instead of num_boxes_remaining
                # because num_boxes_remaining has been adjusted for the high IOU boxes above
                sort[_greater_than](
                    Span[box_idxs.T, __origin_of(box_idxs)](
                        ptr=box_idxs.data + pred_idx,
                        length=num_boxes_curr_pred,
                    )
                )

            @always_inline
            fn sorted() -> Bool:
                for i in range(len(box_idxs) - 1):
                    if (
                        per_class_scores[Int(box_idxs[i])]
                        < per_class_scores[Int(box_idxs[i + 1])]
                    ):
                        return False
                return True

            debug_assert(
                sorted(), "NonMaxSuppression boxes not sorted correctly"
            )
