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

from math import iota

from layout import LayoutTensor, RuntimeTuple

from utils import IndexList


@fieldwise_init
struct BoundingBox[dtype: DType](ImplicitlyCopyable, Movable):
    """Represents a 2D bounding box for object detection.

    The box is stored using two corner points: `nw` and `se`.
    **Note:** In this implementation, `nw` stores the maximum coordinates (max y, max x)
    and `se` stores the minimum coordinates (min y, min x). This differs from the typical
    interpretation of "northwest" (usually min x, max y) and "southeast" (usually max x, min y).
    This representation allows efficient computation of intersection and union areas.

    Parameters:
        dtype: The data type for coordinate values.

    Fields:
        nw: Corner storing the maximum coordinates (max y, max x).
        se: Corner storing the minimum coordinates (min y, min x).
    """

    var nw: SIMD[dtype, 2]
    var se: SIMD[dtype, 2]

    fn __init__(
        out self,
        y1: Scalar[dtype],
        x1: Scalar[dtype],
        y2: Scalar[dtype],
        x2: Scalar[dtype],
    ):
        """Initialize a bounding box from two diagonal corner coordinates.

        Args:
            y1: Y-coordinate of first corner.
            x1: X-coordinate of first corner.
            y2: Y-coordinate of second corner.
            x2: X-coordinate of second corner.

        Note:
            The corners are automatically ordered to ensure nw contains the
            maximum coordinates and se contains the minimum coordinates.
        """
        self.nw = SIMD[dtype, 2](max(y1, y2), max(x1, x2))
        self.se = SIMD[dtype, 2](min(y1, y2), min(x1, x2))

    fn iou(self, other: BoundingBox[dtype]) -> Scalar[dtype]:
        """Calculate Intersection over Union (IoU) with another bounding box.

        Args:
            other: The other bounding box to compare with.

        Returns:
            The IoU value, ranging from 0 (no overlap) to 1 (perfect overlap).
        """
        var intersection_area = self.intersection_area(other)

        var union_area = self.area() + other.area() - intersection_area
        var iou_val = abs(intersection_area) / abs(union_area)
        return iou_val

    fn intersection_area(self, other: BoundingBox[dtype]) -> Scalar[dtype]:
        """Calculate the area of intersection with another bounding box.

        Args:
            other: The other bounding box to intersect with.

        Returns:
            The intersection area, or 0 if boxes don't overlap.
        """
        var nw = min(self.nw, other.nw)
        var se = max(self.se, other.se)

        # Check if boxes don't overlap (invalid intersection)
        if nw[1] < se[1] or nw[0] < se[0]:
            return 0

        return Self(nw, se).area()

    fn area(self) -> Scalar[dtype]:
        """Calculate the area of this bounding box.

        Returns:
            The area of the box.
        """
        return (self.se[0] - self.nw[0]) * (self.se[1] - self.nw[1])


@always_inline
fn _get_bounding_box[
    dtype: DType
](
    batch_size: Int,
    box_idx: Int,
    boxes: LayoutTensor[dtype, **_],
) -> BoundingBox[dtype]:
    """Extract a bounding box from a tensor of boxes.

    Args:
        batch_size: The batch index to extract from.
        box_idx: The box index within the batch.
        boxes: A rank-3 tensor containing boxes with shape (batch, num_boxes, 4).
               The last dimension contains [y1, x1, y2, x2] coordinates.

    Returns:
        A BoundingBox instance constructed from the extracted coordinates.
    """
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    var y1 = boxes[batch_size, box_idx, 0][0]
    var x1 = boxes[batch_size, box_idx, 1][0]
    var y2 = boxes[batch_size, box_idx, 2][0]
    var x2 = boxes[batch_size, box_idx, 3][0]
    return BoundingBox(y1, x1, y2, x2)


fn non_max_suppression[
    dtype: DType
](
    boxes: LayoutTensor[dtype, **_],
    scores: LayoutTensor[dtype, **_],
    output: LayoutTensor[mut=True, DType.int64, **_],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
):
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.

    This is a buffer semantic overload that writes results directly to an output tensor.
    NMS iteratively selects boxes with highest scores while suppressing nearby boxes
    with high overlap (IoU).

    Parameters:
        dtype: The data type for box coordinates and scores.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape (batch, num_boxes, 4).
               Each box is [y1, x1, y2, x2].
        scores: Rank-3 tensor of scores with shape (batch, num_classes, num_boxes).
        output: Rank-2 output tensor to store selected boxes as (N, 3) where each
                row is [batch_idx, class_idx, box_idx].
        max_output_boxes_per_class: Maximum number of boxes to select per class.
        iou_threshold: IoU threshold for suppression. Boxes with IoU > threshold
                       are suppressed.
        score_threshold: Minimum score threshold. Boxes with score < threshold
                        are filtered out.
    """
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    constrained[scores.rank == 3, "scores must be of rank 3"]()
    constrained[output.rank == 2, "output must be of rank 2"]()

    var pred_count = 0

    @parameter
    @always_inline
    fn store_to_outputs(batch_idx: Int64, class_idx: Int64, box_idx: Int64):
        """Store selected box indices to output tensor."""
        output[pred_count, 0] = batch_idx
        output[pred_count, 1] = class_idx
        output[pred_count, 2] = box_idx
        pred_count += 1

    non_max_suppression[dtype, store_to_outputs](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )


fn non_max_suppression_shape_func[
    dtype: DType
](
    boxes: LayoutTensor[dtype, **_],
    scores: LayoutTensor[dtype, **_],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> IndexList[2]:
    """Compute the output shape for NMS without allocating the output buffer.

    This function performs a dry-run of NMS to determine how many boxes will be
    selected, allowing proper output buffer allocation. Can be removed once the
    graph compiler supports value semantic kernels that allocate their own output.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape (batch, num_boxes, 4).
        scores: Rank-3 tensor of scores with shape (batch, num_classes, num_boxes).
        max_output_boxes_per_class: Maximum number of boxes to select per class.
        iou_threshold: IoU threshold for suppression.
        score_threshold: Minimum score threshold.

    Returns:
        A 2-element IndexList specifying the output shape (num_selected_boxes, 3).
    """
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    constrained[scores.rank == 3, "scores must be of rank 3"]()

    var box_pred_count: Int64 = 0

    @parameter
    @always_inline
    fn incr_pred_count(batch_idx: Int64, class_idx: Int64, box_idx: Int64):
        """Count selected boxes without storing them."""
        box_pred_count += 1

    non_max_suppression[dtype, incr_pred_count](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )

    return IndexList[2](Int(box_pred_count), 3)


fn non_max_suppression[
    dtype: DType,
    func: fn (Int64, Int64, Int64) capturing [_] -> None,
](
    boxes: LayoutTensor[dtype, **_],
    scores: LayoutTensor[dtype, **_],
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
    var per_class_scores = List[Scalar[dtype]](unsafe_uninit_length=num_boxes)

    for b in range(batch_size):
        for c in range(num_classes):
            # Entries of per_class_scores_ptr are set to neginf when they no longer
            # correspond to an eligible box
            # this happens when:
            #   1. Score does not meet score threshold (filtered out initially)
            #   2. IoU with an already-selected boc is above IOU threshold (suppressed)
            var offset = scores.runtime_layout(
                RuntimeTuple[scores.layout.shape](b, c, 0)
            )
            var per_class_scores_ptr = scores.ptr.offset(offset)

            # Filter boxes by score threshold
            # This reduces the number of boxes to sort and process
            var num_boxes_remaining = 0
            for i in range(num_boxes):
                var score = per_class_scores_ptr.load(i)
                if score > score_threshold.cast[dtype]():
                    per_class_scores[i] = score
                    num_boxes_remaining += 1
                else:
                    per_class_scores[i] = Scalar[dtype].MIN  # ~ -inf

            # Initialize box indices [0, 1, 2, ..., num_boxes-1]
            iota(box_idxs)

            @parameter
            @always_inline
            fn _greater_than(lhs: Int64, rhs: Int64) -> Bool:
                """Compare boxes by their scores in descending order."""
                return per_class_scores[Int(lhs)] > per_class_scores[Int(rhs)]

            # Sort box indices by descending score
            sort[_greater_than](box_idxs)

            # Iteratively select boxes and suppress overlapping ones
            var pred_idx = 0
            while (
                pred_idx < max_output_boxes_per_class
                and num_boxes_remaining > 0
            ):
                # Select the highest-scoring remaining box
                var pred = _get_bounding_box(b, Int(box_idxs[pred_idx]), boxes)
                num_boxes_remaining -= 1

                # Each output prediction contains 3 values: [batch_index, class_index, box_index]
                func(b, c, box_idxs[pred_idx])

                # At the beginning of this loop box_idxs are sorted such that scores[box_idxs] looks like this:
                # [1st best score, 2nd best score, ..., num_boxes_remaining'th best score, -inf, ..., -inf]
                var num_boxes_curr_pred = num_boxes_remaining

                # Suppress boxes with high IoU overlap
                # Iterate over remaining candidate boxes and mark those with IoU > threshold
                # as suppressed by setting their scores to MIN
                for i in range(
                    pred_idx + 1, pred_idx + 1 + num_boxes_curr_pred
                ):
                    var next_box = _get_bounding_box(b, Int(box_idxs[i]), boxes)

                    if pred.iou(next_box) > iou_threshold.cast[dtype]():
                        per_class_scores[Int(box_idxs[i])] = Scalar[dtype].MIN
                        num_boxes_remaining -= 1

                pred_idx += 1

                # We don't need to sort the entire array because:
                #   1. The start contains already-selected boxes (order doesn't matter)
                #   2. The end contains suppressed boxes with score=-inf (order doesn't matter)
                # Note: Use num_boxes_curr_pred (not num_boxes_remaining) because it
                # represents the count before we marked boxes as suppressed above
                sort[_greater_than](
                    Span[box_idxs.T, __origin_of(box_idxs)](
                        ptr=box_idxs.unsafe_ptr() + pred_idx,
                        length=UInt(num_boxes_curr_pred),
                    )
                )

            @always_inline
            @parameter
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
